# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast

from transformers.configuration_utils import PretrainedConfig
from transformers.activations import silu as SiLUActivation, gelu, gelu_new
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
)
from transformers.modeling_utils import PreTrainedModel, prune_linear_layer
from transformers.models.bert.modeling_bert import (
    BertIntermediate,
    BertOutput,
    BertPooler,
    BertSelfOutput,
)
from transformers.models.roberta.modeling_roberta import (
    RobertaEmbeddings,
    RobertaLMHead,
)
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models import GenericModel
from unitorch.cli import (
    add_default_section_for_init,
    add_default_section_for_function,
    register_model,
)
from unitorch.cli.models import ClassificationOutputs, LossOutputs
from unitorch_microsoft import cached_path
from unitorch_microsoft.models.tulr import (
    pretrained_tulr_infos,
    TULRV6Config,
    TULRV6Model,
)


@register_model("microsoft/msan/l1/model/classification/tulr/v6")
class TULRV6ForClassification(GenericModel):
    def __init__(
        self,
        config_path: str,
        num_classes: Optional[int] = 1,
        gradient_checkpointing: Optional[bool] = False,
    ):
        super().__init__()
        self.config = TULRV6Config.from_json_file(config_path)
        self.config.gradient_checkpointing = gradient_checkpointing
        self.tulrv6 = TULRV6Model(self.config)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        self.init_weights()

    @classmethod
    @add_default_section_for_init("microsoft/msan/l1/model/classification/tulr/v6")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/msan/l1/model/classification/tulr/v6")
        pretrained_name = config.getoption("pretrained_name", "tulrv6-base")
        config_path = config.getoption("config_path", None)

        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_tulr_infos, pretrained_name, "config"),
        )
        config_path = cached_path(config_path)

        num_classes = config.getoption("num_classes", 1)

        gradient_checkpointing = config.getoption("gradient_checkpointing", False)

        inst = cls(config_path, num_classes, gradient_checkpointing)

        weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            weight_path,
            nested_dict_value(pretrained_tulr_infos, pretrained_name, "weight"),
            check_none=False,
        )
        if weight_path is not None:
            inst.from_pretrained(weight_path)

        return inst

    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        outputs = self.tulrv6(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return ClassificationOutputs(outputs=logits)


@register_model("microsoft/msan/l1/model/classification/tulr/v6/v2")
class TULRV6ForClassificationV2(GenericModel):
    def __init__(
        self,
        config_path: str,
        num_classes: Optional[int] = 1,
        gradient_checkpointing: Optional[bool] = False,
    ):
        super().__init__()
        self.config = TULRV6Config.from_json_file(config_path)
        self.config.gradient_checkpointing = gradient_checkpointing
        self.tulrv6 = TULRV6Model(self.config)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.attn = nn.Linear(self.config.hidden_size, 1, bias=False)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        self.init_weights()

    @classmethod
    @add_default_section_for_init("microsoft/msan/l1/model/classification/tulr/v6/v2")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/msan/l1/model/classification/tulr/v6/v2")
        pretrained_name = config.getoption("pretrained_name", "tulrv6-base")
        config_path = config.getoption("config_path", None)

        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_tulr_infos, pretrained_name, "config"),
        )
        config_path = cached_path(config_path)

        num_classes = config.getoption("num_classes", 1)

        gradient_checkpointing = config.getoption("gradient_checkpointing", False)

        inst = cls(config_path, num_classes, gradient_checkpointing)

        weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            weight_path,
            nested_dict_value(pretrained_tulr_infos, pretrained_name, "weight"),
            check_none=False,
        )
        if weight_path is not None:
            inst.from_pretrained(weight_path)

        return inst

    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        num_attention_mask: Optional[torch.Tensor] = None,
    ):
        batch, num, seq_len = input_ids.shape
        outputs = self.tulrv6(
            input_ids.view(-1, seq_len),
            attention_mask=attention_mask.view(-1, seq_len),
            token_type_ids=token_type_ids.view(-1, seq_len),
            position_ids=position_ids.view(-1, seq_len),
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        attention_score = self.attn(pooled_output).view(batch, num)
        attention_score = attention_score + (1 - num_attention_mask) * -10000.0
        attention_score = F.softmax(attention_score, dim=-1)
        final_output = torch.bmm(
            attention_score.unsqueeze(1), pooled_output.view(batch, num, -1)
        ).squeeze(1)

        logits = self.classifier(final_output)
        return ClassificationOutputs(outputs=logits)


@register_model("microsoft/msan/l1/model/classification/tulr/v6/v3")
class TULRV6ForClassificationV3(GenericModel):
    def __init__(
        self,
        config_path: str,
        num_tgs: Optional[int] = 50,
        num_demands: Optional[int] = 10,
        num_markets: Optional[int] = 150,
        num_positions: Optional[int] = 13000,
        projection_dim: Optional[int] = 128,
        num_classes: Optional[int] = 1,
        gradient_checkpointing: Optional[bool] = False,
    ):
        super().__init__()
        self.config = TULRV6Config.from_json_file(config_path)
        self.config.gradient_checkpointing = gradient_checkpointing
        self.tulrv6 = TULRV6Model(self.config)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.attn = nn.Linear(self.config.hidden_size, 1, bias=False)
        self.position_embedding = nn.Embedding(num_positions, projection_dim)
        self.position_layer_norm = nn.LayerNorm(projection_dim)
        self.position = nn.Linear(projection_dim, num_classes)
        self.tg_embedding = nn.Embedding(num_tgs, projection_dim)
        self.tg_layer_norm = nn.LayerNorm(projection_dim)
        self.tg = nn.Linear(projection_dim, num_classes)
        self.demand_embedding = nn.Embedding(num_demands, projection_dim)
        self.demand_layer_norm = nn.LayerNorm(projection_dim)
        self.demand = nn.Linear(projection_dim, num_classes)
        self.market_embedding = nn.Embedding(num_markets, projection_dim)
        self.market_layer_norm = nn.LayerNorm(projection_dim)
        self.market = nn.Linear(projection_dim, num_classes)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        self.init_weights()

    @classmethod
    @add_default_section_for_init("microsoft/msan/l1/model/classification/tulr/v6/v3")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/msan/l1/model/classification/tulr/v6/v3")
        pretrained_name = config.getoption("pretrained_name", "tulrv6-base")
        config_path = config.getoption("config_path", None)

        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_tulr_infos, pretrained_name, "config"),
        )
        config_path = cached_path(config_path)

        num_tgs = config.getoption("num_tgs", 50)
        num_demands = config.getoption("num_demands", 10)
        num_markets = config.getoption("num_markets", 150)
        num_positions = config.getoption("num_positions", 13000)
        projection_dim = config.getoption("projection_dim", 128)
        num_classes = config.getoption("num_classes", 1)

        gradient_checkpointing = config.getoption("gradient_checkpointing", False)

        inst = cls(
            config_path,
            num_tgs=num_tgs,
            num_demands=num_demands,
            num_markets=num_markets,
            num_positions=num_positions,
            projection_dim=projection_dim,
            num_classes=num_classes,
            gradient_checkpointing=gradient_checkpointing,
        )

        weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            weight_path,
            nested_dict_value(pretrained_tulr_infos, pretrained_name, "weight"),
            check_none=False,
        )
        if weight_path is not None:
            inst.from_pretrained(weight_path)

        return inst

    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(
        self,
        input_ids: torch.Tensor,
        tg_ids: torch.Tensor,
        demand_ids: torch.Tensor,
        market_ids: torch.Tensor,
        pos_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        num_attention_mask: Optional[torch.Tensor] = None,
    ):
        batch, num, seq_len = input_ids.shape
        outputs = self.tulrv6(
            input_ids.view(-1, seq_len),
            attention_mask=attention_mask.view(-1, seq_len),
            token_type_ids=token_type_ids.view(-1, seq_len),
            position_ids=position_ids.view(-1, seq_len),
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        attention_score = self.attn(pooled_output).view(batch, num)
        attention_score = attention_score + (1 - num_attention_mask) * -10000.0
        attention_score = F.softmax(attention_score, dim=-1)
        final_output = torch.bmm(
            attention_score.unsqueeze(1), pooled_output.view(batch, num, -1)
        ).squeeze(1)

        logits = self.classifier(final_output)
        pos_emb = self.position_layer_norm(self.position_embedding(pos_ids))
        tg_emb = self.tg_layer_norm(self.tg_embedding(tg_ids))
        demand_emb = self.demand_layer_norm(self.demand_embedding(demand_ids))
        market_emb = self.market_layer_norm(self.market_embedding(market_ids))
        logits += (
            self.position(pos_emb)
            + self.tg(tg_emb)
            + self.demand(demand_emb)
            + self.market(market_emb)
        )
        return ClassificationOutputs(outputs=logits)
