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


@register_model("microsoft/adinsights/model/classification/tulr/v6")
class TULRV6ForClassification(GenericModel):
    def __init__(
        self,
        config_path: str,
        num_positions: Optional[int] = None,
        num_types: Optional[int] = None,
        projection_dim: Optional[int] = 128,
        num_classes: Optional[int] = 1,
        gradient_checkpointing: Optional[bool] = False,
    ):
        super().__init__()
        self.config = TULRV6Config.from_json_file(config_path)
        self.config.gradient_checkpointing = gradient_checkpointing
        self.tulrv6 = TULRV6Model(self.config)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        self.num_positions = num_positions
        self.num_types = num_types
        if num_positions is not None:
            self.position_embedding = nn.Embedding(num_positions, projection_dim)
            self.position_layer_norm = nn.LayerNorm(projection_dim)
            self.position = nn.Linear(projection_dim, num_classes)

        if num_types is not None:
            self.type_embedding = nn.Embedding(num_types, projection_dim)
            self.type_layer_norm = nn.LayerNorm(projection_dim)
            self.type = nn.Linear(projection_dim, num_classes)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        self.init_weights()

    @classmethod
    @add_default_section_for_init("microsoft/adinsights/model/classification/tulr/v6")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/adinsights/model/classification/tulr/v6")
        pretrained_name = config.getoption("pretrained_name", "tulrv6-base")
        config_path = config.getoption("config_path", None)

        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_tulr_infos, pretrained_name, "config"),
        )
        config_path = cached_path(config_path)

        num_positions = config.getoption("num_positions", 50)
        num_types = config.getoption("num_types", None)
        projection_dim = config.getoption("projection_dim", 128)
        num_classes = config.getoption("num_classes", 1)

        gradient_checkpointing = config.getoption("gradient_checkpointing", False)

        inst = cls(
            config_path,
            num_positions=num_positions,
            num_types=num_types,
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
        pos_ids: Optional[torch.Tensor] = None,
        type_ids: Optional[torch.Tensor] = None,
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
        if pos_ids is not None and self.num_positions is not None:
            pos_emb = self.position_layer_norm(self.position_embedding(pos_ids))
            logits += self.position(pos_emb)
        if type_ids is not None and self.num_types is not None:
            type_emb = self.type_layer_norm(self.type_embedding(type_ids))
            logits += self.type(type_emb)
        return ClassificationOutputs(outputs=logits)
