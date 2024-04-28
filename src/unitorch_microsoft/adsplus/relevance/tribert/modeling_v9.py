# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from transformers.models.bert.modeling_bert import (
    BertConfig,
    BertEmbeddings,
    BertEncoder,
    BertPooler,
    BertModel,
    BertPreTrainedModel,
)
from unitorch_microsoft.adsplus.relevance.tribert.modeling_v8 import MultiTriPostLayer
from unitorch.models import GenericModel
from unitorch.utils import pop_value, nested_dict_value
from unitorch.cli import (
    add_default_section_for_init,
    add_default_section_for_function,
    register_model,
)
from unitorch.cli.models import ClassificationOutputs
from unitorch_microsoft import cached_path
from unitorch_microsoft.adsplus.relevance.tribert import pretrained_bert_infos


class TriTwinBertEncoder(nn.Module):
    def __init__(self, config, add_pooling_layer):
        super().__init__()
        self.bert = BertModel(
            config,
            add_pooling_layer=add_pooling_layer,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        return outputs


@register_model("microsoft/adsplus/relevance/classification/tribert/v9")
class TribertForClassification(GenericModel):
    replace_keys_in_state_dict = {"gamma": "weight", "beta": "bias"}
    prefix_keys_in_state_dict = {}

    def __init__(
        self,
        config_path,
        num_tasks: int = 1,
        num_classes: int = 1,
        sim4score: str = "cosine",
        pooltype: str = "bert",
        hidden_downscale_size: int = 128,
        reslayer_use_bn: bool = True,
        reslayer_downscale_size: int = 128,
        reslayer_hidden_size: int = 64,
    ):
        super().__init__()
        self.config = BertConfig.from_json_file(config_path)
        self.config.gradient_checkpointing = False
        self.query_encoder = TriTwinBertEncoder(
            self.config,
            add_pooling_layer=pooltype == "bert",
        )

        self.doc_encoder = TriTwinBertEncoder(
            self.config,
            add_pooling_layer=pooltype == "bert",
        )

        self.ads_encoder = TriTwinBertEncoder(
            self.config,
            add_pooling_layer=pooltype == "bert",
        )

        self.postlayer = MultiTriPostLayer(
            num_tasks=num_tasks,
            num_classes=num_classes,
            sim4score=sim4score,
            pooltype=pooltype,
            hidden_size=self.config.hidden_size,
            hidden_downscale_size=hidden_downscale_size,
            reslayer_use_bn=reslayer_use_bn,
            reslayer_downscale_size=reslayer_downscale_size,
            reslayer_hidden_size=reslayer_hidden_size,
        )

        self.init_weights()

    def from_pretrained(self, weight_path):
        if not os.path.exists(weight_path):
            return
        state_dict = torch.load(weight_path, map_location="cpu")
        _keys = [key for key in state_dict.keys() if key.startswith("bert")]
        for _key in _keys:
            _value = state_dict.pop(_key)
            state_dict["query_encoder." + _key] = _value
            state_dict["doc_encoder." + _key] = _value
            state_dict["ads_encoder." + _key] = _value

        super().from_pretrained(state_dict=state_dict)

    @classmethod
    @add_default_section_for_init(
        "microsoft/adsplus/relevance/classification/tribert/v9"
    )
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section(
            "microsoft/adsplus/relevance/classification/tribert/v9"
        )
        pretrained_name = config.getoption("pretrained_name", "default-bert")
        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_bert_infos, pretrained_name, "config"),
        )
        config_path = cached_path(config_path)
        num_tasks = config.getoption("num_tasks", 1)
        num_classes = config.getoption("num_classes", 1)
        sim4score = config.getoption("sim4score", "cosine")
        pooltype = config.getoption("pooltype", "bert")
        hidden_downscale_size = config.getoption("hidden_downscale_size", 128)
        reslayer_use_bn = config.getoption("reslayer_use_bn", True)
        reslayer_downscale_size = config.getoption("reslayer_downscale_size", 128)
        reslayer_hidden_size = config.getoption("reslayer_hidden_size", 64)
        inst = cls(
            config_path=config_path,
            num_tasks=num_tasks,
            num_classes=num_classes,
            sim4score=sim4score,
            pooltype=pooltype,
            hidden_downscale_size=hidden_downscale_size,
            reslayer_use_bn=reslayer_use_bn,
            reslayer_downscale_size=reslayer_downscale_size,
            reslayer_hidden_size=reslayer_hidden_size,
        )

        weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            weight_path,
            nested_dict_value(pretrained_bert_infos, pretrained_name, "weight"),
            check_none=False,
        )
        if weight_path is not None:
            weight_path = cached_path(weight_path)
            inst.from_pretrained(weight_path)
        return inst

    @autocast()
    def forward(
        self,
        task,
        query_input_ids,
        query_attention_mask,
        query_token_type_ids,
        query_position_ids,
        doc_input_ids,
        doc_attention_mask,
        doc_token_type_ids,
        doc_position_ids,
        ads_input_ids,
        ads_attention_mask,
        ads_token_type_ids,
        ads_position_ids,
    ):
        query_outputs = self.query_encoder(
            query_input_ids,
            query_attention_mask,
            query_token_type_ids,
            query_position_ids,
        )
        doc_outputs = self.doc_encoder(
            doc_input_ids, doc_attention_mask, doc_token_type_ids, doc_position_ids
        )
        ads_outputs = self.ads_encoder(
            ads_input_ids, ads_attention_mask, ads_token_type_ids, ads_position_ids
        )
        outputs = self.postlayer(
            task=task,
            qseq_in=query_outputs.last_hidden_state,
            qpool_in=query_outputs.pooler_output,
            q_mask=query_attention_mask,
            dseq_in=doc_outputs.last_hidden_state,
            dpool_in=doc_outputs.pooler_output,
            d_mask=doc_attention_mask,
            adsseq_in=ads_outputs.last_hidden_state,
            adspool_in=ads_outputs.pooler_output,
            ads_mask=ads_attention_mask,
        )
        return ClassificationOutputs(outputs=outputs)
