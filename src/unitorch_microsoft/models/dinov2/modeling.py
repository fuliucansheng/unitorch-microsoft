# Copyright (c) MICROSOFT.
# Licensed under the MIT License.
import os
import torch
import json
import torch.nn as nn
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torch import autocast
from transformers.utils import is_remote_url
from transformers.models.dinov2 import Dinov2Config, Dinov2Model
from peft import LoraConfig
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models import GenericModel
from unitorch.models.peft import (
    GenericPeftModel,
    PeftModelForSequenceClassification,
    PeftWeightLoaderMixin,
)
from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
    register_model,
)
from unitorch.cli.models import ClassificationOutputs, GenerationOutputs, LossOutputs
from unitorch.cli.models.dinov2 import pretrained_dinov2_infos


class DinoV2ForImageMatching(GenericModel):
    def __init__(
        self,
        query_config_path: str,
        doc_config_path: str,
    ):
        super().__init__()
        query_config = Dinov2Config.from_json_file(query_config_path)
        self.query_encoder = Dinov2Model(query_config)
        doc_config = Dinov2Config.from_json_file(doc_config_path)
        self.doc_encoder = Dinov2Model(doc_config)

        self.init_weights()

    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(
        self,
        query_pixel_values=None,
        doc_pixel_values=None,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        query_outputs = self.query_encoder(pixel_values=query_pixel_values)[0]
        doc_outputs = self.doc_encoder(pixel_values=doc_pixel_values)[0]

        query_embeds = query_outputs[:, 0]
        doc_embeds = doc_outputs[:, 0]

        return (query_embeds, doc_embeds)


@register_model("microsoft/model/matching/peft/lora/dinov2")
class DinoV2LoraForImageMatching(GenericPeftModel, PeftWeightLoaderMixin):
    prefix_keys_in_state_dict = {
        "^(?!peft_model\.base_model\.model\.).*": "peft_model.base_model.model."
    }
    replace_keys_in_state_dict = {
        "attention.key.weight": "attention.key.base_layer.weight",
        "attention.key.bias": "attention.key.base_layer.bias",
        "attention.value.weight": "attention.value.base_layer.weight",
        "attention.value.bias": "attention.value.base_layer.bias",
    }
    modules_to_save_checkpoints = ["lora", "output_projection", "classifier"]
    replace_keys_in_peft_state_dict = {
        ".weight": ".base_layer.weight",
        ".bias": ".base_layer.bias",
    }

    def __init__(
        self,
        query_config_path: str,
        doc_config_path: str,
        lora_r: Optional[int] = 16,
        lora_alpha: Optional[int] = 32,
        lora_dropout: Optional[float] = 0.05,
        fan_in_fan_out: Optional[bool] = True,
        target_modules: Optional[Union[List[str], str]] = [
            "attention.key",
            "attention.value",
        ],
        output_embed_dim: Optional[int] = None,
    ):
        super().__init__()
        self.peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            target_modules=target_modules,
        )
        self.peft_model = PeftModelForSequenceClassification(
            DinoV2ForImageMatching(
                query_config_path,
                doc_config_path,
            ),
            self.peft_config,
        )
        self.output_embed_dim = output_embed_dim
        if output_embed_dim is not None:
            self.output_projection = nn.Linear(
                projection_dim,
                output_embed_dim,
            )
        else:
            self.output_projection = None
        self.classifier = nn.Linear(1, 1)

        self.init_weights()
        self.classifier.weight.data.fill_(5.0)

    @classmethod
    @add_default_section_for_init("microsoft/model/matching/peft/lora/dinov2")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/model/matching/peft/lora/dinov2")
        pretrained_name = config.getoption("pretrained_name", "dinov2-base")
        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_dinov2_infos, pretrained_name, "config"),
        )
        query_config_path = config.getoption("query_config_path", config_path)
        query_config_path = cached_path(query_config_path)
        doc_config_path = config.getoption("doc_config_path", config_path)
        doc_config_path = cached_path(doc_config_path)
        lora_r = config.getoption("lora_r", 16)
        lora_alpha = config.getoption("lora_alpha", 32)
        lora_dropout = config.getoption("lora_dropout", 0.05)
        fan_in_fan_out = config.getoption("fan_in_fan_out", True)
        target_modules = config.getoption(
            "target_modules", ["attention.key", "attention.value"]
        )
        output_embed_dim = config.getoption("output_embed_dim", None)

        inst = cls(
            query_config_path=query_config_path,
            doc_config_path=doc_config_path,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            target_modules=target_modules,
            output_embed_dim=output_embed_dim,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_dinov2_infos, pretrained_name, "weight"),
            check_none=False,
        )
        if weight_path is not None:
            inst.from_pretrained(weight_path)

        pretrained_lora_weight_path = config.getoption(
            "pretrained_lora_weight_path", None
        )
        pretrained_lora_weight = config.getoption("pretrained_lora_weight", 1.0)
        pretrained_lora_alpha = config.getoption("pretrained_lora_alpha", 32.0)
        if pretrained_lora_weight_path is not None:
            inst.load_lora_weights(
                pretrained_lora_weight_path,
                lora_weights=pretrained_lora_weight,
                lora_alphas=pretrained_lora_alpha,
            )

        return inst

    def from_pretrained(self, weight_path):
        weight_path = cached_path(weight_path)
        if not os.path.exists(weight_path):
            return
        state_dict = torch.load(weight_path, map_location="cpu")
        _keys = [
            key
            for key in state_dict.keys()
            if any(
                key.startswith(prefix)
                for prefix in ["embeddings.", "layernorm.", "encoder."]
            )
        ]
        for _key in _keys:
            _value = state_dict.pop(_key)
            state_dict["query_encoder." + _key] = _value
            state_dict["doc_encoder." + _key] = _value

        super().from_pretrained(state_dict=state_dict)

    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(
        self,
        query_pixel_values=None,
        doc_pixel_values=None,
    ):
        query_embeds, doc_embeds = self.peft_model(
            query_pixel_values=query_pixel_values,
            doc_pixel_values=doc_pixel_values,
            return_dict=False,
        )
        if self.output_projection is not None:
            query_embeds = self.output_projection(query_embeds)
            doc_embeds = self.output_projection(doc_embeds)

        query_embeds = query_embeds / query_embeds.norm(dim=-1, keepdim=True)
        doc_embeds = doc_embeds / doc_embeds.norm(dim=-1, keepdim=True)
        scores = torch.sum(query_embeds * doc_embeds, dim=-1, keepdim=True)

        outputs = self.classifier(scores)
        return ClassificationOutputs(outputs=outputs)
