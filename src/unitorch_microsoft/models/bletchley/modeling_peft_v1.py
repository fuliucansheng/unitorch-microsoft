# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import random
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torch.cuda.amp import autocast
from peft import LoraConfig
from unitorch.models import GenericModel
from unitorch.models.peft import GenericPeftModel, PeftModelForSequenceClassification
from unitorch.cli.models import (
    EmbeddingOutputs,
    LossOutputs,
    ClassificationOutputs,
)
from unitorch.cli import (
    add_default_section_for_init,
    add_default_section_for_function,
    cached_path,
    register_model,
)
from unitorch_microsoft.models.bletchley.modeling_v1 import (
    BletchleyTextEncoder,
    BletchleyImageEncoder,
    get_bletchley_text_config,
    get_bletchley_image_config,
)


class BletchleyForMatching(GenericModel):
    replace_keys_in_state_dict = {
        "text_encoder.projection": "text_projection",
        "image_encoder.projection": "image_projection",
    }

    def __init__(
        self,
        config_type: str,
        projection_dim: Optional[int] = 1024,
        gradient_checkpointing: Optional[bool] = False,
    ):
        super().__init__()
        text_config = get_bletchley_text_config(config_type, gradient_checkpointing)
        image_config = get_bletchley_image_config(config_type, gradient_checkpointing)

        self.text_embed_dim = text_config.hidden_size
        self.image_embed_dim = image_config.hidden_size

        self.text_encoder = BletchleyTextEncoder(
            text_config, add_projection_layer=False
        )
        self.image_encoder = BletchleyImageEncoder(
            image_config, add_projection_layer=False
        )

        self.image_projection = nn.Linear(
            self.image_embed_dim,
            projection_dim,
        )
        self.text_projection = nn.Linear(
            self.text_embed_dim,
            projection_dim,
        )

        self.init_weights()

    @autocast()
    def forward(
        self,
        input_ids=None,
        images=None,
        attention_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        image_outputs = self.image_encoder(
            images=images,
        )
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        image_embeds = image_outputs[:, 0]
        image_embeds = self.image_projection(image_embeds)
        text_embeds = text_outputs[:, 0]
        text_embeds = self.text_projection(text_embeds)

        return (text_embeds, image_embeds)


@register_model("microsoft/model/matching/peft/lora/bletchley/v1")
class BletchleyLoraForMatching(GenericPeftModel):
    prefix_keys_in_state_dict = {
        "^(?!peft_model\.base_model\.model\.).*": "peft_model.base_model.model."
    }
    replace_keys_in_state_dict = {
        "query.weight": "query.base_layer.weight",
        "query.bias": "query.base_layer.bias",
        "value.weight": "value.base_layer.weight",
        "value.bias": "value.base_layer.bias",
        "text_encoder.projection": "text_projection",
        "image_encoder.projection": "image_projection",
    }

    def __init__(
        self,
        config_type: str,
        projection_dim: Optional[int] = 1024,
        lora_r: Optional[int] = 16,
        lora_alpha: Optional[int] = 32,
        lora_dropout: Optional[float] = 0.05,
        fan_in_fan_out: Optional[bool] = True,
        target_modules: Optional[Union[List[str], str]] = ["query", "value"],
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
            BletchleyForMatching(config_type, projection_dim=projection_dim),
            self.peft_config,
        )
        self.classifier = nn.Linear(1, 1)

        self.init_weights()
        self.classifier.weight.data.fill_(5.0)

    @classmethod
    @add_default_section_for_init("microsoft/model/matching/peft/lora/bletchley/v1")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/model/matching/peft/lora/bletchley/v1")
        config_type = config.getoption("config_type", "0.8B")

        projection_dim = config.getoption("projection_dim", 1024)
        lora_r = config.getoption("lora_r", 16)
        lora_alpha = config.getoption("lora_alpha", 32)
        lora_dropout = config.getoption("lora_dropout", 0.05)
        fan_in_fan_out = config.getoption("fan_in_fan_out", True)
        target_modules = config.getoption("target_modules", ["query", "value"])

        inst = cls(
            config_type=config_type,
            projection_dim=projection_dim,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            target_modules=target_modules,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        if pretrained_weight_path is not None:
            inst.from_pretrained(pretrained_weight_path)

        return inst

    @autocast()
    def forward(
        self,
        input_ids: torch.Tensor = None,
        images: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
    ):
        text_embeds, image_embeds = self.peft_model(
            input_ids=input_ids,
            images=images,
            attention_mask=attention_mask,
            return_dict=False,
        )

        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        scores = torch.sum(text_embeds * image_embeds, dim=-1, keepdim=True)

        outputs = self.classifier(scores)
        return ClassificationOutputs(outputs=outputs)
