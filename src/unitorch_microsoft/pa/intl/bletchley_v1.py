# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import logging
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torch.cuda.amp import autocast
from unitorch.models import GenericModel
from unitorch.models.clip.modeling import AllGather, _clip_loss
from unitorch.cli.models import (
    EmbeddingOutputs,
    LossOutputs,
    ClassificationOutputs,
)
from unitorch.cli import (
    add_default_section_for_init,
    add_default_section_for_function,
    register_model,
)
from unitorch_microsoft import cached_path
from unitorch_microsoft.models.bletchley.modeling_v1 import (
    get_bletchley_text_config,
    get_bletchley_image_config,
    BletchleyTextEncoder,
    BletchleyImageEncoder,
)


@register_model("microsoft/pa/intl/matching/bletchley/v1")
class BletchleyForMatching(GenericModel):
    replace_keys_in_state_dict = {
        "text_encoder.projection": "text_projection",
        "image_encoder.projection": "image_projection",
    }

    def __init__(
        self,
        config_type: str,
        projection_dim: Optional[int] = 1024,
        freeze_base_model: Optional[bool] = True,
        gradient_checkpointing: Optional[bool] = False,
        output_query_embed: Optional[bool] = False,
        output_text_embed: Optional[bool] = False,
        output_image_embed: Optional[bool] = False,
    ):
        super().__init__()
        text_config = get_bletchley_text_config(config_type, gradient_checkpointing)
        image_config = get_bletchley_image_config(config_type, gradient_checkpointing)

        self.output_query_embed = output_query_embed
        self.output_text_embed = output_text_embed
        self.output_image_embed = output_image_embed

        self.text_embed_dim = text_config.hidden_size
        self.image_embed_dim = image_config.hidden_size

        self.query_encoder = BletchleyTextEncoder(
            text_config, add_projection_layer=False
        )
        self.text_encoder = BletchleyTextEncoder(
            text_config, add_projection_layer=False
        )
        self.image_encoder = BletchleyImageEncoder(
            image_config, add_projection_layer=False
        )

        self.image_projection = nn.Linear(
            self.image_embed_dim,
            projection_dim,
        )  # text_encoder.projection.weight, text_encoder.projection.bias
        self.query_projection = nn.Linear(
            self.text_embed_dim,
            projection_dim,
        )
        self.text_projection = nn.Linear(
            self.text_embed_dim,
            projection_dim,
        )  # image_encoder.projection.weight,  image_encoder.projection.bias

        self.classifier1 = nn.Linear(1, 1)
        self.classifier2 = nn.Linear(1, 1)

        self.init_weights()
        self.classifier1.weight.data.fill_(5.0)
        self.classifier2.weight.data.fill_(5.0)

        if freeze_base_model:
            for p in self.query_encoder.parameters():
                p.requires_grad = False

            for p in self.text_encoder.parameters():
                p.requires_grad = False

            for p in self.image_encoder.parameters():
                p.requires_grad = False

    @classmethod
    @add_default_section_for_init("microsoft/pa/intl/matching/bletchley/v1")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/pa/intl/matching/bletchley/v1")
        config_type = config.getoption("config_type", "0.8B")

        projection_dim = config.getoption("projection_dim", 1024)
        freeze_base_model = config.getoption("freeze_base_model", True)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)
        output_query_embed = config.getoption("output_query_embed", False)
        output_text_embed = config.getoption("output_text_embed", False)
        output_image_embed = config.getoption("output_image_embed", False)

        inst = cls(
            config_type=config_type,
            projection_dim=projection_dim,
            freeze_base_model=freeze_base_model,
            gradient_checkpointing=gradient_checkpointing,
            output_query_embed=output_query_embed,
            output_text_embed=output_text_embed,
            output_image_embed=output_image_embed,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        if pretrained_weight_path is not None:
            pretrained_weight_path = cached_path(pretrained_weight_path)
            inst.from_pretrained(pretrained_weight_path)

        return inst
    
    def from_pretrained(self, weight_path):
        if not os.path.exists(weight_path):
            return
        state_dict = torch.load(weight_path, map_location="cpu")
        _keys = [key for key in state_dict.keys() if key.startswith("text_encoder")]
        for _key in _keys:
            _value = state_dict.get(_key)
            state_dict["query_encoder" + _key[12:]] = _value

        super().from_pretrained(state_dict=state_dict)

    @autocast()
    def forward(
        self,
        query_input_ids: torch.Tensor = None,
        input_ids: torch.Tensor = None,
        images: torch.Tensor = None,
        query_attention_mask: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
    ):
        if not self.training and self.output_image_embed:
            image_outputs = self.image_encoder(
                images=images,
            )
            image_embeds = image_outputs[:, 0]
            image_embeds = self.image_projection(image_embeds)
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            return EmbeddingOutputs(embedding=image_embeds)

        if not self.training and self.output_query_embed:
            query_outputs = self.query_encoder(
                input_ids=query_input_ids,
                attention_mask=query_attention_mask,
            )
            query_embeds = query_outputs[:, 0]
            query_embeds = self.query_projection(query_embeds)
            query_embeds = query_embeds / query_embeds.norm(dim=-1, keepdim=True)
            return EmbeddingOutputs(embedding=query_embeds)

        if not self.training and self.output_text_embed:
            text_outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            text_embeds = text_outputs[:, 0]
            text_embeds = self.text_projection(text_embeds)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            return EmbeddingOutputs(embedding=text_embeds)

        image_outputs = self.image_encoder(
            images=images,
        )
        query_outputs = self.query_encoder(
            input_ids=query_input_ids,
            attention_mask=query_attention_mask,
        )
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        image_embeds = image_outputs[:, 0]
        image_embeds = self.image_projection(image_embeds)
        query_embeds = query_outputs[:, 0]
        query_embeds = self.query_projection(query_embeds)
        text_embeds = text_outputs[:, 0]
        text_embeds = self.text_projection(text_embeds)

        query_embeds = query_embeds / query_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

        scores1 = torch.sum(query_embeds * image_embeds, dim=-1, keepdim=True)
        scores2 = torch.sum(query_embeds * text_embeds, dim=-1, keepdim=True)

        outputs = self.classifier1(scores1) + self.classifier2(scores2)
        return ClassificationOutputs(outputs=outputs)
