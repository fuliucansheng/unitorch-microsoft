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
from torch import autocast
from transformers.activations import quick_gelu
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
from unitorch_microsoft.models.bletchley.modeling_v1 import (
    get_bletchley_text_config,
    get_bletchley_image_config,
    BletchleyTextEncoder,
    BletchleyImageEncoder,
)


@register_model("microsoft/omnipixel/click/bletchley/image")
class BletchleyForImageClickModel(GenericModel):
    replace_keys_in_state_dict = {
        "text_encoder.projection": "text_projection",
        "image_encoder.projection": "image_projection",
    }

    def __init__(
        self,
        config_type: str,
        projection_dim: Optional[int] = 512,
        num_classes: Optional[int] = 1,
        num_positions: Optional[int] = None,
        freeze_base_model: Optional[bool] = False,
        freeze_image_model: Optional[bool] = False,
        enable_text_encoder: Optional[bool] = False,
        gradient_checkpointing: Optional[bool] = False,
        output_text_embed: Optional[bool] = False,
        output_image_embed: Optional[bool] = False,
    ):
        super().__init__()

        self.enable_text_encoder = enable_text_encoder
        self.output_image_embed = output_image_embed
        self.output_text_embed = output_text_embed
        if self.enable_text_encoder:
            text_config = get_bletchley_text_config(config_type, gradient_checkpointing)
            self.text_embed_dim = text_config.hidden_size
            self.text_encoder = BletchleyTextEncoder(
                text_config, add_projection_layer=False
            )
            self.text_projection = nn.Linear(
                self.text_embed_dim,
                projection_dim,
            )

        image_config = get_bletchley_image_config(config_type, gradient_checkpointing)
        self.image_embed_dim = image_config.hidden_size
        self.image_encoder = BletchleyImageEncoder(
            image_config, add_projection_layer=False
        )
        self.image_projection = nn.Linear(
            self.image_embed_dim,
            projection_dim,
        )

        self.num_positions = num_positions
        if num_positions is not None:
            self.position_embedding = nn.Embedding(num_positions, projection_dim)
            self.position_layer_norm = nn.LayerNorm(projection_dim)
            self.position = nn.Linear(projection_dim, num_classes)

        if self.enable_text_encoder:
            self.classifier = nn.Linear(projection_dim * 2, num_classes)
        else:
            self.classifier = nn.Linear(projection_dim, num_classes)
        self.init_weights()

        if freeze_base_model:
            if enable_text_encoder:
                for p in self.text_encoder.parameters():
                    p.requires_grad = False

            for p in self.image_encoder.parameters():
                p.requires_grad = False

        if freeze_image_model:
            for p in self.image_encoder.parameters():
                p.requires_grad = False

    @classmethod
    @add_default_section_for_init("microsoft/omnipixel/click/bletchley/image")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/omnipixel/click/bletchley/image")
        config_type = config.getoption("config_type", "0.8B")
        projection_dim = config.getoption("projection_dim", 1024)
        num_classes = config.getoption("num_classes", 1)
        num_positions = config.getoption("num_positions", 50)
        freeze_base_model = config.getoption("freeze_base_model", True)
        freeze_image_model = config.getoption("freeze_image_model", True)
        enable_text_encoder = config.getoption("enable_text_encoder", False)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)
        output_text_embed = config.getoption("output_text_embed", False)
        output_image_embed = config.getoption("output_image_embed", False)

        inst = cls(
            config_type=config_type,
            projection_dim=projection_dim,
            num_classes=num_classes,
            num_positions=num_positions,
            freeze_image_model=freeze_image_model,
            freeze_base_model=freeze_base_model,
            enable_text_encoder=enable_text_encoder,
            gradient_checkpointing=gradient_checkpointing,
            output_text_embed=output_text_embed,
            output_image_embed=output_image_embed,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        if pretrained_weight_path is not None:
            inst.from_pretrained(pretrained_weight_path)

        return inst

    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(
        self,
        input_ids: torch.Tensor = None,
        pos_ids: Optional[torch.Tensor] = None,
        images: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
    ):
        if not self.training and self.output_image_embed:
            image_outputs = self.image_encoder(images=images)
            image_embeds = image_outputs[:, 0]
            image_embeds = self.image_projection(image_embeds)
            return EmbeddingOutputs(embedding=image_embeds)

        if not self.training and self.output_text_embed:
            text_outputs = self.text_encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )
            text_embeds = text_outputs[:, 0]
            text_embeds = self.text_projection(text_embeds)
            return EmbeddingOutputs(embedding=text_embeds)

        image_outputs = self.image_encoder(images=images)
        image_embeds = image_outputs[:, 0]
        image_embeds = self.image_projection(image_embeds)

        if self.enable_text_encoder:
            text_outputs = self.text_encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )
            text_embeds = text_outputs[:, 0]
            text_embeds = self.text_projection(text_embeds)

            outputs = self.classifier(
                F.relu(torch.cat([image_embeds, text_embeds], axis=1))
            )
        else:
            outputs = self.classifier(image_embeds)

        if pos_ids is not None and self.num_positions is not None:
            pos_emb = self.position_layer_norm(self.position_embedding(pos_ids))
            outputs += self.position(pos_emb)

        return ClassificationOutputs(outputs=outputs)
