# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import logging
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from functools import partial
from PIL import Image
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
from unitorch_microsoft.models.bletchley.processing_v1 import BletchleyProcessor


@register_model("microsoft/omnipixel/model/click/bletchley/image")
class BletchleyForImageClickModel(GenericModel):
    replace_keys_in_state_dict = {
        "text_encoder.projection": "text_projection",
        "image_encoder.projection": "image_projection",
    }

    def __init__(
        self,
        config_type: str,
        projection_dim: Optional[int] = 1024,
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
    @add_default_section_for_init("microsoft/omnipixel/model/click/bletchley/image")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/omnipixel/model/click/bletchley/image")
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


class BletchleyForImageClickModelPipeline(BletchleyForImageClickModel):
    def __init__(
        self,
        config_type: str,
        projection_dim: Optional[int] = 512,
        num_classes: Optional[int] = 1,
        num_positions: Optional[int] = None,
        max_seq_length: Optional[int] = 120,
        enable_text_encoder: Optional[bool] = False,
        pretrained_weight_path: Optional[str] = None,
        device: Optional[Union[str, int]] = "cpu",
    ):
        super().__init__(
            config_type=config_type,
            projection_dim=projection_dim,
            num_classes=num_classes,
            num_positions=num_positions,
            enable_text_encoder=enable_text_encoder,
        )
        self.processor = BletchleyProcessor(max_seq_length=max_seq_length)
        self._device = "cpu" if device == "cpu" else int(device)

        if pretrained_weight_path is not None:
            self.from_pretrained(pretrained_weight_path)

        self.to(device=self._device)

    @classmethod
    @add_default_section_for_init("microsoft/omnipixel/pipeline/click/bletchley/image")
    def from_core_configure(
        cls,
        config,
        config_type: Optional[str] = None,
        projection_dim: Optional[int] = None,
        num_classes: Optional[int] = None,
        num_positions: Optional[int] = None,
        max_seq_length: Optional[int] = None,
        enable_text_encoder: Optional[bool] = False,
        pretrained_weight_path: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        config.set_default_section("microsoft/omnipixel/pipeline/click/bletchley/image")
        config_type = (
            config.getoption("config_type", "0.8B")
            if config_type is None
            else config_type
        )
        projection_dim = (
            config.getoption("projection_dim", 1024)
            if projection_dim is None
            else projection_dim
        )
        num_classes = (
            config.getoption("num_classes", 1) if num_classes is None else num_classes
        )
        num_positions = (
            config.getoption("num_positions", 50)
            if num_positions is None
            else num_positions
        )
        enable_text_encoder = (
            config.getoption("enable_text_encoder", False)
            if enable_text_encoder is None
            else enable_text_encoder
        )
        max_seq_length = (
            config.getoption("max_seq_length", 120)
            if max_seq_length is None
            else max_seq_length
        )
        pretrained_weight_path = (
            config.getoption("pretrained_weight_path", None)
            if pretrained_weight_path is None
            else pretrained_weight_path
        )
        device = config.getoption("device", None) if device is None else device
        inst = cls(
            config_type=config_type,
            projection_dim=projection_dim,
            num_classes=num_classes,
            num_positions=num_positions,
            max_seq_length=max_seq_length,
            enable_text_encoder=enable_text_encoder,
            pretrained_weight_path=pretrained_weight_path,
            device=device,
        )

        return inst

    @torch.no_grad()
    def __call__(
        self,
        image: Image.Image,
        text: Optional[str] = None,
    ):
        inputs = self.processor._image_classification(image).dict()
        inputs = {k: v.unsqueeze(0) if v is not None else v for k, v in inputs.items()}
        if text is not None:
            text_inputs = self.processor._text_classification(text).dict()
            inputs.update(
                {
                    k: v.unsqueeze(0) if v is not None else v
                    for k, v in text_inputs.items()
                }
            )

        inputs = {
            k: v.to(self._device) if v is not None else v for k, v in inputs.items()
        }
        outputs = self.forward(**inputs).outputs
        scores = outputs.sigmoid().squeeze(0)
        return scores[0].item()
