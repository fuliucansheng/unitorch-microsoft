# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import json
import random
import logging
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch import autocast
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from transformers.models.clip.modeling_clip import (
    CLIPConfig,
    CLIPTextTransformer,
    CLIPVisionTransformer,
)
from transformers.models.siglip import (
    SiglipConfig,
    SiglipTextModel,
    SiglipVisionModel,
)
from unitorch.models import GenericModel
from unitorch.utils import pop_value, nested_dict_value
from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
    register_model,
)
from unitorch.cli.models import ClassificationOutputs, LossOutputs
from unitorch.cli.models.clip import pretrained_clip_infos
from unitorch.cli.models.siglip import pretrained_siglip_infos


@register_model("omnipixel/model/siglip/image")
class SiglipForImageClassification(GenericModel):
    """CLIP model for image classification."""

    def __init__(
        self,
        config_path: str,
        freeze_base_model: Optional[bool] = True,
        gradient_checkpointing: Optional[bool] = False,
    ):
        """
        Initialize the ClipForImageClassification model.

        Args:
            config_path (str): The path to the model configuration file.
            freeze_base_model (bool, optional): Whether to freeze the base model. Defaults to True.
            gradient_checkpointing (bool, optional): Whether to use gradient checkpointing. Defaults to False.
        """
        super().__init__()
        config = SiglipConfig.from_json_file(config_path)
        vision_config = config.vision_config
        vision_config.gradient_checkpointing = gradient_checkpointing

        self.vision_embed_dim = vision_config.hidden_size
        self.vision_model = SiglipVisionModel(vision_config).vision_model
        self.scoring_head = nn.Sequential(
            nn.Linear(self.vision_embed_dim, 1024),
            nn.Dropout(0.5),
            nn.Linear(1024, 128),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.Dropout(0.5),
            nn.Linear(64, 16),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
        )
        self.init_weights()

        if freeze_base_model:
            for p in self.vision_model.parameters():
                p.requires_grad = False

    @classmethod
    @add_default_section_for_init("omnipixel/model/siglip/image")
    def from_core_configure(cls, config, **kwargs):
        """
        Create an instance of ClipForImageClassification from a core configuration.

        Args:
            config: The core configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            ClipForImageClassification: An instance of the ClipForImageClassification model.
        """
        config.set_default_section("omnipixel/model/siglip/image")
        pretrained_name = config.getoption(
            "pretrained_name", "siglip-so400m-patch14-384"
        )
        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_siglip_infos, pretrained_name, "config"),
        )

        config_path = cached_path(config_path)

        freeze_base_model = config.getoption("freeze_base_model", True)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)

        inst = cls(
            config_path=config_path,
            freeze_base_model=freeze_base_model,
            gradient_checkpointing=gradient_checkpointing,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_siglip_infos, pretrained_name, "weight"),
            check_none=False,
        )
        if weight_path is not None:
            weight_path = [
                weight_path,
                "https://unitorchazureblob.blob.core.windows.net/shares/models/adsplus/diffusion/pytorch_model.siglip.msra.bin",
            ]
            inst.from_pretrained(weight_path)

        return inst

    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(
        self,
        pixel_values: torch.Tensor,
    ):
        """
        Perform a forward pass through the model.

        Args:
            pixel_values (torch.Tensor): Input pixel values.

        Returns:
            ClassificationOutputs: The classification outputs.
        """
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
        )

        image_embeds = vision_outputs[1]

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        outputs = self.scoring_head(image_embeds)
        return ClassificationOutputs(outputs=outputs)


@register_model("omnipixel/model/laion_clip/image")
class LAIONClipForImageClassification(GenericModel):
    """CLIP model for image classification."""

    def __init__(
        self,
        config_path: str,
        freeze_base_model: Optional[bool] = True,
        gradient_checkpointing: Optional[bool] = False,
    ):
        """
        Initialize the ClipForImageClassification model.

        Args:
            config_path (str): The path to the model configuration file.
            freeze_base_model (bool, optional): Whether to freeze the base model. Defaults to True.
            gradient_checkpointing (bool, optional): Whether to use gradient checkpointing. Defaults to False.
        """
        super().__init__()
        config = CLIPConfig.from_json_file(config_path)
        vision_config = config.vision_config
        vision_config.gradient_checkpointing = gradient_checkpointing

        self.vision_embed_dim = vision_config.hidden_size
        self.vision_model = CLIPVisionTransformer(vision_config)
        self.visual_projection = nn.Linear(
            self.vision_embed_dim,
            config.projection_dim,
            bias=False,
        )
        self.layers = nn.Sequential(
            nn.Linear(config.projection_dim, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )
        self.init_weights()

        if freeze_base_model:
            for p in self.vision_model.parameters():
                p.requires_grad = False

        self.vision_model.encoder.gradient_checkpointing = gradient_checkpointing

    @classmethod
    @add_default_section_for_init("omnipixel/model/laion_clip/image")
    def from_core_configure(cls, config, **kwargs):
        """
        Create an instance of ClipForImageClassification from a core configuration.

        Args:
            config: The core configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            ClipForImageClassification: An instance of the ClipForImageClassification model.
        """
        config.set_default_section("omnipixel/model/laion_clip/image")
        pretrained_name = config.getoption("pretrained_name", "clip-vit-large-patch16")
        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_clip_infos, pretrained_name, "config"),
        )

        config_path = cached_path(config_path)

        freeze_base_model = config.getoption("freeze_base_model", True)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)

        inst = cls(
            config_path=config_path,
            freeze_base_model=freeze_base_model,
            gradient_checkpointing=gradient_checkpointing,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_clip_infos, pretrained_name, "weight"),
            check_none=False,
        )
        if weight_path is not None:
            weight_path = [
                weight_path,
                "https://unitorchazureblob.blob.core.windows.net/shares/models/adsplus/diffusion/pytorch_model.laion_clip.sac.logos.ava1.l14.msra.bin",
            ]
            inst.from_pretrained(weight_path)

        return inst

    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(
        self,
        pixel_values: torch.Tensor,
    ):
        """
        Perform a forward pass through the model.

        Args:
            pixel_values (torch.Tensor): Input pixel values.

        Returns:
            ClassificationOutputs: The classification outputs.
        """
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
        )

        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        outputs = self.layers(image_embeds)
        return ClassificationOutputs(outputs=outputs)
