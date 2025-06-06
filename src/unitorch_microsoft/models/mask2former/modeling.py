# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import json
import random
import logging
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import autocast
from unitorch.utils import pop_value, nested_dict_value
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerConfig,
    Mask2FormerModel,
    Mask2FormerLoss,
)
from unitorch.models import GenericModel, GenericOutputs
from unitorch.cli import (
    add_default_section_for_init,
    add_default_section_for_function,
    register_model,
)
from unitorch.cli.models import SegmentationOutputs, LossOutputs
from unitorch.cli.models import segmentation_model_decorator
from unitorch.cli.models.mask2former import pretrained_mask2former_infos
from unitorch_microsoft import cached_path


@register_model(
    "microsoft/model/segmentation/mask2former", segmentation_model_decorator
)
class Mask2FormerForSegmentation(GenericModel):
    def __init__(
        self,
        config_path: str,
        num_queries: Optional[int] = 100,
    ):
        """
        Initializes a SamForSegmentation model for segmentation tasks.

        Args:
            config_path (str): The path to the Sam Transformer configuration file.
        """
        super().__init__()
        config = Mask2FormerConfig.from_json_file(config_path)
        config.num_queries = num_queries

        self.model = Mask2FormerModel(config)
        self.conv = nn.Conv2d(num_queries, 1, kernel_size=(1, 1), stride=(1, 1))
        self.init_weights()

    @classmethod
    @add_default_section_for_init("microsoft/model/segmentation/mask2former")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/model/segmentation/mask2former")
        pretrained_name = config.getoption(
            "pretrained_name", "mask2former-swin-tiny-ade-semantic"
        )
        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_mask2former_infos, pretrained_name, "config"),
        )
        config_path = cached_path(config_path)
        num_queries = config.getoption("num_queries", 100)

        inst = cls(
            config_path=config_path,
            num_queries=num_queries,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_mask2former_infos, pretrained_name, "weight"),
            check_none=False,
        )
        if weight_path is not None:
            inst.from_pretrained(weight_path)

        return inst

    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(self):
        """
        Performs a forward pass of the SamForSegmentation model.
        """
        raise NotImplementedError

    @add_default_section_for_function("microsoft/model/segmentation/mask2former")
    @torch.no_grad()
    def segment(
        self,
        pixel_values: torch.Tensor,
        pixel_mask: Optional[torch.Tensor] = None,
    ):
        """
        Performs a forward pass of the SamForSegmentation model.
        """
        outputs = self.model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            output_hidden_states=True,
        )

        masks_queries_logits = outputs.masks_queries_logits[-1]
        masks_queries_logits = self.conv(torch.relu(masks_queries_logits))
        masks_queries_logits = nn.functional.interpolate(
            masks_queries_logits,
            scale_factor=4,
            mode="bilinear",
            align_corners=False,
        )
        masks_queries_logits = torch.sigmoid(masks_queries_logits)
        return SegmentationOutputs(masks=list(masks_queries_logits.squeeze(1)))
