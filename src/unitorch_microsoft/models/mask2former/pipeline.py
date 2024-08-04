# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import re
import torch
import numpy as np
from PIL import Image, ImageDraw
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.utils import pop_value, nested_dict_value
from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
)
from unitorch_microsoft.models.mask2former import pretrained_mask2former_infos
from unitorch_microsoft.models.mask2former import (
    Mask2FormerProcessor,
    Mask2FormerForSegmentation as _Mask2FormerForSegmentation,
)


class Mask2FormerPipeline(_Mask2FormerForSegmentation):
    def __init__(
        self,
        config_path: str,
        vision_config_path: str,
        weight_path: Optional[Union[str, List[str]]] = None,
        state_dict: Optional[Dict[str, Any]] = None,
        device: Optional[Union[str, int]] = "cpu",
    ):
        super().__init__(
            config_path=config_path,
        )
        self.processor = Mask2FormerProcessor(
            vision_config_path=vision_config_path,
        )
        self._device = "cpu" if device == "cpu" else int(device)

        self.from_pretrained(weight_path, state_dict=state_dict)
        self.to(device=self._device)
        self.eval()

    @classmethod
    @add_default_section_for_init("microsoft/pipeline/mask2former")
    def from_core_configure(
        cls,
        config,
        pretrained_name: Optional[str] = "mask2former-swin-tiny-ade-semantic",
        config_path: Optional[str] = None,
        vision_config_path: Optional[str] = None,
        pretrained_weight_path: Optional[str] = None,
        **kwargs,
    ):
        config.set_default_section("microsoft/pipeline/mask2former")
        pretrained_name = config.getoption("pretrained_name", pretrained_name)

        config_path = config.getoption("config_path", config_path)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_mask2former_infos, pretrained_name, "config"),
        )
        config_path = cached_path(config_path)

        vision_config_path = config.getoption("vision_config_path", vision_config_path)
        vision_config_path = pop_value(
            vision_config_path,
            nested_dict_value(
                pretrained_mask2former_infos, pretrained_name, "vision_config"
            ),
        )
        vision_config_path = cached_path(vision_config_path)

        device = config.getoption("device", "cpu")
        pretrained_weight_path = config.getoption(
            "pretrained_weight_path", pretrained_weight_path
        )
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_mask2former_infos, pretrained_name, "weight"),
            check_none=False,
        )

        inst = cls(
            config_path,
            vision_config_path,
            weight_path=weight_path,
            device=device,
        )

        return inst

    @torch.no_grad()
    @add_default_section_for_function("microsoft/pipeline/mask2former")
    def __call__(
        self,
        image: Union[Image.Image, str],
        threshold: Optional[float] = 0.5,
    ):
        inputs = self.processor.classification(image)
        pixel_values = inputs.pixel_values.unsqueeze(0).to(self._device)
        outputs = self.segment(
            pixel_values,
        )
        masks = [
            (mask.cpu().numpy() > threshold).astype(np.uint8) for mask in outputs.masks
        ][0]
        result_image = Image.fromarray(masks * 255)
        result_image = result_image.resize(image.size)

        return result_image
