# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import torch
import random
import numpy as np
from PIL import Image
from typing import List, Tuple, Union, Optional
from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize, Compose
from transformers import SamImageProcessor
from transformers.image_utils import to_numpy_array, ChannelDimension
from transformers.image_transforms import to_channel_dimension_format
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models import HfImageClassificationProcessor, GenericOutputs
from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
    register_process,
)
from unitorch.cli.models import SegmentationOutputs, TensorsInputs
from unitorch.cli.models.sam import pretrained_sam_infos


def get_random_points(mask: Image.Image, num_points: int):
    mask = np.array(mask.convert("L"))
    non_zeros = np.argwhere(mask > 0)
    if len(non_zeros) < num_points:
        return np.array(non_zeros)
    return np.array(random.choices(non_zeros, k=num_points))


def get_mask_box(mask: Image.Image):
    width, height = mask.size
    mask = np.array(mask.convert("L"))
    non_zeros = np.argwhere(mask > 0)
    if len(non_zeros) == 0:
        return (0, 0, width, height)
    return (
        np.min(non_zeros[:, 1]),
        np.min(non_zeros[:, 0]),
        np.max(non_zeros[:, 1]),
        np.max(non_zeros[:, 0]),
    )


class SamProcessor:
    def __init__(
        self,
        vision_config_path: str,
    ):
        """
        Initializes a SwinProcessor for image classification tasks.

        Args:
            vision_config_path (str): The path to the SamImageProcessor configuration file.
        """
        self.vision_processor = SamImageProcessor.from_json_file(vision_config_path)

    @classmethod
    @add_default_section_for_init("microsoft/process/sam")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/process/sam")
        pretrained_name = config.getoption("pretrained_name", "sam-vit-base")
        vision_config_path = config.getoption("vision_config_path", None)
        vision_config_path = pop_value(
            vision_config_path,
            nested_dict_value(pretrained_sam_infos, pretrained_name, "vision_config"),
        )

        vision_config_path = cached_path(vision_config_path)

        return {
            "vision_config_path": vision_config_path,
        }

    @register_process("microsoft/process/sam/segmentation")
    def _segmentation(
        self,
        image: Union[Image.Image, str],
        mask: Union[Image.Image, str],
        # num_points: Optional[int] = 1,
    ):
        if isinstance(image, str):
            image = Image.open(image)

        if isinstance(mask, str):
            mask = Image.open(mask)

        pixel_results = self.vision_processor(image)
        pixel_values = torch.tensor(pixel_results.get("pixel_values")[0])
        original_sizes = pixel_results.get("original_sizes")[0] #h, w
        reshaped_input_sizes = pixel_results.get("reshaped_input_sizes")[0] #h, w
        height, width = reshaped_input_sizes
        # mask = mask.resize((width, height)).convert("L")
        # pixel_targets = torch.zeros_like(pixel_values[0]).float()
        # pixel_targets[:height, :width] = torch.tensor(np.array(mask)).float() / 255.0

        mask = mask.resize((width, height)).convert("1")
        pixel_targets = torch.zeros_like(pixel_values[0]).long()
        pixel_targets[:height, :width] = torch.tensor(np.array(mask)).long()

        # input_points = get_random_points(mask, random.randint(1, num_points))
        # input_boxes = torch.tensor([get_mask_box(mask)]).float()
        input_boxes = torch.tensor([(0, 0, width, height)]).float()

        return TensorsInputs(
            pixel_values=pixel_values,
            # input_points=torch.tensor(input_points),
            input_boxes=input_boxes,
            pixel_targets=pixel_targets,
            reshaped_input_sizes=torch.tensor(reshaped_input_sizes),
        )

