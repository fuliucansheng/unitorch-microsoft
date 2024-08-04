# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import torch
import numpy as np
from PIL import Image
from typing import List, Tuple, Union, Optional
from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize, Compose
from transformers import Mask2FormerImageProcessor
from transformers.image_utils import to_numpy_array, ChannelDimension
from transformers.image_transforms import to_channel_dimension_format
from unitorch.utils import pop_value, nested_dict_value, is_opencv_available
from unitorch.models import HfImageClassificationProcessor, GenericOutputs
from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
    register_process,
)
from unitorch.cli import WriterOutputs
from unitorch.cli.models import (
    TensorsInputs,
    GenerationOutputs,
    GenerationTargets,
)
from unitorch.cli.models.mask2former import pretrained_mask2former_infos
from unitorch.cli.models.segmentation_utils import (
    SegmentationOutputs,
    SegmentationTargets,
    SegmentationProcessor,
    numpy_to_pil,
)

if is_opencv_available():
    import cv2


class Mask2FormerProcessor(HfImageClassificationProcessor):
    def __init__(
        self,
        vision_config_path: str,
        mask_threshold: Optional[float] = 0.5,
        background_filter_alpha: Optional[float] = 0.0005,
        foreground_filter_alpha: Optional[float] = 0.0001,
    ):
        """
        Initializes a Mask2FormerProcessor for image classification tasks.

        Args:
            vision_config_path (str): The path to the Mask2FormerImageProcessor configuration file.
        """
        vision_processor = Mask2FormerImageProcessor.from_json_file(vision_config_path)

        super().__init__(
            vision_processor=vision_processor,
        )
        self.mask_threshold = mask_threshold
        self.background_filter_alpha = background_filter_alpha
        self.foreground_filter_alpha = foreground_filter_alpha

    @classmethod
    @add_default_section_for_init("microsoft/process/mask2former")
    def from_core_configure(cls, config, **kwargs):
        """
        Create an instance of Mask2FormerProcessor from a core configuration.

        Args:
            config: The core configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: The processed arguments for initializing the processor.
        """
        config.set_default_section("microsoft/process/mask2former")
        pretrained_name = config.getoption(
            "pretrained_name", "mask2former-swin-tiny-ade-semantic"
        )
        vision_config_path = config.getoption("vision_config_path", None)
        vision_config_path = pop_value(
            vision_config_path,
            nested_dict_value(
                pretrained_mask2former_infos, pretrained_name, "vision_config"
            ),
        )

        vision_config_path = cached_path(vision_config_path)

        return {
            "vision_config_path": vision_config_path,
        }

    @register_process("microsoft/postprocess/mask2former/segmentation/mask")
    def _segmentation_mask(
        self,
        outputs: SegmentationOutputs,
    ):
        results = outputs.to_pandas()
        assert results.shape[0] == 0 or results.shape[0] == len(outputs.masks)
        assert all([m.ndim == 3 for m in outputs.masks])
        if is_opencv_available():
            for mask in masks:
                background = np.where(
                    mask <= self.mask_threshold, np.uint8(1), np.uint8(0)
                )
                _, label, stats, _ = cv2.connectedComponentsWithStats(
                    background, connectivity=8
                )
                areas = stats[:, -1]
                filter_area = (
                    background.shape[0]
                    * background.shape[1]
                    * self.background_filter_alpha
                )
                mask[np.where(areas[label] < filter_area)] = 1.0
                foreground = np.where(
                    mask > self.mask_threshold, np.uint8(1), np.uint8(0)
                )
                _, label, stats, _ = cv2.connectedComponentsWithStats(
                    foreground, connectivity=8
                )
                areas = stats[:, -1]
                filter_area = (
                    foreground.shape[0]
                    * foreground.shape[1]
                    * self.foreground_filter_alpha
                )
                mask[np.where(areas[label] < filter_area)] = 0.0
        else:
            masks = [(mask <= self.mask_threshold).astype(int) for mask in masks]

        classes = [c.numpy() for c in outputs.classes]
        classes = [c if c.ndim == 1 else c.argmax(-1) for c in classes]
        results["mask_images"] = [
            ";".join(
                [
                    self.save_image(numpy_to_pil(_mask_image))
                    for _mask_image in _mask_images
                ]
            )
            for _mask_images in masks
        ]
        results["mask_classes"] = [
            ";".join([str(_class) for _class in _classes]) for _classes in classes
        ]
        return WriterOutputs(results)
