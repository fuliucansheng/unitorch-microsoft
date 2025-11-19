# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import io
import requests
import time
import base64
import json
import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from random import random
from PIL import Image, ImageOps, ImageFile, ImageFilter
from unitorch.utils import is_opencv_available
from unitorch.cli import (
    add_default_section_for_init,
    add_default_section_for_function,
    register_process,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageProcessor:
    """
    Processor for image-related operations.
    """

    def __init__(
        self,
    ):
        """
        Initializes a new instance of the ImageProcessor.
        """
        pass

    @classmethod
    @add_default_section_for_init("microsoft/process/image")
    def from_core_configure(cls, config, **kwargs):
        """
        Creates a new instance of the ImageProcessor using the configuration from the core.

        Args:
            config: The configuration object.
            **kwargs: Additional keyword arguments.

        Returns:
            An instance of the ImageProcessor.
        """
        pass

    @register_process("microsoft/process/image/center_crop")
    def _center_crop(
        self,
        image: Image.Image,
        size: Optional[Tuple[int, int]] = (224, 224),
        do_resize: bool = True,
    ):
        """
        Crops the image to the center.

        Args:
            image (Image.Image): The image to crop.
            size (Optional[Tuple[int, int]]): The size of the cropped image. Defaults to (224, 224).

        Returns:
            The cropped image as a PIL Image object.
        """
        ratio = size[0] / size[1]
        if do_resize:
            if image.width / image.height > ratio:
                new_height = size[1]
                new_width = int(new_height * (image.width / image.height))
            else:
                new_width = size[0]
                new_height = int(new_width * (image.height / image.width))
            image = image.resize((new_width, new_height), Image.LANCZOS)

        width, height = image.size
        left = (width - size[0]) // 2
        top = (height - size[1]) // 2
        left, top = max(0, left), max(0, top)
        right = left + size[0]
        bottom = top + size[1]
        right, bottom = min(width, right), min(height, bottom)
        return image.crop((left, top, right, bottom))

    @register_process("microsoft/process/image/padding")
    def _padding(
        self,
        image: Image.Image,
        pad_ratios: Union[float, Tuple[float, float]] = (0.4, 0.4),
        pad_pixels: Union[int, Tuple[int, int]] = (0, 0),
        color: Union[int, Tuple[int, int, int]] = (255, 255, 255),
    ):
        width, height = image.size
        if isinstance(pad_ratios, float):
            pad_x = int(pad_ratios * width)
            pad_y = int(pad_ratios * height)
        else:
            pad_x = int(pad_ratios[0] * width)
            pad_y = int(pad_ratios[1] * height)

        if isinstance(pad_pixels, int):
            pad_x += pad_pixels
            pad_y += pad_pixels
        else:
            pad_x += pad_pixels[0]
            pad_y += pad_pixels[1]

        new_width = int(width + pad_x)
        new_height = int(height + pad_y)
        new_image = Image.new("RGB", (new_width, new_height), color)
        new_image.paste(image, ((new_width - width) // 2, (new_height - height) // 2))
        return new_image
