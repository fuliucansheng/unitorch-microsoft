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

    @register_process("microsoft/process/image/canny")
    def _canny(
        self,
        image: Image.Image,
    ):
        """
        Detects edges in the image using the Canny algorithm.

        Args:
            image (Image.Image): The image to detect edges in.

        Returns:
            The image with detected edges as a PIL Image object.
        """
        if is_opencv_available():
            import cv2

            image = np.array(image, np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = cv2.Canny(image, 100, 200)
            image = Image.fromarray(image)
        else:
            image = image.convert("L")
            image = image.filter(ImageFilter.FIND_EDGES)
        return image

    @register_process("microsoft/process/image/mask")
    def _mask(
        self,
        image: Image.Image,
        mask: Image.Image,
        threshold: Optional[int] = -1,
    ):
        result = Image.new("RGBA", image.size, (0, 0, 0, 0))
        mask = mask.convert("L").resize(image.size)
        result.paste(image, (0, 0), mask)
        if threshold >= 0:
            ys, xs = np.where(np.array(mask.convert("L")) > threshold)
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            result = result.crop((min_x, min_y, max_x, max_y))
        return result
