# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import math
import random
import numpy as np
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from PIL import Image, ImageOps, ImageFile, ImageFilter, ImageChops, ImageDraw
from random import randint, shuffle, choice
from unitorch.cli import (
    add_default_section_for_init,
    add_default_section_for_function,
    register_process,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True


class OmniPixelProcessor:
    """
    Processor for image-related operations.
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (512, 512),
        use_soft_mask: Optional[bool] = True,
    ):
        """
        Initializes a new instance of the ImageProcessor.
        """
        self.image_size = (
            image_size if isinstance(image_size, tuple) else (image_size, image_size)
        )
        self.use_soft_mask = use_soft_mask

    @classmethod
    @add_default_section_for_init("microsoft/omnipixel/process")
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
