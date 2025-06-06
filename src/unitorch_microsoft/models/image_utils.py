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
