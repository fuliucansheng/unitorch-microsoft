# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import torch
import random
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from PIL import Image
from random import randint, shuffle, choice
from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize, Compose
from transformers import XLMRobertaTokenizer
from unitorch.utils import pop_value, truncate_sequence_pair
from unitorch.models import GenericOutputs
from unitorch.cli import (
    add_default_section_for_init,
    add_default_section_for_function,
    register_process,
)
from unitorch.cli.models import TensorsInputs
from unitorch_microsoft import cached_path
from unitorch_microsoft.models.bletchley.processing_v1 import BletchleyProcessor


class BletchleyProcessorV2(BletchleyProcessor):
    def __init__(
        self,
        max_seq_length: Optional[int] = 128,
        pixel_mean: List[float] = [0.5, 0.5, 0.5],
        pixel_std: List[float] = [0.5, 0.5, 0.5],
        resize_shape: Optional[List[int]] = [224, 224],
        crop_shape: Optional[List[int]] = [224, 224],
        max_num_text: Optional[int] = 5,
    ):
        super().__init__(
            max_seq_length, pixel_mean, pixel_std, resize_shape, crop_shape
        )
        self.max_num_text = max_num_text

    @classmethod
    @add_default_section_for_init("microsoft/process/china/selection/bletchley/v1")
    def from_core_configure(cls, config, **kwargs):
        pass

    @register_process(
        "microsoft/process/china/selection/bletchley/v1/text_classification_list"
    )
    def _text_classification_list(
        self,
        text: Union[str, List[str]],
        text_pair: Optional[str] = None,
        max_num_text: Optional[int] = None,
        max_seq_length: Optional[int] = None,
        prefix: Optional[str] = None,
    ):
        if isinstance(text, str):
            text = [text]
        max_num_text = int(pop_value(max_num_text, self.max_num_text))
        num_attention_mask = [1] * len(text[:max_num_text]) + [0] * (
            max_num_text - len(text[:max_num_text])
        )
        texts = text[:max_num_text] + [""] * (max_num_text - len(text[:max_num_text]))
        outputs = [
            self._text_classification(text, text_pair, max_seq_length) for text in texts
        ]
        inputs = dict(
            input_ids=torch.stack([output.input_ids for output in outputs]),
            attention_mask=torch.stack([output.attention_mask for output in outputs]),
            num_attention_mask=torch.tensor(num_attention_mask),
        )

        if prefix is not None:
            inputs = {prefix + k: v for k, v in inputs.items()}
        return TensorsInputs(inputs)
