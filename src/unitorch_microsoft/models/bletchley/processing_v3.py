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


class BletchleyProcessor:
    def __init__(
        self,
        max_seq_length: Optional[int] = 128,
        pixel_mean: List[float] = [0.5, 0.5, 0.5],
        pixel_std: List[float] = [0.5, 0.5, 0.5],
        resize_shape: Optional[List[int]] = [224, 224],
        crop_shape: Optional[List[int]] = [224, 224],
    ):
        self.pixel_mean = torch.tensor(pixel_mean)
        self.pixel_std = torch.tensor(pixel_std)
        self.resize_shape = resize_shape
        self.crop_shape = crop_shape
        self.image_transform = Compose(
            [
                Resize(self.resize_shape),
                CenterCrop(self.crop_shape),
                ToTensor(),
                Normalize(self.pixel_mean, self.pixel_std),
            ]
        )
        self.tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-large")
        self.pad_token = self.tokenizer.pad_token
        self.sep_token = self.tokenizer.sep_token
        self.cls_token = self.tokenizer.cls_token
        self.mask_token_id = self.tokenizer.mask_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.max_seq_length = max_seq_length
        self.vocab_size = self.tokenizer.vocab_size

    @classmethod
    @add_default_section_for_init("microsoft/process/bletchley/v3")
    def from_core_configure(cls, config, **kwargs):
        pass

    def _tokenize(
        self,
        text,
        text_pair: Optional[str] = None,
        max_seq_length: Optional[int] = None,
    ):
        tokens = self.tokenizer.tokenize(str(text))
        if text_pair is None:
            tokens = tokens[: max_seq_length - 2]
            tokens = [self.cls_token] + tokens + [self.sep_token]
        else:
            tokens_pair = self.tokenizer.tokenize(str(text_pair))
            truncate_sequence_pair(tokens, tokens_pair, max_seq_length - 4)
            tokens = (
                [self.cls_token]
                + tokens
                + [self.sep_token, self.sep_token]
                + tokens_pair
                + [self.sep_token]
            )

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += len(padding) * [self.pad_token_id]
        attention_mask += padding
        return GenericOutputs(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            attention_mask=torch.tensor(attention_mask, dtype=torch.long),
        )

    @register_process("microsoft/process/bletchley/v3/classification")
    def _classification(
        self,
        text: str,
        image: Union[Image.Image, str],
        text_pair: Optional[str] = None,
        max_seq_length: Optional[int] = None,
        prefix: Optional[str] = None,
    ):
        max_seq_length = pop_value(
            max_seq_length,
            self.max_seq_length,
        )
        if isinstance(image, str):
            image = Image.open(image)
        image = self.image_transform(image)

        outputs = self._tokenize(
            text=text,
            text_pair=text_pair,
            max_seq_length=max_seq_length,
        )

        inputs = dict(
            input_ids=outputs.input_ids,
            attention_mask=outputs.attention_mask,
            images=image,
        )
        if prefix is not None:
            inputs = {prefix + k: v for k, v in inputs.items()}

        return TensorsInputs(inputs)

    @register_process("microsoft/process/bletchley/v3/text_classification")
    def _text_classification(
        self,
        text: str,
        text_pair: Optional[str] = None,
        max_seq_length: Optional[int] = None,
        prefix: Optional[str] = None,
    ):
        max_seq_length = pop_value(
            max_seq_length,
            self.max_seq_length,
        )
        outputs = self._tokenize(
            text=text,
            text_pair=text_pair,
            max_seq_length=max_seq_length,
        )
        inputs = dict(
            input_ids=outputs.input_ids,
            attention_mask=outputs.attention_mask,
        )
        if prefix is not None:
            inputs = {prefix + k: v for k, v in inputs.items()}
        return TensorsInputs(inputs)

    @register_process("microsoft/process/bletchley/v3/image_classification")
    def _image_classification(
        self,
        image: Union[Image.Image, str],
        prefix: Optional[str] = None,
    ):
        if isinstance(image, str):
            image = Image.open(image)
        image = self.image_transform(image)
        inputs = dict(
            images=image,
        )
        if prefix is not None:
            inputs = {prefix + k: v for k, v in inputs.items()}
        return TensorsInputs(inputs)
