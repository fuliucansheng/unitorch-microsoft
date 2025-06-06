# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import random
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from functools import partial

import torch
from transformers import XLMRobertaTokenizer
from unitorch.utils import pop_value, nested_dict_value, truncate_sequence_pair
from unitorch.models import GenericOutputs, HfTextClassificationProcessor
from unitorch.models.bert.processing import get_random_mask_indexes, get_random_word
from unitorch.cli import (
    add_default_section_for_init,
    add_default_section_for_function,
    register_process,
)
from unitorch.cli.models import TensorsInputs
from unitorch_microsoft import cached_path
from unitorch_microsoft.models.tulr import pretrained_tulr_infos


def get_xlm_roberta_tokenizer(vocab_path):
    assert os.path.exists(vocab_path)
    tokenizer = XLMRobertaTokenizer(vocab_path)
    return tokenizer


class TULRV6Processor(HfTextClassificationProcessor):
    def __init__(
        self,
        vocab_path: str,
        max_seq_length: Optional[int] = 128,
        source_type_id: Optional[int] = 0,
        target_type_id: Optional[int] = 0,
        max_num_text: Optional[int] = 25,
    ):
        tokenizer = get_xlm_roberta_tokenizer(
            vocab_path,
        )
        super().__init__(
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            source_type_id=source_type_id,
            target_type_id=target_type_id,
            position_start_id=tokenizer.pad_token_id + 1,
        )
        self.max_num_text = max_num_text
        self.vocab_words = list(self.tokenizer.get_vocab().keys())

    @classmethod
    @add_default_section_for_init("microsoft/msan/l1/process/tulr/v6")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/msan/l1/process/tulr/v6")
        pretrained_name = config.getoption("pretrained_name", "tulrv6-base")
        vocab_path = config.getoption("vocab_path", None)
        vocab_path = pop_value(
            vocab_path,
            nested_dict_value(pretrained_tulr_infos, pretrained_name, "vocab"),
        )
        vocab_path = cached_path(vocab_path)

        return {
            "vocab_path": vocab_path,
        }

    @register_process("microsoft/msan/l1/process/tulr/v6/classification/v2")
    def _classification(
        self,
        text: Union[str, List[str]],
        text_pair: Optional[str] = None,
        max_num_text: Optional[int] = None,
        max_seq_length: Optional[int] = None,
    ):
        if isinstance(text, str):
            text = [text]
        max_num_text = int(pop_value(max_num_text, self.max_num_text))
        num_attention_mask = [1] * len(text[:max_num_text]) + [0] * (
            max_num_text - len(text[:max_num_text])
        )
        texts = text[:max_num_text] + [""] * (max_num_text - len(text[:max_num_text]))
        outputs = [
            self.classification(text, text_pair, max_seq_length) for text in texts
        ]
        return TensorsInputs(
            input_ids=torch.stack([output.input_ids for output in outputs]),
            token_type_ids=torch.stack([output.token_type_ids for output in outputs]),
            attention_mask=torch.stack([output.attention_mask for output in outputs]),
            position_ids=torch.stack([output.position_ids for output in outputs]),
            num_attention_mask=torch.tensor(num_attention_mask),
        )
