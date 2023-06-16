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
        do_whole_word_mask: Optional[bool] = True,
        masked_lm_prob: Optional[float] = 0.15,
        max_predictions_per_seq: Optional[int] = 20,
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
        self.do_whole_word_mask = do_whole_word_mask
        self.masked_lm_prob = masked_lm_prob
        self.max_predictions_per_seq = max_predictions_per_seq
        self.vocab_words = list(self.tokenizer.get_vocab().keys())

    @classmethod
    @add_default_section_for_init("microsoft/adsplus/relevance/process/tulr/v6")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/adsplus/relevance/process/tulr/v6")
        pretrained_name = config.getoption("pretrained_name", "default-tulrv6")
        vocab_path = config.getoption("vocab_path", None)
        vocab_path = pop_value(
            vocab_path,
            nested_dict_value(pretrained_tulr_infos, pretrained_name, "vocab"),
        )
        vocab_path = cached_path(vocab_path)

        return {
            "vocab_path": vocab_path,
        }

    @register_process("microsoft/adsplus/relevance/process/tulr/v6/classification")
    def _classification(
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

        tokens = self.tokenizer.tokenize(str(text))
        if text_pair is None:
            tokens = tokens[: max_seq_length - 2]
            tokens = [self.cls_token] + tokens + [self.sep_token]
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            token_type_ids = [self.source_type_id] * len(input_ids)
            attention_mask = [1] * len(input_ids)
        else:
            tokens_pair = self.tokenizer.tokenize(str(text_pair))
            truncate_sequence_pair(tokens, tokens_pair, max_seq_length - 4)
            token_type_ids = (
                [self.source_type_id]
                + [self.source_type_id] * len(tokens)
                + [self.source_type_id, self.target_type_id]
                + [self.target_type_id] * len(tokens_pair)
                + [self.target_type_id]
            )
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
        token_type_ids += len(padding) * [self.target_type_id]

        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(token_type_ids) == max_seq_length

        inputs = dict(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            token_type_ids=torch.tensor(token_type_ids, dtype=torch.long),
            attention_mask=torch.tensor(attention_mask, dtype=torch.long),
            position_ids=torch.tensor(
                list(
                    range(
                        self.position_start_id,
                        self.position_start_id + max_seq_length,
                    )
                ),
                dtype=torch.long,
            ),
        )
        if prefix is not None:
            inputs = {prefix + k: v for k, v in inputs.items()}
        return TensorsInputs(inputs)
