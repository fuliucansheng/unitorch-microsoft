# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import re
import torch
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models.llama import LlamaProcessor as _LlamaProcessor
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
from unitorch.cli.models.llama import pretrained_llama_infos


class LlamaProcessor(_LlamaProcessor):
    """Processor for Llama models."""

    def __init__(
        self,
        vocab_path,
        max_seq_length: Optional[int] = 128,
        max_gen_seq_length: Optional[int] = 128,
    ):
        """
        Initialize the LlamaProcessor.

        Args:
            vocab_path (str): The path to the vocabulary file.
            max_seq_length (int, optional): The maximum sequence length. Defaults to 128.
            max_gen_seq_length (int, optional): The maximum generated sequence length. Defaults to 128.
        """
        super().__init__(
            vocab_path=vocab_path,
            max_seq_length=max_seq_length,
            max_gen_seq_length=max_gen_seq_length,
        )

    @classmethod
    @add_default_section_for_init("microsoft/process/llama")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/process/llama")
        pretrained_name = config.getoption("pretrained_name", "llama-7b")
        vocab_path = config.getoption("vocab_path", None)
        vocab_path = pop_value(
            vocab_path,
            nested_dict_value(pretrained_llama_infos, pretrained_name, "vocab"),
        )
        vocab_path = cached_path(vocab_path)

        return {
            "vocab_path": vocab_path,
        }
