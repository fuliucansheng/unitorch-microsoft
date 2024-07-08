# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import re
import torch
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models.bloom import BloomProcessor as _BloomProcessor
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
from unitorch.cli.models.bloom import pretrained_bloom_infos


class BloomProcessor(_BloomProcessor):
    """Processor for Bloom language models."""

    def __init__(
        self,
        tokenizer_file: str,
        max_seq_length: Optional[int] = 128,
        max_gen_seq_length: Optional[int] = 128,
    ):
        """
        Initialize the BloomProcessor.

        Args:
            tokenizer_file (str): The path to the tokenizer file.
            max_seq_length (int, optional): The maximum sequence length. Defaults to 128.
            max_gen_seq_length (int, optional): The maximum generation sequence length. Defaults to 128.
        """
        super().__init__(
            tokenizer_file=tokenizer_file,
            max_seq_length=max_seq_length,
            max_gen_seq_length=max_gen_seq_length,
        )

    @classmethod
    @add_default_section_for_init("microsoft/process/bloom")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/process/bloom")
        pretrained_name = config.getoption("pretrained_name", "default-bloom")
        tokenizer_file = config.getoption("tokenizer_file", None)
        tokenizer_file = pop_value(
            tokenizer_file,
            nested_dict_value(pretrained_bloom_infos, pretrained_name, "tokenizer"),
        )
        tokenizer_file = cached_path(tokenizer_file)

        return {
            "tokenizer_file": tokenizer_file,
        }

    def _instrution_tokenize(self, instruction, encode, max_seq_length):
        tokens1 = self.tokenizer.tokenize(instruction.format(""))
        tokens2 = self.tokenizer.tokenize(str(encode))[
            : max_seq_length - len(tokens1) - 2
        ]
        encode2 = self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(tokens2))
        encode = instruction.format(encode2)
        return self.tokenizer.tokenize(encode)[:max_seq_length]

    @register_process("microsoft/process/bloom/generation/inputs")
    def _generation_inputs(
        self,
        instruction: str,
        encode: str,
        max_seq_length: Optional[int] = None,
    ):
        max_seq_length = pop_value(
            max_seq_length,
            self.max_seq_length,
        )
        tokens = self._instrution_tokenize(instruction, encode, max_seq_length)
        padding = [self.pad_token] * (max_seq_length - len(tokens))
        tokens = padding + tokens
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        assert len(input_ids) == max_seq_length
        return TensorsInputs(input_ids=torch.tensor(input_ids, dtype=torch.long))

    @register_process("microsoft/process/bloom/generation/labels")
    def _generation_labels(
        self,
        decode: str,
        max_gen_seq_length: Optional[int] = None,
    ):
        outputs = super().generation_labels(
            text=decode,
            max_gen_seq_length=max_gen_seq_length,
        )
        return GenerationTargets(
            refs=outputs.input_ids,
            masks=outputs.attention_mask,
        )

    @register_process("microsoft/process/bloom/generation")
    def _generation(
        self,
        instruction: str,
        encode: str,
        decode: str,
        max_seq_length: Optional[int] = None,
        max_gen_seq_length: Optional[int] = None,
    ):
        max_seq_length = pop_value(
            max_seq_length,
            self.max_seq_length,
        )
        max_gen_seq_length = pop_value(
            max_gen_seq_length,
            self.max_gen_seq_length,
        )

        tokens = self._instrution_tokenize(instruction, encode, max_seq_length)
        tokens_pair = self.tokenizer.tokenize(str(decode))[: max_gen_seq_length - 1] + [
            self.eos_token
        ]
        padding_a = [self.pad_token] * (max_seq_length - len(tokens))
        padding_b = [self.pad_token] * (max_gen_seq_length - len(tokens_pair))
        attention_mask = (
            [0] * len(padding_a)
            + [1] * (len(tokens) + len(tokens_pair))
            + [0] * len(padding_b)
        )
        _tokens = padding_a + tokens + tokens_pair + padding_b
        input_ids = self.tokenizer.convert_tokens_to_ids(_tokens)

        tokens_label = tokens_pair + [self.pad_token] * (
            max_gen_seq_length - len(tokens_pair) + 1
        )
        input_ids_label = self.tokenizer.convert_tokens_to_ids(tokens_label)
        input_ids_label = [0] * (max_seq_length - 1) + input_ids_label
        attention_mask_label = [1] * len(tokens_pair) + [0] * (
            max_gen_seq_length - len(tokens_pair) + 1
        )
        attention_mask_label = [0] * (max_seq_length - 1) + attention_mask_label

        return TensorsInputs(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            attention_mask=torch.tensor(attention_mask, dtype=torch.long),
        ), GenerationTargets(
            refs=torch.tensor(input_ids_label, dtype=torch.long),
            masks=torch.tensor(attention_mask_label, dtype=torch.long),
        )

    @register_process("microsoft/postprocess/bloom/detokenize")
    def _detokenize(
        self,
        outputs: GenerationOutputs,
    ):
        """
        Detokenize the generated sequences.

        Args:
            outputs (GenerationOutputs): The generation outputs.

        Returns:
            WriterOutputs: The detokenized writer outputs.
        """
        results = outputs.to_pandas()
        assert results.shape[0] == 0 or results.shape[0] == outputs.sequences.shape[0]

        decoded = super().detokenize(sequences=outputs.sequences)
        cleanup_string = lambda text: re.sub(r"\n", " ", text)
        if isinstance(decoded[0], list):
            decoded = [list(map(cleanup_string, sequence)) for sequence in decoded]
        elif isinstance(decoded[0], str):
            decoded = list(map(cleanup_string, decoded))
        else:
            raise ValueError(
                f"Unsupported type for bloom detokenize: {type(decoded[0])}"
            )
        results["decoded"] = decoded
        return WriterOutputs(results)
