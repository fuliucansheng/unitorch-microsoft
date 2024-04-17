# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import re
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
from unitorch_microsoft.models.mistral import pretrained_mistral_infos
import torch


class MistralProcessor(_LlamaProcessor):
    """Processor for Mistral models."""

    def __init__(
        self,
        vocab_path,
        max_seq_length: Optional[int] = 128,
        max_gen_seq_length: Optional[int] = 128,
    ):
        """
        Initialize the MistralProcessor.

        Args:
            vocab_path (str): The path to the vocabulary file.
            max_seq_length (int, optional): The maximum sequence length. Defaults to 128.
            max_gen_seq_length (int, optional): The maximum generated sequence length. Defaults to 128.
        """
        super().__init__(
            vocab_file=vocab_path,
            max_seq_length=max_seq_length,
            max_gen_seq_length=max_gen_seq_length,
        )

    @classmethod
    @add_default_section_for_init("microsoft/process/mistral")
    def from_core_configure(cls, config, **kwargs):
        """
        Create an instance of MistralProcessor from a core configuration.

        Args:
            config: The core configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            MistralProcessor: An instance of MistralProcessor.
        """
        config.set_default_section("microsoft/process/mistral")
        pretrained_name = config.getoption("pretrained_name", "default-mistral")
        vocab_path = config.getoption("vocab_path", None)
        vocab_path = pop_value(
            vocab_path,
            nested_dict_value(pretrained_mistral_infos, pretrained_name, "vocab"),
        )
        vocab_path = cached_path(vocab_path)
        return {
            "vocab_path": vocab_path,
        }

    @register_process("microsoft/process/mistral/classification")
    def _classification(
        self,
        text: str,
        text_pair: Optional[str] = None,
        max_seq_length: Optional[int] = None,
    ):
        """
        Process inputs for classification.

        Args:
            text (str): The input text.
            text_pair (str, optional): The second input text for sequence pair classification. Defaults to None.
            max_seq_length (int, optional): The maximum sequence length. Defaults to None.

        Returns:
            TensorsInputs: Processed tensors inputs.
        """
        outputs = super().classification(
            text=text,
            text_pair=text_pair,
            max_seq_length=max_seq_length,
        )
        return TensorsInputs(
            input_ids=outputs.input_ids,
            attention_mask=outputs.attention_mask,
        )

    @register_process("microsoft/process/mistral/pretrain")
    def _pretrain(
        self,
        text: str,
        text_pair: Optional[str] = None,
        max_seq_length: Optional[int] = None,
        max_gen_seq_length: Optional[int] = None,
    ):
        """
        Process inputs for pretraining/generation.

        Args:
            text (str): The input text.
            text_pair (str, optional): The second input text for sequence pair pretraining. Defaults to None.
            max_seq_length (int, optional): The maximum sequence length. Defaults to None.
            max_gen_seq_length (int, optional): The maximum generated sequence length. Defaults to None.

        Returns:
            TensorsInputs: Processed tensors inputs.
        """
        outputs = super().generation(
            text=text,
            text_pair=text_pair,
            max_seq_length=max_seq_length,
            max_gen_seq_length=max_gen_seq_length,
        )
        return TensorsInputs(
            input_ids=outputs.input_ids,
            attention_mask=outputs.attention_mask,
            input_ids_label=outputs.input_ids_label,
            attention_mask_label=outputs.attention_mask_label,
        )

    @register_process("microsoft/process/mistral/prompt")
    def _prompt(
        self,
        text: str,
        max_seq_length: Optional[int] = None,
    ):
        """
        Process inputs for prompt-based generation.

        Args:
            text (str): The input text.
            max_seq_length (int, optional): The maximum sequence length. Defaults to None.

        Returns:
            TensorsInputs: Processed tensors inputs.
        """
        outputs = super().prompt(
            text=text,
            max_seq_length=max_seq_length,
        )
        return TensorsInputs(input_ids=outputs.input_ids)

    def _instrution_tokenize(self, instruction, encode, max_seq_length):
        tokens1 = self.tokenizer.tokenize(instruction.format(""))
        tokens2 = self.tokenizer.tokenize(str(encode))[
            : max_seq_length - len(tokens1) - 2
        ]
        encode2 = self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(tokens2))
        encode = instruction.format(encode2)
        return self.tokenizer.tokenize(encode)[: max_seq_length - 1]

    @register_process("microsoft/process/mistral/generation/inputs")
    def _generation_inputs(
        self,
        instruction: str,
        encode: str,
        max_seq_length: Optional[int] = None,
    ):
        """
        Preprocess the input text for generation tasks.

        Args:
            text (str): The input text.
            max_seq_length (int, optional): The maximum sequence length. Defaults to None.

        Returns:
            TensorsInputs: The processed input tensors.
        """
        max_seq_length = pop_value(
            max_seq_length,
            self.max_seq_length,
        )
        tokens = [self.bos_token] + self._instrution_tokenize(
            instruction, encode, max_seq_length
        )
        padding = [self.pad_token] * (max_seq_length - len(tokens))
        tokens = padding + tokens
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        assert len(input_ids) == max_seq_length
        return TensorsInputs(input_ids=torch.tensor(input_ids, dtype=torch.long))

    @register_process("microsoft/process/mistral/generation/labels")
    def _generation_labels(
        self,
        text: str,
        max_gen_seq_length: Optional[int] = None,
    ):
        """
        Preprocess the target text for generation tasks.

        Args:
            text (str): The target text.
            max_gen_seq_length (int, optional): The maximum generation sequence length. Defaults to None.

        Returns:
            GenerationTargets: The processed generation targets.
        """
        outputs = super().generation_labels(
            text=text,
            max_gen_seq_length=max_gen_seq_length,
        )
        return GenerationTargets(
            refs=outputs.input_ids,
            masks=outputs.attention_mask,
        )

    @register_process("microsoft/process/mistral/generation")
    def _generation(
        self,
        instruction: str,
        encode: str,
        decode: str,
        max_seq_length: Optional[int] = None,
        max_gen_seq_length: Optional[int] = None,
    ):
        """
        Preprocess the input and target texts for generation tasks.

        Args:
            text (str): The input text.
            text_pair (str, optional): The paired input text. Defaults to None.
            max_seq_length (int, optional): The maximum sequence length. Defaults to None.
            max_gen_seq_length (int, optional): The maximum generation sequence length. Defaults to None.

        Returns:
            Tuple[TensorsInputs, GenerationTargets]: The processed input tensors and generation targets.
        """
        max_seq_length = pop_value(
            max_seq_length,
            self.max_seq_length,
        )
        max_gen_seq_length = pop_value(
            max_gen_seq_length,
            self.max_gen_seq_length,
        )

        tokens = [self.bos_token] + self._instrution_tokenize(
            instruction, encode, max_seq_length
        )
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

    @register_process("microsoft/postprocess/mistral/detokenize")
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
                f"Unsupported type for minigpt4 detokenize: {type(decoded[0])}"
            )
        results["decoded"] = decoded
        return WriterOutputs(results)
