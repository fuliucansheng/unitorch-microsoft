# Copyright (c) MICROSOFT.
# Licensed under the MIT License.
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.utils import pop_value, nested_dict_value
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
from unitorch.cli.models.bert import pretrained_bert_infos
from unitorch.cli.models.bert import BertProcessor


class TribertV9Processor(BertProcessor):
    def __init__(
        self,
        vocab_path: str,
        max_seq_length: Optional[int] = 128,
        special_input_ids: Optional[Dict] = dict(),
        do_lower_case: Optional[bool] = True,
        do_basic_tokenize: Optional[bool] = True,
        do_whole_word_mask: Optional[bool] = True,
        masked_lm_prob: Optional[float] = 0.15,
        max_predictions_per_seq: Optional[int] = 20,
    ):
        """
        Initialize TribertProcessor.

        Args:
            vocab_path (str): The path to the vocabulary file.
            max_seq_length (int, optional): The maximum sequence length. Defaults to 128.
            special_input_ids (Dict, optional): Special input IDs. Defaults to an empty dictionary.
            do_lower_case (bool, optional): Whether to lower case the input text. Defaults to True.
            do_basic_tokenize (bool, optional): Whether to perform basic tokenization. Defaults to True.
            do_whole_word_mask (bool, optional): Whether to use whole word masking. Defaults to True.
            masked_lm_prob (float, optional): The probability of masking a token for masked language modeling. Defaults to 0.15.
            max_predictions_per_seq (int, optional): The maximum number of masked LM predictions per sequence. Defaults to 20.
        """
        super().__init__(
            vocab_path=vocab_path,
            max_seq_length=max_seq_length,
            special_input_ids=special_input_ids,
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            do_whole_word_mask=do_whole_word_mask,
            masked_lm_prob=masked_lm_prob,
            max_predictions_per_seq=max_predictions_per_seq,
        )

    @classmethod
    @add_default_section_for_init("microsoft/process/tribert/v9")
    def from_core_configure(cls, config, **kwargs):
        """
        Create an instance of BertProcessor from a core configuration.

        Args:
            config: The core configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            BertProcessor: An instance of BertProcessor.
        """
        config.set_default_section("microsoft/process/tribert/v9")
        pretrained_name = config.getoption("pretrained_name", "bert-base-uncased")
        vocab_path = config.getoption("vocab_path", None)
        vocab_path = pop_value(
            vocab_path,
            nested_dict_value(pretrained_bert_infos, pretrained_name, "vocab"),
        )
        vocab_path = cached_path(vocab_path)

        return {
            "vocab_path": vocab_path,
        }

    @register_process("microsoft/process/tribert/v9/classification")
    def _classification(
        self,
        text,
        max_seq_length: Optional[int] = None,
        input_ids_name: Optional[str] = "input_ids",
        attention_mask_name: Optional[str] = "attention_mask",
        token_type_ids_name: Optional[str] = "token_type_ids",
        position_ids_name: Optional[str] = "position_ids",
    ):
        outputs = super().classification(
            text=text,
            max_seq_length=max_seq_length,
        )

        return TensorsInputs(
            {
                input_ids_name: outputs.input_ids,
                attention_mask_name: outputs.attention_mask,
                token_type_ids_name: outputs.token_type_ids,
                position_ids_name: outputs.position_ids,
            }
        )
