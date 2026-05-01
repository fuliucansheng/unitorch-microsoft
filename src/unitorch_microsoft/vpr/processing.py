# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models.bert import BertProcessor
from unitorch.cli import (
    add_default_section_for_init,
    add_default_section_for_function,
    register_process,
)
from unitorch.cli import WriterOutputs
from unitorch.cli.models import (
    TensorInputs,
    GenerationOutputs,
    GenerationTargets,
)
from unitorch.cli.models.visualbert import pretrained_visualbert_infos
from unitorch_microsoft import cached_path


class VisualBertProcessor(BertProcessor):
    def __init__(
        self,
        vocab_path,
        max_seq_length: Optional[int] = 128,
        special_input_ids: Optional[Dict] = dict(),
        do_lower_case: Optional[bool] = True,
        do_basic_tokenize: Optional[bool] = True,
        do_whole_word_mask: Optional[bool] = True,
        masked_lm_prob: Optional[float] = 0.15,
        max_predictions_per_seq: Optional[int] = 20,
    ):
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
    @add_default_section_for_init("microsoft/vpr/process/visualbert")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/vpr/process/visualbert")
        pretrained_name = config.getoption("pretrained_name", "visualbert-vqa-coco-pre")
        vocab_path = config.getoption("vocab_path", None)
        vocab_path = pop_value(
            vocab_path,
            nested_dict_value(
                pretrained_visualbert_infos,
                pretrained_name,
                "vocab",
            ),
        )
        vocab_path = cached_path(vocab_path)

        return {
            "vocab_path": vocab_path,
        }

    @register_process("microsoft/vpr/process/visualbert/classification")
    def _classification(
        self,
        text: str,
        text_pair: Optional[str] = None,
        max_seq_length: Optional[int] = None,
        prefix: Optional[str] = None,
    ):
        outputs = super().classification(
            text=text,
            text_pair=text_pair,
            max_seq_length=max_seq_length,
        )
        inputs = dict(
            input_ids=outputs.input_ids,
            attention_mask=outputs.attention_mask,
            token_type_ids=outputs.token_type_ids,
            position_ids=outputs.position_ids,
        )
        if prefix is not None:
            inputs = {prefix + k: v for k, v in inputs.items()}
        return TensorInputs(inputs)
