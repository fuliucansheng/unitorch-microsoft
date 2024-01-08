# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torch.cuda.amp import autocast
from transformers.utils import is_remote_url
from unitorch.utils import pop_value, nested_dict_value
from transformers import MistralConfig, MistralForCausalLM, MistralModel
from unitorch.models import (
    GenericModel,
    GenericOutputs,
    QuantizationConfig,
    QuantizationMixin,
)
from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
    register_model,
)

from unitorch.models.quantization import quantize_model

from unitorch.cli.models import generation_model_decorator
from unitorch.cli.models import ClassificationOutputs, GenerationOutputs, LossOutputs
from unitorch_microsoft.models.mistral import pretrained_mistral_infos


@register_model("microsoft/model/generation/mistral", generation_model_decorator)
class MistralForGeneration(GenericModel, QuantizationMixin):
    prefix_keys_in_state_dict = {
        "^(?!model\.model\.|model\.lm_head\.)model\.": "model.",
        "^lm_head.": "model.",
    }

    def __init__(
        self,
        config_path: str,
        quant_config_path: Optional[str] = None,
        gradient_checkpointing: Optional[bool] = False,
        pad_token_id: Optional[int] = 0,
    ):
        super().__init__()
        self.config = MistralConfig.from_json_file(config_path)
        self.config.gradient_checkpointing = gradient_checkpointing
        self.config.pad_token_id = pad_token_id
        self.model = MistralForCausalLM(self.config)
        self.init_weights()

        if quant_config_path is not None:
            self.quant_config = QuantizationConfig.from_json_file(quant_config_path)
            self.quantize(self.quant_config, ignore_modules=["lm_head"])

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass of the generation model.

        Args:
            input_ids (torch.Tensor, optional): Input tensor of shape (batch_size, sequence_length). Defaults to None.
            attention_mask (torch.Tensor, optional): Attention mask tensor of shape (batch_size, sequence_length). Defaults to None.
            position_ids (torch.Tensor, optional): Position IDs tensor of shape (batch_size, sequence_length). Defaults to None.

        Returns:
            torch Output logits.Tensor: tensor of shape (batch_size, sequence_length, vocab_size).
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
        )
        logits = outputs.logits
        return GenerationOutputs(sequences=logits)

    @classmethod
    @add_default_section_for_init("microsoft/model/generation/mistral")
    def from_core_configure(cls, config, **kwargs):
        """
        Create an instance of MistralLoraForGeneration from a core configuration.

        Args:
            config: The core configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            MistralLoraForGeneration: The initialized MistralLoraForGeneration instance.
        """
        config.set_default_section("microsoft/model/generation/mistral")
        pretrained_name = config.getoption("pretrained_name", "default-mistral")
        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_mistral_infos, pretrained_name, "config"),
        )
        config_path = cached_path(config_path)
        quant_config_path = config.getoption("quant_config_path", None)
        if quant_config_path is not None:
            quant_config_path = cached_path(quant_config_path)

        gradient_checkpointing = config.getoption("gradient_checkpointing", False)

        inst = cls(
            config_path,
            quant_config_path=quant_config_path,
            gradient_checkpointing=gradient_checkpointing,
        )

        weight_path = []
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        pretrained_weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_mistral_infos, pretrained_name, "weight"),
            check_none=False,
        )
        if pretrained_weight_path is not None:
            if isinstance(pretrained_weight_path, str):
                weight_path.append(pretrained_weight_path)
            elif isinstance(pretrained_weight_path, list):
                weight_path.extend(pretrained_weight_path)

        if len(weight_path) > 0:
            inst.from_pretrained(
                weight_path=weight_path,
            )

        return inst

    @add_default_section_for_function("microsoft/model/generation/mistral")
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        num_beams: Optional[int] = 5,
        decoder_start_token_id: Optional[int] = 1,
        decoder_end_token_id: Optional[Union[int, List[int]]] = 2,
        num_return_sequences: Optional[int] = 1,
        min_gen_seq_length: Optional[int] = 0,
        max_gen_seq_length: Optional[int] = 48,
        repetition_penalty: Optional[float] = 1.0,
        no_repeat_ngram_size: Optional[int] = 0,
        early_stopping: Optional[bool] = True,
        length_penalty: Optional[float] = 1.0,
        num_beam_groups: Optional[int] = 1,
        diversity_penalty: Optional[float] = 0.0,
        do_sample: Optional[bool] = False,
        temperature: Optional[float] = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 1.0,
    ):
        """
        Generate text using the generation model.

        Args:
            input_ids: Input tensor of shape (batch_size, sequence_length).
            num_beams (int, optional): Number of beams for beam search. Defaults to 5.
            decoder_start_token_id (int, optional): The ID of the decoder start token. Defaults to 2.
            decoder_end_token_id (int or List[int], optional): The ID(s) of the decoder end token(s). Defaults to 2.
            num_return_sequences (int, optional): Number of generated sequences to return. Defaults to 1.
            min_gen_seq_length (int, optional): Minimum length of generated sequences. Defaults to 0.
            max_gen_seq_length (int, optional): Maximum length of generated sequences. Defaults to 48.
            repetition_penalty (float, optional): Penalty for repeated tokens. Defaults to 1.0.
            no_repeat_ngram_size (int, optional): Size of n-grams to avoid repeating. Defaults to 0.
            early_stopping (bool, optional): Whether to stop generation early. Defaults to True.
            length_penalty (float, optional): Penalty for longer sequences. Defaults to 1.0.
            num_beam_groups (int, optional): Number of beam groups for diverse beam search. Defaults to 1.
            diversity_penalty (float, optional): Penalty for diverse sequences in diverse beam search. Defaults to 0.0.
            do_sample (bool, optional): Whether to use sampling for generation. Defaults to False.
            temperature (float, optional): Sampling temperature. Defaults to 1.0.
            top_k (int, optional): Top-k value for sampling. Defaults to 50.
            top_p (float, optional): Top-p value for sampling. Defaults to 1.0.

        Returns:
            GenericOutputs: Generated sequences and their scores.
        """
        input_seq_length = input_ids.size(1)
        outputs = self.model.generate(
            input_ids=input_ids,
            max_length=max_gen_seq_length + input_seq_length,
            min_length=min_gen_seq_length + input_seq_length,
            num_beams=num_beams,
            do_sample=do_sample,
            decoder_start_token_id=decoder_start_token_id,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=early_stopping,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_return_sequences,
            bos_token_id=decoder_start_token_id,
            eos_token_id=decoder_end_token_id,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            return_dict_in_generate=True,
            output_scores=True,
        )
        sequences = outputs.sequences.reshape(
            -1, num_return_sequences, outputs.sequences.size(-1)
        )
        outputs.sequences = torch.zeros(
            sequences.size(0), num_return_sequences, max_gen_seq_length
        ).to(device=sequences.device)
        outputs.sequences[:, :, : sequences.size(-1) - input_seq_length].copy_(
            sequences[:, :, input_seq_length : sequences.size(-1)]
        )

        if num_return_sequences == 1:
            outputs.sequences = outputs.sequences.reshape(-1, max_gen_seq_length)
        output = GenericOutputs(
            sequences=outputs.sequences.long(),
            sequences_scores=outputs.sequences_scores,
        )
        return GenerationOutputs(
            sequences=outputs.sequences,
            sequences_scores=outputs.sequences_scores,
        )
