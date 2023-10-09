# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
import json
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torch.cuda.amp import autocast
from transformers.utils import is_remote_url
from transformers import BloomModel, BloomConfig, BloomForCausalLM
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models import GenericModel
from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
    register_model,
)
from unitorch.cli.models import generation_model_decorator
from unitorch.cli.models import ClassificationOutputs, GenerationOutputs, LossOutputs
from unitorch.cli.models.bloom import pretrained_bloom_infos
from unitorch_microsoft import is_auto_gptq_available


@register_model("microsoft/model/generation/bloom/gptq", generation_model_decorator)
class BloomGPTQForGeneration(GenericModel):
    """Bloom model for text generation."""

    prefix_keys_in_state_dict = {
        "^(?!model\.).*": "model.model.transformer.",
        "^(?!model\.model\.).*": "model.",
    }

    def __init__(
        self,
        config_path: str,
        gptq_quant_config_path: str,
        gradient_checkpointing: Optional[bool] = False,
    ):
        """
        Initialize the BloomForGeneration model.

        Args:
            config_path (str): The path to the model configuration file.
            gradient_checkpointing (bool, optional): Whether to use gradient checkpointing during training. Defaults to False.
        """
        super().__init__()
        self.config = BloomConfig.from_json_file(config_path)
        self.config.gradient_checkpointing = gradient_checkpointing
        self.model = BloomForCausalLM(self.config)
        assert (
            is_auto_gptq_available()
        ), "Please install auto-gptq to use BloomGPTQForGeneration."

        from auto_gptq import BaseQuantizeConfig
        from auto_gptq.modeling import BloomGPTQForCausalLM

        quant_config_dict = json.load(open(gptq_quant_config_path))
        quant_config = BaseQuantizeConfig(**quant_config_dict)

        self.model = BloomGPTQForCausalLM(
            BloomForCausalLM(self.config), quantized=True, quantize_config=quant_config
        )
        self.init_weights()

    @classmethod
    @add_default_section_for_init("microsoft/model/generation/bloom/gptq")
    def from_core_configure(cls, config, **kwargs):
        """
        Create an instance of BloomForGeneration from the core configuration.

        Args:
            config (Config): The core configuration object.

        Returns:
            BloomForGeneration: An instance of BloomForGeneration initialized with the provided configuration.
        """
        config.set_default_section("microsoft/model/generation/bloom/gptq")
        pretrained_name = config.getoption("pretrained_name", "default-bloom")
        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_bloom_infos, pretrained_name, "config"),
        )

        config_path = cached_path(config_path)
        gptq_quant_config_path = config.getoption("gptq_quant_config_path", None)
        gptq_quant_config_path = cached_path(gptq_quant_config_path)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)

        inst = cls(config_path, gptq_quant_config_path, gradient_checkpointing)
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_bloom_infos, pretrained_name, "weight"),
            check_none=False,
        )

        if weight_path is not None:
            inst.from_pretrained(weight_path)

        return inst

    def forward(
        self,
    ):
        raise NotImplementedError

    @add_default_section_for_function("microsoft/model/generation/bloom/gptq")
    @torch.no_grad()
    @autocast()
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
        Generate sequences using the Bloom model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            num_beams (int, optional): Number of beams for beam search. Defaults to 5.
            decoder_start_token_id (int, optional): Decoder start token ID. Defaults to 0.
            decoder_end_token_id (int or List[int], optional): The ID(s) of the decoder end token(s). Defaults to 1.
            num_return_sequences (int, optional): Number of generated sequences to return. Defaults to 1.
            min_gen_seq_length (int, optional): Minimum generation sequence length. Defaults to 0.
            max_gen_seq_length (int, optional): Maximum generation sequence length. Defaults to 48.
            repetition_penalty (float, optional): Repetition penalty. Defaults to 1.0.
            no_repeat_ngram_size (int, optional): Size of n-grams to prevent repetition. Defaults to 0.
            early_stopping (bool, optional): Whether to perform early stopping. Defaults to True.
            length_penalty (float, optional): Length penalty. Defaults to 1.0.
            num_beam_groups (int, optional): Number of beam groups for diverse beam search. Defaults to 1.
            diversity_penalty (float, optional): Diversity penalty for diverse beam search. Defaults to 0.0.
            do_sample (bool, optional): Whether to use sampling for generation. Defaults to False.
            temperature (float, optional): Sampling temperature. Defaults to 1.0.
            top_k (int, optional): Top-k sampling parameter. Defaults to 50.
            top_p (float, optional): Top-p sampling parameter. Defaults to 1.0.

        Returns:
            GenerationOutputs: The generation outputs.
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

        return GenerationOutputs(
            sequences=outputs.sequences.long(),
            sequences_scores=outputs.sequences_scores,
        )
