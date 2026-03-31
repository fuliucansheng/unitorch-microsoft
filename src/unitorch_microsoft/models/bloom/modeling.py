# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import torch
import json
import importlib
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torch import autocast
import torch.nn as nn
from transformers import BloomModel, BloomConfig, BloomForCausalLM
from peft import LoraConfig, PeftModelForCausalLM
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models import GenericModel
from unitorch.models.peft import PeftWeightLoaderMixin, GenericPeftModel
from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
    register_model,
)
from unitorch.cli.models import generation_model_decorator
from unitorch.cli.models import ClassificationOutputs, GenerationOutputs, LossOutputs
from unitorch_microsoft.models.bloom import pretrained_bloom_infos, pretrained_bloom_extensions_infos

@register_model("microsoft/model/generation/bloom", generation_model_decorator)
class BloomForGeneration(GenericModel, PeftWeightLoaderMixin):
    """Bloom model for text generation."""

    def __init__(
        self,
        config_path: str,
        num_classes: Optional[int] = 1,
        hidden_dropout_prob: Optional[float] = 0.1,
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
        self.transformer = BloomModel(self.config)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        self.init_weights()

    @classmethod
    @add_default_section_for_init("microsoft/model/generation/bloom")
    def from_core_configure(cls, config, **kwargs):
        """
        Create an instance of BloomForGeneration from the core configuration.

        Args:
            config (Config): The core configuration object.

        Returns:
            BloomForGeneration: An instance of BloomForGeneration initialized with the provided configuration.
        """
        config.set_default_section("microsoft/model/generation/bloom")
        pretrained_name = config.getoption("pretrained_name", "bloom-560m")
        pretrained_lora_name = config.getoption(
            "pretrained_lora_name", "bloom-560m-lora"
        )
        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_bloom_infos, pretrained_name, "config"),
        )

        config_path = cached_path(config_path)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)

        inst = cls(config_path, gradient_checkpointing)
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_bloom_infos, pretrained_name, "weight"),
            check_none=False,
        )

        if weight_path is not None:
            inst.from_pretrained(weight_path)

        pretrained_lora_weight_path = config.getoption(
            "pretrained_lora_weight_path", None
        )
        lora_weight_path = pop_value(
            pretrained_lora_weight_path,
            nested_dict_value(pretrained_bloom_extensions_infos, pretrained_lora_name),
            check_none=False,
        )
        pretrained_lora_weight = config.getoption("pretrained_lora_weight", 1.0)
        pretrained_lora_alpha = config.getoption("pretrained_lora_alpha", 32.0)
        if lora_weight_path is not None:
            inst.load_lora_weights(
                lora_weight_path,
                lora_weights=pretrained_lora_weight,
                lora_alphas=pretrained_lora_alpha,
                save_base_state=False,
            )

        return inst

    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Perform forward pass of the BloomForGeneration model.

        Args:
            input_ids (torch.Tensor): The input token IDs.
            attention_mask (torch.Tensor, optional): The attention mask. Defaults to None.

        Returns:
            GenerationOutputs: The generation outputs.
        """
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        logits = outputs.logits
        return GenerationOutputs(sequences=logits)

    @add_default_section_for_function("microsoft/model/generation/bloom")
    @torch.no_grad()
    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def generate(
        self,
        input_ids: torch.Tensor,
        num_beams: Optional[int] = 5,
        decoder_start_token_id: Optional[int] = 1,
        decoder_end_token_id: Optional[Union[int, List[int]]] = 2,
        decoder_pad_token_id: Optional[int] = 1,
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
            input_ids,
            max_length=max_gen_seq_length + input_seq_length,
            min_length=min_gen_seq_length + input_seq_length,
            num_beams=num_beams,
            do_sample=do_sample,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=early_stopping,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_return_sequences,
            bos_token_id=decoder_start_token_id,
            eos_token_id=decoder_end_token_id,
            pad_token_id=decoder_pad_token_id,
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
        outputs.sequences = (
            torch.zeros(sequences.size(0), num_return_sequences, max_gen_seq_length).to(
                device=sequences.device
            )
            + decoder_start_token_id
        )
        outputs.sequences[:, :, : sequences.size(-1) - input_seq_length].copy_(
            sequences[:, :, input_seq_length : sequences.size(-1)]
        )

        if num_return_sequences == 1:
            outputs.sequences = outputs.sequences.reshape(-1, max_gen_seq_length)

        return GenerationOutputs(
            sequences=outputs.sequences.long(),
            sequences_scores=outputs.sequences_scores,
        )

@register_model("microsoft/model/generation/peft/lora/bloom", generation_model_decorator)
class BloomLoraForGeneration(GenericPeftModel):
    """BloomLora model for generation tasks."""

    prefix_keys_in_state_dict = {"^(?!model\.).*": "model.transformer."}
    replace_keys_in_state_dict = {
        "query_key_value.weight": "query_key_value.base_layer.weight",
        "query_key_value.bias": "query_key_value.base_layer.bias",
    }

    def __init__(
        self,
        config_path: str,
        lora_r: Optional[int] = 16,
        lora_alpha: Optional[int] = 32,
        lora_dropout: Optional[float] = 0.05,
        fan_in_fan_out: Optional[bool] = True,
        target_modules: Optional[Union[List[str], str]] = ["query_key_value"],
        gradient_checkpointing: Optional[bool] = False,
    ):
        """
        Initialize the BloomLoraForGeneration model.

        Args:
            config_path (str): The path to the model configuration file.
            lora_r (int, optional): The number of Lora ranks. Defaults to 16.
            lora_alpha (int, optional): The Lora alpha value. Defaults to 32.
            lora_dropout (float, optional): The Lora dropout rate. Defaults to 0.05.
            fan_in_fan_out (bool, optional): Whether to use fan-in/fan-out weight initialization. Defaults to True.
            target_modules (Union[List[str], str], optional): The target modules for Lora regularization. Defaults to ["q_proj", "v_proj"].
            gradient_checkpointing (bool, optional): Whether to use gradient checkpointing during training. Defaults to False.
        """
        super().__init__()
        self.config = BloomConfig.from_json_file(config_path)
        self.config.gradient_checkpointing = gradient_checkpointing
        self.peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            target_modules=target_modules,
        )
        self.model = BloomForCausalLM(self.config)
        self.model.add_adapter(self.peft_config)
        self.init_weights()

    @classmethod
    @add_default_section_for_init("microsoft/model/generation/peft/lora/bloom")
    def from_core_configure(cls, config, **kwargs):
        """
        Create an instance of BloomLoraForGeneration from a core configuration.

        Args:
            config: The core configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            BloomLoraForGeneration: The initialized BloomLoraForGeneration instance.
        """
        config.set_default_section("microsoft/model/generation/peft/lora/bloom")
        pretrained_name = config.getoption("pretrained_name", "bloom-560m")
        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_bloom_infos, pretrained_name, "config"),
        )
        config_path = cached_path(config_path)

        lora_r = config.getoption("lora_r", 16)
        lora_alpha = config.getoption("lora_alpha", 32)
        lora_dropout = config.getoption("lora_dropout", 0.05)
        fan_in_fan_out = config.getoption("fan_in_fan_out", True)
        target_modules = config.getoption("target_modules", ["query_key_value"])

        gradient_checkpointing = config.getoption("gradient_checkpointing", False)

        inst = cls(
            config_path,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            target_modules=target_modules,
            gradient_checkpointing=gradient_checkpointing,
        )

        weight_path = []
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        pretrained_weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_bloom_infos, pretrained_name, "weight"),
            check_none=False,
        )
        if pretrained_weight_path is not None:
            if isinstance(pretrained_weight_path, str):
                weight_path.append(pretrained_weight_path)
            elif isinstance(pretrained_weight_path, list):
                weight_path.extend(pretrained_weight_path)

        pretrained_lora_weight_path = config.getoption(
            "pretrained_lora_weight_path", None
        )
        if pretrained_lora_weight_path is not None:
            weight_path.append(pretrained_lora_weight_path)

        if len(weight_path) > 0:
            inst.from_pretrained(
                weight_path=weight_path,
            )

        return inst

    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        """
        Perform forward pass of the BloomLoraForGeneration model.

        Args:
            input_ids (torch.Tensor, optional): The input IDs.
            attention_mask (torch.Tensor, optional): The attention mask.
            position_ids (torch.Tensor, optional): The position IDs.

        Returns:
            GenerationOutputs: The output of the generation task.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
        )
        logits = outputs.logits
        return GenerationOutputs(sequences=logits)

    @add_default_section_for_function("microsoft/model/generation/peft/lora/bloom")
    @torch.no_grad()
    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
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
        outputs.sequences = (
            torch.zeros(sequences.size(0), num_return_sequences, max_gen_seq_length).to(
                device=sequences.device
            )
            + decoder_start_token_id
        )
        outputs.sequences[:, :, : sequences.size(-1) - input_seq_length].copy_(
            sequences[:, :, input_seq_length : sequences.size(-1)]
        )

        if num_return_sequences == 1:
            outputs.sequences = outputs.sequences.reshape(-1, max_gen_seq_length)

        return GenerationOutputs(
            sequences=outputs.sequences.long(),
            sequences_scores=outputs.sequences_scores,
        )