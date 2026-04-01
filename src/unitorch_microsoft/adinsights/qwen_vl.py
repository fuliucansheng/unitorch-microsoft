# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import re
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import torch
from torch import autocast
from transformers.models.qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLModel,
    Qwen3VLForConditionalGeneration,
)
from unitorch.utils import pop_value, nested_dict_value, is_bfloat16_available
from unitorch.models import (
    GenericModel,
    GenericOutputs,
)

from unitorch.models.peft import PeftWeightLoaderMixin
from unitorch.models.qwen import (
    QWen3VLForGeneration as _QWen3VLForGeneration,
    QWenVLProcessor as _QWenVLProcessor,
)
from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
    register_model,
    register_process,
    hf_endpoint_url,
)
from unitorch.cli import WriterOutputs
from unitorch.cli.models import generation_model_decorator
from unitorch.cli.models import (
    TensorsInputs,
    GenerationOutputs,
    GenerationTargets,
)
from unitorch.cli.models.qwen import pretrained_qwen_infos, pretrained_qwen_extensions_infos

pretrained_qwen_infos.update(
    {
        "qwen3-vl-2b-instruct-img-lp-relevance": {
            "config": hf_endpoint_url(
                "/Qwen/Qwen3-VL-2B-Instruct/resolve/main/config.json"
            ),
            "tokenizer": hf_endpoint_url(
                "/Qwen/Qwen3-VL-2B-Instruct/resolve/main/tokenizer.json"
            ),
            "vision_config": hf_endpoint_url(
                "/Qwen/Qwen3-VL-2B-Instruct/resolve/main/preprocessor_config.json"
            ),
            "tokenizer_config": hf_endpoint_url(
                "/Qwen/Qwen3-VL-2B-Instruct/resolve/main/tokenizer_config.json"
            ),
            "chat_template": hf_endpoint_url(
                "/Qwen/Qwen3-VL-2B-Instruct/resolve/main/chat_template.json"
            ),
            "weight": hf_endpoint_url(
                "/Qwen/Qwen3-VL-2B-Instruct/resolve/main/model.safetensors"
            ),
        },
    }
)

@register_model("microsoft/model/generation/qwen3_vl/lp_image_relevance/v1", generation_model_decorator)
class QWen3VLForGeneration(GenericModel, PeftWeightLoaderMixin):
    """Qwen3 model for text generation."""
    prefix_keys_in_state_dict = {
        "^model.visual.": "model.",
        "^model(?!\.model).": "model.",
    }

    def __init__(
        self,
        config_path: str,
        gradient_checkpointing: Optional[bool] = False,
    ):
        """
        Initialize the BloomForGeneration model.

        Args:
            config_path (str): The path to the model configuration file.
            gradient_checkpointing (bool, optional): Whether to use gradient checkpointing during training. Defaults to False.
        """
        super().__init__()
        self.config = Qwen3VLConfig.from_json_file(config_path)
        self.config.gradient_checkpointing = gradient_checkpointing
        self.model = Qwen3VLForConditionalGeneration(self.config)
        self.init_weights()
        self.good_token_id = 15216
        self.fair_token_id = 60795
        self.bad_token_id = 17082

    @classmethod
    @add_default_section_for_init("microsoft/model/generation/qwen3_vl/lp_image_relevance/v1")
    def from_core_configure(cls, config, **kwargs):
        """
        Create an instance of BloomForGeneration from the core configuration.

        Args:
            config (Config): The core configuration object.

        Returns:
            BloomForGeneration: An instance of BloomForGeneration initialized with the provided configuration.
        """
        config.set_default_section("microsoft/model/generation/qwen3_vl/lp_image_relevance/v1")
        pretrained_name = config.getoption("pretrained_name", "qwen3-vl-8b-instruct")
        pretrained_lora_name = config.getoption("pretrained_lora_name", None)
        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_qwen_infos, pretrained_name, "config"),
        )

        config_path = cached_path(config_path)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)

        inst = cls(config_path, gradient_checkpointing)
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_qwen_infos, pretrained_name, "weight"),
            check_none=False,
        )

        if weight_path is not None:
            inst.from_pretrained(weight_path)

        pretrained_lora_weight_path = config.getoption(
            "pretrained_lora_weight_path", None
        )
        lora_weight_path = pop_value(
            pretrained_lora_weight_path,
            nested_dict_value(pretrained_qwen_extensions_infos, pretrained_lora_name),
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

    @autocast(
        device_type=("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=(torch.bfloat16 if is_bfloat16_available() else torch.float32),
    )
    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
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
        image_grid_thw = image_grid_thw.view(-1, image_grid_thw.size(-1))
        pixel_values = pixel_values.view(-1, pixel_values.size(-1))
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
        )

        logits = outputs.logits
        return GenerationOutputs(sequences=logits)

    @add_default_section_for_function("microsoft/model/generation/qwen3_vl/lp_image_relevance/v1")
    @torch.no_grad()
    @autocast(
        device_type=("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=(torch.bfloat16 if is_bfloat16_available() else torch.float32),
    )
    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        num_beams: Optional[int] = 5,
        decoder_start_token_id: Optional[int] = 151643,
        decoder_end_token_id: Optional[Union[int, List[int]]] = 151645,
        decoder_pad_token_id: Optional[int] = 151643,
        num_return_sequences: Optional[int] = 1,
        min_gen_seq_length: Optional[int] = 0,
        max_gen_seq_length: Optional[int] = 512,
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
        image_grid_thw = image_grid_thw.view(-1, image_grid_thw.size(-1))
        pixel_values = pixel_values.view(-1, pixel_values.size(-1))

        outputs = self.model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
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
            output_logits=True,
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

        logits = outputs.logits[0].softmax(dim=-1)
        logits = logits.reshape(-1, num_beams, logits.size(-1))[:, 0, :]
        good = logits[:, self.good_token_id]
        fair = logits[:, self.fair_token_id]
        bad = logits[:, self.bad_token_id]
        scores = torch.maximum(good, fair) - bad

        return GenerationOutputs(
            sequences=outputs.sequences.long(),
            sequences_scores=scores,
        )

class QWenVLProcessor(_QWenVLProcessor):
    """Processor for Bloom language models."""

    def __init__(
        self,
        tokenizer_file: str,
        vision_config_path: str,
        tokenizer_config: Optional[str] = None,
        special_tokens_map: Optional[str] = None,
        chat_template: Optional[str] = None,
        max_seq_length: Optional[int] = 128,
        max_gen_seq_length: Optional[int] = 128,
    ):
        """
        Initialize the BloomProcessor.

        Args:
            vocab_path (str): The path to the vocabulary file.
            merge_path (str): The path to the merges file.
            max_seq_length (int, optional): The maximum sequence length. Defaults to 128.
            max_gen_seq_length (int, optional): The maximum generation sequence length. Defaults to 128.
        """
        super().__init__(
            tokenizer_file=tokenizer_file,
            vision_config_path=vision_config_path,
            tokenizer_config=tokenizer_config,
            special_tokens_map=special_tokens_map,
            chat_template=chat_template,
            max_seq_length=max_seq_length,
            max_gen_seq_length=max_gen_seq_length,
        )

    @classmethod
    @add_default_section_for_init("microsoft/process/qwen_vl/lp_image_relevance/v1")
    def from_core_configure(cls, config, **kwargs):
        """
        Create an instance of BloomProcessor from the core configuration.

        Args:
            config (Config): The core configuration object.

        Returns:
            BloomProcessor: An instance of BloomProcessor initialized with the provided configuration.
        """
        config.set_default_section("microsoft/process/qwen_vl/lp_image_relevance/v1")
        pretrained_name = config.getoption("pretrained_name", "qwen3-vl-8b-instruct")
        tokenizer_file = config.getoption("tokenizer_file", None)
        tokenizer_file = pop_value(
            tokenizer_file,
            nested_dict_value(pretrained_qwen_infos, pretrained_name, "tokenizer"),
        )
        tokenizer_file = cached_path(tokenizer_file)

        vision_config_path = config.getoption("vision_config_path", None)
        vision_config_path = pop_value(
            vision_config_path,
            nested_dict_value(pretrained_qwen_infos, pretrained_name, "vision_config"),
        )
        vision_config_path = cached_path(vision_config_path)

        tokenizer_config = config.getoption("tokenizer_config", None)
        tokenizer_config = pop_value(
            tokenizer_config,
            nested_dict_value(
                pretrained_qwen_infos, pretrained_name, "tokenizer_config"
            ),
            check_none=False,
        )
        tokenizer_config = (
            cached_path(tokenizer_config) if tokenizer_config is not None else None
        )

        special_tokens_map = config.getoption("special_tokens_map", None)
        special_tokens_map = pop_value(
            special_tokens_map,
            nested_dict_value(
                pretrained_qwen_infos, pretrained_name, "special_tokens_map"
            ),
            check_none=False,
        )
        special_tokens_map = (
            cached_path(special_tokens_map) if special_tokens_map is not None else None
        )

        chat_template = config.getoption("chat_template", None)
        chat_template = pop_value(
            chat_template,
            nested_dict_value(pretrained_qwen_infos, pretrained_name, "chat_template"),
            check_none=False,
        )
        chat_template = (
            cached_path(chat_template) if chat_template is not None else None
        )

        return {
            "tokenizer_file": tokenizer_file,
            "vision_config_path": vision_config_path,
            "tokenizer_config": tokenizer_config,
            "special_tokens_map": special_tokens_map,
            "chat_template": chat_template,
        }


    @register_process("microsoft/postprocess/qwen_vl/lp_image_relevance/v1")
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
        scores = outputs.sequences_scores.cpu().numpy() if outputs.sequences_scores is not None else None

        decoded = super().detokenize(sequences=outputs.sequences)
        cleanup_string = lambda text: re.sub(r"\n", " ", text)
        if isinstance(decoded[0], list):
            decoded = [list(map(cleanup_string, sequence)) for sequence in decoded]
        elif isinstance(decoded[0], str):
            decoded = list(map(cleanup_string, decoded))
        else:
            raise ValueError(
                f"Unsupported type for Qwen detokenize: {type(decoded[0])}"
            )
        results["decoded"] = decoded
        results["scores"] = scores
        return WriterOutputs(results)