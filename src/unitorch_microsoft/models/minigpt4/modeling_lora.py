# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torch.cuda.amp import autocast
from transformers.utils import is_remote_url
from unitorch.utils import pop_value, nested_dict_value
from transformers import LlamaConfig, LlamaForCausalLM, LlamaModel
from peft import LoraConfig, PeftModelForCausalLM
from transformers.models.blip_2.modeling_blip_2 import (
    Blip2Config,
    Blip2VisionModel,
    Blip2QFormerModel,
)
from unitorch.models import GenericModel, GenericOutputs, QuantizationConfig, QuantizationMixin
from unitorch.models.quantization import quantize_model
from unitorch.models.peft import PeftModelForSequenceClassification, GenericPeftModel
from unitorch.models.minigpt4.modeling import MiniGPT4Blip2LlamaModel
from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
    register_model,
)
from unitorch.cli.models import generation_model_decorator
from unitorch.cli.models import ClassificationOutputs, GenerationOutputs, LossOutputs
from unitorch.cli.models.minigpt4 import pretrained_minigpt4_infos


class MiniGPT4Blip2LlamaLoraClassificationModel(nn.Module):
    """
    MiniGPT4Blip2LlamaModel is a model that combines the Blip2VisionModel, Blip2QFormerModel, and LlamaForCausalLM
    models for generation. It inherits from the nn.Module class.
    """

    def __init__(
        self,
        blip2_config: Blip2Config,
        llama_config: LlamaConfig,
        peft_config: LoraConfig,
        quant_config: QuantizationConfig = None,
    ):
        """
        Initializes a MiniGPT4Blip2LlamaModel instance.

        Args:
            blip2_config (Blip2Config): The configuration for the Blip2 model.
            llama_config (LlamaConfig): The configuration for the Llama model.
        """
        super().__init__()
        self.blip2_config = blip2_config
        self.vision_model = Blip2VisionModel(self.blip2_config.vision_config)

        self.query_tokens = nn.Parameter(
            torch.zeros(
                1,
                self.blip2_config.num_query_tokens,
                self.blip2_config.qformer_config.hidden_size,
            )
        )
        self.qformer = Blip2QFormerModel(self.blip2_config.qformer_config)

        self.llama_config = llama_config
        self.language_projection = nn.Linear(
            self.blip2_config.qformer_config.hidden_size, self.llama_config.hidden_size
        )
        model = LlamaModel(self.llama_config)
        self.quant_config = quant_config
        if self.quant_config is not None:
            ignore_modules = peft_config.target_modules + ["lm_head"]
            model = quantize_model(model, self.quant_config, ignore_modules=ignore_modules)

        self.peft_config = peft_config
        self.model = PeftModelForSequenceClassification(
            model,
            self.peft_config,
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        prefix_input_ids: torch.Tensor,
        suffix_input_ids: torch.Tensor,
        prefix_attention_mask: Optional[torch.Tensor] = None,
        suffix_attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Performs a forward pass of the MiniGPT4Blip2LlamaModel.

        Args:
            pixel_values (torch.Tensor): The pixel values.
            prefix_input_ids (torch.Tensor): The input IDs for the prefix tokens.
            suffix_input_ids (torch.Tensor): The input IDs for the suffix tokens.
            prefix_attention_mask (torch.Tensor, optional): The attention mask for the prefix tokens.
            suffix_attention_mask (torch.Tensor, optional): The attention mask for the suffix tokens.

        Returns:
            outputs: The model outputs.
        """
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
        )
        image_embeds = vision_outputs[0]
        image_attention_mask = torch.ones(
            image_embeds.size()[:-1],
            dtype=torch.long,
            device=image_embeds.device,
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
        )
        query_embeds = query_outputs[0]

        language_model_inputs = self.language_projection(query_embeds)
        language_model_attention_mask = torch.ones(
            language_model_inputs.size()[:-1],
            dtype=torch.long,
            device=language_model_inputs.device,
        )
        prefix_inputs_embeds = self.model.get_input_embeddings()(prefix_input_ids)
        suffix_inputs_embeds = self.model.get_input_embeddings()(suffix_input_ids)
        inputs_embeds = torch.cat(
            [
                prefix_inputs_embeds,
                language_model_inputs,
                suffix_inputs_embeds,
            ],
            dim=1,
        )
        expected_device = language_model_attention_mask.device

        if prefix_attention_mask is None:
            prefix_attention_mask = torch.ones(
                prefix_inputs_embeds.size()[:-1],
                dtype=torch.long,
                device=expected_device,
            )

        if suffix_attention_mask is None:
            suffix_attention_mask = torch.ones(
                suffix_inputs_embeds.size()[:-1],
                dtype=torch.long,
                device=expected_device,
            )

        attention_mask = torch.cat(
            [
                prefix_attention_mask,
                language_model_attention_mask,
                suffix_attention_mask,
            ],
            dim=1,
        )
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return outputs


class MiniGPT4Blip2LlamaLoraGenerationModel(nn.Module):
    """
    MiniGPT4Blip2LlamaModel is a model that combines the Blip2VisionModel, Blip2QFormerModel, and LlamaForCausalLM
    models for generation. It inherits from the nn.Module class.
    """

    def __init__(
        self,
        blip2_config: Blip2Config,
        llama_config: LlamaConfig,
        peft_config: LoraConfig,
        quant_config: QuantizationConfig = None,
    ):
        """
        Initializes a MiniGPT4Blip2LlamaModel instance.

        Args:
            blip2_config (Blip2Config): The configuration for the Blip2 model.
            llama_config (LlamaConfig): The configuration for the Llama model.
        """
        super().__init__()
        self.blip2_config = blip2_config
        self.vision_model = Blip2VisionModel(self.blip2_config.vision_config)

        self.query_tokens = nn.Parameter(
            torch.zeros(
                1,
                self.blip2_config.num_query_tokens,
                self.blip2_config.qformer_config.hidden_size,
            )
        )
        self.qformer = Blip2QFormerModel(self.blip2_config.qformer_config)

        self.llama_config = llama_config
        self.language_projection = nn.Linear(
            self.blip2_config.qformer_config.hidden_size, self.llama_config.hidden_size
        )
        model = LlamaForCausalLM(self.llama_config)
        self.quant_config = quant_config
        if self.quant_config is not None:
            ignore_modules = peft_config.target_modules + ["lm_head"]
            model = quantize_model(model, self.quant_config, ignore_modules=ignore_modules)

        self.peft_config = peft_config
        self.llama = PeftModelForCausalLM(
            model,
            self.peft_config,
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        prefix_input_ids: torch.Tensor,
        suffix_input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        prefix_attention_mask: Optional[torch.Tensor] = None,
        suffix_attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Performs a forward pass of the MiniGPT4Blip2LlamaModel.

        Args:
            pixel_values (torch.Tensor): The pixel values.
            prefix_input_ids (torch.Tensor): The input IDs for the prefix tokens.
            suffix_input_ids (torch.Tensor): The input IDs for the suffix tokens.
            decoder_input_ids (torch.Tensor): The input IDs for the decoder tokens.
            prefix_attention_mask (torch.Tensor, optional): The attention mask for the prefix tokens.
            suffix_attention_mask (torch.Tensor, optional): The attention mask for the suffix tokens.
            decoder_attention_mask (torch.Tensor, optional): The attention mask for the decoder tokens.

        Returns:
            outputs: The model outputs.
        """
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
        )
        image_embeds = vision_outputs[0]
        image_attention_mask = torch.ones(
            image_embeds.size()[:-1],
            dtype=torch.long,
            device=image_embeds.device,
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
        )
        query_embeds = query_outputs[0]

        language_model_inputs = self.language_projection(query_embeds)
        language_model_attention_mask = torch.ones(
            language_model_inputs.size()[:-1],
            dtype=torch.long,
            device=language_model_inputs.device,
        )
        prefix_inputs_embeds = self.llama.get_input_embeddings()(prefix_input_ids)
        suffix_inputs_embeds = self.llama.get_input_embeddings()(suffix_input_ids)
        decoder_input_embeds = self.llama.get_input_embeddings()(decoder_input_ids)
        inputs_embeds = torch.cat(
            [
                prefix_inputs_embeds,
                language_model_inputs,
                suffix_inputs_embeds,
                decoder_input_embeds,
            ],
            dim=1,
        )
        expected_device = language_model_attention_mask.device

        if prefix_attention_mask is None:
            prefix_attention_mask = torch.ones(
                prefix_inputs_embeds.size()[:-1],
                dtype=torch.long,
                device=expected_device,
            )

        if suffix_attention_mask is None:
            suffix_attention_mask = torch.ones(
                suffix_inputs_embeds.size()[:-1],
                dtype=torch.long,
                device=expected_device,
            )

        if decoder_attention_mask is None:
            decoder_attention_mask = torch.ones(
                decoder_input_embeds.size()[:-1],
                dtype=torch.long,
                device=expected_device,
            )

        attention_mask = torch.cat(
            [
                prefix_attention_mask,
                language_model_attention_mask,
                suffix_attention_mask,
                decoder_attention_mask.to(expected_device),
            ],
            dim=1,
        )
        outputs = self.llama(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return outputs

    def generate(
        self,
        pixel_values: torch.FloatTensor,
        prefix_input_ids: Optional[torch.Tensor] = None,
        suffix_input_ids: Optional[torch.Tensor] = None,
        **generate_kwargs,
    ):
        """
        Generates sequences using the MiniGPT4Blip2LlamaModel.

        Args:
            pixel_values (torch.FloatTensor): The pixel values.
            prefix_input_ids (torch.Tensor, optional): The input IDs for the prefix tokens. Defaults to None.
            suffix_input_ids (torch.Tensor, optional): The input IDs for the suffix tokens. Defaults to None.
            **generate_kwargs: Additional keyword arguments for sequence generation.

        Returns:
            outputs: The generation outputs.
        """
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
        )
        image_embeds = vision_outputs[0]
        image_attention_mask = torch.ones(
            image_embeds.size()[:-1],
            dtype=torch.long,
            device=image_embeds.device,
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
        )
        query_embeds = query_outputs[0]

        inputs_embeds = self.language_projection(query_embeds)

        attention_mask = torch.ones(
            inputs_embeds.size(0), inputs_embeds.size(1), dtype=torch.bool
        ).to(inputs_embeds.device)

        if prefix_input_ids is not None:
            prefix_inputs_embeds = self.llama.get_input_embeddings()(prefix_input_ids)
            inputs_embeds = torch.cat(
                [prefix_inputs_embeds, inputs_embeds],
                dim=1,
            )
            attention_mask = torch.cat(
                [prefix_input_ids.ne(self.blip2_config.pad_token_id), attention_mask],
                dim=1,
            )

        if suffix_input_ids is not None:
            suffix_inputs_embeds = self.llama.get_input_embeddings()(suffix_input_ids)
            inputs_embeds = torch.cat(
                [inputs_embeds, suffix_inputs_embeds],
                dim=1,
            )
            attention_mask = torch.cat(
                [attention_mask, suffix_input_ids.ne(self.blip2_config.pad_token_id)],
                dim=1,
            )

        outputs = self.llama.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

        return outputs


@register_model("microsoft/model/classification/peft/lora/minigpt4")
class MiniGPT4Blip2LlamaLoraForClassification(GenericPeftModel):
    prefix_keys_in_state_dict = {
        "^qformer.": "peft_model.",
        "^query_tokens": "peft_model.",
        "^vision_model.": "peft_model.",
        "^language_projection.": "peft_model.",
        "^model\.": "peft_model.model.base_model.",
    }
    modules_to_save_checkpoints = ["vision_model", "query_tokens", "qformer", "language_projection", "lora", "classifier"]

    def __init__(
        self,
        blip2_config_path: str,
        llama_config_path: str,
        quant_config_path: Optional[str] = None,
        pad_token_id: Optional[int] = 0,
        freeze_vision_model: Optional[bool] = True,
        freeze_qformer_model: Optional[bool] = True,
        lora_r: Optional[int] = 16,
        lora_alpha: Optional[int] = 32,
        lora_dropout: Optional[float] = 0.05,
        fan_in_fan_out: Optional[bool] = True,
        target_modules: Optional[Union[List[str], str]] = ["q_proj", "v_proj"],
        num_classes: Optional[int] = 1,
        hidden_dropout_prob: Optional[float] = 0.1,
        gradient_checkpointing: Optional[bool] = False,
    ):
        super().__init__()
        self.blip2_config = Blip2Config.from_json_file(blip2_config_path)
        self.blip2_config.pad_token_id = pad_token_id
        self.llama_config = LlamaConfig.from_json_file(llama_config_path)
        self.llama_config.gradient_checkpointing = gradient_checkpointing
        self.peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            target_modules=target_modules,
        )
        if quant_config_path is not None:
            self.quant_config = QuantizationConfig.from_json_file(quant_config_path)
        else:
            self.quant_config = None
        self.peft_model = MiniGPT4Blip2LlamaLoraClassificationModel(
            self.blip2_config,
            self.llama_config,
            self.peft_config,
            quant_config=self.quant_config,
        )
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Linear(self.llama_config.hidden_size, num_classes)
        self.init_weights()

        if freeze_vision_model:
            for param in self.peft_model.vision_model.parameters():
                param.requires_grad = False

        if freeze_qformer_model:
            for param in self.peft_model.qformer.parameters():
                param.requires_grad = False

    @classmethod
    @add_default_section_for_init("microsoft/model/classification/peft/lora/minigpt4")
    def from_core_configure(cls, config, **kwargs):
        """
        Creates a MiniGPT4Blip2LlamaForGeneration instance from a core configuration.

        Args:
            config: The configuration object.
            **kwargs: Additional keyword arguments.

        Returns:
            MiniGPT4Blip2LlamaForGeneration: The created instance.
        """
        config.set_default_section("microsoft/model/classification/peft/lora/minigpt4")
        pretrained_name = config.getoption("pretrained_name", "default-minigpt4")

        blip2_config_path = config.getoption("blip2_config_path", None)
        blip2_config_path = pop_value(
            blip2_config_path,
            nested_dict_value(
                pretrained_minigpt4_infos, pretrained_name, "blip2_config_path"
            ),
        )
        blip2_config_path = cached_path(blip2_config_path)

        llama_config_path = config.getoption("llama_config_path", None)
        llama_config_path = pop_value(
            llama_config_path,
            nested_dict_value(
                pretrained_minigpt4_infos, pretrained_name, "llama_config_path"
            ),
        )
        llama_config_path = cached_path(llama_config_path)

        quant_config_path = config.getoption("quant_config_path", None)
        if quant_config_path is not None:
            quant_config_path = cached_path(quant_config_path)

        lora_r = config.getoption("lora_r", 16)
        lora_alpha = config.getoption("lora_alpha", 32)
        lora_dropout = config.getoption("lora_dropout", 0.05)
        fan_in_fan_out = config.getoption("fan_in_fan_out", True)
        target_modules = config.getoption("target_modules", ["q_proj", "v_proj"])

        freeze_vision_model = config.getoption("freeze_vision_model", True)
        freeze_qformer_model = config.getoption("freeze_qformer_model", True)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)
        num_classes = config.getoption("num_classes", 1)

        inst = cls(
            blip2_config_path,
            llama_config_path,
            quant_config_path=quant_config_path,
            freeze_vision_model=freeze_vision_model,
            freeze_qformer_model=freeze_qformer_model,
            gradient_checkpointing=gradient_checkpointing,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            target_modules=target_modules,
            num_classes=num_classes,
        )

        weight_path = []
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        pretrained_weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_minigpt4_infos, pretrained_name, "weight"),
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

    @autocast()
    def forward(
        self,
        pixel_values: torch.Tensor,
        prefix_input_ids: torch.Tensor,
        suffix_input_ids: torch.Tensor,
        prefix_attention_mask: Optional[torch.Tensor] = None,
        suffix_attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass of the classification model.

        Args:
            input_ids (torch.Tensor): Input tensor of shape (batch_size, sequence_length).
            attention_mask (torch.Tensor, optional): Attention mask tensor of shape (batch_size, sequence_length). Defaults to None.
            position_ids (torch.Tensor, optional): Position IDs tensor of shape (batch_size, sequence_length). Defaults to None.

        Returns:
            torch Output logits.Tensor: tensor of shape (batch_size, num_classes).
        """
        outputs = self.peft_model(
            pixel_values=pixel_values,
            prefix_input_ids=prefix_input_ids,
            suffix_input_ids=suffix_input_ids,
            prefix_attention_mask=prefix_attention_mask,
            suffix_attention_mask=suffix_attention_mask,
        )[0]
        pooled_output = outputs[:, -1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return ClassificationOutputs(outputs=logits)


@register_model(
    "microsoft/model/generation/peft/lora/minigpt4", generation_model_decorator
)
class MiniGPT4Blip2LlamaLoraForGeneration(GenericPeftModel):
    """
    MiniGPT4Blip2LlamaForGeneration is a class for generating sequences using the MiniGPT4 model with Blip2 and Llama.
    It inherits from the _MiniGPT4Blip2LlamaForGeneration class.
    """

    prefix_keys_in_state_dict = {
        "^qformer.": "peft_model.",
        "^query_tokens": "peft_model.",
        "^vision_model.": "peft_model.",
        "^language_projection.": "peft_model.",
        "^model\.": "peft_model.llama.base_model.model.",
        "^lm_head.": "peft_model.llama.base_model.model.",
    }
    modules_to_save_checkpoints = ["vision_model", "query_tokens", "qformer", "language_projection", "lora"]

    def __init__(
        self,
        blip2_config_path: str,
        llama_config_path: str,
        quant_config_path: Optional[str] = None,
        pad_token_id: Optional[int] = 0,
        freeze_vision_model: Optional[bool] = True,
        freeze_qformer_model: Optional[bool] = True,
        gradient_checkpointing: Optional[bool] = False,
        lora_r: Optional[int] = 16,
        lora_alpha: Optional[int] = 32,
        lora_dropout: Optional[float] = 0.05,
        fan_in_fan_out: Optional[bool] = True,
        target_modules: Optional[Union[List[str], str]] = ["q_proj", "v_proj"],
    ):
        """
        Initializes a MiniGPT4Blip2LlamaForGeneration instance.

        Args:
            blip2_config_path (str): The file path to the Blip2 configuration.
            llama_config_path (str): The file path to the Llama configuration.
            pad_token_id (int, optional): The ID of the padding token. Defaults to 0.
            freeze_vision_model (bool, optional): Whether to freeze the vision model. Defaults to True.
            freeze_qformer_model (bool, optional): Whether to freeze the query transformer model. Defaults to True.
            gradient_checkpointing (bool, optional): Whether to use gradient checkpointing. Defaults to False.
        """
        super().__init__()
        self.blip2_config = Blip2Config.from_json_file(blip2_config_path)
        self.blip2_config.pad_token_id = pad_token_id
        self.llama_config = LlamaConfig.from_json_file(llama_config_path)
        self.llama_config.gradient_checkpointing = gradient_checkpointing
        self.peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            target_modules=target_modules,
        )
        if quant_config_path is not None:
            self.quant_config = QuantizationConfig.from_json_file(quant_config_path)
        else:
            self.quant_config = None
        self.peft_model = MiniGPT4Blip2LlamaLoraGenerationModel(
            self.blip2_config,
            self.llama_config,
            self.peft_config,
            quant_config=self.quant_config,
        )
        self.init_weights()

        if freeze_vision_model:
            for param in self.peft_model.vision_model.parameters():
                param.requires_grad = False

        if freeze_qformer_model:
            for param in self.peft_model.qformer.parameters():
                param.requires_grad = False

    @classmethod
    @add_default_section_for_init("microsoft/model/generation/peft/lora/minigpt4")
    def from_core_configure(cls, config, **kwargs):
        """
        Creates a MiniGPT4Blip2LlamaForGeneration instance from a core configuration.

        Args:
            config: The configuration object.
            **kwargs: Additional keyword arguments.

        Returns:
            MiniGPT4Blip2LlamaForGeneration: The created instance.
        """
        config.set_default_section("microsoft/model/generation/peft/lora/minigpt4")
        pretrained_name = config.getoption("pretrained_name", "default-minigpt4")

        blip2_config_path = config.getoption("blip2_config_path", None)
        blip2_config_path = pop_value(
            blip2_config_path,
            nested_dict_value(
                pretrained_minigpt4_infos, pretrained_name, "blip2_config_path"
            ),
        )
        blip2_config_path = cached_path(blip2_config_path)

        llama_config_path = config.getoption("llama_config_path", None)
        llama_config_path = pop_value(
            llama_config_path,
            nested_dict_value(
                pretrained_minigpt4_infos, pretrained_name, "llama_config_path"
            ),
        )
        llama_config_path = cached_path(llama_config_path)

        quant_config_path = config.getoption("quant_config_path", None)
        if quant_config_path is not None:
            quant_config_path = cached_path(quant_config_path)

        lora_r = config.getoption("lora_r", 16)
        lora_alpha = config.getoption("lora_alpha", 32)
        lora_dropout = config.getoption("lora_dropout", 0.05)
        fan_in_fan_out = config.getoption("fan_in_fan_out", True)
        target_modules = config.getoption("target_modules", ["q_proj", "v_proj"])

        freeze_vision_model = config.getoption("freeze_vision_model", True)
        freeze_qformer_model = config.getoption("freeze_qformer_model", True)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)

        inst = cls(
            blip2_config_path,
            llama_config_path,
            quant_config_path=quant_config_path,
            freeze_vision_model=freeze_vision_model,
            freeze_qformer_model=freeze_qformer_model,
            gradient_checkpointing=gradient_checkpointing,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            target_modules=target_modules,
        )

        weight_path = []
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        pretrained_weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_minigpt4_infos, pretrained_name, "weight"),
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

    @autocast()
    def forward(
        self,
        pixel_values: torch.Tensor,
        prefix_input_ids: torch.Tensor,
        suffix_input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        prefix_attention_mask: Optional[torch.Tensor] = None,
        suffix_attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Performs a forward pass through the model.

        Args:
            pixel_values (torch.Tensor): The pixel values.
            prefix_input_ids (torch.Tensor): The input IDs for the prefix tokens.
            suffix_input_ids (torch.Tensor): The input IDs for the suffix tokens.
            decoder_input_ids (torch.Tensor): The input IDs for the decoder tokens.
            prefix_attention_mask (torch.Tensor, optional): The attention mask for the prefix tokens.
                Defaults to None.
            suffix_attention_mask (torch.Tensor, optional): The attention mask for the suffix tokens.
                Defaults to None.
            decoder_attention_mask (torch.Tensor, optional): The attention mask for the decoder tokens.
                Defaults to None.

        Returns:
            GenerationOutputs: The generation outputs.
        """
        outputs = self.peft_model(
            pixel_values=pixel_values,
            prefix_input_ids=prefix_input_ids,
            suffix_input_ids=suffix_input_ids,
            decoder_input_ids=decoder_input_ids,
            prefix_attention_mask=prefix_attention_mask,
            suffix_attention_mask=suffix_attention_mask,
            decoder_attention_mask=decoder_attention_mask,
        )
        logits = outputs.logits[
            :, -suffix_input_ids.size(1) - decoder_input_ids.size(1) :, :
        ]
        return GenerationOutputs(sequences=logits)

    @add_default_section_for_function("microsoft/model/generation/peft/lora/minigpt4")
    @torch.no_grad()
    @autocast()
    def generate(
        self,
        pixel_values: torch.Tensor,
        prefix_input_ids: torch.Tensor,
        suffix_input_ids: torch.Tensor,
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
        Generates sequences using the model.

        Args:
            pixel_values (torch.Tensor): The pixel values.
            prefix_input_ids (torch.Tensor): The input IDs for the prefix tokens.
            suffix_input_ids (torch.Tensor): The input IDs for the suffix tokens.
            num_beams (int, optional): The number of beams for beam search. Defaults to 5.
            decoder_start_token_id (int, optional): The ID of the decoder start token. Defaults to 1.
            decoder_end_token_id (int or List[int], optional): The ID(s) of the decoder end token(s).
                Defaults to 2.
            num_return_sequences (int, optional): The number of sequences to return. Defaults to 1.
            min_gen_seq_length (int, optional): The minimum generated sequence length. Defaults to 0.
            max_gen_seq_length (int, optional): The maximum generated sequence length. Defaults to 48.
            repetition_penalty (float, optional): The repetition penalty. Defaults to 1.0.
            no_repeat_ngram_size (int, optional): The size of the n-grams to avoid repeating. Defaults to 0.
            early_stopping (bool, optional): Whether to perform early stopping. Defaults to True.
            length_penalty (float, optional): The length penalty. Defaults to 1.0.
            num_beam_groups (int, optional): The number of beam groups for diverse beam search.
                Defaults to 1.
            diversity_penalty (float, optional): The diversity penalty for diverse beam search. Defaults to 0.0.
            do_sample (bool, optional): Whether to use sampling instead of beam search. Defaults to False.
            temperature (float, optional): The temperature value for sampling. Defaults to 1.0.
            top_k (int, optional): The value for top-k sampling. Defaults to 50.
            top_p (float, optional): The value for top-p (nucleus) sampling. Defaults to 1.0.

        Returns:
            GenerationOutputs: The generation outputs.
        """
        outputs = self.peft_model.generate(
            pixel_values=pixel_values,
            prefix_input_ids=prefix_input_ids,
            suffix_input_ids=suffix_input_ids,
            max_length=max_gen_seq_length,
            min_length=min_gen_seq_length,
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
        outputs.sequences[:, :, : sequences.size(-1)].copy_(
            sequences[:, :, : sequences.size(-1)]
        )

        if num_return_sequences == 1:
            outputs.sequences = outputs.sequences.reshape(-1, max_gen_seq_length)

        return GenerationOutputs(
            sequences=outputs.sequences.long(),
            sequences_scores=outputs.sequences_scores,
        )
