# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torch.cuda.amp import autocast
from transformers.utils import is_remote_url
from unitorch.utils import pop_value, nested_dict_value
from transformers import LlamaConfig, LlamaForCausalLM, LlamaModel
from transformers.models.blip_2.modeling_blip_2 import (
    Blip2Config,
    Blip2VisionModel,
    Blip2QFormerModel,
)
from unitorch.models import (
    GenericModel,
    GenericOutputs,
    QuantizationConfig,
    QuantizationMixin,
)
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


class MiniGPT4Blip2LlamaClassificationModel(nn.Module):
    """
    MiniGPT4Blip2LlamaModel is a model that combines the Blip2VisionModel, Blip2QFormerModel, and LlamaForCausalLM
    models for generation. It inherits from the nn.Module class.
    """

    def __init__(
        self,
        blip2_config: Blip2Config,
        llama_config: LlamaConfig,
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
        self.llama = LlamaModel(self.llama_config)
        self.quant_config = quant_config
        if self.quant_config is not None:
            ignore_modules = ["lm_head"]
            self.llama = quantize_model(
                self.llama, self.quant_config, ignore_modules=ignore_modules
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
        prefix_inputs_embeds = self.llama.get_input_embeddings()(prefix_input_ids)
        suffix_inputs_embeds = self.llama.get_input_embeddings()(suffix_input_ids)
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
        outputs = self.llama(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return outputs


@register_model("microsoft/model/classification/minigpt4")
class MiniGPT4Blip2LlamaForClassification(GenericPeftModel):
    prefix_keys_in_state_dict = {
        "^qformer.": "base_model.",
        "^query_tokens": "base_model.",
        "^vision_model.": "base_model.",
        "^language_projection.": "base_model.",
        "^(?!model\.language_projection\.)model\.": "base_model.llama.",
        "^lm_head.": "base_model.llama.",
    }
    replace_keys_in_state_dict = {
        "llama.model.": "llama.",
    }

    def __init__(
        self,
        blip2_config_path: str,
        llama_config_path: str,
        quant_config_path: Optional[str] = None,
        pad_token_id: Optional[int] = 0,
        freeze_vision_model: Optional[bool] = True,
        freeze_qformer_model: Optional[bool] = True,
        freeze_language_projection: Optional[bool] = True,
        freeze_llama_model: Optional[bool] = True,
        num_classes: Optional[int] = 1,
        hidden_dropout_prob: Optional[float] = 0.1,
        gradient_checkpointing: Optional[bool] = False,
    ):
        super().__init__()
        self.blip2_config = Blip2Config.from_json_file(blip2_config_path)
        self.blip2_config.pad_token_id = pad_token_id
        self.llama_config = LlamaConfig.from_json_file(llama_config_path)
        self.llama_config.gradient_checkpointing = gradient_checkpointing
        if quant_config_path is not None:
            self.quant_config = QuantizationConfig.from_json_file(quant_config_path)
        else:
            self.quant_config = None
        self.base_model = MiniGPT4Blip2LlamaClassificationModel(
            self.blip2_config,
            self.llama_config,
            quant_config=self.quant_config,
        )
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Linear(self.llama_config.hidden_size, num_classes)
        self.init_weights()

        if freeze_vision_model:
            for param in self.base_model.vision_model.parameters():
                param.requires_grad = False

        if freeze_qformer_model:
            for param in self.base_model.qformer.parameters():
                param.requires_grad = False

        if freeze_language_projection:
            for param in self.base_model.language_projection.parameters():
                param.requires_grad = False

        if freeze_llama_model:
            for param in self.base_model.llama.parameters():
                param.requires_grad = False

    @classmethod
    @add_default_section_for_init("microsoft/model/classification/minigpt4")
    def from_core_configure(cls, config, **kwargs):
        """
        Creates a MiniGPT4Blip2LlamaForGeneration instance from a core configuration.

        Args:
            config: The configuration object.
            **kwargs: Additional keyword arguments.

        Returns:
            MiniGPT4Blip2LlamaForGeneration: The created instance.
        """
        config.set_default_section("microsoft/model/classification/minigpt4")
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

        freeze_vision_model = config.getoption("freeze_vision_model", True)
        freeze_qformer_model = config.getoption("freeze_qformer_model", True)
        freeze_language_projection = config.getoption(
            "freeze_language_projection", True
        )
        freeze_llama_model = config.getoption("freeze_llama_model", True)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)
        num_classes = config.getoption("num_classes", 1)
        hidden_dropout_prob = config.getoption("hidden_dropout_prob", 0.1)

        inst = cls(
            blip2_config_path,
            llama_config_path,
            quant_config_path=quant_config_path,
            freeze_vision_model=freeze_vision_model,
            freeze_qformer_model=freeze_qformer_model,
            freeze_language_projection=freeze_language_projection,
            freeze_llama_model=freeze_llama_model,
            gradient_checkpointing=gradient_checkpointing,
            num_classes=num_classes,
            hidden_dropout_prob=hidden_dropout_prob,
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

        if len(weight_path) > 0:
            inst.from_pretrained(
                weight_path=weight_path,
            )

        return inst

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
        outputs = self.base_model(
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
