# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torch.cuda.amp import autocast
from transformers.utils import is_remote_url
from unitorch.utils import pop_value, nested_dict_value
from transformers import MistralConfig, MistralForCausalLM, MistralModel
from peft import LoraConfig, PeftModelForCausalLM
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
from unitorch.models.peft import PeftModelForSequenceClassification, GenericPeftModel

from unitorch.cli.models import generation_model_decorator
from unitorch.cli.models import ClassificationOutputs, GenerationOutputs, LossOutputs, EmbeddingOutputs
from unitorch_microsoft.models.mistral import pretrained_mistral_infos


@register_model("microsoft/msan/user_profile/classification/peft/lora/mistral")
class MistralLoraForClassification(GenericPeftModel):
    prefix_keys_in_state_dict = {
        "^(?!peft_model\.base_model\.model\.).*": "peft_model.base_model.",
    }
    replace_keys_in_state_dict = {
        "q_proj.weight": "q_proj.base_layer.weight",
        "v_proj.weight": "v_proj.base_layer.weight",
    }

    def __init__(
        self,
        config_path: str,
        num_age_classes: Optional[int] = 5,
        num_gender_classes: Optional[int] = 2,
        loss_weight: Optional[float] = 0.5,
        hidden_dropout_prob: Optional[float] = 0.1,
        quant_config_path: Optional[str] = None,
        lora_r: Optional[int] = 16,
        lora_alpha: Optional[int] = 32,
        lora_dropout: Optional[float] = 0.05,
        fan_in_fan_out: Optional[bool] = True,
        target_modules: Optional[Union[List[str], str]] = ["q_proj", "v_proj"],
        gradient_checkpointing: Optional[bool] = False,
        pad_token_id: Optional[int] = 0,
    ):
        super().__init__()
        self.config = MistralConfig.from_json_file(config_path)
        self.config.gradient_checkpointing = gradient_checkpointing
        self.config.pad_token_id = pad_token_id
        self.peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            target_modules=target_modules,
        )
        model = MistralModel(self.config)
        if quant_config_path is not None:
            quant_config = QuantizationConfig.from_json_file(quant_config_path)
            ignore_modules = target_modules + ["lm_head"]
            model = quantize_model(model, quant_config, ignore_modules=ignore_modules)
        self.peft_model = PeftModelForSequenceClassification(model, self.peft_config)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.age_classifier = nn.Linear(self.peft_model.config.hidden_size, num_age_classes)
        self.gender_classifier = nn.Linear(
            self.peft_model.config.hidden_size,
            num_gender_classes,
        )
        self.loss_weight = loss_weight
        self.init_weights()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        age_labels: Optional[torch.Tensor] = None,
        gender_labels: Optional[torch.Tensor] = None,
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
        outputs = self.peft_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )[0]
        pooled_output = outputs[:, -1]
        pooled_output = self.dropout(pooled_output)
        age_logits = self.age_classifier(pooled_output)
        gender_logits = self.gender_classifier(pooled_output)
        if age_labels is not None and gender_labels is not None:
            age_loss = nn.CrossEntropyLoss()(age_logits, age_labels.long())
            gender_loss = nn.CrossEntropyLoss()(gender_logits, gender_labels.long())
            loss = age_loss * self.loss_weight + gender_loss * (1 - self.loss_weight)
            return LossOutputs(loss=loss)

        return EmbeddingOutputs(embedding=age_logits.softmax(-1), embedding1=gender_logits.softmax(-1))

    @classmethod
    @add_default_section_for_init("microsoft/msan/user_profile/classification/peft/lora/mistral")
    def from_core_configure(cls, config, **kwargs):
        """
        Create an instance of MistralLoraForGeneration from a core configuration.

        Args:
            config: The core configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            MistralLoraForGeneration: The initialized MistralLoraForGeneration instance.
        """
        config.set_default_section("microsoft/msan/user_profile/classification/peft/lora/mistral")
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

        lora_r = config.getoption("lora_r", 16)
        lora_alpha = config.getoption("lora_alpha", 32)
        lora_dropout = config.getoption("lora_dropout", 0.05)
        fan_in_fan_out = config.getoption("fan_in_fan_out", True)
        target_modules = config.getoption("target_modules", ["q_proj", "v_proj"])

        gradient_checkpointing = config.getoption("gradient_checkpointing", False)

        inst = cls(
            config_path,
            quant_config_path=quant_config_path,
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
            nested_dict_value(pretrained_mistral_infos, pretrained_name, "weight"),
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
