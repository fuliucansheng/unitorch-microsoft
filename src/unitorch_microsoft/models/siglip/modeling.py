# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import json
import random
import logging
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch import autocast
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from transformers.models.siglip.modeling_siglip import (
    SiglipConfig,
    SiglipTextTransformer,
    SiglipVisionTransformer,
)
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models import GenericModel
from unitorch.models.siglip import SiglipProcessor
from unitorch.models.peft import PeftWeightLoaderMixin
from unitorch.models.clip.modeling import _clip_loss, AllGather
from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
    register_model,
)
from unitorch.cli.models import ClassificationOutputs, LossOutputs
from unitorch.cli.models.siglip import pretrained_siglip_infos


@register_model("microsoft/model/matching/siglip/v2")
class SiglipForMatchingV2(GenericModel, PeftWeightLoaderMixin):
    """
    Siglip model for pretraining.
    """

    replace_keys_in_peft_state_dict = {"peft_model.base_model.model.": ""}

    def __init__(
        self,
        config_path: str,
        freeze_base_model: Optional[bool] = True,
        gradient_checkpointing: Optional[bool] = False,
        labels: Optional[List[str]] = None,
        vocab_path: Optional[str] = None,
        vision_config_path: Optional[str] = None,
        max_seq_length: Optional[int] = 128,
    ):
        """
        Initializes the SiglipForPretrain model.

        Args:
            config_path (str): Path to the model configuration file.
            projection_dim (int, optional): Dimension of the projected embeddings. Defaults to 512.
            freeze_base_model (bool, optional): Whether to freeze the base model parameters. Defaults to True.
            gradient_checkpointing (bool, optional): Whether to use gradient checkpointing. Defaults to False.
            use_all_gather (bool, optional): Whether to use all-gather operation. Defaults to True.
        """
        super().__init__()

        config = SiglipConfig.from_json_file(config_path)
        text_config = config.text_config
        vision_config = config.vision_config
        text_config.gradient_checkpointing = gradient_checkpointing
        vision_config.gradient_checkpointing = gradient_checkpointing

        self.text_embed_dim = text_config.projection_size
        self.vision_embed_dim = vision_config.hidden_size
        vision_config.vision_use_head = True

        self.text_model = SiglipTextTransformer(text_config)
        self.vision_model = SiglipVisionTransformer(vision_config)

        self.classifier = nn.Linear(1, 1)
        self.init_weights()
        self.classifier.weight.data.fill_(5.0)

        self.processor = SiglipProcessor(
            vocab_path=vocab_path,
            vision_config_path=vision_config_path,
            max_seq_length=max_seq_length,
        )

        assert labels is not None
        self.labels_inputs = self.get_label_inputs(labels)
        self.labels_embs = None

        if freeze_base_model:
            for param in self.text_model.parameters():
                param.requires_grad = False
            for param in self.vision_model.parameters():
                param.requires_grad = False

    def get_label_inputs(self, texts):
        input_ids, attention_mask, position_ids = [], [], []
        for text in texts:
            inputs = self.processor.text_classification(text)
            input_ids.append(inputs.input_ids)
            attention_mask.append(inputs.attention_mask)
            position_ids.append(inputs.position_ids)
        input_ids = torch.stack(input_ids, dim=0)
        attention_mask = torch.stack(attention_mask, dim=0)
        position_ids = torch.stack(position_ids, dim=0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }

    @classmethod
    @add_default_section_for_init("microsoft/model/matching/siglip/v2")
    def from_core_configure(cls, config, **kwargs):
        """
        Create an instance of SiglipForClassification from a core configuration.

        Args:
            config: The core configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            SiglipForClassification: An instance of the SiglipForClassification model.
        """
        config.set_default_section("microsoft/model/matching/siglip/v2")
        pretrained_name = config.getoption("pretrained_name", "siglip-base-patch16-224")
        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_siglip_infos, pretrained_name, "config"),
        )

        config_path = cached_path(config_path)

        vocab_path = config.getoption("vocab_path", None)
        vocab_path = pop_value(
            vocab_path,
            nested_dict_value(pretrained_siglip_infos, pretrained_name, "vocab"),
        )
        vocab_path = cached_path(vocab_path)

        vision_config_path = config.getoption("vision_config_path", None)
        vision_config_path = pop_value(
            vision_config_path,
            nested_dict_value(
                pretrained_siglip_infos, pretrained_name, "vision_config"
            ),
        )

        vision_config_path = cached_path(vision_config_path)

        freeze_base_model = config.getoption("freeze_base_model", True)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)
        labels = config.getoption("labels", None)
        max_seq_length = config.getoption("max_seq_length", 128)

        inst = cls(
            config_path=config_path,
            freeze_base_model=freeze_base_model,
            gradient_checkpointing=gradient_checkpointing,
            labels=labels,
            vocab_path=vocab_path,
            vision_config_path=vision_config_path,
            max_seq_length=max_seq_length,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_siglip_infos, pretrained_name, "weight"),
            check_none=False,
        )
        if weight_path is not None:
            inst.from_pretrained(weight_path)

        pretrained_lora_weight_path = config.getoption(
            "pretrained_lora_weight_path", None
        )
        pretrained_lora_weight = config.getoption("pretrained_lora_weight", 1.0)
        pretrained_lora_alpha = config.getoption("pretrained_lora_alpha", 32.0)
        if pretrained_lora_weight_path is not None:
            inst.load_lora_weights(
                pretrained_lora_weight_path,
                lora_weights=pretrained_lora_weight,
                lora_alphas=pretrained_lora_alpha,
                save_base_state=False,
            )

        return inst

    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(
        self,
        pixel_values: torch.Tensor,
    ):
        """
        Perform a forward pass through the model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            pixel_values (torch.Tensor): Input pixel values.
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            position_ids (torch.Tensor, optional): Position IDs. Defaults to None.

        Returns:
            ClassificationOutputs: The classification outputs.
        """
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
        )
        image_embeds = vision_outputs[1]
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

        if self.labels_embs is None or self.training:
            text_outputs = self.text_model(
                input_ids=self.labels_inputs["input_ids"].to(self.device),
                attention_mask=self.labels_inputs["attention_mask"].to(self.device),
                position_ids=self.labels_inputs["position_ids"].to(self.device),
            )
            text_embeds = text_outputs[1]
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            self.labels_embs = text_embeds

        self.labels_embs = self.labels_embs.to(image_embeds.device)
        scores = torch.einsum("ij,kj->ik", image_embeds, self.labels_embs)
        scores = self.classifier(scores.view(-1, 1)).view(-1, self.labels_embs.size(0))
        return ClassificationOutputs(outputs=scores)
