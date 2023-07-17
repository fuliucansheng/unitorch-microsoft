# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torch.cuda.amp import autocast
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models.bert import BertForClassification as _BertForClassification
from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
    register_model,
)
from unitorch.cli.models import ClassificationOutputs
from unitorch.cli.models.bert import pretrained_bert_infos


@register_model("microsoft/adsplus/slab/classification/bert")
class BertForClassification(_BertForClassification):
    """BERT model for classification tasks."""

    def __init__(
        self,
        config_path: str,
        num_classes: Optional[int] = 10,
        label_gains: Optional[List[float]] = None,
        gradient_checkpointing: Optional[bool] = False,
    ):
        """
        Initialize BertForClassification.

        Args:
            config_path (str): The path to the model configuration file.
            num_classes (int, optional): The number of classes for classification. Defaults to 1.
            gradient_checkpointing (bool, optional): Whether to use gradient checkpointing. Defaults to False.
        """
        super().__init__(
            config_path=config_path,
            num_classes=num_classes,
            gradient_checkpointing=gradient_checkpointing,
        )
        if label_gains is None:
            label_gains = [i for i in range(1, num_classes + 1)]
        self.label_gains = label_gains

    @classmethod
    @add_default_section_for_init("microsoft/adsplus/slab/classification/bert")
    def from_core_configure(cls, config, **kwargs):
        """
        Create an instance of BertForClassification from a core configuration.

        Args:
            config: The core configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            BertForClassification: An instance of BertForClassification.
        """
        config.set_default_section("microsoft/adsplus/slab/classification/bert")
        pretrained_name = config.getoption("pretrained_name", "default-bert")
        config_path = config.getoption("config_path", None)
        num_classes = config.getoption("num_classes", 10)

        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_bert_infos, pretrained_name, "config"),
        )
        config_path = cached_path(config_path)
        label_gains = config.getoption("label_gains", None)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)

        inst = cls(
            config_path,
            num_classes,
            label_gains,
            gradient_checkpointing,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_bert_infos, pretrained_name, "weight"),
            check_none=False,
        )
        if weight_path is not None:
            weight_path = cached_path(weight_path)
            inst.from_pretrained(weight_path)

        return inst

    @autocast()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass of the BertForClassification model.

        Args:
            input_ids (torch.Tensor): Input IDs.
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            token_type_ids (torch.Tensor, optional): Token type IDs. Defaults to None.
            position_ids (torch.Tensor, optional): Position IDs. Defaults to None.

        Returns:
            ClassificationOutputs: Model outputs for classification.
        """
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        if self.training:
            return ClassificationOutputs(outputs=outputs)

        scores = torch.zeros(outputs.size(0), 1).to(outputs.device)
        outputs = torch.sigmoid(outputs)
        for i in range(outputs.size(-1)):
            scores += outputs[:, i].unsqueeze(-1) * self.label_gains[i]

        return ClassificationOutputs(outputs=scores)
