# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models import GenericModel
from unitorch.modules.classifier import mlplayer, reslayer
from unitorch.cli import (
    add_default_section_for_init,
    add_default_section_for_function,
    register_model,
)
from unitorch.cli.models import ClassificationOutputs, LossOutputs
from unitorch_microsoft import cached_path
from unitorch_microsoft.models.tulr import pretrained_tulr_infos
from unitorch_microsoft.models.tulr.modeling_v6 import TULRV6Config, TULRV6Model


@register_model("microsoft/adsplus/relevance/classification/tulr/v6")
class TULRV6ForClassification(GenericModel):
    def __init__(
        self,
        config_path: str,
        used_tasks: List[int] = [4, 5, 6, 7],
        num_classes: Optional[int] = 1,
        gradient_checkpointing: Optional[bool] = False,
    ):
        super().__init__()
        self.config = TULRV6Config.from_json_file(config_path)
        self.config.gradient_checkpointing = gradient_checkpointing
        self.used_tasks = used_tasks
        self.num_classes = num_classes
        self.tulrv6 = TULRV6Model(self.config)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.ModuleDict(
            {
                str(k): nn.Sequential(
                    mlplayer(
                        self.config.hidden_size,
                        self.config.hidden_size // 2,
                        self.config.hidden_size,
                    ),
                    nn.Linear(self.config.hidden_size, num_classes),
                )
                for k in self.used_tasks
            }
        )
        self.init_weights()

    @classmethod
    @add_default_section_for_init("microsoft/adsplus/relevance/classification/tulr/v6")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/adsplus/relevance/classification/tulr/v6")
        pretrained_name = config.getoption("pretrained_name", "tulrv6-base")
        config_path = config.getoption("config_path", None)
        num_classes = config.getoption("num_classes", 1)
        used_tasks = config.getoption("used_tasks", [4, 5, 6, 7])

        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_tulr_infos, pretrained_name, "config"),
        )
        config_path = cached_path(config_path)

        gradient_checkpointing = config.getoption("gradient_checkpointing", False)

        inst = cls(config_path, used_tasks, num_classes, gradient_checkpointing)

        weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            weight_path,
            nested_dict_value(pretrained_tulr_infos, pretrained_name, "weight"),
            check_none=False,
        )
        if weight_path is not None:
            inst.from_pretrained(weight_path)

        return inst

    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(
        self,
        task: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        outputs = self.tulrv6(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = (
            torch.zeros(pooled_output.size(0), self.num_classes)
            .to(pooled_output)
            .float()
        )
        for _task in self.used_tasks:
            logits += self.classifier[str(_task)](pooled_output) * (
                task == _task
            ).float().unsqueeze(-1)
        return ClassificationOutputs(outputs=logits)
