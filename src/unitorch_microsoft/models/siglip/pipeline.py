# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import re
import torch
import numpy as np
import pandas as pd
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from transformers.models.siglip.modeling_siglip import (
    SiglipConfig,
    SiglipTextTransformer,
    SiglipVisionTransformer,
)
from unitorch.models import GenericModel
from unitorch.models.siglip import SiglipProcessor
from unitorch.models.peft import PeftWeightLoaderMixin
from unitorch.utils import (
    pop_value,
    nested_dict_value,
    read_file,
    read_json_file,
    is_remote_url,
)

from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
)
from unitorch_microsoft.models.siglip import (
    pretrained_siglip_infos,
    pretrained_siglip_extensions_infos,
)
from unitorch_microsoft.models.siglip.modeling import SiglipForMatchingV2

ACT2FN = {
    "sigmoid": torch.sigmoid,
    "softmax": torch.nn.Softmax(dim=1),
}


class Siglip2ForMatchingV2Pipeline(SiglipForMatchingV2):
    def __init__(
        self,
        config_path: str,
        vocab_path: Optional[str] = None,
        vision_config_path: Optional[str] = None,
        max_seq_length: Optional[int] = 64,
        weight_path: Optional[Union[str, List[str]]] = None,
        lora_weight_path: Optional[Union[str, List[str]]] = None,
        lora_weight: Optional[float] = 1.0,
        lora_alpha: Optional[float] = 32.0,
        state_dict: Optional[Dict[str, Any]] = None,
        label_dict: Optional[Dict[str, str]] = None,
        device: Optional[Union[str, int]] = "cpu",
        act_fn: Optional[str] = None,
    ):
        assert label_dict is not None, "label_dict must be provided"
        self.label_keys = list(label_dict.keys())
        self.label_values = list(label_dict.values())
        self.act_fn = ACT2FN.get(act_fn, None)
        super().__init__(
            config_path=config_path,
            labels=self.label_values,
            vocab_path=vocab_path,
            vision_config_path=vision_config_path,
            max_seq_length=max_seq_length,
        )
        self.processor = SiglipProcessor(
            vocab_path=vocab_path,
            vision_config_path=vision_config_path,
            max_seq_length=max_seq_length,
        )
        self._device = "cpu" if device == "cpu" else int(device)

        self.from_pretrained(weight_path, state_dict=state_dict)

        if lora_weight_path is not None:
            self.load_lora_weights(
                lora_weight_path,
                lora_weights=lora_weight,
                lora_alphas=lora_alpha,
                save_base_state=False,
            )

        self.to(device=self._device)

    @classmethod
    @add_default_section_for_init("microsoft/models/siglip/pipeline/matching/v2")
    def from_core_configure(
        cls,
        config,
        pretrained_name: Optional[str] = None,
        config_path: Optional[str] = None,
        vocab_path: Optional[str] = None,
        vision_config_path: Optional[str] = None,
        pretrained_weight_path: Optional[str] = None,
        pretrained_lora_weight_path: Optional[str] = None,
        label_dict: Optional[Dict[str, str]] = None,
        device: Optional[str] = None,
        act_fn: Optional[str] = None,
        **kwargs,
    ):
        config.set_default_section("microsoft/models/siglip/pipeline/matching/v2")

        config_path = config_path or config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_siglip_infos, pretrained_name, "config"),
        )
        config_path = cached_path(config_path)

        vocab_path = vocab_path or config.getoption("vocab_path", None)
        vocab_path = pop_value(
            vocab_path,
            nested_dict_value(pretrained_siglip_infos, pretrained_name, "vocab"),
        )
        vocab_path = cached_path(vocab_path)

        vision_config_path = vision_config_path or config.getoption(
            "vision_config_path", None
        )
        vision_config_path = pop_value(
            vision_config_path,
            nested_dict_value(
                pretrained_siglip_infos, pretrained_name, "vision_config"
            ),
        )

        vision_config_path = cached_path(vision_config_path)

        max_seq_length = config.getoption("max_seq_length", 64)
        device = config.getoption("device", None) if device is None else device
        pretrained_weight_path = (
            config.getoption("pretrained_weight_path", None)
            if pretrained_weight_path is None
            else pretrained_weight_path
        )
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_siglip_infos, pretrained_name, "weight"),
            check_none=False,
        )

        lora_weight_path = (
            config.getoption("pretrained_lora_weight_path", None)
            if pretrained_lora_weight_path is None
            else pretrained_lora_weight_path
        )
        lora_weight = config.getoption("pretrained_lora_weight", 1.0)
        lora_alpha = config.getoption("pretrained_lora_alpha", 32.0)
        label_dict = (
            config.getoption("label_dict", None) if label_dict is None else label_dict
        )
        act_fn = config.getoption("act_fn", None) if act_fn is None else act_fn

        inst = cls(
            config_path=config_path,
            vocab_path=vocab_path,
            vision_config_path=vision_config_path,
            max_seq_length=max_seq_length,
            weight_path=weight_path,
            lora_weight_path=lora_weight_path,
            lora_weight=lora_weight,
            lora_alpha=lora_alpha,
            label_dict=label_dict,
            device=device,
            act_fn=act_fn,
        )

        return inst

    @torch.no_grad()
    @add_default_section_for_function("microsoft/models/siglip/pipeline/matching/v2")
    def __call__(
        self,
        image,
    ):
        assert image is not None, "image must be provided"
        inputs = self.processor.image_classification(image)
        inputs = {k: v.unsqueeze(0) if v is not None else v for k, v in inputs.items()}
        inputs = {
            k: v.to(device=self._device) if v is not None else v
            for k, v in inputs.items()
        }
        results = self.forward(
            pixel_values=inputs["pixel_values"],
        ).outputs
        if self.act_fn is not None:
            results = self.act_fn(results)
        return {k: v for k, v in zip(self.label_keys, results[0])}
