# Copyright (c) MICROSOFT.
# Licensed under the MIT License.
import re
import torch
import numpy as np
import pandas as pd
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.models import GenericOutputs
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
from unitorch_microsoft.models.bletchley import (
    pretrained_bletchley_v1_infos,
    pretrained_bletchley_v1_extensions_infos,
)
from unitorch_microsoft.models.bletchley.modeling_v1 import (
    BletchleyForPretrain as _BletchleyForPretrain,
    BletchleyForMatching as _BletchleyForMatching,
    BletchleyForMatchingV2 as _BletchleyForMatchingV2,
)
from unitorch_microsoft.models.bletchley.modeling_v1 import (
    get_bletchley_text_config,
    get_bletchley_image_config,
)
from unitorch_microsoft.models.bletchley.processing_v1 import BletchleyProcessor

ACT2FN = {
    "sigmoid": torch.sigmoid,
    "softmax": torch.nn.Softmax(dim=1),
}


class BletchleyForMatchingPipeline(_BletchleyForMatching):
    def __init__(
        self,
        config_type: str,
        max_seq_length: Optional[int] = 120,
        weight_path: Optional[Union[str, List[str]]] = None,
        lora_weight_path: Optional[Union[str, List[str]]] = None,
        lora_weight: Optional[float] = 1.0,
        lora_alpha: Optional[float] = 32.0,
        state_dict: Optional[Dict[str, Any]] = None,
        device: Optional[Union[str, int]] = "cpu",
    ):
        projection_dim = get_bletchley_text_config(config_type).global_vector_size
        super().__init__(
            config_type=config_type,
            projection_dim=projection_dim,
        )
        self.processor = BletchleyProcessor(
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
    @add_default_section_for_init("microsoft/models/bletchley/pipeline/v1/matching")
    def from_core_configure(
        cls,
        config,
        config_type: Optional[str] = None,
        pretrained_weight_path: Optional[str] = None,
        pretrained_lora_weight_path: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        config.set_default_section("microsoft/models/bletchley/pipeline/v1/matching")
        config_type = (
            config.getoption("config_type", "2.5B")
            if config_type is None
            else config_type
        )

        max_seq_length = config.getoption("max_seq_length", 120)
        device = config.getoption("device", None) if device is None else device
        pretrained_weight_path = (
            config.getoption("pretrained_weight_path", None)
            if pretrained_weight_path is None
            else pretrained_weight_path
        )
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_bletchley_v1_infos, config_type),
            check_none=False,
        )
        lora_weight_path = (
            config.getoption("pretrained_lora_weight_path", None)
            if pretrained_lora_weight_path is None
            else pretrained_lora_weight_path
        )
        lora_weight = config.getoption("pretrained_lora_weight", 1.0)
        lora_alpha = config.getoption("pretrained_lora_alpha", 32.0)

        inst = cls(
            config_type,
            max_seq_length=max_seq_length,
            weight_path=weight_path,
            lora_weight_path=lora_weight_path,
            lora_weight=lora_weight,
            lora_alpha=lora_alpha,
            device=device,
        )

        return inst

    @torch.no_grad()
    @add_default_section_for_function("microsoft/models/bletchley/pipeline/v1/matching")
    def __call__(
        self,
        text: str,
        image: Image.Image,
        max_seq_length: Optional[int] = 120,
    ):
        inputs = self.processor._classification(
            text=text,
            image=image,
            max_seq_length=max_seq_length,
        ).dict()
        inputs = {k: v.unsqueeze(0) if v is not None else v for k, v in inputs.items()}
        inputs = {
            k: v.to(device=self._device) if v is not None else v
            for k, v in inputs.items()
        }

        outputs = (
            super()
            .forward(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                images=inputs["images"],
            )
            .outputs
        )
        scores = outputs.sigmoid().squeeze(0)
        return scores[0].item()


class BletchleyForMatchingV2Pipeline(_BletchleyForMatchingV2):
    def __init__(
        self,
        config_type: str,
        max_seq_length: Optional[int] = 120,
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

        projection_dim = get_bletchley_text_config(config_type).global_vector_size
        self.label_keys = list(label_dict.keys())
        self.label_values = list(label_dict.values())
        self.act_fn = ACT2FN.get(act_fn, None)
        super().__init__(
            config_type=config_type,
            projection_dim=projection_dim,
            labels=self.label_values,
            max_seq_length=max_seq_length,
        )
        self.processor = BletchleyProcessor(
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
    @add_default_section_for_init("microsoft/models/bletchley/pipeline/v1/matching/v2")
    def from_core_configure(
        cls,
        config,
        config_type: Optional[str] = None,
        pretrained_weight_path: Optional[str] = None,
        pretrained_lora_weight_path: Optional[str] = None,
        label_dict: Optional[Dict[str, str]] = None,
        device: Optional[str] = None,
        act_fn: Optional[str] = None,
        **kwargs,
    ):
        config.set_default_section("microsoft/models/bletchley/pipeline/v1/matching/v2")
        config_type = (
            config.getoption("config_type", "2.5B")
            if config_type is None
            else config_type
        )

        max_seq_length = config.getoption("max_seq_length", 120)
        device = config.getoption("device", None) if device is None else device
        pretrained_weight_path = (
            config.getoption("pretrained_weight_path", None)
            if pretrained_weight_path is None
            else pretrained_weight_path
        )
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_bletchley_v1_infos, config_type),
            check_none=False,
        )

        lora_weight_path = (
            config.getoption("pretrained_lora_weight_path", None)
            if pretrained_lora_weight_path is None
            else pretrained_lora_weight_path
        )
        lora_weight = config.getoption("pretrained_lora_weight", 1.0)
        lora_alpha = config.getoption("pretrained_lora_alpha", 32.0)
        label_dict = config.getoption("label_dict", label_dict)
        act_fn = config.getoption("act_fn", None) if act_fn is None else act_fn

        inst = cls(
            config_type,
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
    @add_default_section_for_function(
        "microsoft/models/bletchley/pipeline/v1/matching/v2"
    )
    def __call__(
        self,
        image,
    ):
        assert image is not None, "image must be provided"
        inputs = self.processor._image_classification(image)
        inputs = {
            k: v.unsqueeze(0) if v is not None else v for k, v in inputs.dict().items()
        }
        inputs = {
            k: v.to(device=self._device) if v is not None else v
            for k, v in inputs.items()
        }
        results = self.forward(
            images=inputs["images"],
        ).outputs
        if self.act_fn is not None:
            results = self.act_fn(results)
        return {k: v for k, v in zip(self.label_keys, results[0])}
