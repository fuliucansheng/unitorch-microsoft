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
    pretrained_bletchley_v3_infos,
    pretrained_bletchley_v3_extensions_infos,
)
from unitorch_microsoft.models.bletchley.modeling_v3 import (
    BletchleyForMatching as _BletchleyForMatching,
)
from unitorch_microsoft.models.bletchley.processing_v3 import BletchleyProcessor


class BletchleyForMatchingPipeline(_BletchleyForMatching):
    def __init__(
        self,
        config_type: str,
        max_seq_length: Optional[int] = 120,
        weight_path: Optional[Union[str, List[str]]] = None,
        state_dict: Optional[Dict[str, Any]] = None,
        device: Optional[Union[str, int]] = "cpu",
    ):
        projection_dim = 1024 if config_type == "2.5B" else 768
        super().__init__(
            config_type=config_type,
            projection_dim=projection_dim,
        )
        self.processor = BletchleyProcessor(
            max_seq_length=max_seq_length,
        )
        self._device = "cpu" if device == "cpu" else int(device)

        self.from_pretrained(weight_path, state_dict=state_dict)
        self.to(device=self._device)

    @classmethod
    @add_default_section_for_init("microsoft/models/bletchley/pipeline/v3/matching")
    def from_core_configure(
        cls,
        config,
        config_type: Optional[str] = "2.5B",
        pretrained_weight_path: Optional[str] = None,
        device: Optional[str] = "cpu",
        **kwargs,
    ):
        config.set_default_section("microsoft/models/bletchley/pipeline/v3/matching")
        config_type = config.getoption("config_type", config_type)

        max_seq_length = config.getoption("max_seq_length", 77)
        device = config.getoption("device", device)
        pretrained_weight_path = config.getoption(
            "pretrained_weight_path", pretrained_weight_path
        )
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_bletchley_v3_infos, config_type),
            check_none=False,
        )

        inst = cls(
            config_type,
            max_seq_length=max_seq_length,
            weight_path=weight_path,
            device=device,
        )

        return inst

    @torch.no_grad()
    @add_default_section_for_function("microsoft/models/bletchley/pipeline/v3/matching")
    def __call__(
        self,
        text: str,
        image: Image.Image,
        max_seq_length: Optional[int] = 120,
        lora_checkpoints: Optional[Union[str, List[str]]] = None,
        lora_weights: Optional[Union[float, List[float]]] = 1.0,
        lora_alphas: Optional[Union[float, List[float]]] = 32,
        lora_urls: Optional[Union[str, List[str]]] = None,
        lora_files: Optional[Union[str, List[str]]] = None,
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

        if isinstance(lora_checkpoints, str):
            lora_checkpoints = [lora_checkpoints]
        if isinstance(lora_weights, float):
            lora_weights = [lora_weights]
        if isinstance(lora_alphas, float):
            lora_alphas = [lora_alphas]
        if isinstance(lora_urls, str):
            lora_urls = [lora_urls]
        if isinstance(lora_files, str):
            lora_files = [lora_files]

        assert (
            len(lora_checkpoints) == len(lora_weights)
            and len(lora_checkpoints) == len(lora_alphas)
            and len(lora_checkpoints) == len(lora_urls)
            and len(lora_checkpoints) == len(lora_files)
        )
        processed_lora_files, processed_lora_weights, processed_lora_alphas = [], [], []
        for ckpt, url, file, weight, alpha in zip(
            lora_checkpoints, lora_urls, lora_files, lora_weights, lora_alphas
        ):
            if ckpt is not None:
                lora_file = nested_dict_value(
                    pretrained_bletchley_v3_extensions_infos, ckpt, "weight"
                )
                processed_lora_files.append(lora_file)
                processed_lora_weights.append(weight)
                processed_lora_alphas.append(alpha)
            elif url is not None and is_remote_url(url):
                processed_lora_files.append(url)
                processed_lora_weights.append(weight)
                processed_lora_alphas.append(alpha)
            elif file is not None:
                processed_lora_files.append(file)
                processed_lora_weights.append(weight)
                processed_lora_alphas.append(alpha)

        if len(processed_lora_files) > 0:
            self.load_lora_weights(
                processed_lora_files,
                lora_weights=processed_lora_weights,
                lora_alphas=processed_lora_alphas,
            )

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
        self.unload_lora_weights()
        return scores[0].item()
