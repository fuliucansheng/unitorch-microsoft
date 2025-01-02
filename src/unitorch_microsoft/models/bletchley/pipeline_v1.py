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
)
from unitorch_microsoft.models.bletchley.modeling_v1 import (
    get_bletchley_text_config,
    get_bletchley_image_config,
)
from unitorch_microsoft.models.bletchley.processing_v1 import BletchleyProcessor


class BletchleyForMatchingPipeline(_BletchleyForMatching):
    def __init__(
        self,
        config_type: str,
        max_seq_length: Optional[int] = 120,
        weight_path: Optional[Union[str, List[str]]] = None,
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
        self.to(device=self._device)

    @classmethod
    @add_default_section_for_init("microsoft/models/bletchley/pipeline/v1/matching")
    def from_core_configure(
        cls,
        config,
        config_type: Optional[str] = "2.5B",
        pretrained_weight_path: Optional[str] = None,
        device: Optional[str] = "cpu",
        **kwargs,
    ):
        config.set_default_section("microsoft/models/bletchley/pipeline/v1/matching")
        config_type = config.getoption("config_type", config_type)

        max_seq_length = config.getoption("max_seq_length", 120)
        device = config.getoption("device", device)
        pretrained_weight_path = config.getoption(
            "pretrained_weight_path", pretrained_weight_path
        )
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_bletchley_v1_infos, config_type),
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
    @add_default_section_for_function("microsoft/models/bletchley/pipeline/v1/matching")
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
                    pretrained_bletchley_v1_extensions_infos, ckpt, "weight"
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


class BletchleyForMatchingV2Pipeline(_BletchleyForPretrain):
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

        assert label_dict is not None, "label_dict must be provided"
        self.label_dict = label_dict

        self.label_embs = {
            k: self.get_text_embeds(v).cpu().numpy().reshape(-1)
            for k, v in self.label_dict.items()
        }

    @classmethod
    @add_default_section_for_init("microsoft/models/bletchley/pipeline/v1/matching/v2")
    def from_core_configure(
        cls,
        config,
        config_type: Optional[str] = "2.5B",
        pretrained_weight_path: Optional[str] = None,
        pretrained_lora_weight_path: Optional[str] = None,
        label_dict: Optional[Dict[str, str]] = None,
        device: Optional[str] = "cpu",
        **kwargs,
    ):
        config.set_default_section("microsoft/models/bletchley/pipeline/v1/matching/v2")
        config_type = config.getoption("config_type", config_type)

        max_seq_length = config.getoption("max_seq_length", 120)
        device = config.getoption("device", device)
        pretrained_weight_path = config.getoption(
            "pretrained_weight_path", pretrained_weight_path
        )
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_bletchley_v1_infos, config_type),
            check_none=False,
        )

        lora_weight_path = config.getoption(
            "pretrained_lora_weight_path", pretrained_lora_weight_path
        )
        lora_weight = config.getoption("pretrained_lora_weight", 1.0)
        lora_alpha = config.getoption("pretrained_lora_alpha", 32.0)
        label_dict = config.getoption("label_dict", label_dict)

        inst = cls(
            config_type,
            max_seq_length=max_seq_length,
            weight_path=weight_path,
            lora_weight_path=lora_weight_path,
            lora_weight=lora_weight,
            lora_alpha=lora_alpha,
            label_dict=label_dict,
            device=device,
        )

        return inst

    @torch.no_grad()
    def get_image_embeds(self, image: Image.Image):
        inputs = self.processor._image_classification(image)
        inputs = {
            k: v.unsqueeze(0) if v is not None else v for k, v in inputs.dict().items()
        }
        inputs = {
            k: v.to(device=self._device) if v is not None else v
            for k, v in inputs.items()
        }
        image_outputs = self.image_encoder(inputs["images"])
        image_embeds = self.image_projection(image_outputs[:, 0])
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        return image_embeds

    @torch.no_grad()
    def get_text_embeds(self, text: str):
        inputs = self.processor._text_classification(text)
        inputs = {
            k: v.unsqueeze(0) if v is not None else v for k, v in inputs.dict().items()
        }
        inputs = {
            k: v.to(device=self._device) if v is not None else v
            for k, v in inputs.items()
        }
        text_outputs = self.text_encoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        text_embeds = self.text_projection(text_outputs[:, 0])
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        return text_embeds

    @torch.no_grad()
    @add_default_section_for_function(
        "microsoft/models/bletchley/pipeline/v1/matching/v2"
    )
    def __call__(
        self,
        image,
    ):
        assert image is not None, "image must be provided"
        image_embeds = self.get_image_embeds(image)
        image_embeds = image_embeds.cpu().numpy().reshape(1, -1)

        results = {
            k: (1 + np.dot(image_embeds, v)) / 2 for k, v in self.label_embs.items()
        }
        return results
