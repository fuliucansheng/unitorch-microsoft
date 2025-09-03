# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import re
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image
from torch import autocast
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from transformers.models.siglip.modeling_siglip import (
    SiglipConfig,
    SiglipTextTransformer,
    SiglipVisionTransformer,
)
from unitorch.utils import pop_value, nested_dict_value, read_file, read_json_file
from unitorch.models import GenericModel
from unitorch.models.siglip import SiglipProcessor
from unitorch.models.peft import PeftWeightLoaderMixin
from unitorch.cli import (
    hf_endpoint_url,
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
    register_model,
)
from unitorch.cli import CoreConfigureParser, GenericScript
from unitorch.cli import register_script
from unitorch.cli.models.clip import pretrained_clip_infos
from unitorch.cli import WriterOutputs, register_process
from unitorch.cli.models import (
    TensorsInputs,
    GenerationOutputs,
    GenerationTargets,
)
from unitorch.cli.models import TensorsOutputs, ClassificationOutputs
from unitorch.cli.models.siglip import pretrained_siglip_infos


@register_model("microsoft/picasso/model/bad_crop/siglip")
class SiglipForBadCropModel(GenericModel, PeftWeightLoaderMixin):
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
    @add_default_section_for_init("microsoft/picasso/model/bad_crop/siglip")
    def from_core_configure(cls, config, **kwargs):
        """
        Create an instance of SiglipForClassification from a core configuration.

        Args:
            config: The core configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            SiglipForClassification: An instance of the SiglipForClassification model.
        """
        config.set_default_section("microsoft/picasso/model/bad_crop/siglip")
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
        is_valid: torch.Tensor = None,
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
        if is_valid is not None:
            scores = scores * is_valid + (1 - is_valid) * -10000.0
        return ClassificationOutputs(outputs=scores)


class BadCropProcessor(SiglipProcessor):
    def __init__(
        self,
        vocab_path: str,
        vision_config_path: str,
        max_seq_length: Optional[int] = 128,
        position_start_id: Optional[int] = 0,
    ):
        super().__init__(
            vocab_path=vocab_path,
            vision_config_path=vision_config_path,
            max_seq_length=max_seq_length,
            position_start_id=position_start_id,
        )

    @classmethod
    @add_default_section_for_init("microsoft/picasso/process/bad_crop")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/picasso/process/bad_crop")
        pretrained_name = config.getoption("pretrained_name", "siglip-base-patch16-224")
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

        return {
            "vocab_path": vocab_path,
            "vision_config_path": vision_config_path,
        }

    def _processing_center_crop(self, image, ratio):
        image_width, image_height = image.size
        image_ratio = image_width / image_height

        if image_ratio > ratio:
            # Image is too wide
            new_height = image_height
            new_width = int(ratio * new_height)
            new_x = (image_width - new_width) // 2
            new_y = 0
        else:
            # Image is too tall
            new_width = image_width
            new_height = int(new_width / ratio)
            new_x = 0
            new_y = (image_height - new_height) // 2

        cropped_image = image.crop(
            (new_x, new_y, new_x + new_width, new_y + new_height)
        )
        return cropped_image

    def _processing_smart_crop(self, image, roi, ratio):
        x1, y1, x2, y2 = roi
        x = x1
        y = y1
        w = x2 - x1
        h = y2 - y1

        image_width, image_height = image.size
        image_ratio = image_width / image_height
        ratio_range = (w / image_height, image_width / h)

        if ratio < ratio_range[0] or ratio > ratio_range[1]:
            return image

        if image_ratio > ratio:
            new_h = image_height
            new_w = ratio * new_h
            w_diff = new_w - w
            left_distance = x
            right_distance = image_width - x - w

            if left_distance >= w_diff / 2 and right_distance >= w_diff / 2:
                new_x = x - w_diff / 2
            elif left_distance < w_diff / 2:
                new_x = 0
            else:
                new_x = image_width - new_w

            new_y = 0
        else:
            new_w = image_width
            new_h = new_w / ratio
            h_diff = new_h - h
            top_distance = y
            bottom_distance = image_height - y - h

            if top_distance >= h_diff / 2 and bottom_distance >= h_diff / 2:
                new_y = y - h_diff / 2
            elif top_distance < h_diff / 2:
                new_y = 0
            else:
                new_y = image_height - new_h

            new_x = 0

        # Clamp and round values
        new_x = int(round(max(0, min(new_x, image_width - new_w))))
        new_y = int(round(max(0, min(new_y, image_height - new_h))))
        new_w = int(round(min(new_w, image_width - new_x)))
        new_h = int(round(min(new_h, image_height - new_y)))

        return image.crop((new_x, new_y, new_x + new_w, new_y + new_h))

    def _processing_campaign_crop(self, image, rois, ratio):
        if len(rois) == 0:
            return image

        rois.sort(
            key=lambda x: abs((x[2] - x[0]) / (x[3] - x[1]) - ratio), reverse=True
        )
        roi = None
        if abs((rois[0][2] - rois[0][0]) / (rois[0][3] - rois[0][1]) - ratio) < 0.01:
            roi = rois[0]
        else:
            for item in rois:
                min_ratio = (item[2] - item[0]) / image.height
                max_ratio = image.width / (item[3] - item[1])
                if ratio >= min_ratio and ratio <= max_ratio:
                    roi = item
                    break

        if roi is None:
            return image

        return self._processing_smart_crop(
            image,
            roi,
            ratio,
        )

    @register_process("microsoft/picasso/process/bad_crop/image_classification")
    def _image_classification(
        self,
        image: Union[Image.Image, str],
        crop: Optional[str] = None,
        rois: Optional[str] = None,
        ratio: Optional[float] = None,
    ):
        """
        Process image inputs for image classification.

        Args:
            image (Union[Image.Image, str]): The input image.

        Returns:
            TensorsInputs: The processed inputs as tensors.
        """
        if isinstance(image, str):
            image = Image.open(image)

        def process(_rois):
            _rois = _rois.split(";")
            res = []
            for _roi in _rois:
                _roi = _roi.split(",")
                if len(_roi) != 4:
                    continue
                _roi = [float(x) for x in _roi]
                _roi[2] = _roi[2] + _roi[0]
                _roi[3] = _roi[3] + _roi[1]
                res.append(_roi)
            return res

        if crop == "CenterROI":
            image = self._processing_center_crop(image, ratio)
        elif crop == "SmartROI":
            if rois is None:
                raise ValueError("rois must be provided for SmartCrop")
            rois = [float(x) for x in rois.split(",")]
            rois[2] = rois[2] + rois[0]
            rois[3] = rois[3] + rois[1]
            if len(rois) != 4:
                raise ValueError("rois must contain 4 values")
            image = self._processing_smart_crop(image, rois, ratio)
        elif crop == "CampaignROI":
            if rois is None:
                raise ValueError("rois must be provided for CampaignCrop")
            rois = process(rois)
            image = self._processing_campaign_crop(image, rois, ratio)
        outputs = super().image_classification(image=image.convert("RGB"))
        if ratio is not None and image.height > 0:
            _ratio = image.width / image.height
            is_valid = torch.tensor(
                [1.0 if abs(_ratio - ratio) < 0.01 else 0.0],
                dtype=torch.float32,
            )
        else:
            is_valid = torch.tensor([1.0], dtype=torch.float32)
        return TensorsInputs(pixel_values=outputs.pixel_values, is_valid=is_valid)
