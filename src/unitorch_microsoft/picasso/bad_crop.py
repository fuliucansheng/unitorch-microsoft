# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import json
import math
import re
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image
from torch import autocast
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
try:
    from transformers.models.siglip.modeling_siglip import (
        SiglipConfig,
        SiglipTextTransformer,
        SiglipVisionTransformer,
    )
except ImportError:
    from transformers.models.siglip.modeling_siglip import (
        SiglipConfig,
        SiglipTextModel as SiglipTextTransformer,
        SiglipVisionModel as SiglipVisionTransformer,
    )
from unitorch.utils import pop_value, nested_dict_value, read_file, read_json_file
from unitorch.models import GenericModel
from unitorch.models.siglip import SiglipProcessor
from unitorch.models.peft import PeftWeightLoaderMixin
from unitorch.cli import (
    hf_endpoint_url,
    cached_path,
    config_defaults_init,
    config_defaults_method,
    register_model,
)
from unitorch.cli import Config
from unitorch.cli.models.clip import pretrained_clip_infos
from unitorch.cli import WriterOutputs, register_process
from unitorch.cli.models import (
    TensorInputs,
    GenerationOutputs,
    GenerationTargets,
)
from unitorch.cli.models import TensorOutputs, ClassificationOutputs
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
    @config_defaults_init("microsoft/picasso/model/bad_crop/siglip")
    def from_config(cls, config, **kwargs):
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
        meta_infos: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            vocab_path=vocab_path,
            vision_config_path=vision_config_path,
            max_seq_length=max_seq_length,
            position_start_id=position_start_id,
        )
        if isinstance(meta_infos, str):
            try:
                meta_infos = json.loads(meta_infos)
            except json.JSONDecodeError:
                meta_infos = {}
        self.meta_infos = meta_infos or {}

    @classmethod
    @config_defaults_init("microsoft/picasso/process/bad_crop")
    def from_config(cls, config, **kwargs):
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
        meta_infos = config.getoption("meta_infos", None)

        return {
            "vocab_path": vocab_path,
            "vision_config_path": vision_config_path,
            "meta_infos": meta_infos,
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

    def _meta_value(self, key, default=None):
        if not isinstance(self.meta_infos, dict):
            return default
        return self.meta_infos.get(key, default)

    def _meta_bool(self, key, default=False):
        value = self._meta_value(key, default)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y"}
        return bool(value)

    def _meta_float(self, key, default=0.0):
        try:
            return float(self._meta_value(key, default))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _to_bool(value):
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y"}
        return bool(value)

    @staticmethod
    def _to_float(value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _get_any(data, *keys, default=None):
        for key in keys:
            if isinstance(data, dict) and key in data and data[key] is not None:
                return data[key]
        return default

    @staticmethod
    def _json_items(value):
        if value is None:
            return []
        if isinstance(value, float) and math.isnan(value):
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            return [value]

        value = str(value).strip()
        if not value or value.lower() in {"nan", "none", "null"}:
            return []

        items = []
        for part in value.split("|"):
            part = part.strip()
            if not part:
                continue
            try:
                parsed = json.loads(part)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, list):
                items.extend(parsed)
            elif isinstance(parsed, dict):
                items.append(parsed)
        return items

    @staticmethod
    def _clamp_box(box, image_size):
        image_width, image_height = image_size
        x, y, width, height = box
        x = max(0.0, min(float(x), float(image_width)))
        y = max(0.0, min(float(y), float(image_height)))
        width = max(0.0, min(float(width), float(image_width) - x))
        height = max(0.0, min(float(height), float(image_height) - y))
        return (x, y, width, height)

    @staticmethod
    def _contains(outer, inner):
        ox, oy, ow, oh = outer
        ix, iy, iw, ih = inner
        return ix >= ox and iy >= oy and ix + iw <= ox + ow and iy + ih <= oy + oh

    def _parse_manual_rois_v2(self, manual_info, image_size):
        rois = []
        for item in self._json_items(manual_info):
            if not isinstance(item, dict):
                continue

            crop_type = self._get_any(
                item, "image_crop_type", "CroppingType", "croppingType"
            )
            if str(crop_type).lower() in {"2", "smartcrop"}:
                continue

            box = (
                self._get_any(
                    item, "top_left_x", "x", "OffsetX", "SourceX", "cropX", default=0
                ),
                self._get_any(
                    item, "top_left_y", "y", "OffsetY", "SourceY", "cropY", default=0
                ),
                self._get_any(
                    item, "width", "Width", "SourceWidth", "cropWidth", default=0
                ),
                self._get_any(
                    item, "height", "Height", "SourceHeight", "cropHeight", default=0
                ),
            )
            box = tuple(self._to_float(v) for v in box)
            box = self._clamp_box(box, image_size)
            if box[2] > 0 and box[3] > 0:
                rois.append(box)
        return rois

    def _parse_smart_info_v2(self, smart_info, image_size):
        items = self._json_items(smart_info)
        data = items[0] if items and isinstance(items[0], dict) else None
        if not data:
            return {}

        image_width, image_height = image_size
        width = int(self._to_float(data.get("ThumbnailWidth"), image_width))
        height = int(self._to_float(data.get("ThumbnailHeight"), image_height))

        def parsed(box):
            if not box or len(box) < 4:
                return None
            box = tuple(self._to_float(v) for v in box[:4])
            box = self._clamp_box(box, (width, height))
            return box if box[2] > 0 and box[3] > 0 else None

        strict_rois = data.get("StrictROIs") or data.get("StrictROIInOriginalImage") or []
        return {
            "size": (width, height),
            "roi": parsed(data.get("ROI") or data.get("ROIInOriginalImage")),
            "original_roi": parsed(data.get("RawROI")),
            "strict_rois": [box for box in (parsed(item) for item in strict_rois) if box],
            "background_type": str(data.get("BackgroundType", "")).strip().lower(),
            "padding_color": str(data.get("PaddingColor", "")).strip(),
        }

    def _target_size_v2(self, image_size, ratio=None, width=None, height=None):
        image_width, image_height = image_size

        if ratio is None:
            width = int(self._to_float(width, 0.0))
            height = int(self._to_float(height, 0.0))
            if width > 0 and height > 0:
                return width / height, width, height

            width = int(self._meta_float("target_width", 0.0))
            height = int(self._meta_float("target_height", 0.0))
            if width > 0 and height > 0:
                return width / height, width, height

        if ratio is None:
            return None, None, None

        ratio = float(ratio)
        image_ratio = image_width / image_height
        if ratio >= image_ratio:
            return ratio, image_width, max(1, int(image_width / ratio))
        return ratio, max(1, int(image_height * ratio)), image_height

    @staticmethod
    def _try_position_v2(dist1, dist2, backfill, original_position, start_point):
        if dist1 >= backfill / 2.0 and dist2 >= backfill / 2.0:
            return original_position - backfill / 2.0
        if dist1 < backfill / 2.0 and dist1 + dist2 >= backfill:
            return start_point
        if dist2 < backfill / 2.0 and dist1 + dist2 >= backfill:
            return original_position - (backfill - dist2)
        return None

    @staticmethod
    def _round_calc_v2(calc):
        eps = 1e-9
        return (
            int(math.ceil(calc[0] - eps)),
            int(math.ceil(calc[1] - eps)),
            int(math.floor(calc[2] + eps)),
            int(math.floor(calc[3] + eps)),
        )

    @staticmethod
    def _validate_calc_v2(calc, boundary_size, req_w, req_h):
        width, height = boundary_size
        return (
            calc is not None
            and calc[0] >= 0
            and calc[1] >= 0
            and calc[2] > 0
            and calc[3] > 0
            and calc[0] + calc[2] <= width
            and calc[1] + calc[3] <= height
            and calc[2] >= req_w
            and calc[3] >= req_h
        )

    @staticmethod
    def _validate_region_v2(region, boundary_size, req_w=0, req_h=0):
        width, height = boundary_size
        return (
            region is not None
            and region[0] >= 0
            and region[1] >= 0
            and region[2] > 0
            and region[3] > 0
            and region[0] + region[2] <= width
            and region[1] + region[3] <= height
            and region[2] >= req_w
            and region[3] >= req_h
        )

    def _fit_requirement_v2(self, box, boundary_size, req_w, req_h):
        x, y, width, height = box
        if width <= 0 or height <= 0:
            return None

        boundary_width, boundary_height = boundary_size
        ratio = req_w / req_h
        roi_ratio = width / height
        if ratio == roi_ratio:
            return (x, y, width, height)
        if ratio > roi_ratio:
            new_width = ratio * height
            backfill = new_width - width
            new_x = self._try_position_v2(
                x, boundary_width - x - width, backfill, x, 0.0
            )
            return None if new_x is None else (new_x, y, new_width, height)

        new_height = width / ratio
        backfill = new_height - height
        new_y = self._try_position_v2(
            y, boundary_height - y - height, backfill, y, 0.0
        )
        return None if new_y is None else (x, new_y, width, new_height)

    def _fit_required_size_v2(self, calc, boundary_size, req_w, req_h):
        if calc[2] >= req_w and calc[3] >= req_h:
            return calc

        boundary_width, boundary_height = boundary_size
        x, y, width, height = calc
        new_x = self._try_position_v2(
            x, boundary_width - x - width, req_w - width, x, 0.0
        )
        if new_x is None:
            return None
        x, width = new_x, req_w

        new_y = self._try_position_v2(
            y, boundary_height - y - height, req_h - height, y, 0.0
        )
        if new_y is None:
            return None
        return (x, new_y, width, req_h)

    def _adjust_exact_v2(self, box, boundary_size, req_w, req_h):
        x, y, width, height = box
        if width <= 0 or height <= 0:
            return None

        boundary_width, boundary_height = boundary_size
        target_ratio = req_w / req_h
        source_ratio = width / height
        if target_ratio < source_ratio:
            new_width = height * target_ratio
            diff = new_width - width
            new_x = self._try_position_v2(
                x, boundary_width - x - width, diff, x, 0.0
            )
            return None if new_x is None else (new_x, y, new_width, height)
        if target_ratio > source_ratio:
            new_height = width / target_ratio
            diff = new_height - height
            new_y = self._try_position_v2(
                y, boundary_height - y - height, diff, y, 0.0
            )
            return None if new_y is None else (x, new_y, width, new_height)
        return (x, y, width, height)

    @staticmethod
    def _unified_check_v2(box, boundary_size, req_w, req_h):
        if not box or box[2] <= 0 or box[3] <= 0:
            return False
        boundary_width, boundary_height = boundary_size
        ratio = req_w / req_h
        return box[2] / boundary_height <= ratio <= boundary_width / box[3]

    def _adjust_roi_v2(self, box, boundary_size, req_w, req_h, strict_rois=None):
        if not box or box[2] <= 0 or box[3] <= 0:
            return None

        x, y, width, height = box
        boundary_width, boundary_height = boundary_size
        ratio = req_w / req_h
        min_ratio = width / boundary_height
        max_ratio = boundary_width / height
        threshold = self._meta_float("roi_shrink_threshold", 0.01)

        if ratio < min_ratio:
            shrink = 1.0 - (boundary_height * ratio / width)
            if shrink > threshold:
                return None
            new_width = int(width * (1.0 - shrink))
            new_x = self._try_position_v2(
                x, boundary_width - x - width, new_width - width, x, 0.0
            )
            adjusted = None if new_x is None else (int(new_x), y, new_width, height)
        elif ratio > max_ratio:
            shrink = 1.0 - (boundary_width / (ratio * height))
            if shrink > threshold:
                return None
            new_height = int(height * (1.0 - shrink))
            new_y = self._try_position_v2(
                y, boundary_height - y - height, new_height - height, y, 0.0
            )
            adjusted = None if new_y is None else (x, int(new_y), width, new_height)
        else:
            return None

        if adjusted and strict_rois:
            if not all(self._contains(adjusted, roi) for roi in strict_rois):
                return None
        return adjusted

    @staticmethod
    def _calc_in_box_v2(calc, box):
        return (
            calc[0] >= box[0]
            and calc[1] >= box[1]
            and calc[0] + calc[2] <= box[0] + box[2]
            and calc[1] + calc[3] <= box[1] + box[3]
        )

    def _scale_calc_v2(self, calc, area_box, req_w, req_h):
        if not self._meta_bool("enable_maximize_image_area", True):
            return calc
        if not area_box or not self._calc_in_box_v2(calc, area_box):
            return calc

        x, y, width, height = calc
        area_x, area_y, area_width, area_height = area_box
        scale_h = area_width / width
        scale_v = area_height / height
        if scale_h <= 1.0001 or scale_v <= 1.0001:
            return calc

        if scale_h <= scale_v:
            backfill = height * scale_h - height
            new_y = self._try_position_v2(
                y - area_y,
                area_y + area_height - y - height,
                backfill,
                y,
                area_y,
            )
            if new_y is None:
                return calc
            scaled = (area_x, new_y, area_width, height * scale_h)
        else:
            backfill = width * scale_v - width
            new_x = self._try_position_v2(
                x - area_x,
                area_x + area_width - x - width,
                backfill,
                x,
                area_x,
            )
            if new_x is None:
                return calc
            scaled = (new_x, area_y, width * scale_v, area_height)

        if self._validate_calc_v2(scaled, (area_width, area_height), req_w, req_h):
            return scaled
        return calc

    def _fit_box_region_v2(
        self,
        box,
        boundary_size,
        req_w,
        req_h,
        adjust=False,
        strict_rois=None,
        scale_box=None,
    ):
        work_box = box
        if adjust:
            work_box = self._adjust_roi_v2(
                box, boundary_size, req_w, req_h, strict_rois=strict_rois
            )
            if work_box is None:
                return None
        elif not self._unified_check_v2(box, boundary_size, req_w, req_h):
            return None

        calc = self._fit_requirement_v2(work_box, boundary_size, req_w, req_h)
        if calc is None:
            return None
        calc = self._fit_required_size_v2(calc, boundary_size, req_w, req_h)
        if calc is None or not self._validate_calc_v2(calc, boundary_size, req_w, req_h):
            return None

        calc = self._scale_calc_v2(calc, scale_box, req_w, req_h)
        region = self._round_calc_v2(calc)
        return region if self._validate_region_v2(region, boundary_size, req_w, req_h) else None

    @staticmethod
    def _crop_region_v2(image, region):
        x, y, width, height = region
        return image.crop((x, y, x + width, y + height))

    @staticmethod
    def _parse_color_v2(color):
        color = str(color or "").strip().lstrip("#")
        if len(color) == 6:
            try:
                return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))
            except ValueError:
                pass
        return (255, 255, 255)

    def _pad_to_ratio_v2(
        self,
        image,
        req_w,
        req_h,
        smart_info,
        is_pa=False,
        require_non_empty=True,
    ):
        if self._meta_bool("disable_padding", False):
            return None

        bg_type = (smart_info or {}).get("background_type", "")
        if require_non_empty and bg_type not in {"white", "simple", "complex"}:
            return None

        image_width, image_height = image.size
        if req_w > image_width and req_h > image_height:
            return None

        ratio = req_w / req_h
        image_ratio = image_width / image_height
        if ratio > image_ratio:
            pad_width, pad_height = image_height * ratio, image_height
        else:
            pad_width, pad_height = image_width, image_width / ratio
        pad_width_i, pad_height_i = int(pad_width), int(pad_height)
        offset = (int((pad_width - image_width) / 2), int((pad_height - image_height) / 2))

        algo = str(self._meta_value("padding_strategy_algo", "")).lower()
        if bg_type == "white":
            color = self._parse_color_v2((smart_info or {}).get("padding_color", ""))
            canvas = Image.new("RGB", (pad_width_i, pad_height_i), color)
        elif algo == "blurry":
            from PIL import ImageFilter

            canvas = image.resize((pad_width_i, pad_height_i), Image.LANCZOS).filter(
                ImageFilter.GaussianBlur(radius=20)
            )
        elif algo == "color":
            color = self._parse_color_v2(
                (smart_info or {}).get("padding_color", "C8C8C8")
            )
            canvas = Image.new("RGB", (pad_width_i, pad_height_i), color)
        elif is_pa:
            color = self._parse_color_v2((smart_info or {}).get("padding_color", ""))
            canvas = Image.new("RGB", (pad_width_i, pad_height_i), color)
        else:
            return None

        canvas.paste(image.convert("RGB"), offset)
        return canvas

    def _crop_by_rois_v2(self, image, rois, req_w, req_h, exact=False, adjust=False):
        tolerance = self._meta_float("similarity_tolerance", 0.01)
        ratio = req_w / req_h
        boundary_size = image.size
        scale_box = (0.0, 0.0, float(image.width), float(image.height))
        for roi in rois:
            if exact:
                if abs(roi[2] / roi[3] - ratio) > tolerance:
                    continue
                calc = self._adjust_exact_v2(roi, boundary_size, req_w, req_h)
                if calc is None:
                    continue
                calc = self._fit_required_size_v2(calc, boundary_size, req_w, req_h)
                region = self._round_calc_v2(calc) if calc else None
                if not self._validate_region_v2(region, boundary_size, req_w, req_h):
                    continue
            else:
                region = self._fit_box_region_v2(
                    roi,
                    boundary_size,
                    req_w,
                    req_h,
                    adjust=adjust,
                    scale_box=scale_box,
                )
            if region:
                return self._crop_region_v2(image, region)
        return None

    def _crop_by_original_exact_v2(self, image, req_w, req_h):
        boundary_size = image.size
        full_image = (0.0, 0.0, float(image.width), float(image.height))
        calc = self._adjust_exact_v2(full_image, boundary_size, req_w, req_h)
        if calc is None:
            return None
        calc = self._fit_required_size_v2(calc, boundary_size, req_w, req_h)
        region = self._round_calc_v2(calc) if calc else None
        return (
            self._crop_region_v2(image, region)
            if self._validate_region_v2(region, boundary_size, req_w, req_h)
            else None
        )

    def _crop_by_smart_v2(self, image, smart_info, req_w, req_h, adjust=False):
        if self._meta_bool("ai_enhancement_crop_optout", False):
            return None
        roi = (smart_info or {}).get("roi")
        if not roi:
            return None
        boundary_size = smart_info.get("size", image.size)
        scale_box = smart_info.get("original_roi") or (
            0.0,
            0.0,
            float(boundary_size[0]),
            float(boundary_size[1]),
        )
        region = self._fit_box_region_v2(
            roi,
            boundary_size,
            req_w,
            req_h,
            adjust=adjust,
            strict_rois=(smart_info or {}).get("strict_rois", []),
            scale_box=scale_box,
        )
        return self._crop_region_v2(image, region) if region else None

    def _center_crop_v2(self, image, req_w, req_h, roi=None):
        boundary_size = image.size
        if req_w > image.width or req_h > image.height:
            return None

        ratio = req_w / req_h
        image_ratio = image.width / image.height
        if image_ratio > ratio:
            width, height = image.height * ratio, float(image.height)
        else:
            width, height = float(image.width), image.width / ratio

        if roi is None:
            center_x, center_y = image.width / 2.0, image.height / 2.0
        else:
            center_x, center_y = roi[0] + roi[2] / 2.0, roi[1] + roi[3] / 2.0

        candidate = (
            math.floor(center_x - width / 2.0),
            math.floor(center_y - height / 2.0),
            math.floor(width),
            math.floor(height),
        )
        calc = self._adjust_exact_v2(candidate, boundary_size, req_w, req_h) or candidate
        calc = self._fit_required_size_v2(calc, boundary_size, req_w, req_h)
        region = self._round_calc_v2(calc) if calc else None
        return (
            self._crop_region_v2(image, region)
            if self._validate_region_v2(region, boundary_size, req_w, req_h)
            else None
        )

    def _center_by_rois_v2(self, image, rois, req_w, req_h):
        ratio = req_w / req_h
        image_ratio = image.width / image.height
        for roi in rois:
            roi_ratio = roi[2] / roi[3]
            if abs(image_ratio - ratio) <= abs(roi_ratio - ratio):
                continue
            cropped = self._center_crop_v2(image, req_w, req_h, roi)
            if cropped is not None:
                return cropped
        return None

    def _compute_bad_crop_image_v2(
        self,
        image,
        manual_info,
        smart_info,
        ratio,
        is_pa,
        width=None,
        height=None,
    ):
        ratio, req_w, req_h = self._target_size_v2(image.size, ratio, width, height)
        if ratio is None or ratio <= 0 or req_w <= 0 or req_h <= 0:
            return image

        is_pa = self._to_bool(is_pa)
        manual_rois = self._parse_manual_rois_v2(manual_info, image.size)
        smart = self._parse_smart_info_v2(smart_info, image.size)
        image_ratio = image.width / image.height
        tolerance = self._meta_float("similarity_tolerance", 0.01)

        cropped = self._crop_by_rois_v2(image, manual_rois, req_w, req_h, exact=True)
        if cropped is None and abs(image_ratio - ratio) <= tolerance * ratio:
            cropped = self._crop_by_original_exact_v2(image, req_w, req_h)
        if cropped is None:
            cropped = self._crop_by_smart_v2(image, smart, req_w, req_h, adjust=False)
        if cropped is None:
            cropped = self._crop_by_rois_v2(
                image, manual_rois, req_w, req_h, adjust=False
            )
        if cropped is None and smart.get("background_type") == "white":
            cropped = self._pad_to_ratio_v2(
                image, req_w, req_h, smart, is_pa=is_pa
            )
        if cropped is None and smart.get("background_type") != "white":
            cropped = self._crop_by_smart_v2(image, smart, req_w, req_h, adjust=True)
        if cropped is None and smart.get("background_type") != "white":
            cropped = self._crop_by_rois_v2(
                image, manual_rois, req_w, req_h, adjust=True
            )
        if cropped is None:
            cropped = self._pad_to_ratio_v2(
                image, req_w, req_h, smart, is_pa=is_pa
            )

        if cropped is None and is_pa:
            cropped = self._pad_to_ratio_v2(
                image,
                req_w,
                req_h,
                smart,
                is_pa=True,
                require_non_empty=False,
            )
        elif cropped is None:
            cropped = self._center_by_rois_v2(image, manual_rois, req_w, req_h)
            if cropped is None:
                cropped = self._center_crop_v2(image, req_w, req_h)
        return cropped or image

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
            TensorInputs: The processed inputs as tensors.
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
        elif crop in ["CampaignROI", "CampaignManualCropROI", "CampaignCenterCropROI"]:
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
        return TensorInputs(pixel_values=outputs.pixel_values, is_valid=is_valid)

    @register_process("microsoft/picasso/process/bad_crop/image_classification/v2")
    def _image_classification(
        self,
        image: Union[Image.Image, str],
        manual_info: Optional[str] = None,
        smart_info: Optional[str] = None,
        ratio: Optional[float] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        is_pa: Optional[bool] = False,
    ):
        if isinstance(image, str):
            image = Image.open(image)

        if ratio is not None:
            ratio = float(ratio)
        image = self._compute_bad_crop_image_v2(
            image=image,
            manual_info=manual_info,
            smart_info=smart_info,
            ratio=ratio,
            is_pa=is_pa,
            width=width,
            height=height,
        )

        outputs = super().image_classification(image=image.convert("RGB"))
        ratio, _, _ = self._target_size_v2(image.size, ratio, width, height)

        if ratio is not None and image.height > 0:
            image_ratio = image.width / image.height
            is_valid = torch.tensor(
                [1.0 if abs(image_ratio - ratio) < 0.01 else 0.0],
                dtype=torch.float32,
            )
        else:
            is_valid = torch.tensor([1.0], dtype=torch.float32)
        return TensorInputs(pixel_values=outputs.pixel_values, is_valid=is_valid)
