# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import random
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torch.cuda.amp import autocast
from peft import LoraConfig
from transformers.models.sam.modeling_sam import SamConfig, SamModel
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models import GenericModel
from unitorch.models.peft import GenericPeftModel, PeftModelForSequenceClassification
from unitorch.models.sam import SamProcessor
from unitorch.cli.models import SegmentationOutputs, LossOutputs
from unitorch.cli.models import segmentation_model_decorator
from unitorch.cli import (
    add_default_section_for_init,
    add_default_section_for_function,
    cached_path,
    register_model,
)
from unitorch.cli.models.sam import pretrained_sam_infos
from unitorch_microsoft.models.sam.utils import FocalLoss, DiceLoss, SSIMLoss, SSIM, calc_iou


class SamForSegmentation(GenericModel):
    def __init__(
        self,
        config_path: str,
    ):
        """
        Initializes a SamForSegmentation model for segmentation tasks.

        Args:
            config_path (str): The path to the Sam Transformer configuration file.
        """
        super().__init__()
        config = SamConfig.from_json_file(config_path)

        self.sam = SamModel(config)
        self.init_weights()

    def forward(
        self,
        pixel_values=None,
        input_points=None,
        input_boxes=None,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        Performs a forward pass of the SamForSegmentation model.
        """
        outputs = self.sam(
            pixel_values=pixel_values,
            input_points=input_points,
            input_boxes=input_boxes,
            multimask_output=False,
        )
        pred_masks = nn.functional.interpolate(
            outputs.pred_masks.squeeze(1),
            scale_factor=4,
            mode="bilinear",
            align_corners=False,
        )
        iou_scores = outputs.iou_scores.squeeze(1)
        return pred_masks, iou_scores


@register_model("microsoft/model/segmentation/peft/lora/sam")
class SamLoraForSegmentation(GenericPeftModel):
    prefix_keys_in_state_dict = {
        "^(?!peft_model\.base_model\.model\.sam\.).*": "peft_model.base_model.model.sam.",
    }
    replace_keys_in_state_dict = {
        "attn.proj.weight": "attn.proj.base_layer.weight",
        "attn.proj.bias": "attn.proj.base_layer.bias",
        "q_proj.weight": "q_proj.base_layer.weight",
        "q_proj.bias": "q_proj.base_layer.bias",
        "k_proj.weight": "k_proj.base_layer.weight",
        "k_proj.bias": "k_proj.base_layer.bias",
        "v_proj.weight": "v_proj.base_layer.weight",
        "v_proj.bias": "v_proj.base_layer.bias",
    }

    def __init__(
        self,
        config_path: str,
        lora_r: Optional[int] = 16,
        lora_alpha: Optional[int] = 32,
        lora_dropout: Optional[float] = 0.05,
        fan_in_fan_out: Optional[bool] = True,
        target_modules: Optional[Union[List[str], str]] = [
            "attn.proj",
            "q_proj",
            "k_proj",
            "v_proj",
        ],
    ):
        """
        Initializes a SamForSegmentation model for segmentation tasks.

        Args:
            config_path (str): The path to the Sam Transformer configuration file.
        """
        super().__init__()
        self.peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            target_modules=target_modules,
        )
        self.peft_model = PeftModelForSequenceClassification(
            SamForSegmentation(config_path),
            self.peft_config,
        )
        self.init_weights()
        # self.focal_loss = FocalLoss()
        # self.dice_loss = DiceLoss()
        # self.ssim_loss = SSIMLoss()
        

    @classmethod
    @add_default_section_for_init("microsoft/model/segmentation/peft/lora/sam")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/model/segmentation/peft/lora/sam")
        pretrained_name = config.getoption("pretrained_name", "sam-vit-base")
        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_sam_infos, pretrained_name, "config"),
        )

        config_path = cached_path(config_path)

        lora_r = config.getoption("lora_r", 16)
        lora_alpha = config.getoption("lora_alpha", 32)
        lora_dropout = config.getoption("lora_dropout", 0.05)
        fan_in_fan_out = config.getoption("fan_in_fan_out", True)
        target_modules = config.getoption(
            "target_modules",
            [
                "attn.proj",
                "q_proj",
                "k_proj",
                "v_proj",
            ],
        )

        inst = cls(
            config_path=config_path,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            target_modules=target_modules,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_sam_infos, pretrained_name, "weight"),
            check_none=False,
        )
        if weight_path is not None:
            inst.from_pretrained(weight_path)

        return inst

    @autocast()
    def forward(
        self,
        pixel_values: torch.Tensor,
        # input_points: torch.Tensor,
        input_boxes: torch.Tensor,
        pixel_targets: torch.Tensor,
        reshaped_input_sizes: torch.Tensor,
    ):
        """
        Performs a forward pass of the SamForSegmentation model.
        """
        pred_masks, iou_scores = self.peft_model(
            pixel_values=pixel_values,
            # input_points=input_points.unsqueeze(1),
            input_boxes=input_boxes,
            return_dict=False,
        )
        pred_masks = pred_masks.squeeze(1)
        iou_scores = iou_scores.squeeze(1)

        # loss_focal = torch.tensor(0.0, device=pred_masks.device)
        # loss_dice = torch.tensor(0.0, device=pred_masks.device)
        loss_iou = torch.tensor(0.0, device=pred_masks.device)
        loss_l1 = torch.tensor(0.0, device=pred_masks.device)
        # loss_l2 = torch.tensor(0.0, device=pred_masks.device)
        loss_bce = torch.tensor(0.0, device=pred_masks.device)
        loss_ssim = torch.tensor(0.0, device=pred_masks.device)
        for pred_mask, pixel_target, iou_score, reshaped_input_size in zip(
            pred_masks, pixel_targets, iou_scores, reshaped_input_sizes
        ):
            height, width = reshaped_input_size
            pred_mask = pred_mask[:height, :width]
            pixel_target = pixel_target[:height, :width]
            iou_label = calc_iou(pred_mask, pixel_target)
            # loss_focal += self.focal_loss(pred_mask, pixel_target)
            # loss_dice += self.dice_loss(pred_mask, pixel_target)
            loss_iou += F.mse_loss(iou_score, iou_label, reduction="sum")
            loss_l1 += F.l1_loss(pred_mask.reshape(-1).sigmoid(), pixel_target.reshape(-1), reduction="mean")
            # loss_l2 += F.mse_loss(pred_mask.reshape(-1).sigmoid(), pixel_target.reshape(-1), reduction="mean")
            loss_bce += F.binary_cross_entropy_with_logits(pred_mask.reshape(-1), pixel_target.reshape(-1).float(), reduction="mean")
            # loss_ssim += self.ssim_loss(pred_mask.sigmoid(), pixel_target)
            loss_ssim += SSIM(pred_mask.sigmoid(), pixel_target)
        
        loss = loss_iou + loss_l1 * 3 + loss_bce * 3 + loss_ssim
        return LossOutputs(loss=loss / len(pred_masks))
