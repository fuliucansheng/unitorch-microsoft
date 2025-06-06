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
from PIL import Image
from torch import autocast
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from transformers.models.clip.modeling_clip import (
    CLIPConfig,
    CLIPTextTransformer,
    CLIPVisionTransformer,
)
from transformers.models.siglip import (
    SiglipConfig,
    SiglipTextModel,
    SiglipVisionModel,
)
from unitorch.models import GenericModel
from unitorch.utils import pop_value, nested_dict_value
from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
    register_model,
    register_process,
)
from unitorch.cli.models import ClassificationOutputs, LossOutputs
from unitorch.cli.models.clip import pretrained_clip_infos
from unitorch.cli.models.siglip import pretrained_siglip_infos
import open_clip
import numpy as np
from torchvision import transforms
import kornia.augmentation as K
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from unitorch.models import (
    GenericOutputs,
    HfTextClassificationProcessor,
    HfImageClassificationProcessor,
)
from unitorch.cli.models import TensorsInputs


@register_model("microsoft/omnipixel/model/siglip/image")
class SiglipForImageClassification(GenericModel):
    """CLIP model for image classification."""

    def __init__(
        self,
        config_path: str,
        freeze_base_model: Optional[bool] = True,
        gradient_checkpointing: Optional[bool] = False,
    ):
        """
        Initialize the ClipForImageClassification model.

        Args:
            config_path (str): The path to the model configuration file.
            freeze_base_model (bool, optional): Whether to freeze the base model. Defaults to True.
            gradient_checkpointing (bool, optional): Whether to use gradient checkpointing. Defaults to False.
        """
        super().__init__()
        config = SiglipConfig.from_json_file(config_path)
        vision_config = config.vision_config
        vision_config.gradient_checkpointing = gradient_checkpointing

        self.vision_embed_dim = vision_config.hidden_size
        self.vision_model = SiglipVisionModel(vision_config).vision_model
        self.scoring_head = nn.Sequential(
            nn.Linear(self.vision_embed_dim, 1024),
            nn.Dropout(0.5),
            nn.Linear(1024, 128),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.Dropout(0.5),
            nn.Linear(64, 16),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
        )
        self.init_weights()

        if freeze_base_model:
            for p in self.vision_model.parameters():
                p.requires_grad = False

    @classmethod
    @add_default_section_for_init("microsoft/omnipixel/model/siglip/image")
    def from_core_configure(cls, config, **kwargs):
        """
        Create an instance of ClipForImageClassification from a core configuration.

        Args:
            config: The core configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            ClipForImageClassification: An instance of the ClipForImageClassification model.
        """
        config.set_default_section("microsoft/omnipixel/model/siglip/image")
        pretrained_name = config.getoption(
            "pretrained_name", "siglip-so400m-patch14-384"
        )
        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_siglip_infos, pretrained_name, "config"),
        )

        config_path = cached_path(config_path)

        freeze_base_model = config.getoption("freeze_base_model", True)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)

        inst = cls(
            config_path=config_path,
            freeze_base_model=freeze_base_model,
            gradient_checkpointing=gradient_checkpointing,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_siglip_infos, pretrained_name, "weight"),
            check_none=False,
        )
        if weight_path is not None:
            weight_path = [
                weight_path,
                "https://huggingface.co/datasets/fuliucansheng/unitorchblobfuse/resolve/main/models/adsplus/diffusion/pytorch_model.siglip.msra.bin",
            ]
            inst.from_pretrained(weight_path)

        return inst

    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(
        self,
        pixel_values: torch.Tensor,
    ):
        """
        Perform a forward pass through the model.

        Args:
            pixel_values (torch.Tensor): Input pixel values.

        Returns:
            ClassificationOutputs: The classification outputs.
        """
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
        )

        image_embeds = vision_outputs[1]

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        outputs = self.scoring_head(image_embeds)
        return ClassificationOutputs(outputs=outputs)


@register_model("microsoft/omnipixel/model/laion_clip/image")
class LAIONClipForImageClassification(GenericModel):
    """CLIP model for image classification."""

    def __init__(
        self,
        config_path: str,
        freeze_base_model: Optional[bool] = True,
        gradient_checkpointing: Optional[bool] = False,
    ):
        """
        Initialize the ClipForImageClassification model.

        Args:
            config_path (str): The path to the model configuration file.
            freeze_base_model (bool, optional): Whether to freeze the base model. Defaults to True.
            gradient_checkpointing (bool, optional): Whether to use gradient checkpointing. Defaults to False.
        """
        super().__init__()
        config = CLIPConfig.from_json_file(config_path)
        vision_config = config.vision_config
        vision_config.gradient_checkpointing = gradient_checkpointing

        self.vision_embed_dim = vision_config.hidden_size
        self.vision_model = CLIPVisionTransformer(vision_config)
        self.visual_projection = nn.Linear(
            self.vision_embed_dim,
            config.projection_dim,
            bias=False,
        )
        self.layers = nn.Sequential(
            nn.Linear(config.projection_dim, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )
        self.init_weights()

        if freeze_base_model:
            for p in self.vision_model.parameters():
                p.requires_grad = False

        self.vision_model.encoder.gradient_checkpointing = gradient_checkpointing

    @classmethod
    @add_default_section_for_init("microsoft/omnipixel/model/laion_clip/image")
    def from_core_configure(cls, config, **kwargs):
        """
        Create an instance of ClipForImageClassification from a core configuration.

        Args:
            config: The core configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            ClipForImageClassification: An instance of the ClipForImageClassification model.
        """
        config.set_default_section("microsoft/omnipixel/model/laion_clip/image")
        pretrained_name = config.getoption("pretrained_name", "clip-vit-large-patch16")
        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_clip_infos, pretrained_name, "config"),
        )

        config_path = cached_path(config_path)

        freeze_base_model = config.getoption("freeze_base_model", True)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)

        inst = cls(
            config_path=config_path,
            freeze_base_model=freeze_base_model,
            gradient_checkpointing=gradient_checkpointing,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_clip_infos, pretrained_name, "weight"),
            check_none=False,
        )
        if weight_path is not None:
            weight_path = [
                weight_path,
                "https://huggingface.co/datasets/fuliucansheng/unitorchblobfuse/resolve/main/models/adsplus/diffusion/pytorch_model.laion_clip.sac.logos.ava1.l14.msra.bin",
            ]
            inst.from_pretrained(weight_path)

        return inst

    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(
        self,
        pixel_values: torch.Tensor,
    ):
        """
        Perform a forward pass through the model.

        Args:
            pixel_values (torch.Tensor): Input pixel values.

        Returns:
            ClassificationOutputs: The classification outputs.
        """
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
        )

        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        outputs = self.layers(image_embeds)
        return ClassificationOutputs(outputs=outputs)


## image reward model
from ImageReward.models.BLIP import init_tokenizer
from ImageReward.models.BLIP.vit import VisionTransformer
from ImageReward.models.BLIP.med import BertConfig, BertModel
from ImageReward.ImageReward import MLP, _transform
from unitorch.models import (
    GenericOutputs,
    HfTextClassificationProcessor,
    HfImageClassificationProcessor,
)
from unitorch.cli.models import TensorsInputs


class ImageRewardProcessor(HfTextClassificationProcessor):
    def __init__(self):
        tokenizer = init_tokenizer()
        super().__init__(
            tokenizer=tokenizer,
            max_seq_length=35,
            source_type_id=0,
            target_type_id=0,
            position_start_id=tokenizer.pad_token_id + 1,
        )
        self.preprocess = _transform(224)

    @classmethod
    @add_default_section_for_init("microsoft/omnipixel/process/image_reward")
    def from_core_configure(cls, config, **kwargs):
        pass

    @register_process("microsoft/omnipixel/process/image_reward/classification")
    def _classification(
        self,
        text: str,
        image: Union[str, Image.Image],
        max_seq_length: Optional[int] = None,
    ):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        outputs = super().classification(text, max_seq_length)
        pixel_values = self.preprocess(image)
        return TensorsInputs(
            input_ids=outputs.input_ids,
            attention_mask=outputs.attention_mask,
            pixel_values=pixel_values,
        )


@register_model("microsoft/omnipixel/model/image_reward")
class ImageRewardModel(GenericModel):
    replace_keys_in_state_dict = {"blip.": ""}

    def __init__(self, config_path):
        super().__init__()
        self.visual_encoder = VisionTransformer(
            img_size=224,
            patch_size=16,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            use_grad_checkpointing=False,
            ckpt_layer=0,
            drop_path_rate=0.1,
        )
        config = BertConfig.from_json_file(config_path)
        config.encoder_width = 1024
        self.text_encoder = BertModel(config=config, add_pooling_layer=False)
        self.mlp = MLP(768)
        self.mean = 0.16717362830052426
        self.std = 1.0333394966054072

    @classmethod
    @add_default_section_for_init("microsoft/omnipixel/model/image_reward")
    def from_core_configure(cls, config, **kwargs):
        """
        Create an instance of ClipForImageClassification from a core configuration.

        Args:
            config: The core configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            ClipForImageClassification: An instance of the ClipForImageClassification model.
        """
        config.set_default_section("microsoft/omnipixel/model/image_reward")
        config_path = config.getoption("config_path", None)
        config_path = cached_path(config_path)

        inst = cls(config_path=config_path)
        weight_path = config.getoption("pretrained_weight_path", None)
        if weight_path is not None:
            inst.from_pretrained(weight_path)

        return inst

    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        image_embeds = self.visual_encoder(pixel_values)
        image_attn_mask = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device
        )
        text_outputs = self.text_encoder(
            input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attn_mask,
            return_dict=True,
        )
        txt_features = text_outputs.last_hidden_state[:, 0, :].float()  # (feature_dim)
        rewards = self.mlp(txt_features)
        rewards = (rewards - self.mean) / self.std
        return ClassificationOutputs(outputs=rewards)


# ## vqa model
# from transformers import CLIPImageProcessor, AutoTokenizer
# from t2v_metrics.models.vqascore_models.clip_t5_model import (
#     CLIPT5ForConditionalGeneration,
#     expand2square, load_pretrained_model, t5_tokenizer_image_token,
#     default_question_template,
#     default_answer_template,
#     format_question,
#     format_answer,
# )
# from unitorch.cli.models import TensorsInputs

# class VQACLIPT5Processor:
#     def __init__(self, max_seq_length=256):
#         self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xxl")
#         self.image_processor = CLIPImageProcessor.from_pretrained("zhiqiulin/clip-flant5-xxl")
#         self.max_seq_length = max_seq_length

#     @classmethod
#     @add_default_section_for_init("microsoft/omnipixel/process/vqa_clip_t5")
#     def from_core_configure(cls, config, **kwargs):
#         pass

#     @register_process("microsoft/omnipixel/process/vqa_clip_t5/generation/inputs")
#     def _generation_inputs(
#         self,
#         text: str,
#         image: Union[str, Image.Image],
#     ):
#         if isinstance(image, str):
#             image = Image.open(image).convert("RGB")

#         question = default_question_template.format(text)
#         answer = default_answer_template.format(text)

#         question = format_question(question, conversation_style=self.conversational_style)
#         answer = format_answer(answer, conversation_style=self.conversational_style)
#         input_ids = t5_tokenizer_image_token(question, self.tokenizer, return_tensors='pt')
#         labels = t5_tokenizer_image_token(answer, self.tokenizer, return_tensors='pt')
#         input_ids = input_ids[:self.max_seq_length] + [self.tokenizer.pad_token_id] * (self.max_seq_length - len(input_ids))
#         labels = labels[:self.max_seq_length] + [-100] * (self.max_seq_length - len(labels))
#         attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
#         decoder_attention_mask = labels.ne(-100)

#         image = expand2square(image, tuple(int(x*255) for x in self.image_processor.image_mean))
#         pixel_values = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

#         return TensorsInputs(
#             input_ids=torch.tensor(input_ids, dtype=torch.long),
#             attention_mask=torch.tensor(attention_mask, dtype=torch.long),
#             decoder_attention_mask=torch.tensor(decoder_attention_mask, dtype=torch.long),
#             labels=torch.tensor(labels, dtype=torch.long),
#             pixel_values=torch.tensor(pixel_values),
#         )

# @register_model("microsoft/omnipixel/model/vqa_clip_t5")
# class VQACLIPT5Model(GenericModel):
#     replace_keys_in_state_dict = {

#     }
#     def __init__(self):
#         super().__init__()
#         self.model = CLIPT5ForConditionalGeneration.from_pretrained("zhiqiulin/clip-flant5-xxl")

#     @classmethod
#     @add_default_section_for_init("microsoft/omnipixel/model/vqa_clip_t5")
#     def from_core_configure(cls, config, **kwargs):
#         """
#         Create an instance of ClipForImageClassification from a core configuration.

#         Args:
#             config: The core configuration.
#             **kwargs: Additional keyword arguments.

#         Returns:
#             ClipForImageClassification: An instance of the ClipForImageClassification model.
#         """
#         config.set_default_section("microsoft/omnipixel/model/vqa_clip_t5")
#         inst = cls()
#         return inst

#     @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
#     def forward(
#         self,
#         input_ids: torch.Tensor,
#         pixel_values: torch.Tensor,
#         labels: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#         decoder_attention_mask: Optional[torch.Tensor] = None,
#     ):
#         outputs = self.model(
#             input_ids=input_ids,
#             images=pixel_values,
#             labels=labels,
#             attention_mask=attention_mask,
#             decoder_attention_mask=decoder_attention_mask,
#         )

#         logits = outputs.logits
#         lm_prob = torch.zeros(logits.shape[0])
#         loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
#         for k in range(lm_prob.shape[0]):
#             lm_prob[k] = (-loss_fct(logits[k], labels[k])).exp() # exp to cancel the log and get raw prob between 0 and 1
#         return ClassificationOutputs(outputs=lm_prob)


## image quality model: AIGC Detection: https://github.com/shilinyan99/AIDE
# '0_real', '1_fake'


def DCT_mat(size):
    m = [
        [
            (np.sqrt(1.0 / size) if i == 0 else np.sqrt(2.0 / size))
            * np.cos((j + 0.5) * np.pi * i / size)
            for j in range(size)
        ]
        for i in range(size)
    ]
    return m


def generate_filter(start, end, size):
    return [
        [0.0 if i + j > end or i + j < start else 1.0 for j in range(size)]
        for i in range(size)
    ]


def norm_sigma(x):
    return 2.0 * torch.sigmoid(x) - 1.0


class Filter(nn.Module):
    def __init__(self, size, band_start, band_end, use_learnable=False, norm=False):
        super(Filter, self).__init__()
        self.use_learnable = use_learnable

        self.base = nn.Parameter(
            torch.tensor(generate_filter(band_start, band_end, size)),
            requires_grad=False,
        )
        if self.use_learnable:
            self.learnable = nn.Parameter(torch.randn(size, size), requires_grad=True)
            self.learnable.data.normal_(0.0, 0.1)
        self.norm = norm
        if norm:
            self.ft_num = nn.Parameter(
                torch.sum(torch.tensor(generate_filter(band_start, band_end, size))),
                requires_grad=False,
            )

    def forward(self, x):
        if self.use_learnable:
            filt = self.base + norm_sigma(self.learnable)
        else:
            filt = self.base

        if self.norm:
            y = x * filt / self.ft_num
        else:
            y = x * filt
        return y


class DCT_base_Rec_Module(nn.Module):
    """_summary_

    Args:
        x: [C, H, W] -> [C*level, output, output]
    """

    def __init__(
        self, window_size=32, stride=16, output=256, grade_N=6, level_fliter=[0]
    ):
        super().__init__()

        assert output % window_size == 0
        assert len(level_fliter) > 0

        self.window_size = window_size
        self.grade_N = grade_N
        self.level_N = len(level_fliter)
        self.N = (output // window_size) * (output // window_size)

        self._DCT_patch = nn.Parameter(
            torch.tensor(DCT_mat(window_size)).float(), requires_grad=False
        )
        self._DCT_patch_T = nn.Parameter(
            torch.transpose(torch.tensor(DCT_mat(window_size)).float(), 0, 1),
            requires_grad=False,
        )

        self.unfold = nn.Unfold(kernel_size=(window_size, window_size), stride=stride)
        self.fold0 = nn.Fold(
            output_size=(window_size, window_size),
            kernel_size=(window_size, window_size),
            stride=window_size,
        )

        lm, mh = 2.82, 2
        level_f = [Filter(window_size, 0, window_size * 2)]

        self.level_filters = nn.ModuleList([level_f[i] for i in level_fliter])
        self.grade_filters = nn.ModuleList(
            [
                Filter(
                    window_size,
                    window_size * 2.0 / grade_N * i,
                    window_size * 2.0 / grade_N * (i + 1),
                    norm=True,
                )
                for i in range(grade_N)
            ]
        )

    def forward(self, x):
        N = self.N
        grade_N = self.grade_N
        level_N = self.level_N
        window_size = self.window_size
        C, W, H = x.shape
        x_unfold = self.unfold(x.unsqueeze(0)).squeeze(0)

        _, L = x_unfold.shape
        x_unfold = x_unfold.transpose(0, 1).reshape(L, C, window_size, window_size)
        x_dct = self._DCT_patch @ x_unfold @ self._DCT_patch_T

        y_list = []
        for i in range(self.level_N):
            x_pass = self.level_filters[i](x_dct)
            y = self._DCT_patch_T @ x_pass @ self._DCT_patch
            y_list.append(y)
        level_x_unfold = torch.cat(y_list, dim=1)

        grade = torch.zeros(L).to(x.device)
        w, k = 1, 2
        for _ in range(grade_N):
            _x = torch.abs(x_dct)
            _x = torch.log(_x + 1)
            _x = self.grade_filters[_](_x)
            _x = torch.sum(_x, dim=[1, 2, 3])
            grade += w * _x
            w *= k

        _, idx = torch.sort(grade)
        max_idx = torch.flip(idx, dims=[0])[:N]
        maxmax_idx = max_idx[0]
        if len(max_idx) == 1:
            maxmax_idx1 = max_idx[0]
        else:
            maxmax_idx1 = max_idx[1]

        min_idx = idx[:N]
        minmin_idx = idx[0]
        if len(min_idx) == 1:
            minmin_idx1 = idx[0]
        else:
            minmin_idx1 = idx[1]

        x_minmin = torch.index_select(level_x_unfold, 0, minmin_idx)
        x_maxmax = torch.index_select(level_x_unfold, 0, maxmax_idx)
        x_minmin1 = torch.index_select(level_x_unfold, 0, minmin_idx1)
        x_maxmax1 = torch.index_select(level_x_unfold, 0, maxmax_idx1)

        x_minmin = x_minmin.reshape(
            1, level_N * C * window_size * window_size
        ).transpose(0, 1)
        x_maxmax = x_maxmax.reshape(
            1, level_N * C * window_size * window_size
        ).transpose(0, 1)
        x_minmin1 = x_minmin1.reshape(
            1, level_N * C * window_size * window_size
        ).transpose(0, 1)
        x_maxmax1 = x_maxmax1.reshape(
            1, level_N * C * window_size * window_size
        ).transpose(0, 1)

        x_minmin = self.fold0(x_minmin)
        x_maxmax = self.fold0(x_maxmax)
        x_minmin1 = self.fold0(x_minmin1)
        x_maxmax1 = self.fold0(x_maxmax1)

        return x_minmin, x_maxmax, x_minmin1, x_maxmax1


class AIDetectProcessor:
    def __init__(self):
        self.transform_before = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.transform = transforms.Compose(
            [
                transforms.Resize([256, 256]),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.dct = DCT_base_Rec_Module()

    @classmethod
    @add_default_section_for_init("microsoft/omnipixel/process/AIDetect")
    def from_core_configure(cls, config, **kwargs):
        pass

    @register_process("microsoft/omnipixel/process/AIDetect/classification")
    def _classification(
        self,
        image: Union[str, Image.Image],
    ):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        image = self.transform_before(image)

        # x_max, x_min, x_max_min, x_minmin = self.dct(image)

        x_minmin, x_maxmax, x_minmin1, x_maxmax1 = self.dct(image)

        x_0 = self.transform(image)
        x_minmin = self.transform(x_minmin)
        x_maxmax = self.transform(x_maxmax)

        x_minmin1 = self.transform(x_minmin1)
        x_maxmax1 = self.transform(x_maxmax1)

        pixel_values = torch.stack(
            [x_minmin, x_maxmax, x_minmin1, x_maxmax1, x_0], dim=0
        )

        return TensorsInputs(
            Image=pixel_values,
        )


class HPF(nn.Module):
    def __init__(self):
        super(HPF, self).__init__()

        # Load 30 SRM Filters
        all_hpf_list_5x5 = []
        all_normalized_hpf_list = self.srm_filter_kernel()
        for hpf_item in all_normalized_hpf_list:
            if hpf_item.shape[0] == 3:
                hpf_item = np.pad(hpf_item, pad_width=((1, 1), (1, 1)), mode="constant")

            all_hpf_list_5x5.append(hpf_item)

        hpf_weight = torch.Tensor(all_hpf_list_5x5).view(30, 1, 5, 5).contiguous()
        hpf_weight = torch.nn.Parameter(
            hpf_weight.repeat(1, 3, 1, 1), requires_grad=False
        )

        self.hpf = nn.Conv2d(3, 30, kernel_size=5, padding=2, bias=False)
        self.hpf.weight = hpf_weight

    def srm_filter_kernel(self):
        filter_class_1 = [
            np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=np.float32),
            np.array([[0, 1, 0], [0, -1, 0], [0, 0, 0]], dtype=np.float32),
            np.array([[0, 0, 1], [0, -1, 0], [0, 0, 0]], dtype=np.float32),
            np.array([[0, 0, 0], [1, -1, 0], [0, 0, 0]], dtype=np.float32),
            np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]], dtype=np.float32),
            np.array([[0, 0, 0], [0, -1, 0], [1, 0, 0]], dtype=np.float32),
            np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]], dtype=np.float32),
            np.array([[0, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float32),
        ]

        filter_class_2 = [
            np.array([[1, 0, 0], [0, -2, 0], [0, 0, 1]], dtype=np.float32),
            np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]], dtype=np.float32),
            np.array([[0, 0, 1], [0, -2, 0], [1, 0, 0]], dtype=np.float32),
            np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]], dtype=np.float32),
        ]

        filter_class_3 = [
            np.array(
                [
                    [-1, 0, 0, 0, 0],
                    [0, 3, 0, 0, 0],
                    [0, 0, -3, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    [0, 0, -1, 0, 0],
                    [0, 0, 3, 0, 0],
                    [0, 0, -3, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    [0, 0, 0, 0, -1],
                    [0, 0, 0, 3, 0],
                    [0, 0, -3, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 1, -3, 3, -1],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, -3, 0, 0],
                    [0, 0, 0, 3, 0],
                    [0, 0, 0, 0, -1],
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, -3, 0, 0],
                    [0, 0, 3, 0, 0],
                    [0, 0, -1, 0, 0],
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, -3, 0, 0],
                    [0, 3, 0, 0, 0],
                    [-1, 0, 0, 0, 0],
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [-1, 3, -3, 1, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=np.float32,
            ),
        ]

        filter_edge_3x3 = [
            np.array([[-1, 2, -1], [2, -4, 2], [0, 0, 0]], dtype=np.float32),
            np.array([[0, 2, -1], [0, -4, 2], [0, 2, -1]], dtype=np.float32),
            np.array([[0, 0, 0], [2, -4, 2], [-1, 2, -1]], dtype=np.float32),
            np.array([[-1, 2, 0], [2, -4, 0], [-1, 2, 0]], dtype=np.float32),
        ]

        filter_edge_5x5 = [
            np.array(
                [
                    [-1, 2, -2, 2, -1],
                    [2, -6, 8, -6, 2],
                    [-2, 8, -12, 8, -2],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    [0, 0, -2, 2, -1],
                    [0, 0, 8, -6, 2],
                    [0, 0, -12, 8, -2],
                    [0, 0, 8, -6, 2],
                    [0, 0, -2, 2, -1],
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [-2, 8, -12, 8, -2],
                    [2, -6, 8, -6, 2],
                    [-1, 2, -2, 2, -1],
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    [-1, 2, -2, 0, 0],
                    [2, -6, 8, 0, 0],
                    [-2, 8, -12, 0, 0],
                    [2, -6, 8, 0, 0],
                    [-1, 2, -2, 0, 0],
                ],
                dtype=np.float32,
            ),
        ]

        square_3x3 = np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float32)

        square_5x5 = np.array(
            [
                [-1, 2, -2, 2, -1],
                [2, -6, 8, -6, 2],
                [-2, 8, -12, 8, -2],
                [2, -6, 8, -6, 2],
                [-1, 2, -2, 2, -1],
            ],
            dtype=np.float32,
        )

        normalized_filter_class_2 = [hpf / 2 for hpf in filter_class_2]
        normalized_filter_class_3 = [hpf / 3 for hpf in filter_class_3]
        normalized_filter_edge_3x3 = [hpf / 4 for hpf in filter_edge_3x3]
        normalized_square_3x3 = square_3x3 / 4
        normalized_filter_edge_5x5 = [hpf / 12 for hpf in filter_edge_5x5]
        normalized_square_5x5 = square_5x5 / 12

        all_normalized_hpf_list = (
            filter_class_1
            + normalized_filter_class_2
            + normalized_filter_class_3
            + normalized_filter_edge_3x3
            + normalized_filter_edge_5x5
            + [normalized_square_3x3, normalized_square_5x5]
        )

        return all_normalized_hpf_list

    def forward(self, input):
        output = self.hpf(input)

        return output


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=True):
        super(ResNet, self).__init__()

        self.inplanes = 64
        self.conv1 = nn.Conv2d(30, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


@register_model("microsoft/omnipixel/model/AIDetect")
class AIDetectModel(GenericModel):
    # replace_keys_in_state_dict = {"blip.": ""}

    def __init__(self, resnet_path=None):
        super().__init__()
        self.hpf = HPF()
        self.model_min = ResNet(Bottleneck, [3, 4, 6, 3])
        self.model_max = ResNet(Bottleneck, [3, 4, 6, 3])

        if resnet_path is not None:
            pretrained_dict = torch.load(resnet_path, map_location="cpu")

            model_min_dict = self.model_min.state_dict()
            model_max_dict = self.model_max.state_dict()

            for k in pretrained_dict.keys():
                if (
                    k in model_min_dict
                    and pretrained_dict[k].size() == model_min_dict[k].size()
                ):
                    model_min_dict[k] = pretrained_dict[k]
                    model_max_dict[k] = pretrained_dict[k]
                else:
                    print(f"Skipping layer {k} because of size mismatch")

        self.fc = Mlp(2048 + 256, 1024, 2)

        print("build model with convnext_xxl")
        self.openclip_convnext_xxl, _, _ = open_clip.create_model_and_transforms(
            "convnext_xxlarge"
        )

        self.openclip_convnext_xxl = self.openclip_convnext_xxl.visual.trunk
        self.openclip_convnext_xxl.head.global_pool = nn.Identity()
        self.openclip_convnext_xxl.head.flatten = nn.Identity()

        self.openclip_convnext_xxl.eval()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.convnext_proj = nn.Sequential(
            nn.Linear(3072, 256),
        )
        for param in self.openclip_convnext_xxl.parameters():
            param.requires_grad = False

    @classmethod
    @add_default_section_for_init("microsoft/omnipixel/model/AIDetect")
    def from_core_configure(cls, config, **kwargs):
        """
        Create an instance of ClipForImageClassification from a core configuration.

        Args:
            config: The core configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            ClipForImageClassification: An instance of the ClipForImageClassification model.
        """
        config.set_default_section("microsoft/omnipixel/model/AIDetect")
        resnet_path = config.getoption("resnet_path", None)

        inst = cls(resnet_path=resnet_path)
        weight_path = config.getoption("pretrained_weight_path", None)
        if weight_path is not None:
            inst.from_pretrained(weight_path)

        return inst

    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(
        self,
        Image: torch.Tensor,
    ):
        if isinstance(Image, TensorsInputs):
            Image = Image.Image

        b, t, c, h, w = Image.shape

        x_minmin = Image[:, 0]  # [b, c, h, w]
        x_maxmax = Image[:, 1]
        x_minmin1 = Image[:, 2]
        x_maxmax1 = Image[:, 3]
        tokens = Image[:, 4]

        x_minmin = self.hpf(x_minmin)
        x_maxmax = self.hpf(x_maxmax)
        x_minmin1 = self.hpf(x_minmin1)
        x_maxmax1 = self.hpf(x_maxmax1)

        with torch.no_grad():
            clip_mean = torch.Tensor([0.48145466, 0.4578275, 0.40821073])
            clip_mean = clip_mean.to(tokens, non_blocking=True).view(3, 1, 1)
            clip_std = torch.Tensor([0.26862954, 0.26130258, 0.27577711])
            clip_std = clip_std.to(tokens, non_blocking=True).view(3, 1, 1)
            dinov2_mean = (
                torch.Tensor([0.485, 0.456, 0.406])
                .to(tokens, non_blocking=True)
                .view(3, 1, 1)
            )
            dinov2_std = (
                torch.Tensor([0.229, 0.224, 0.225])
                .to(tokens, non_blocking=True)
                .view(3, 1, 1)
            )

            local_convnext_image_feats = self.openclip_convnext_xxl(
                tokens * (dinov2_std / clip_std) + (dinov2_mean - clip_mean) / clip_std
            )  # [b, 3072, 8, 8]
            assert local_convnext_image_feats.size()[1:] == (3072, 8, 8)
            local_convnext_image_feats = self.avgpool(local_convnext_image_feats).view(
                tokens.size(0), -1
            )
            x_0 = self.convnext_proj(local_convnext_image_feats)

        x_min = self.model_min(x_minmin)
        x_max = self.model_max(x_maxmax)
        x_min1 = self.model_min(x_minmin1)
        x_max1 = self.model_max(x_maxmax1)

        x_1 = (x_min + x_max + x_min1 + x_max1) / 4

        x = torch.cat([x_0, x_1], dim=1)

        x = self.fc(x)

        # 确保输出是二维的 [batch_size, num_classes]
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        # 确保输出是 float32 类型
        x = x.to(torch.float32)

        return ClassificationOutputs(outputs=x)  # '0_real', '1_fake'


# https://laion.ai/notes/realfake/
# "real": 0, "fake": 1
class RealFakeProcessor(HfImageClassificationProcessor):
    def __init__(self, img_resize: int = 256, img_crop: int = 224, train: bool = False):
        self.train = train
        self.img_resize = img_resize
        self.img_crop = img_crop
        self.transform = self._get_augs(train)

    def _get_augs(self, train: bool = True) -> A.Compose:
        if train:
            return A.Compose(
                [
                    A.Resize(self.img_resize, self.img_resize),
                    A.RandomCrop(self.img_crop, self.img_crop),
                    A.HorizontalFlip(),
                    A.VerticalFlip(),
                    A.RandomBrightnessContrast(),
                    A.Affine(),
                    A.Rotate(),
                    A.CoarseDropout(),
                    ExpandChannels(),
                    RGBAtoRGB(),
                    A.Normalize(),
                    ToTensorV2(),
                ]
            )
        else:
            return A.Compose(
                [
                    A.Resize(self.img_resize, self.img_resize),
                    A.CenterCrop(self.img_crop, self.img_crop),
                    ExpandChannels(),
                    RGBAtoRGB(),
                    A.Normalize(),
                    ToTensorV2(),
                ]
            )

    @classmethod
    @add_default_section_for_init("microsoft/omnipixel/process/realfake")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/omnipixel/process/realfake")
        img_resize = config.getoption("img_resize", 256)
        img_crop = config.getoption("img_crop", 224)
        train = config.getoption("train", False)
        return cls(img_resize=img_resize, img_crop=img_crop, train=train)

    @register_process("microsoft/omnipixel/process/realfake/classification")
    def _classification(
        self,
        image: Union[str, Image.Image],
    ):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        # 转换为numpy数组
        image = np.array(image)

        # 应用albumentations增强
        transformed = self.transform(image=image)
        image = transformed["image"]

        return TensorsInputs(image=image)


class ExpandChannels(A.ImageOnlyTransform):
    """Expands image up to three channels if the image is grayscale."""

    def __init__(self, always_apply: bool = False, p: float = 0.5):
        super().__init__(True, 1.0)

    def apply(self, image, **params):
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)
        elif image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        return image


class RGBAtoRGB(A.ImageOnlyTransform):
    """Converts RGBA image to RGB."""

    def __init__(self, always_apply: bool = False, p: float = 0.5):
        super().__init__(True, 1.0)

    def apply(self, image, **params):
        if image.shape[2] == 4:
            image = image[:, :, :3]
        return image


@register_model("microsoft/omnipixel/model/realfake")
class RealFakeModel(GenericModel):
    def __init__(
        self,
        model_name: str = "convnext_large",
        num_classes: int = 2,
        pretrained: bool = True,
    ):
        super().__init__()

        # 使用timm创建基础模型
        self.model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=num_classes
        )

    @classmethod
    @add_default_section_for_init("microsoft/omnipixel/model/realfake")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/omnipixel/model/realfake")

        model_name = config.getoption("model_name", "convnext_tiny")
        num_classes = config.getoption("num_classes", 2)
        pretrained = config.getoption("pretrained", True)

        inst = cls(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
        )

        weight_path = config.getoption("pretrained_weight_path", None)
        if weight_path is not None:
            inst.from_pretrained(weight_path)

        return inst

    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(
        self,
        image: torch.Tensor,
    ):
        if isinstance(image, TensorsInputs):
            image = image.image

        # 前向传播
        outputs = self.model(image)

        # 确保输出是float32类型
        outputs = outputs.to(torch.float32)

        return ClassificationOutputs(outputs=outputs)  # "real": 0, "fake": 1
