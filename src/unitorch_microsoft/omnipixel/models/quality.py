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
                "https://unitorchazureblob.blob.core.windows.net/shares/models/adsplus/diffusion/pytorch_model.siglip.msra.bin",
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
                "https://unitorchazureblob.blob.core.windows.net/shares/models/adsplus/diffusion/pytorch_model.laion_clip.sac.logos.ava1.l14.msra.bin",
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

@register_model("microsoft/omnipixel/model/laion_clipV2/image")
class LAIONClipV2ForImageClassification(GenericModel):
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
    @add_default_section_for_init("microsoft/omnipixel/model/laion_clipV2/image")
    def from_core_configure(cls, config, **kwargs):
        """
        Create an instance of ClipForImageClassification from a core configuration.

        Args:
            config: The core configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            ClipForImageClassification: An instance of the ClipForImageClassification model.
        """
        config.set_default_section("microsoft/omnipixel/model/laion_clipV2/image")
        pretrained_name = config.getoption("pretrained_name", "clip-vit-large-patch14")
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
                "https://unitorchazureblob.blob.core.windows.net/shares/models/adsplus/diffusion/LaionV2.sac.logos.ava1-l14-linearMSE.bin"
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
from unitorch.models import GenericOutputs, HfTextClassificationProcessor
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
