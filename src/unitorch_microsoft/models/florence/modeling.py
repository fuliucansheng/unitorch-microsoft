# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torch.cuda.amp import autocast

from transformers import PreTrainedModel, PretrainedConfig
from unitorch.models import GenericModel
from unitorch.models.clip.modeling import AllGather, _clip_loss
from unitorch.cli import (
    add_default_section_for_init,
    add_default_section_for_function,
    register_model,
)
from unitorch.cli.models import EmbeddingOutputs, ClassificationOutputs, LossOutputs
from unitorch_microsoft.models.florence.transformer import Transformer
from unitorch_microsoft.models.florence.davit import DaViT


def get_florence_config(
    config_type: str,
    gradient_checkpointing: Optional[bool] = False,
):
    config_types = {
        "davit-d3-224": OrderedDict(
            max_seq_length=77,
            vocab_size=49408,
            width=512,
            layers=12,
            heads=8,
            is_autogressive=False,
            depths=[1, 1, 9, 1],
            embed_dims=[128, 256, 512, 1024],
            num_heads=[4, 8, 16, 32],
            num_groups=[4, 8, 16, 32],
            patch_size=[7, 2, 2, 2],
            patch_stride=[4, 2, 2, 2],
            patch_padding=[3, 0, 0, 0],
            patch_prenorm=[False, True, True, True],
            drop_path_rate=0.0,
            image_size=224,
            window_size=7,
            conv_at_attn=True,
            conv_at_ffn=True,
            text_embed_dim=512,
            image_embed_dim=1024,
        ),
        "davit-d5-224": OrderedDict(
            max_seq_length=77,
            vocab_size=49408,
            width=1024,
            layers=16,
            heads=16,
            is_autogressive=False,
            depths=[1, 1, 9, 1],
            embed_dims=[256, 512, 1024, 2048],
            num_heads=[8, 16, 32, 64],
            num_groups=[8, 16, 32, 64],
            patch_size=[7, 3, 3, 3],
            patch_stride=[4, 2, 2, 2],
            patch_padding=[3, 1, 1, 1],
            patch_prenorm=[False, True, True, True],
            drop_path_rate=0.1,
            image_size=224,
            window_size=7,
            conv_at_attn=True,
            conv_at_ffn=True,
            text_embed_dim=1024,
            image_embed_dim=2048,
        ),
        "davit-d7-384": OrderedDict(
            max_seq_length=77,
            vocab_size=49408,
            width=1024,
            layers=24,
            heads=16,
            is_autogressive=False,
            depths=[1, 1, 12, 3],
            embed_dims=[384, 768, 1536, 3072],
            num_heads=[12, 24, 48, 96],
            num_groups=[12, 24, 48, 96],
            patch_size=[7, 3, 3, 3],
            patch_stride=[4, 2, 2, 2],
            patch_padding=[3, 1, 1, 1],
            patch_prenorm=[False, True, True, True],
            drop_path_rate=0.1,
            image_size=384,
            window_size=12,
            conv_at_attn=True,
            conv_at_ffn=True,
            text_embed_dim=1024,
            image_embed_dim=3072,
        ),
        "davit-d9-448": OrderedDict(
            max_seq_length=77,
            vocab_size=49408,
            width=1024,
            layers=24,
            heads=16,
            is_autogressive=False,
            depths=[1, 1, 16, 1],
            embed_dims=[768, 1536, 3072, 6144],
            num_heads=[24, 48, 96, 192],
            num_groups=[24, 48, 96, 192],
            patch_size=[7, 3, 3, 3],
            patch_stride=[4, 2, 2, 2],
            patch_padding=[3, 1, 1, 1],
            patch_prenorm=[False, True, True, True],
            drop_path_rate=0.2,
            image_size=448,
            window_size=14,
            conv_at_attn=True,
            conv_at_ffn=False,
            text_embed_dim=1024,
            image_embed_dim=6144,
        ),
    }

    assert config_type in config_types.keys(), "Invalid config passed"

    params = config_types.get(config_type)
    config = PretrainedConfig()
    for k, v in params.items():
        setattr(config, k, v)
    config.gradient_checkpointing = gradient_checkpointing
    return config


def get_florence_state_dict_from_checkpoint(weight_path):
    image_replace_keys = {
        "conv_embeds": "convs",
        "main_blocks": "blocks",
        "0.cpe.0.proj": "spatial_block.conv1.fn.dw",
        "0.attn": "spatial_block.window_attn.fn",
        "0.cpe.1.proj": "spatial_block.conv2.fn.dw",
        "0.mlp": "spatial_block.ffn.fn.net",
        "1.cpe.0.proj": "channel_block.conv1.fn.dw",
        "1.attn": "channel_block.channel_attn.fn",
        "1.cpe.1.proj": "channel_block.conv2.fn.dw",
        "1.mlp": "channel_block.ffn.fn.net",
        "0.norm1": "spatial_block.window_attn.norm",
        "0.norm2": "spatial_block.ffn.norm",
        "1.norm1": "channel_block.channel_attn.norm",
        "1.norm2": "channel_block.ffn.norm",
    }
    replace_keys = {
        "lang_projection": "text_projection.weight",
        "image_projection": "vision_projection.weight",
    }
    state_dict = torch.load(weight_path, map_location="cpu")

    old_keys = []
    new_keys = []
    for key in state_dict.keys():
        new_key = None
        for rkey, nkey in replace_keys.items():
            if rkey not in key:
                continue
            if new_key is None:
                new_key = key.replace(rkey, nkey)
            else:
                new_key = new_key.replace(rkey, nkey)

        if "image_encoder" in key:
            for rkey, nkey in image_replace_keys.items():
                if rkey not in key:
                    continue
                if new_key is None:
                    new_key = key.replace(rkey, nkey)
                else:
                    new_key = new_key.replace(rkey, nkey)

        if new_key:
            old_keys.append(key)
            new_keys.append(new_key)

    for old_key, new_key in zip(old_keys, new_keys):
        if old_key in ["lang_projection", "image_projection"]:
            state_dict[new_key] = state_dict.pop(old_key).transpose(0, 1)
        else:
            state_dict[new_key] = state_dict.pop(old_key)

    return state_dict


@register_model("microsoft/model/classification/florence")
class FlorenceForClassification(GenericModel):
    def __init__(
        self,
        config_type: str,
        projection_dim: Optional[int] = 512,
        num_classes: Optional[int] = 1,
        freeze_base_model: Optional[bool] = True,
        gradient_checkpointing: Optional[bool] = False,
    ):
        super().__init__()
        config = get_florence_config(config_type, gradient_checkpointing)
        self.lang_encoder = Transformer(
            context_length=config.max_seq_length,
            vocab_size=config.vocab_size,
            width=config.width,
            layers=config.layers,
            heads=config.heads,
            autogressive=config.is_autogressive,
            enable_checkpoint=config.gradient_checkpointing,
        )
        self.image_encoder = DaViT(
            num_classes=0,
            depths=config.depths,
            embed_dims=config.embed_dims,
            num_heads=config.num_heads,
            num_groups=config.num_groups,
            patch_size=config.patch_size,
            patch_stride=config.patch_stride,
            patch_padding=config.patch_padding,
            patch_prenorm=config.patch_prenorm,
            drop_path_rate=config.drop_path_rate,
            img_size=config.image_size,
            window_size=config.window_size,
            enable_checkpoint=config.gradient_checkpointing,
            conv_at_attn=config.conv_at_attn,
            conv_at_ffn=config.conv_at_ffn,
        )

        self.projection_dim = projection_dim
        self.text_embed_dim = config.text_embed_dim
        self.image_embed_dim = config.image_embed_dim
        self.text_projection = nn.Linear(
            self.text_embed_dim,
            self.projection_dim,
            bias=False,
        )
        self.vision_projection = nn.Linear(
            self.image_embed_dim,
            self.projection_dim,
            bias=False,
        )

        self.classifier = nn.Linear(self.projection_dim * 2, num_classes)
        self.init_weights()

        if freeze_base_model:
            for p in self.lang_encoder.parameters():
                p.requires_grad = False

            for p in self.image_encoder.parameters():
                p.requires_grad = False

    @classmethod
    @add_default_section_for_init("microsoft/model/classification/florence")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/model/classification/florence")
        config_type = config.getoption("config_type", "davit-d3-224")
        projection_dim = config.getoption("projection_dim", 512)
        num_classes = config.getoption("num_classes", 1)
        freeze_base_model = config.getoption("freeze_base_model", True)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)

        inst = cls(
            config_type=config_type,
            projection_dim=projection_dim,
            num_classes=num_classes,
            freeze_base_model=freeze_base_model,
            gradient_checkpointing=gradient_checkpointing,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)

        if pretrained_weight_path is not None:
            state_dict = get_florence_state_dict_from_checkpoint(pretrained_weight_path)
            inst.from_pretrained(state_dict=state_dict)

        return inst

    @autocast()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ):
        text_embeds = self.lang_encoder(input_ids, attention_mask)
        text_embeds = self.text_projection(text_embeds)

        image_embeds = self.image_encoder(pixel_values)
        image_embeds = self.vision_projection(image_embeds)
        outputs = self.classifier(
            F.relu(torch.cat([image_embeds, text_embeds], axis=1))
        )
        return ClassificationOutputs(outputs=outputs)


@register_model("microsoft/model/matching/florence")
class FlorenceForMatching(GenericModel):
    def __init__(
        self,
        config_type: str,
        projection_dim: Optional[int] = 512,
        freeze_base_model: Optional[bool] = True,
        gradient_checkpointing: Optional[bool] = False,
        output_text_embed: Optional[bool] = False,
        output_image_embed: Optional[bool] = False,
    ):
        super().__init__()
        config = get_florence_config(config_type, gradient_checkpointing)

        self.output_text_embed = output_text_embed
        self.output_image_embed = output_image_embed

        self.lang_encoder = Transformer(
            context_length=config.max_seq_length,
            vocab_size=config.vocab_size,
            width=config.width,
            layers=config.layers,
            heads=config.heads,
            autogressive=config.is_autogressive,
            enable_checkpoint=config.gradient_checkpointing,
        )
        self.image_encoder = DaViT(
            num_classes=0,
            depths=config.depths,
            embed_dims=config.embed_dims,
            num_heads=config.num_heads,
            num_groups=config.num_groups,
            patch_size=config.patch_size,
            patch_stride=config.patch_stride,
            patch_padding=config.patch_padding,
            patch_prenorm=config.patch_prenorm,
            drop_path_rate=config.drop_path_rate,
            img_size=config.image_size,
            window_size=config.window_size,
            enable_checkpoint=config.gradient_checkpointing,
            conv_at_attn=config.conv_at_attn,
            conv_at_ffn=config.conv_at_ffn,
        )

        self.projection_dim = projection_dim
        self.text_embed_dim = config.text_embed_dim
        self.image_embed_dim = config.image_embed_dim
        self.text_projection = nn.Linear(
            self.text_embed_dim,
            self.projection_dim,
            bias=False,
        )
        self.vision_projection = nn.Linear(
            self.image_embed_dim,
            self.projection_dim,
            bias=False,
        )

        self.classifier = nn.Linear(1, 1)
        self.init_weights()
        self.classifier.weight.data.fill_(5.0)

        if freeze_base_model:
            for p in self.lang_encoder.parameters():
                p.requires_grad = False

            for p in self.image_encoder.parameters():
                p.requires_grad = False

    @classmethod
    @add_default_section_for_init("microsoft/model/matching/florence")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/model/matching/florence")
        config_type = config.getoption("config_type", "davit-d3-224")
        projection_dim = config.getoption("projection_dim", 512)
        freeze_base_model = config.getoption("freeze_base_model", True)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)

        output_text_embed = config.getoption("output_text_embed", False)
        output_image_embed = config.getoption("output_image_embed", False)

        inst = cls(
            config_type=config_type,
            projection_dim=projection_dim,
            freeze_base_model=freeze_base_model,
            gradient_checkpointing=gradient_checkpointing,
            output_text_embed=output_text_embed,
            output_image_embed=output_image_embed,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)

        if pretrained_weight_path is not None:
            state_dict = get_florence_state_dict_from_checkpoint(pretrained_weight_path)
            inst.from_pretrained(state_dict=state_dict)

        return inst

    @autocast()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ):
        if not self.training and self.output_text_embed:
            text_embeds = self.lang_encoder(input_ids, attention_mask)
            text_embeds = self.text_projection(text_embeds)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            return EmbeddingOutputs(embedding=text_embeds)

        if not self.training and self.output_image_embed:
            image_embeds = self.image_encoder(pixel_values)
            image_embeds = self.vision_projection(image_embeds)
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            return EmbeddingOutputs(embedding=image_embeds)

        text_embeds = self.lang_encoder(input_ids, attention_mask)
        text_embeds = self.text_projection(text_embeds)

        image_embeds = self.image_encoder(pixel_values)
        image_embeds = self.vision_projection(image_embeds)

        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        scores = torch.sum(text_embeds * image_embeds, dim=-1, keepdim=True)

        outputs = self.classifier(scores)
        return ClassificationOutputs(outputs=outputs)


@register_model("microsoft/model/pretrain/florence")
class FlorenceForPretrain(GenericModel):
    def __init__(
        self,
        config_type: str,
        projection_dim: Optional[int] = 512,
        freeze_base_model: Optional[bool] = True,
        logit_scale_init_value: Optional[float] = 2.6592,
        gradient_checkpointing: Optional[bool] = False,
        use_all_gather: Optional[bool] = True,
    ):
        super().__init__()
        config = get_florence_config(config_type, gradient_checkpointing)

        # self.output_text_embed = output_text_embed
        # self.output_image_embed = output_image_embed

        self.lang_encoder = Transformer(
            context_length=config.max_seq_length,
            vocab_size=config.vocab_size,
            width=config.width,
            layers=config.layers,
            heads=config.heads,
            autogressive=config.is_autogressive,
            enable_checkpoint=config.gradient_checkpointing,
        )
        self.image_encoder = DaViT(
            num_classes=0,
            depths=config.depths,
            embed_dims=config.embed_dims,
            num_heads=config.num_heads,
            num_groups=config.num_groups,
            patch_size=config.patch_size,
            patch_stride=config.patch_stride,
            patch_padding=config.patch_padding,
            patch_prenorm=config.patch_prenorm,
            drop_path_rate=config.drop_path_rate,
            img_size=config.image_size,
            window_size=config.window_size,
            enable_checkpoint=config.gradient_checkpointing,
            conv_at_attn=config.conv_at_attn,
            conv_at_ffn=config.conv_at_ffn,
        )

        self.projection_dim = projection_dim
        self.text_embed_dim = config.text_embed_dim
        self.image_embed_dim = config.image_embed_dim
        self.use_all_gather = use_all_gather
        self.text_projection = nn.Linear(
            self.text_embed_dim,
            self.projection_dim,
        )
        self.vision_projection = nn.Linear(
            self.image_embed_dim,
            self.projection_dim,
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init_value)

        self.init_weights()

        if freeze_base_model:
            for p in self.lang_encoder.parameters():
                p.requires_grad = False

            for p in self.image_encoder.parameters():
                p.requires_grad = False

    @classmethod
    @add_default_section_for_init("microsoft/model/pretrain/florence")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/model/pretrain/florence")
        config_type = config.getoption("config_type", "davit-d3-224")
        projection_dim = config.getoption("projection_dim", 512)
        freeze_base_model = config.getoption("freeze_base_model", True)
        logit_scale_init_value = config.getoption("logit_scale_init_value", 2.6592)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)
        use_all_gather = config.getoption("use_all_gather", True)

        inst = cls(
            config_type=config_type,
            projection_dim=projection_dim,
            freeze_base_model=freeze_base_model,
            logit_scale_init_value=logit_scale_init_value,
            gradient_checkpointing=gradient_checkpointing,
            use_all_gather=use_all_gather,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        if pretrained_weight_path is not None:
            state_dict = get_florence_state_dict_from_checkpoint(pretrained_weight_path)
            inst.from_pretrained(state_dict=state_dict)

        return inst

    def all_gather(self, input):
        output = AllGather.apply(input)
        output = output.view(-1, *(output.shape[2:]))
        return output

    @autocast()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ):
        text_embeds = self.lang_encoder(input_ids, attention_mask)
        text_embeds = self.text_projection(text_embeds)

        image_embeds = self.image_encoder(pixel_values)
        image_embeds = self.vision_projection(image_embeds)

        # text_outputs = self.lang_encoder(input_ids, attention_mask)
        # text_embeds = self.text_projection(text_outputs[:, 0])

        # image_outputs = self.image_encoder(pixel_values)
        # image_embeds = self.vision_projection(image_outputs[:, 0])

        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        if self.use_all_gather:
            text_embeds = self.all_gather(text_embeds)
            image_embeds = self.all_gather(image_embeds)
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.T

        loss = _clip_loss(logits_per_text)
        return LossOutputs(loss=loss)
