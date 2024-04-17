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
from transformers.activations import quick_gelu

from unitorch.models import GenericModel
from unitorch.models.clip.modeling import AllGather, _clip_loss
from unitorch.cli.models import (
    EmbeddingOutputs,
    LossOutputs,
    ClassificationOutputs,
)
from unitorch.cli import (
    add_default_section_for_init,
    add_default_section_for_function,
    register_model,
)

from unitorch_microsoft.models.bletchley.modeling_v1 import (
    get_bletchley_text_config,
    get_bletchley_image_config,
    BletchleyTextEncoder,
    BletchleyImageEncoder,
)


@register_model("microsoft/adsplus/image/retrieval/pretrain/bletchley/v1")
class BletchleyForImageRetrievalPretrain(GenericModel):
    replace_keys_in_state_dict = {
        "text_encoder.projection": "text_projection",
        "image_encoder.projection": "image_projection",
    }

    def __init__(
        self,
        text_config_type: str,
        image_config_type: str,
        projection_dim: Optional[int] = 64,
        freeze_base_model: Optional[bool] = False,
        freeze_text_base_model: Optional[bool] = False,
        freeze_image_base_model: Optional[bool] = False,
        freeze_all_image_model: Optional[bool] = False,
        logit_scale_init_value: Optional[float] = 2.6592,
        gradient_checkpointing: Optional[bool] = False,
        use_all_gather: Optional[bool] = True,
        output_text_embed: Optional[bool] = False,
        output_image_embed: Optional[bool] = False,
    ):
        super().__init__()
        text_config = get_bletchley_text_config(
            text_config_type, gradient_checkpointing
        )
        image_config = get_bletchley_image_config(
            image_config_type, gradient_checkpointing
        )
        self.text_encoder = BletchleyTextEncoder(
            text_config, add_projection_layer=False
        )
        self.image_encoder = BletchleyImageEncoder(
            image_config, add_projection_layer=False
        )

        self.projection_dim = projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.image_embed_dim = image_config.hidden_size

        self.use_all_gather = use_all_gather
        self.output_text_embed = output_text_embed
        self.output_image_embed = output_image_embed

        self.text_projection = nn.Linear(
            self.text_embed_dim,
            self.projection_dim,
        )
        self.image_projection = nn.Linear(
            self.image_embed_dim,
            self.projection_dim,
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init_value)

        self.init_weights()

        # freeze model parameters
        if freeze_base_model:
            for p in self.text_encoder.parameters():
                p.requires_grad = False
            for p in self.image_encoder.parameters():
                p.requires_grad = False

        if freeze_text_base_model:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        if freeze_image_base_model:
            for p in self.image_encoder.parameters():
                p.requires_grad = False

        if freeze_all_image_model:
            for p in self.image_encoder.parameters():
                p.requires_grad = False
            for p in self.image_projection.parameters():
                p.requires_grad = False

    @classmethod
    @add_default_section_for_init(
        "microsoft/adsplus/image/retrieval/pretrain/bletchley/v1"
    )
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section(
            "microsoft/adsplus/image/retrieval/pretrain/bletchley/v1"
        )
        text_config_type = config.getoption("text_config_type", "0.8B")
        image_config_type = config.getoption("image_config_type", "0.8B")
        projection_dim = config.getoption("projection_dim", 64)
        freeze_base_model = config.getoption("freeze_base_model", False)
        freeze_text_base_model = config.getoption("freeze_text_base_model", False)
        freeze_image_base_model = config.getoption("freeze_image_base_model", False)
        freeze_all_image_model = config.getoption("freeze_all_image_model", False)
        logit_scale_init_value = config.getoption("logit_scale_init_value", 2.6592)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)
        use_all_gather = config.getoption("use_all_gather", True)
        output_text_embed = config.getoption("output_text_embed", False)
        output_image_embed = config.getoption("output_image_embed", False)

        inst = cls(
            text_config_type=text_config_type,
            image_config_type=image_config_type,
            projection_dim=projection_dim,
            freeze_base_model=freeze_base_model,
            freeze_text_base_model=freeze_text_base_model,
            freeze_image_base_model=freeze_image_base_model,
            freeze_all_image_model=freeze_all_image_model,
            logit_scale_init_value=logit_scale_init_value,
            gradient_checkpointing=gradient_checkpointing,
            use_all_gather=use_all_gather,
            output_text_embed=output_text_embed,
            output_image_embed=output_image_embed,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        if pretrained_weight_path is not None:
            inst.from_pretrained(pretrained_weight_path)

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
        images: torch.Tensor,
    ):
        if not self.training and self.output_image_embed:
            image_outputs = self.image_encoder(images)
            image_embeds = self.image_projection(image_outputs[:, 0])
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            return EmbeddingOutputs(embedding=image_embeds)

        if not self.training and self.output_text_embed:
            text_outputs = self.text_encoder(input_ids, attention_mask)
            text_embeds = self.text_projection(text_outputs[:, 0])
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            return EmbeddingOutputs(embedding=text_embeds)

        text_outputs = self.text_encoder(input_ids, attention_mask)
        text_embeds = self.text_projection(text_outputs[:, 0])

        image_outputs = self.image_encoder(images)
        image_embeds = self.image_projection(image_outputs[:, 0])

        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        if self.use_all_gather and dist.is_initialized():
            text_embeds = self.all_gather(text_embeds)
            image_embeds = self.all_gather(image_embeds)
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.T

        loss = _clip_loss(logits_per_text)
        return LossOutputs(loss=loss)


@register_model("microsoft/adsplus/image/retrieval/matching/bletchley/v1")
class BletchleyForImageRetrievalMatching(GenericModel):
    replace_keys_in_state_dict = {
        "text_encoder.projection": "text_projection",
        "image_encoder.projection": "image_projection",
    }

    def __init__(
        self,
        text_config_type: str,
        image_config_type: str,
        projection_dim: Optional[int] = 64,
        freeze_base_model: Optional[bool] = False,
        freeze_text_base_model: Optional[bool] = False,
        freeze_image_base_model: Optional[bool] = False,
        freeze_all_image_model: Optional[bool] = False,
        gradient_checkpointing: Optional[bool] = False,
        use_all_gather: Optional[bool] = True,
        output_text_embed: Optional[bool] = False,
        output_image_embed: Optional[bool] = False,
    ):
        super().__init__()
        text_config = get_bletchley_text_config(
            text_config_type, gradient_checkpointing
        )
        image_config = get_bletchley_image_config(
            image_config_type, gradient_checkpointing
        )

        self.output_text_embed = output_text_embed
        self.output_image_embed = output_image_embed

        self.text_embed_dim = text_config.hidden_size
        self.image_embed_dim = image_config.hidden_size

        self.text_encoder = BletchleyTextEncoder(
            text_config, add_projection_layer=False
        )
        self.image_encoder = BletchleyImageEncoder(
            image_config, add_projection_layer=False
        )

        self.text_projection = nn.Linear(
            self.text_embed_dim,
            projection_dim,
        )
        self.image_projection = nn.Linear(
            self.image_embed_dim,
            projection_dim,
        )

        self.classifier = nn.Linear(1, 1)

        self.init_weights()
        self.classifier.weight.data.fill_(5.0)

        if freeze_base_model:
            for p in self.text_encoder.parameters():
                p.requires_grad = False
            for p in self.image_encoder.parameters():
                p.requires_grad = False

        if freeze_text_base_model:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        if freeze_image_base_model:
            for p in self.image_encoder.parameters():
                p.requires_grad = False

        if freeze_all_image_model:
            for p in self.image_encoder.parameters():
                p.requires_grad = False
            for p in self.image_projection.parameters():
                p.requires_grad = False

    @classmethod
    @add_default_section_for_init(
        "microsoft/adsplus/image/retrieval/matching/bletchley/v1"
    )
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section(
            "microsoft/adsplus/image/retrieval/matching/bletchley/v1"
        )

        text_config_type = config.getoption("text_config_type", "0.8B")
        image_config_type = config.getoption("image_config_type", "0.8B")
        projection_dim = config.getoption("projection_dim", 64)
        freeze_base_model = config.getoption("freeze_base_model", False)
        freeze_text_base_model = config.getoption("freeze_text_base_model", False)
        freeze_image_base_model = config.getoption("freeze_image_base_model", False)
        freeze_all_image_model = config.getoption("freeze_all_image_model", False)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)
        output_text_embed = config.getoption("output_text_embed", False)
        output_image_embed = config.getoption("output_image_embed", False)

        inst = cls(
            text_config_type=text_config_type,
            image_config_type=image_config_type,
            projection_dim=projection_dim,
            freeze_base_model=freeze_base_model,
            freeze_text_base_model=freeze_text_base_model,
            freeze_image_base_model=freeze_image_base_model,
            freeze_all_image_model=freeze_all_image_model,
            gradient_checkpointing=gradient_checkpointing,
            output_text_embed=output_text_embed,
            output_image_embed=output_image_embed,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        if pretrained_weight_path is not None:
            inst.from_pretrained(pretrained_weight_path)

        return inst

    @autocast()
    def forward(
        self,
        input_ids: torch.Tensor = None,
        images: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
    ):
        if not self.training and self.output_image_embed:
            image_outputs = self.image_encoder(images=images)
            image_embeds = image_outputs[:, 0]
            image_embeds = self.image_projection(image_embeds)
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            return EmbeddingOutputs(embedding=image_embeds)

        if not self.training and self.output_text_embed:
            text_outputs = self.text_encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )
            text_embeds = text_outputs[:, 0]
            text_embeds = self.text_projection(text_embeds)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            return EmbeddingOutputs(embedding=text_embeds)

        image_outputs = self.image_encoder(images=images)
        text_outputs = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )

        image_embeds = image_outputs[:, 0]
        image_embeds = self.image_projection(image_embeds)
        text_embeds = text_outputs[:, 0]
        text_embeds = self.text_projection(text_embeds)

        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        scores = torch.sum(text_embeds * image_embeds, dim=-1, keepdim=True)

        outputs = self.classifier(scores)
        return ClassificationOutputs(outputs=outputs)
