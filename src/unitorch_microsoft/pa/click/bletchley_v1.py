# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import logging
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torch import autocast
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


# from unitorch-microsoft/src/unitorch_microsoft/pa/bletchley_v1.py model is microsoft/pa/pretrain/bletchley/v1/v2
@register_model("microsoft/pa/pretrain/bletchley/v1/3tower")
class Bletchley3TowerForPretrainV2(GenericModel):
    def __init__(
        self,
        query_config_type,
        offer_config_type,
        projection_dim: int = 512,
        freeze_offer_model: Optional[bool] = False,
        logit_scale_init_value: Optional[float] = 2.6592,
        gradient_checkpointing: Optional[bool] = False,
        use_all_gather: Optional[bool] = True,
        output_query_embed: Optional[bool] = False,
        output_offer_embed: Optional[bool] = False,
        output_image_embed: Optional[bool] = False,
    ):
        super().__init__()

        query_config = get_bletchley_text_config(
            query_config_type, gradient_checkpointing
        )
        offer_config = get_bletchley_text_config(
            offer_config_type, gradient_checkpointing
        )
        image_config = get_bletchley_image_config(
            offer_config_type, gradient_checkpointing
        )

        self.projection_dim = projection_dim
        self.query_embed_dim = query_config.hidden_size
        self.offer_embed_dim = offer_config.hidden_size
        self.image_embed_dim = image_config.hidden_size
        self.use_all_gather = use_all_gather
        self.output_query_embed = output_query_embed
        self.output_offer_embed = output_offer_embed
        self.output_image_embed = output_image_embed

        self.query_encoder = BletchleyTextEncoder(
            query_config, add_projection_layer=False
        )
        self.offer_encoder = BletchleyTextEncoder(
            offer_config, add_projection_layer=False
        )
        self.image_encoder = BletchleyImageEncoder(
            image_config, add_projection_layer=False
        )

        self.query_projection = nn.Linear(self.query_embed_dim, self.projection_dim)
        self.offer_projection = nn.Linear(
            self.offer_embed_dim,
            self.projection_dim,
        )
        self.image_projection = nn.Linear(
            self.image_embed_dim,
            self.projection_dim,
        )

        self.logit_scale_query = nn.Parameter(torch.ones([]) * logit_scale_init_value)
        self.logit_scale_image = nn.Parameter(torch.ones([]) * logit_scale_init_value)
        self.logit_scale_offer = nn.Parameter(torch.ones([]) * logit_scale_init_value)

        self.init_weights()

        if freeze_offer_model:
            for p in self.offer_encoder.parameters():
                p.requires_grad = False

            for p in self.image_encoder.parameters():
                p.requires_grad = False

    @classmethod
    @add_default_section_for_init("microsoft/pa/pretrain/bletchley/v1/3tower")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/pa/pretrain/bletchley/v1/3tower")
        query_config_type = config.getoption("query_config_type", "0.3B")
        offer_config_type = config.getoption("offer_config_type", "0.3B")
        projection_dim = config.getoption("projection_dim", 512)
        freeze_offer_model = config.getoption("freeze_offer_model", False)
        logit_scale_init_value = config.getoption("logit_scale_init_value", 2.6592)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)
        use_all_gather = config.getoption("use_all_gather", True)
        output_query_embed = config.getoption("output_query_embed", False)
        output_offer_embed = config.getoption("output_offer_embed", False)
        output_image_embed = config.getoption("output_image_embed", False)

        inst = cls(
            query_config_type=query_config_type,
            offer_config_type=offer_config_type,
            projection_dim=projection_dim,
            freeze_offer_model=freeze_offer_model,
            logit_scale_init_value=logit_scale_init_value,
            gradient_checkpointing=gradient_checkpointing,
            use_all_gather=use_all_gather,
            output_query_embed=output_query_embed,
            output_offer_embed=output_offer_embed,
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

    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(
        self,
        query_input_ids: torch.Tensor = None,
        query_attention_mask: torch.Tensor = None,
        images: torch.Tensor = None,
        offer_input_ids: torch.Tensor = None,
        offer_attention_mask: torch.Tensor = None,
    ):
        if not self.training and self.output_query_embed:
            query_outputs = self.query_encoder(query_input_ids, query_attention_mask)
            query_embeds = self.query_projection(query_outputs[:, 0])
            query_embeds = query_embeds / query_embeds.norm(dim=-1, keepdim=True)
            return EmbeddingOutputs(embedding=query_embeds)

        if not self.training and self.output_offer_embed:
            offer_outputs = self.offer_encoder(offer_input_ids, offer_attention_mask)
            offer_embeds = self.offer_projection(offer_outputs[:, 0])
            offer_embeds = offer_embeds / offer_embeds.norm(dim=-1, keepdim=True)
            return EmbeddingOutputs(embedding=offer_embeds)

        if not self.training and self.output_image_embed:
            image_outputs = self.image_encoder(images)
            image_embeds = self.image_projection(image_outputs[:, 0])
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            return EmbeddingOutputs(embedding=image_embeds)

        query_outputs = self.query_encoder(query_input_ids, query_attention_mask)
        query_embeds = self.query_projection(query_outputs[:, 0])

        offer_outputs = self.offer_encoder(offer_input_ids, offer_attention_mask)
        offer_embeds = self.offer_projection(offer_outputs[:, 0])

        image_outputs = self.image_encoder(images)
        image_embeds = self.image_projection(image_outputs[:, 0])

        query_embeds = query_embeds / query_embeds.norm(dim=-1, keepdim=True)
        offer_embeds = offer_embeds / offer_embeds.norm(dim=-1, keepdim=True)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

        logit_scale_query = self.logit_scale_query.exp()
        logit_scale_image = self.logit_scale_image.exp()
        logit_scale_offer = self.logit_scale_offer.exp()
        if self.use_all_gather and dist.is_initialized():
            query_embeds = self.all_gather(query_embeds)
            offer_embeds = self.all_gather(offer_embeds)
            image_embeds = self.all_gather(image_embeds)
        logits_per_query = (
            torch.matmul(query_embeds, image_embeds.t()) * logit_scale_query
        )
        logits_per_image = (
            torch.matmul(image_embeds, offer_embeds.t()) * logit_scale_image
        )
        logits_per_offer = (
            torch.matmul(offer_embeds, query_embeds.t()) * logit_scale_offer
        )

        loss = (
            _clip_loss(logits_per_query)
            + _clip_loss(logits_per_image)
            + _clip_loss(logits_per_offer)
        )
        loss = loss / 3.0
        return LossOutputs(loss=loss)
