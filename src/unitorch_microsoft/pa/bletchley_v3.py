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
from unitorch_microsoft.models.bletchley.modeling_v3 import (
    BletchleyTextEncoder,
    BletchleyImageEncoder,
)


@register_model("microsoft/pa/pretrain/bletchley/v3")
class BletchleyForPretrain(GenericModel):
    replace_keys_in_state_dict = {
        "text_encoder.projection": "text_projection",
        "image_encoder.projection": "image_projection",
        "^query_encoder.": "text_encoder.",
        "query_projection": "text_projection",
    }

    def __init__(
        self,
        config_type,
        projection_dim: int = 1024,
        freeze_text_model: Optional[bool] = False,
        logit_scale_init_value: Optional[float] = 2.6592,
        gradient_checkpointing: Optional[bool] = False,
        use_all_gather: Optional[bool] = True,
    ):
        super().__init__()
        self.text_encoder = BletchleyTextEncoder(
            config_type,
            add_projection_layer=False,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.image_encoder = BletchleyImageEncoder(
            config_type,
            add_projection_layer=False,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.projection_dim = projection_dim
        self.text_embed_dim = self.text_encoder.hidden_size
        self.image_embed_dim = self.image_encoder.hidden_size

        self.use_all_gather = use_all_gather
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

        if freeze_text_model:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

    @classmethod
    @add_default_section_for_init("microsoft/pa/pretrain/bletchley/v3")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/pa/pretrain/bletchley/v3")
        config_type = config.getoption("config_type", "0.3B")
        projection_dim = config.getoption("projection_dim", 1024)
        freeze_text_model = config.getoption("freeze_text_model", False)
        logit_scale_init_value = config.getoption("logit_scale_init_value", 2.6592)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)
        use_all_gather = config.getoption("use_all_gather", True)

        inst = cls(
            config_type=config_type,
            projection_dim=projection_dim,
            freeze_text_model=freeze_text_model,
            logit_scale_init_value=logit_scale_init_value,
            gradient_checkpointing=gradient_checkpointing,
            use_all_gather=use_all_gather,
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
        input_ids,
        attention_mask,
        images,
    ):
        text_outputs = self.text_encoder(input_ids, attention_mask)
        text_embeds = self.text_projection(text_outputs[:, 0])

        image_outputs = self.image_encoder(images)
        image_embeds = self.image_projection(image_outputs[:, 0])

        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        if self.use_all_gather and dist.is_initialized():
            text_embeds = self.all_gather(text_embeds)
            image_embeds = self.all_gather(image_embeds)
        logits = torch.matmul(text_embeds, image_embeds.t()) * logit_scale

        loss = _clip_loss(logits)

        return LossOutputs(loss=loss)


@register_model("microsoft/pa/pretrain/bletchley/v3/v2")
class BletchleyForPretrainV2(GenericModel):
    def __init__(
        self,
        query_config_type,
        offer_config_type,
        projection_dim: int = 1024,
        freeze_offer_model: Optional[bool] = False,
        freeze_text_model: Optional[bool] = False,
        logit_scale_init_value: Optional[float] = 2.6592,
        gradient_checkpointing: Optional[bool] = False,
        use_all_gather: Optional[bool] = True,
    ):
        super().__init__()
        self.query_encoder = BletchleyTextEncoder(
            query_config_type,
            add_projection_layer=False,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.offer_encoder = BletchleyTextEncoder(
            offer_config_type,
            add_projection_layer=False,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.image_encoder = BletchleyImageEncoder(
            offer_config_type,
            add_projection_layer=False,
            gradient_checkpointing=gradient_checkpointing,
        )

        self.projection_dim = projection_dim
        self.query_embed_dim = self.query_encoder.hidden_size
        self.offer_embed_dim = self.offer_encoder.hidden_size
        self.image_embed_dim = self.image_encoder.hidden_size
        self.use_all_gather = use_all_gather
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

        if freeze_text_model:
            for p in self.query_encoder.parameters():
                p.requires_grad = False

            for p in self.offer_encoder.parameters():
                p.requires_grad = False

    @classmethod
    @add_default_section_for_init("microsoft/pa/pretrain/bletchley/v3/v2")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/pa/pretrain/bletchley/v3/v2")
        query_config_type = config.getoption("query_config_type", "0.3B")
        offer_config_type = config.getoption("offer_config_type", "0.8B")
        projection_dim = config.getoption("projection_dim", 1024)
        freeze_offer_model = config.getoption("freeze_offer_model", False)
        freeze_text_model = config.getoption("freeze_text_model", False)
        logit_scale_init_value = config.getoption("logit_scale_init_value", 2.6592)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)
        use_all_gather = config.getoption("use_all_gather", True)

        inst = cls(
            query_config_type=query_config_type,
            offer_config_type=offer_config_type,
            projection_dim=projection_dim,
            freeze_offer_model=freeze_offer_model,
            freeze_text_model=freeze_text_model,
            logit_scale_init_value=logit_scale_init_value,
            gradient_checkpointing=gradient_checkpointing,
            use_all_gather=use_all_gather,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        if pretrained_weight_path is not None:
            inst.from_pretrained(pretrained_weight_path)

        return inst

    def from_pretrained(self, weight_path):
        if not os.path.exists(weight_path):
            return
        state_dict = torch.load(weight_path, map_location="cpu")
        _keys = [key for key in state_dict.keys() if key.startswith("text_encoder")]
        for _key in _keys:
            _value = state_dict.pop(_key)
            state_dict["query_encoder" + _key[12:]] = _value
            state_dict["offer_encoder" + _key[12:]] = _value

        super().from_pretrained(state_dict=state_dict)

    def all_gather(self, input):
        output = AllGather.apply(input)
        output = output.view(-1, *(output.shape[2:]))
        return output

    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(
        self,
        query_input_ids,
        query_attention_mask,
        images,
        offer_input_ids,
        offer_attention_mask,
    ):
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


@register_model("microsoft/pa/pretrain/bletchley/v3/v2/text")
class BletchleyTextForPretrainV2(GenericModel):
    def __init__(
        self,
        query_config_type,
        offer_config_type,
        projection_dim: int = 1024,
        freeze_offer_model: Optional[bool] = False,
        logit_scale_init_value: Optional[float] = 2.6592,
        gradient_checkpointing: Optional[bool] = False,
        use_all_gather: Optional[bool] = True,
    ):
        super().__init__()
        self.query_encoder = BletchleyTextEncoder(
            query_config_type,
            add_projection_layer=False,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.offer_encoder = BletchleyTextEncoder(
            offer_config_type,
            add_projection_layer=False,
            gradient_checkpointing=gradient_checkpointing,
        )

        self.projection_dim = projection_dim
        self.query_embed_dim = self.query_encoder.hidden_size
        self.offer_embed_dim = self.offer_encoder.hidden_size
        self.use_all_gather = use_all_gather
        self.query_projection = nn.Linear(self.query_embed_dim, self.projection_dim)
        self.offer_projection = nn.Linear(
            self.offer_embed_dim,
            self.projection_dim,
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init_value)

        self.init_weights()

        if freeze_offer_model:
            for p in self.offer_encoder.parameters():
                p.requires_grad = False

    @classmethod
    @add_default_section_for_init("microsoft/pa/pretrain/bletchley/v3/v2/text")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/pa/pretrain/bletchley/v3/v2/text")
        query_config_type = config.getoption("query_config_type", "0.3B")
        offer_config_type = config.getoption("offer_config_type", "0.8B")
        projection_dim = config.getoption("projection_dim", 1024)
        freeze_offer_model = config.getoption("freeze_offer_model", False)
        logit_scale_init_value = config.getoption("logit_scale_init_value", 2.6592)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)
        use_all_gather = config.getoption("use_all_gather", True)

        inst = cls(
            query_config_type=query_config_type,
            offer_config_type=offer_config_type,
            projection_dim=projection_dim,
            freeze_offer_model=freeze_offer_model,
            logit_scale_init_value=logit_scale_init_value,
            gradient_checkpointing=gradient_checkpointing,
            use_all_gather=use_all_gather,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        if pretrained_weight_path is not None:
            inst.from_pretrained(pretrained_weight_path)

        return inst

    def from_pretrained(self, weight_path):
        if not os.path.exists(weight_path):
            return
        state_dict = torch.load(weight_path, map_location="cpu")
        _keys = [key for key in state_dict.keys() if key.startswith("text_encoder")]
        for _key in _keys:
            _value = state_dict.pop(_key)
            state_dict["query_encoder" + _key[12:]] = _value
            state_dict["offer_encoder" + _key[12:]] = _value

        super().from_pretrained(state_dict=state_dict)

    def all_gather(self, input):
        output = AllGather.apply(input)
        output = output.view(-1, *(output.shape[2:]))
        return output

    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(
        self,
        query_input_ids,
        query_attention_mask,
        offer_input_ids,
        offer_attention_mask,
    ):
        query_outputs = self.query_encoder(query_input_ids, query_attention_mask)
        query_embeds = self.query_projection(query_outputs[:, 0])

        offer_outputs = self.offer_encoder(offer_input_ids, offer_attention_mask)
        offer_embeds = self.offer_projection(offer_outputs[:, 0])

        query_embeds = query_embeds / query_embeds.norm(dim=-1, keepdim=True)
        offer_embeds = offer_embeds / offer_embeds.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        if self.use_all_gather and dist.is_initialized():
            query_embeds = self.all_gather(query_embeds)
            offer_embeds = self.all_gather(offer_embeds)
        logits = torch.matmul(offer_embeds, query_embeds.t()) * logit_scale

        loss = _clip_loss(logits)

        return LossOutputs(loss=loss)
