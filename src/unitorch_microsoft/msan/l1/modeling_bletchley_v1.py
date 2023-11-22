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
from torch.cuda.amp import autocast
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
from unitorch_microsoft import cached_path

@register_model("microsoft/msan/l1/pretrain/bletchley/v1")
class BletchleyForPretrain(GenericModel):
    def __init__(
        self,
        config_type,
        projection_dim: int = 32,
        logit_scale_init_value: Optional[float] = 2.6592,
        gradient_checkpointing: Optional[bool] = False,
        use_all_gather: Optional[bool] = True,
    ):
        super().__init__()

        text_config = get_bletchley_text_config(config_type, gradient_checkpointing)

        self.projection_dim = projection_dim
        self.text_embed_dim = text_config.hidden_size

        self.user_encoder = BletchleyTextEncoder(
            text_config, add_projection_layer=False
        )
        self.ads_encoder = BletchleyTextEncoder(
            text_config, add_projection_layer=False
        )

        self.use_all_gather = use_all_gather
        self.user_projection = nn.Linear(
            self.text_embed_dim,
            self.projection_dim,
        )
        self.attn = nn.Linear(text_config.hidden_size, 1, bias=False)
        self.ads_projection = nn.Linear(
            self.text_embed_dim,
            self.projection_dim,
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init_value)

        self.init_weights()

    @classmethod
    @add_default_section_for_init("microsoft/msan/l1/pretrain/bletchley/v1")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/msan/l1/pretrain/bletchley/v1")
        config_type = config.getoption("config_type", "0.15B")
        projection_dim = config.getoption("projection_dim", 32)
        logit_scale_init_value = config.getoption("logit_scale_init_value", 2.6592)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)
        use_all_gather = config.getoption("use_all_gather", True)

        inst = cls(
            config_type=config_type,
            projection_dim=projection_dim,
            logit_scale_init_value=logit_scale_init_value,
            gradient_checkpointing=gradient_checkpointing,
            use_all_gather=use_all_gather,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        if pretrained_weight_path is not None:
            pretrained_weight_path = cached_path(pretrained_weight_path)
            inst.from_pretrained(pretrained_weight_path)

        return inst

    def from_pretrained(self, weight_path):
        if not os.path.exists(weight_path):
            return
        state_dict = torch.load(weight_path, map_location="cpu")
        _keys = [key for key in state_dict.keys() if key.startswith("text_encoder")]
        for _key in _keys:
            _value = state_dict.pop(_key)
            state_dict["user_encoder" + _key[12:]] = _value
            state_dict["ads_encoder" + _key[12:]] = _value

        super().from_pretrained(state_dict=state_dict)

    def all_gather(self, input):
        output = AllGather.apply(input)
        output = output.view(-1, *(output.shape[2:]))
        return output

    @autocast()
    def forward(
        self,
        user_input_ids,
        user_attention_mask,
        user_num_attention_mask,
        ads_input_ids,
        ads_attention_mask,
    ):
        batch, num, seq_len = user_input_ids.shape
        user_outputs = self.user_encoder(user_input_ids.view(-1, seq_len), user_attention_mask.view(-1, seq_len))
        pooled_output = user_outputs[:, 0]
        attention_score = self.attn(pooled_output).view(batch, num)
        attention_score = attention_score + (1 - user_num_attention_mask) * -10000.0
        attention_score = F.softmax(attention_score, dim=-1)
        user_attn_outputs = torch.bmm(
            attention_score.unsqueeze(1), pooled_output.view(batch, num, -1)
        ).squeeze(1)
        user_embeds = self.user_projection(user_attn_outputs)

        ads_outputs = self.ads_encoder(ads_input_ids, ads_attention_mask)
        ads_embeds = self.ads_projection(ads_outputs[:, 0])

        user_embeds = user_embeds / user_embeds.norm(dim=-1, keepdim=True)
        ads_embeds = ads_embeds / ads_embeds.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        if self.use_all_gather and dist.is_initialized():
            user_embeds = self.all_gather(user_embeds)
            ads_embeds = self.all_gather(ads_embeds)
        logits = torch.matmul(user_embeds, ads_embeds.t()) * logit_scale

        loss = _clip_loss(logits)

        return LossOutputs(loss=loss)
