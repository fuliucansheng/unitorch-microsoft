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
from unitorch.cli import (
    add_default_section_for_init,
    add_default_section_for_function,
    register_model,
)
from unitorch.cli.models import (
    EmbeddingOutputs,
    LossOutputs,
    ClassificationOutputs,
)
from unitorch_microsoft.models.bletchley.modeling_v3 import (
    BletchleyTextEncoder,
    BletchleyImageEncoder,
)


@register_model("microsoft/vpr/pretrain/bletchley/v3/argus")
class BletchleyForPretrain(GenericModel):
    def __init__(
        self,
        config_type,
        image_embed_dim: Optional[int] = 100,
        projection_dim: Optional[int] = 64,
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

        self.projection_dim = projection_dim
        self.image_embed_dim = image_embed_dim
        self.text_embed_dim = self.text_encoder.hidden_size
        self.use_all_gather = use_all_gather
        self.image_projection = nn.Linear(self.image_embed_dim, self.projection_dim)
        self.text_projection = nn.Linear(
            self.text_embed_dim,
            self.projection_dim,
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init_value)

        self.init_weights()

    @classmethod
    @add_default_section_for_init("microsoft/vpr/pretrain/bletchley/v3/argus")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/vpr/pretrain/bletchley/v3/argus")
        config_type = config.getoption("config_type", "0.3B")
        image_embed_dim = config.getoption("image_embed_dim", 100)
        projection_dim = config.getoption("projection_dim", 64)
        logit_scale_init_value = config.getoption("logit_scale_init_value", 2.6592)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)
        use_all_gather = config.getoption("use_all_gather", True)

        inst = cls(
            config_type=config_type,
            image_embed_dim=image_embed_dim,
            projection_dim=projection_dim,
            logit_scale_init_value=logit_scale_init_value,
            gradient_checkpointing=gradient_checkpointing,
            use_all_gather=use_all_gather,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        if pretrained_weight_path is not None:
            inst.from_pretrained(
                pretrained_weight_path,
                replace_keys={"^offer_encoder": "text_encoder"},
            )

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
        image_embeds,
    ):
        text_outputs = self.text_encoder(input_ids, attention_mask)
        text_embeds = self.text_projection(text_outputs[:, 0])
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        image_embeds = self.image_projection(image_embeds)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        if self.use_all_gather and dist.is_initialized():
            text_embeds = self.all_gather(text_embeds)
            image_embeds = self.all_gather(image_embeds)
        logits = torch.matmul(text_embeds, image_embeds.t()) * logit_scale

        loss = _clip_loss(logits)

        return LossOutputs(loss=loss)
