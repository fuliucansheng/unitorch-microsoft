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
from unitorch_microsoft.models.bletchley.modeling_v1 import (
    get_bletchley_text_config,
    get_bletchley_image_config,
    BletchleyTextEncoder,
    BletchleyImageEncoder,
)


@register_model("microsoft/vpr/pretrain/bletchley/v1/argus")
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
        config = get_bletchley_text_config(
            config_type,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.text_encoder = BletchleyTextEncoder(
            config,
            add_projection_layer=False,
        )

        self.projection_dim = projection_dim
        self.image_embed_dim = image_embed_dim
        self.text_embed_dim = config.hidden_size
        self.use_all_gather = use_all_gather
        self.image_projection = nn.Linear(self.image_embed_dim, self.projection_dim)
        self.text_projection = nn.Linear(
            self.text_embed_dim,
            self.projection_dim,
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init_value)

        self.init_weights()

    @classmethod
    @add_default_section_for_init("microsoft/vpr/pretrain/bletchley/v1/argus")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/vpr/pretrain/bletchley/v1/argus")
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


@register_model("microsoft/vpr/classification/bletchley/v1/argus")
class BletchleyForClassification(GenericModel):
    def __init__(
        self,
        config_type,
        image_embed_dim: Optional[int] = 100,
        projection_dim: Optional[int] = 64,
        gradient_checkpointing: Optional[bool] = False,
        output_query_embed: Optional[bool] = False,
        output_final_query_embed: Optional[bool] = False,
        output_offer_embed: Optional[bool] = False,
        output_final_offer_embed: Optional[bool] = False,
    ):
        super().__init__()
        config = get_bletchley_text_config(
            config_type,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.projection_dim = projection_dim
        self.image_embed_dim = image_embed_dim
        self.text_embed_dim = config.hidden_size

        self.output_query_embed = output_query_embed
        self.output_final_query_embed = output_final_query_embed
        self.output_offer_embed = output_offer_embed
        self.output_final_offer_embed = output_final_offer_embed

        self.query_text_encoder = BletchleyTextEncoder(
            config,
            add_projection_layer=False,
        )
        self.query_image_projection = nn.Linear(
            self.image_embed_dim, self.projection_dim
        )
        self.query_text_projection = nn.Linear(
            self.text_embed_dim,
            self.projection_dim,
        )
        self.final_query_projection = nn.Linear(
            self.projection_dim * 2,
            self.projection_dim,
        )

        self.offer_text_encoder = BletchleyTextEncoder(
            config,
            add_projection_layer=False,
        )
        self.offer_image_projection = nn.Linear(
            self.image_embed_dim, self.projection_dim
        )
        self.offer_text_projection = nn.Linear(
            self.text_embed_dim,
            self.projection_dim,
        )
        self.final_offer_projection = nn.Linear(
            self.projection_dim * 2,
            self.projection_dim,
        )

        self.classifier = nn.Linear(1, 1)

        self.init_weights()
        self.classifier.weight.data.fill_(5.0)

    @classmethod
    @add_default_section_for_init("microsoft/vpr/classification/bletchley/v1/argus")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/vpr/classification/bletchley/v1/argus")
        config_type = config.getoption("config_type", "0.3B")
        image_embed_dim = config.getoption("image_embed_dim", 100)
        projection_dim = config.getoption("projection_dim", 64)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)
        output_query_embed = config.getoption("output_query_embed", False)
        output_final_query_embed = config.getoption("output_final_query_embed", False)
        output_offer_embed = config.getoption("output_offer_embed", False)
        output_final_offer_embed = config.getoption("output_final_offer_embed", False)

        inst = cls(
            config_type=config_type,
            image_embed_dim=image_embed_dim,
            projection_dim=projection_dim,
            gradient_checkpointing=gradient_checkpointing,
            output_query_embed=output_query_embed,
            output_final_query_embed=output_final_query_embed,
            output_offer_embed=output_offer_embed,
            output_final_offer_embed=output_final_offer_embed,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        if pretrained_weight_path is not None:
            inst.from_pretrained(pretrained_weight_path)

        return inst

    def from_pretrained(self, weight_path):
        if not os.path.exists(weight_path):
            return
        state_dict = torch.load(weight_path, map_location="cpu")
        _keys = [
            key
            for key in state_dict.keys()
            if key.startswith("text_") or key.startswith("image_")
        ]
        for _key in _keys:
            _value = state_dict.pop(_key)
            state_dict["query_" + _key] = _value
            state_dict["offer_" + _key] = _value

        super().from_pretrained(state_dict=state_dict)

    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(
        self,
        query_input_ids=None,
        query_attention_mask=None,
        query_image_embeds=None,
        offer_input_ids=None,
        offer_attention_mask=None,
        offer_image_embeds=None,
    ):
        if self.output_query_embed:
            query_text_outputs = self.query_text_encoder(
                query_input_ids, query_attention_mask
            )
            query_text_embeds = self.query_text_projection(query_text_outputs[:, 0])
            # query_text_embeds = query_text_embeds / query_text_embeds.norm(dim=-1, keepdim=True)
            query_image_embeds = self.query_image_projection(query_image_embeds)
            # query_image_embeds = query_image_embeds / query_image_embeds.norm(dim=-1, keepdim=True)
            query_emb = torch.cat([query_text_embeds, query_image_embeds], dim=-1)
            return EmbeddingOutputs(embedding=query_emb)

        if self.output_final_query_embed:
            query_text_outputs = self.query_text_encoder(
                query_input_ids, query_attention_mask
            )
            query_text_embeds = self.query_text_projection(query_text_outputs[:, 0])
            # query_text_embeds = query_text_embeds / query_text_embeds.norm(dim=-1, keepdim=True)
            query_image_embeds = self.query_image_projection(query_image_embeds)
            # query_image_embeds = query_image_embeds / query_image_embeds.norm(dim=-1, keepdim=True)
            query_emb = torch.cat([query_text_embeds, query_image_embeds], dim=-1)
            query_emb = self.final_query_projection(query_emb)
            query_emb = query_emb / query_emb.norm(dim=-1, keepdim=True)
            return EmbeddingOutputs(embedding=query_emb)

        if self.output_offer_embed:
            offer_text_outputs = self.offer_text_encoder(
                offer_input_ids, offer_attention_mask
            )
            offer_text_embeds = self.offer_text_projection(offer_text_outputs[:, 0])
            # offer_text_embeds = offer_text_embeds / offer_text_embeds.norm(dim=-1, keepdim=True)
            offer_image_embeds = self.offer_image_projection(offer_image_embeds)
            # offer_image_embeds = offer_image_embeds / offer_image_embeds.norm(dim=-1, keepdim=True)
            offer_emb = torch.cat([offer_text_embeds, offer_image_embeds], dim=-1)
            return EmbeddingOutputs(embedding=offer_emb)

        if self.output_final_offer_embed:
            offer_text_outputs = self.offer_text_encoder(
                offer_input_ids, offer_attention_mask
            )
            offer_text_embeds = self.offer_text_projection(offer_text_outputs[:, 0])
            # offer_text_embeds = offer_text_embeds / offer_text_embeds.norm(dim=-1, keepdim=True)
            offer_image_embeds = self.offer_image_projection(offer_image_embeds)
            # offer_image_embeds = offer_image_embeds / offer_image_embeds.norm(dim=-1, keepdim=True)
            offer_emb = torch.cat([offer_text_embeds, offer_image_embeds], dim=-1)
            offer_emb = self.final_offer_projection(offer_emb)
            offer_emb = offer_emb / offer_emb.norm(dim=-1, keepdim=True)
            return EmbeddingOutputs(embedding=offer_emb)

        query_text_outputs = self.query_text_encoder(
            query_input_ids, query_attention_mask
        )
        query_text_embeds = self.query_text_projection(query_text_outputs[:, 0])
        # query_text_embeds = query_text_embeds / query_text_embeds.norm(dim=-1, keepdim=True)
        query_image_embeds = self.query_image_projection(query_image_embeds)
        # query_image_embeds = query_image_embeds / query_image_embeds.norm(dim=-1, keepdim=True)
        query_emb = torch.cat([query_text_embeds, query_image_embeds], dim=-1)
        query_emb = self.final_query_projection(query_emb)
        query_emb = query_emb / query_emb.norm(dim=-1, keepdim=True)

        offer_text_outputs = self.offer_text_encoder(
            offer_input_ids, offer_attention_mask
        )
        offer_text_embeds = self.offer_text_projection(offer_text_outputs[:, 0])
        # offer_text_embeds = offer_text_embeds / offer_text_embeds.norm(dim=-1, keepdim=True)
        offer_image_embeds = self.offer_image_projection(offer_image_embeds)
        # offer_image_embeds = offer_image_embeds / offer_image_embeds.norm(dim=-1, keepdim=True)
        offer_emb = torch.cat([offer_text_embeds, offer_image_embeds], dim=-1)
        offer_emb = self.final_offer_projection(offer_emb)
        offer_emb = offer_emb / offer_emb.norm(dim=-1, keepdim=True)

        scores = torch.sum(query_emb * offer_emb, dim=-1, keepdim=True)

        outputs = self.classifier(scores)
        # outputs = torch.sigmoid(outputs)

        return ClassificationOutputs(outputs=outputs)
