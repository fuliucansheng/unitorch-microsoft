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
from unitorch_microsoft.models.bletchley.modeling_v3 import (
    BletchleyTextEncoder,
    BletchleyImageEncoder,
)


@register_model("microsoft/pa/l2/classification/mmdnn/bletchley/v3/v2")
class MMDNNBletchleyForClassificationV2(GenericModel):
    def __init__(
        self,
        query_config_type: str,
        offer_config_type: str,
        projection_dim: Optional[int] = 288,
        output_hidden_dim: Optional[int] = 64,
        freeze_base_model: Optional[bool] = True,
        gradient_checkpointing: Optional[bool] = False,
        output_query_embed: Optional[bool] = False,
        output_offer_embed: Optional[bool] = False,
        output_image_embed: Optional[bool] = False,
        output_final_query_embed: Optional[bool] = False,
        output_final_offer_embed: Optional[bool] = False,
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

        self.output_query_embed = output_query_embed
        self.output_offer_embed = output_offer_embed
        self.output_image_embed = output_image_embed
        self.output_final_query_embed = output_final_query_embed
        self.output_final_offer_embed = output_final_offer_embed

        self.query_projection = nn.Linear(self.query_embed_dim, self.projection_dim)
        self.offer_projection = nn.Linear(self.offer_embed_dim, self.projection_dim)
        self.image_projection = nn.Linear(self.image_embed_dim, self.projection_dim)

        self.query_layer_norm = nn.LayerNorm(projection_dim)
        self.offer_layer_norm = nn.LayerNorm(projection_dim)
        self.image_layer_norm = nn.LayerNorm(projection_dim)

        self.final_offer_projection = nn.Linear(
            projection_dim + projection_dim,
            output_hidden_dim,
        )
        self.final_query_projection = nn.Linear(
            projection_dim,
            output_hidden_dim,
        )

        self.classifier = nn.Linear(1, 1)

        self.init_weights()
        self.classifier.weight.data.fill_(5.0)

        if freeze_base_model:
            for p in self.query_encoder.parameters():
                p.requires_grad = False

            for p in self.offer_encoder.parameters():
                p.requires_grad = False

            for p in self.image_encoder.parameters():
                p.requires_grad = False

    @classmethod
    @add_default_section_for_init(
        "microsoft/pa/l2/classification/mmdnn/bletchley/v3/v2"
    )
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section(
            "microsoft/pa/l2/classification/mmdnn/bletchley/v3/v2"
        )
        query_config_type = config.getoption("query_config_type", "0.3B")
        offer_config_type = config.getoption("offer_config_type", "0.3B")
        projection_dim = config.getoption("projection_dim", 288)
        output_hidden_dim = config.getoption("output_hidden_dim", 64)
        freeze_base_model = config.getoption("freeze_base_model", True)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)
        output_query_embed = config.getoption("output_query_embed", False)
        output_offer_embed = config.getoption("output_offer_embed", False)
        output_image_embed = config.getoption("output_image_embed", False)
        output_final_query_embed = config.getoption("output_final_query_embed", False)
        output_final_offer_embed = config.getoption("output_final_offer_embed", False)

        inst = cls(
            query_config_type=query_config_type,
            offer_config_type=offer_config_type,
            projection_dim=projection_dim,
            output_hidden_dim=output_hidden_dim,
            freeze_base_model=freeze_base_model,
            gradient_checkpointing=gradient_checkpointing,
            output_query_embed=output_query_embed,
            output_offer_embed=output_offer_embed,
            output_image_embed=output_image_embed,
            output_final_query_embed=output_final_query_embed,
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
        _keys = [key for key in state_dict.keys() if key.startswith("text_encoder")]
        for _key in _keys:
            _value = state_dict.pop(_key)
            state_dict["query_encoder" + _key[12:]] = _value
            state_dict["offer_encoder" + _key[12:]] = _value

        super().from_pretrained(state_dict=state_dict)

    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(
        self,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        images: torch.Tensor,
        offer_input_ids: torch.Tensor,
        offer_attention_mask: torch.Tensor,
    ):
        if not self.training and self.output_query_embed:
            query_outputs = self.query_encoder(
                input_ids=query_input_ids, attention_mask=query_attention_mask
            )
            query_embeds = query_outputs[:, 0]
            query_embeds = self.query_projection(query_embeds)
            query_embeds = self.query_layer_norm(quick_gelu(query_embeds))

            return EmbeddingOutputs(embedding=query_embeds)

        if not self.training and self.output_offer_embed:
            offer_outputs = self.offer_encoder(
                input_ids=offer_input_ids, attention_mask=offer_attention_mask
            )
            offer_embeds = offer_outputs[:, 0]
            offer_embeds = self.offer_projection(offer_embeds)
            offer_embeds = self.offer_layer_norm(quick_gelu(offer_embeds))

            return EmbeddingOutputs(embedding=offer_embeds)

        if not self.training and self.output_final_query_embed:
            query_outputs = self.query_encoder(
                input_ids=query_input_ids, attention_mask=query_attention_mask
            )
            query_embeds = query_outputs[:, 0]
            query_embeds = self.query_projection(query_embeds)
            query_embeds = self.query_layer_norm(quick_gelu(query_embeds))

            query_embeds = self.final_query_projection(quick_gelu(query_embeds))
            query_embeds = query_embeds / query_embeds.norm(dim=-1, keepdim=True)
            return EmbeddingOutputs(embedding=query_embeds)

        if not self.training and self.output_final_offer_embed:
            offer_outputs = self.offer_encoder(
                input_ids=offer_input_ids, attention_mask=offer_attention_mask
            )
            offer_embeds = offer_outputs[:, 0]
            offer_embeds = self.offer_projection(offer_embeds)
            offer_embeds = self.offer_layer_norm(quick_gelu(offer_embeds))

            image_outputs = self.image_encoder(images=images)
            image_embeds = image_outputs[:, 0]
            image_embeds = self.image_projection(image_embeds)
            image_embeds = self.image_layer_norm(quick_gelu(image_embeds))

            offer_embeds = torch.cat(
                [image_embeds, offer_embeds],
                dim=-1,
            )

            offer_embeds = self.final_offer_projection(quick_gelu(offer_embeds))
            offer_embeds = offer_embeds / offer_embeds.norm(dim=-1, keepdim=True)
            return EmbeddingOutputs(embedding=offer_embeds)

        query_outputs = self.query_encoder(
            input_ids=query_input_ids, attention_mask=query_attention_mask
        )

        query_embeds = query_outputs[:, 0]
        query_embeds = self.query_projection(query_embeds)
        query_embeds = self.query_layer_norm(quick_gelu(query_embeds))

        query_embeds = self.final_query_projection(quick_gelu(query_embeds))
        query_embeds = query_embeds / query_embeds.norm(dim=-1, keepdim=True)

        offer_outputs = self.offer_encoder(
            input_ids=offer_input_ids, attention_mask=offer_attention_mask
        )
        offer_embeds = offer_outputs[:, 0]
        offer_embeds = self.offer_projection(offer_embeds)
        offer_embeds = self.offer_layer_norm(quick_gelu(offer_embeds))

        image_outputs = self.image_encoder(images=images)
        image_embeds = image_outputs[:, 0]
        image_embeds = self.image_projection(image_embeds)
        image_embeds = self.image_layer_norm(quick_gelu(image_embeds))

        offer_embeds = torch.cat(
            [image_embeds, offer_embeds],
            dim=-1,
        )

        offer_embeds = self.final_offer_projection(quick_gelu(offer_embeds))
        offer_embeds = offer_embeds / offer_embeds.norm(dim=-1, keepdim=True)

        scores = torch.sum(query_embeds * offer_embeds, dim=-1, keepdim=True)

        outputs = self.classifier(scores)
        outputs = torch.sigmoid(outputs)

        return ClassificationOutputs(outputs=outputs)


@register_model("microsoft/pa/l2/classification/mmdnn/bletchley/v3/v2/text")
class MMDNNBletchleyTextForClassificationV2(GenericModel):
    def __init__(
        self,
        query_config_type: str,
        offer_config_type: str,
        projection_dim: Optional[int] = 288,
        output_hidden_dim: Optional[int] = 64,
        freeze_base_model: Optional[bool] = True,
        gradient_checkpointing: Optional[bool] = False,
        output_query_embed: Optional[bool] = False,
        output_offer_embed: Optional[bool] = False,
        output_final_query_embed: Optional[bool] = False,
        output_final_offer_embed: Optional[bool] = False,
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

        self.output_query_embed = output_query_embed
        self.output_offer_embed = output_offer_embed
        self.output_final_query_embed = output_final_query_embed
        self.output_final_offer_embed = output_final_offer_embed

        self.query_projection = nn.Linear(self.query_embed_dim, self.projection_dim)
        self.offer_projection = nn.Linear(self.offer_embed_dim, self.projection_dim)

        self.query_layer_norm = nn.LayerNorm(projection_dim)
        self.offer_layer_norm = nn.LayerNorm(projection_dim)

        self.final_offer_projection = nn.Linear(
            projection_dim,
            output_hidden_dim,
        )
        self.final_query_projection = nn.Linear(
            projection_dim,
            output_hidden_dim,
        )

        self.classifier = nn.Linear(1, 1)

        self.init_weights()
        self.classifier.weight.data.fill_(5.0)

        if freeze_base_model:
            for p in self.query_encoder.parameters():
                p.requires_grad = False

            for p in self.offer_encoder.parameters():
                p.requires_grad = False

    @classmethod
    @add_default_section_for_init(
        "microsoft/pa/l2/classification/mmdnn/bletchley/v3/v2/text"
    )
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section(
            "microsoft/pa/l2/classification/mmdnn/bletchley/v3/v2/text"
        )
        query_config_type = config.getoption("query_config_type", "0.3B")
        offer_config_type = config.getoption("offer_config_type", "0.3B")
        projection_dim = config.getoption("projection_dim", 288)
        output_hidden_dim = config.getoption("output_hidden_dim", 64)
        freeze_base_model = config.getoption("freeze_base_model", True)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)
        output_query_embed = config.getoption("output_query_embed", False)
        output_offer_embed = config.getoption("output_offer_embed", False)
        output_final_query_embed = config.getoption("output_final_query_embed", False)
        output_final_offer_embed = config.getoption("output_final_offer_embed", False)

        inst = cls(
            query_config_type=query_config_type,
            offer_config_type=offer_config_type,
            projection_dim=projection_dim,
            output_hidden_dim=output_hidden_dim,
            freeze_base_model=freeze_base_model,
            gradient_checkpointing=gradient_checkpointing,
            output_query_embed=output_query_embed,
            output_offer_embed=output_offer_embed,
            output_final_query_embed=output_final_query_embed,
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
        _keys = [key for key in state_dict.keys() if key.startswith("text_encoder")]
        for _key in _keys:
            _value = state_dict.pop(_key)
            state_dict["query_encoder" + _key[12:]] = _value
            state_dict["offer_encoder" + _key[12:]] = _value

        super().from_pretrained(state_dict=state_dict)

    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(
        self,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        offer_input_ids: torch.Tensor,
        offer_attention_mask: torch.Tensor,
    ):
        if not self.training and self.output_query_embed:
            query_outputs = self.query_encoder(
                input_ids=query_input_ids, attention_mask=query_attention_mask
            )
            query_embeds = query_outputs[:, 0]
            query_embeds = self.query_projection(query_embeds)
            query_embeds = self.query_layer_norm(quick_gelu(query_embeds))

            return EmbeddingOutputs(embedding=query_embeds)

        if not self.training and self.output_offer_embed:
            offer_outputs = self.offer_encoder(
                input_ids=offer_input_ids, attention_mask=offer_attention_mask
            )
            offer_embeds = offer_outputs[:, 0]
            offer_embeds = self.offer_projection(offer_embeds)
            offer_embeds = self.offer_layer_norm(quick_gelu(offer_embeds))

            return EmbeddingOutputs(embedding=offer_embeds)

        if not self.training and self.output_final_query_embed:
            query_outputs = self.query_encoder(
                input_ids=query_input_ids, attention_mask=query_attention_mask
            )
            query_embeds = query_outputs[:, 0]
            query_embeds = self.query_projection(query_embeds)
            query_embeds = self.query_layer_norm(quick_gelu(query_embeds))

            query_embeds = self.final_query_projection(quick_gelu(query_embeds))
            query_embeds = query_embeds / query_embeds.norm(dim=-1, keepdim=True)
            return EmbeddingOutputs(embedding=query_embeds)

        if not self.training and self.output_final_offer_embed:
            offer_outputs = self.offer_encoder(
                input_ids=offer_input_ids, attention_mask=offer_attention_mask
            )
            offer_embeds = offer_outputs[:, 0]
            offer_embeds = self.offer_projection(offer_embeds)
            offer_embeds = self.offer_layer_norm(quick_gelu(offer_embeds))

            offer_embeds = self.final_offer_projection(quick_gelu(offer_embeds))
            offer_embeds = offer_embeds / offer_embeds.norm(dim=-1, keepdim=True)
            return EmbeddingOutputs(embedding=offer_embeds)

        query_outputs = self.query_encoder(
            input_ids=query_input_ids, attention_mask=query_attention_mask
        )

        query_embeds = query_outputs[:, 0]
        query_embeds = self.query_projection(query_embeds)
        query_embeds = self.query_layer_norm(quick_gelu(query_embeds))

        query_embeds = self.final_query_projection(quick_gelu(query_embeds))
        query_embeds = query_embeds / query_embeds.norm(dim=-1, keepdim=True)

        offer_outputs = self.offer_encoder(
            input_ids=offer_input_ids, attention_mask=offer_attention_mask
        )
        offer_embeds = offer_outputs[:, 0]
        offer_embeds = self.offer_projection(offer_embeds)
        offer_embeds = self.offer_layer_norm(quick_gelu(offer_embeds))

        offer_embeds = self.final_offer_projection(quick_gelu(offer_embeds))
        offer_embeds = offer_embeds / offer_embeds.norm(dim=-1, keepdim=True)

        scores = torch.sum(query_embeds * offer_embeds, dim=-1, keepdim=True)

        outputs = self.classifier(scores)
        outputs = torch.sigmoid(outputs)

        return ClassificationOutputs(outputs=outputs)
