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


@register_model("microsoft/adsplus/slab/bletchley/v1/ranking")
class BletchleyForSLABRanking(GenericModel):
    replace_keys_in_state_dict = {
        "text_encoder.projection": "text_projection",
        "image_encoder.projection": "image_projection",
    }

    def __init__(
        self,
        config_type: str,
        projection_dim: Optional[int] = 64,
        num_query_layers: Optional[int] = 4,
        num_positions: Optional[int] = None,
        num_types: Optional[int] = None,
        num_classes: Optional[int] = 1,
        freeze_base_model: Optional[bool] = False,
        enable_quantization: Optional[bool] = False,
        gradient_checkpointing: Optional[bool] = False,
        output_query_embed: Optional[bool] = False,
        output_text_embed: Optional[bool] = False,
    ):
        super().__init__()
        query_config = get_bletchley_text_config(config_type, gradient_checkpointing)
        query_config.num_hidden_layers = num_query_layers
        text_config = get_bletchley_text_config(config_type, gradient_checkpointing)

        self.output_query_embed = output_query_embed
        self.output_text_embed = output_text_embed

        self.query_embed_dim = query_config.hidden_size
        self.text_embed_dim = text_config.hidden_size

        self.num_positions = num_positions
        self.num_types = num_types

        self.query_encoder = BletchleyTextEncoder(
            query_config, add_projection_layer=False
        )
        self.text_encoder = BletchleyTextEncoder(
            text_config, add_projection_layer=False
        )

        self.query_projection = nn.Linear(
            self.query_embed_dim,
            projection_dim,
        )  # text_encoder.projection.weight, text_encoder.projection.bias
        self.text_projection = nn.Linear(
            self.text_embed_dim,
            projection_dim,
        )  # image_encoder.projection.weight,  image_encoder.projection.bias

        if num_positions is not None:
            self.position_embedding = nn.Embedding(num_positions, projection_dim)
            self.position_layer_norm = nn.LayerNorm(projection_dim)
            self.position = nn.Linear(projection_dim, num_classes)

        if num_types is not None:
            self.type_embedding = nn.Embedding(num_types, projection_dim)
            self.type_layer_norm = nn.LayerNorm(projection_dim)
            self.type = nn.Linear(projection_dim, num_classes)

        self.classifier = nn.Linear(projection_dim * 2, num_classes)
        self.init_weights()

        if enable_quantization:
            for __model__ in [
                self.text_encoder,
                self.text_projection,
            ]:
                __model__.qconfig = torch.quantization.get_default_qat_qconfig(
                    version=0
                )
                torch.quantization.prepare_qat(__model__, inplace=True)

        if freeze_base_model:
            for p in self.query_encoder.parameters():
                p.requires_grad = False

            for p in self.text_encoder.parameters():
                p.requires_grad = False

    @classmethod
    @add_default_section_for_init("microsoft/adsplus/slab/bletchley/v1/ranking")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/adsplus/slab/bletchley/v1/ranking")
        config_type = config.getoption("config_type", "0.3B")

        projection_dim = config.getoption("projection_dim", 64)
        num_query_layers = config.getoption("num_query_layers", 4)
        num_positions = config.getoption("num_positions", 50)
        num_types = config.getoption("num_types", None)
        num_classes = config.getoption("num_classes", 1)
        freeze_base_model = config.getoption("freeze_base_model", True)
        enable_quantization = config.getoption("enable_quantization", False)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)
        output_query_embed = config.getoption("output_query_embed", False)
        output_text_embed = config.getoption("output_text_embed", False)

        inst = cls(
            config_type=config_type,
            projection_dim=projection_dim,
            num_query_layers=num_query_layers,
            num_positions=num_positions,
            num_types=num_types,
            num_classes=num_classes,
            freeze_base_model=freeze_base_model,
            enable_quantization=enable_quantization,
            gradient_checkpointing=gradient_checkpointing,
            output_query_embed=output_query_embed,
            output_text_embed=output_text_embed,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        if pretrained_weight_path is not None:
            inst.from_pretrained(pretrained_weight_path)

        return inst

    @autocast()
    def forward(
        self,
        query_input_ids: torch.Tensor,
        text_input_ids: torch.Tensor,
        query_attention_mask: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        pos_ids: Optional[torch.Tensor] = None,
        type_ids: Optional[torch.Tensor] = None,
    ):
        if not self.training and self.output_query_embed:
            query_outputs = self.query_encoder(
                input_ids=query_input_ids,
                attention_mask=query_attention_mask,
            )
            query_embeds = query_outputs[:, 0]
            query_embeds = self.query_projection(query_embeds)
            return EmbeddingOutputs(embedding=query_embeds)

        if not self.training and self.output_text_embed:
            text_outputs = self.text_encoder(
                input_ids=text_input_ids,
                attention_mask=text_attention_mask,
            )
            text_embeds = text_outputs[:, 0]
            text_embeds = self.text_projection(text_embeds)
            return EmbeddingOutputs(embedding=text_embeds)


        query_outputs = self.query_encoder(
            input_ids=query_input_ids,
            attention_mask=query_attention_mask,
        )
        query_embeds = query_outputs[:, 0]
        query_embeds = self.query_projection(query_embeds)

        text_outputs = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
        )
        text_embeds = text_outputs[:, 0]
        text_embeds = self.text_projection(text_embeds)

        outputs = self.classifier(
            F.relu(torch.cat([query_embeds, text_embeds], axis=1))
        )

        if pos_ids is not None and self.num_positions is not None:
            pos_emb = self.position_layer_norm(self.position_embedding(pos_ids))
            outputs += self.position(pos_emb)
        if type_ids is not None and self.num_types is not None:
            type_emb = self.type_layer_norm(self.type_embedding(type_ids))
            outputs += self.type(type_emb)

        return ClassificationOutputs(outputs=outputs)
