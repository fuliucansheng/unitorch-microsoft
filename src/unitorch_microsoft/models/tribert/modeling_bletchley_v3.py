# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import json
import random
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torch.cuda.amp import autocast
from transformers.activations import quick_gelu
from unitorch.models import GenericModel
from unitorch.modules.classifier import reslayer
from unitorch.cli import (
    add_default_section_for_init,
    add_default_section_for_function,
    register_model,
)
from unitorch.cli.models import (
    ClassificationOutputs,
    LossOutputs,
    EmbeddingOutputs,
)
from unitorch_microsoft import cached_path
from unitorch_microsoft.models.bletchley.modeling_v3 import (
    BletchleyTextEncoder,
    BletchleyImageEncoder,
)


@register_model("microsoft/model/classification/twinbert/bletchley/v3")
class TwinBertBletchleyForClassification(GenericModel):
    def __init__(
        self,
        query_config_type: str,
        doc_config_type: str,
        projection_dim: Optional[int] = 128,
        hidden_dropout_prob: Optional[float] = 0.0,
        freeze_base_model: Optional[bool] = True,
        gradient_checkpointing: Optional[bool] = False,
        enable_quantization: Optional[bool] = False,
        output_query_embed: Optional[bool] = False,
        output_doc_embed: Optional[bool] = False,
    ):
        super().__init__()
        self.output_query_embed = output_query_embed
        self.output_doc_embed = output_doc_embed
        self.query_encoder = BletchleyTextEncoder(
            query_config_type,
            add_projection_layer=False,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.doc_encoder = BletchleyTextEncoder(
            doc_config_type,
            add_projection_layer=False,
            gradient_checkpointing=gradient_checkpointing,
        )

        self.projection_dim = projection_dim
        self.query_embed_dim = self.query_encoder.hidden_size
        self.doc_embed_dim = self.doc_encoder.hidden_size

        self.query_projection = nn.Linear(self.query_embed_dim, self.projection_dim)
        self.doc_projection = nn.Linear(self.doc_embed_dim, self.projection_dim)

        self.query_layer_norm = nn.LayerNorm(projection_dim)
        self.doc_layer_norm = nn.LayerNorm(projection_dim)

        self.dropout = nn.Dropout(hidden_dropout_prob)

        self.reslayer = reslayer(
            self.projection_dim,
            self.projection_dim // 2,
            self.projection_dim,
        )

        self.linear = nn.Linear(self.projection_dim, 1)

        self.classifier = nn.Linear(1, 1)

        self.init_weights()
        self.classifier.weight.data.fill_(5.0)

        if enable_quantization:
            for __model__ in [
                self.query_encoder,
                self.query_layer_norm,
            ]:
                __model__.qconfig = torch.quantization.get_default_qat_qconfig(
                    version=0
                )
                torch.quantization.prepare_qat(__model__, inplace=True)

        if freeze_base_model:
            for p in self.query_encoder.parameters():
                p.requires_grad = False

            for p in self.doc_encoder.parameters():
                p.requires_grad = False

    @classmethod
    @add_default_section_for_init(
        "microsoft/model/classification/twinbert/bletchley/v3"
    )
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section(
            "microsoft/model/classification/twinbert/bletchley/v3"
        )
        query_config_type = config.getoption("query_config_type", "0.3B")
        doc_config_type = config.getoption("doc_config_type", "0.8B")

        projection_dim = config.getoption("projection_dim", 128)
        hidden_dropout_prob = config.getoption("hidden_dropout_prob", 0.0)
        freeze_base_model = config.getoption("freeze_base_model", True)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)
        enable_quantization = config.getoption("enable_quantization", False)
        output_query_embed = config.getoption("output_query_embed", False)
        output_doc_embed = config.getoption("output_doc_embed", False)

        inst = cls(
            query_config_type=query_config_type,
            doc_config_type=doc_config_type,
            projection_dim=projection_dim,
            hidden_dropout_prob=hidden_dropout_prob,
            freeze_base_model=freeze_base_model,
            gradient_checkpointing=gradient_checkpointing,
            enable_quantization=enable_quantization,
            output_query_embed=output_query_embed,
            output_doc_embed=output_doc_embed,
        )

        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        if pretrained_weight_path is not None:
            inst.from_pretrained(pretrained_weight_path)

        return inst

    def from_pretrained(self, weight_path):
        if not os.path.exists(weight_path):
            return
        state_dict = torch.load(weight_path, map_location="cpu")
        state_dict = {
            key.replace("offer_encoder", "doc_encoder"): value
            for key, value in state_dict.items()
        }
        _keys = [key for key in state_dict.keys() if key.startswith("text_encoder")]
        for _key in _keys:
            _value = state_dict.pop(_key)
            state_dict["query_encoder" + _key[12:]] = _value
            state_dict["doc_encoder" + _key[12:]] = _value

        super().from_pretrained(state_dict=state_dict)

    def get_query_embedding(self, input_ids, attention_mask):
        query_outputs = self.query_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        query_embeds = query_outputs[:, 0]
        query_embeds = self.query_projection(query_embeds)
        query_embeds = self.query_layer_norm(quick_gelu(query_embeds))
        return query_embeds

    def get_doc_embedding(self, input_ids, attention_mask):
        doc_outputs = self.doc_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        doc_embeds = doc_outputs[:, 0]
        doc_embeds = self.doc_projection(doc_embeds)
        doc_embeds = self.doc_layer_norm(quick_gelu(doc_embeds))
        return doc_embeds

    def forward(
        self,
        query_input_ids: torch.Tensor = None,
        query_attention_mask: torch.Tensor = None,
        doc_input_ids: torch.Tensor = None,
        doc_attention_mask: torch.Tensor = None,
    ):
        if not self.training and self.output_query_embed:
            query_embeds = self.get_query_embedding(
                input_ids=query_input_ids,
                attention_mask=query_attention_mask,
            )

            return EmbeddingOutputs(embedding=query_embeds)

        if not self.training and self.output_doc_embed:
            doc_embeds = self.get_doc_embedding(
                input_ids=doc_input_ids,
                attention_mask=doc_attention_mask,
            )

            return EmbeddingOutputs(embedding=doc_embeds)

        query_embeds = self.get_query_embedding(query_input_ids, query_attention_mask)
        doc_embeds = self.get_doc_embedding(doc_input_ids, doc_attention_mask)

        mix_embeds = query_embeds * doc_embeds
        mix_embeds = self.dropout(mix_embeds)
        mix_embeds = self.reslayer(mix_embeds)
        outputs = self.linear(mix_embeds)
        outputs = self.classifier(outputs)
        outputs = torch.sigmoid(outputs)
        return ClassificationOutputs(outputs=outputs)
