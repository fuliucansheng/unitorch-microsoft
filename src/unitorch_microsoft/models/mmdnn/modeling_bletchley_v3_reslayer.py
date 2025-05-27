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
from torch import autocast
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


@register_model("microsoft/model/classification/mmdnn/bletchley/v3/reslayer")
class MMDNNBletchleyForClassification(GenericModel):
    def __init__(
        self,
        query_config_type: str,
        offer_config_type: str,
        projection_dim: Optional[int] = 288,
        num_ice: Optional[int] = 3181,
        num_seller: Optional[int] = 15020,
        num_brand: Optional[int] = 1000001,
        padding_idx: Optional[int] = -1,
        hidden_dim: Optional[int] = 32,
        output_hidden_dim: Optional[int] = 64,
        hidden_dropout_prob: Optional[float] = 0.0,
        freeze_base_model: Optional[bool] = True,
        freeze_image_model: Optional[bool] = False,
        gradient_checkpointing: Optional[bool] = False,
        enable_quantization: Optional[bool] = False,
        output_text_embed: Optional[bool] = False,
        output_image_embed: Optional[bool] = False,
        output_final_text_embed: Optional[bool] = False,
        output_final_image_embed: Optional[bool] = False,
    ):
        super().__init__()
        self.padding_idx = padding_idx
        self.output_text_embed = output_text_embed
        self.output_image_embed = output_image_embed
        self.output_final_text_embed = output_final_text_embed
        self.output_final_image_embed = output_final_image_embed

        self.text_encoder = BletchleyTextEncoder(
            query_config_type,
            add_projection_layer=False,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.image_encoder = BletchleyImageEncoder(
            offer_config_type,
            add_projection_layer=False,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.text_embed_dim = self.text_encoder.hidden_size
        self.image_embed_dim = self.image_encoder.hidden_size

        self.image_projection = nn.Linear(
            self.image_embed_dim,
            projection_dim,
            bias=False,
        )
        self.text_projection = nn.Linear(
            self.text_embed_dim,
            projection_dim,
            bias=False,
        )

        self.text_layer_norm = nn.LayerNorm(projection_dim)
        self.image_layer_norm = nn.LayerNorm(projection_dim)

        self.ice_embedding = nn.Embedding(num_ice, projection_dim)
        self.ice_layer_norm = nn.LayerNorm(projection_dim)
        self.seller_embedding = nn.Embedding(num_seller, hidden_dim)
        self.seller_layer_norm = nn.LayerNorm(hidden_dim)
        self.brand_embedding = nn.Embedding(num_brand, hidden_dim)
        self.brand_layer_norm = nn.LayerNorm(hidden_dim)

        self.final_visual_projection = nn.Linear(
            projection_dim + hidden_dim * 2,
            output_hidden_dim,
        )
        self.final_text_projection = nn.Linear(
            projection_dim + projection_dim,
            output_hidden_dim,
        )

        self.dropout = nn.Dropout(hidden_dropout_prob)

        self.reslayer = reslayer(
            output_hidden_dim,
            output_hidden_dim // 2,
            output_hidden_dim,
        )

        self.linear = nn.Linear(output_hidden_dim, 1)

        self.classifier = nn.Linear(1, 1)

        self.init_weights()
        self.classifier.weight.data.fill_(5.0)
        if enable_quantization:
            for __model__ in [
                self.text_encoder,
                self.text_layer_norm,
                self.ice_embedding,
                self.ice_layer_norm,
                self.final_text_projection,
            ]:
                __model__.qconfig = torch.quantization.get_default_qat_qconfig(
                    version=0
                )
                torch.quantization.prepare_qat(__model__, inplace=True)

        if freeze_base_model:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

            for p in self.image_encoder.parameters():
                p.requires_grad = False

        if freeze_image_model:
            for p in self.image_encoder.parameters():
                p.requires_grad = False

    @classmethod
    @add_default_section_for_init(
        "microsoft/model/classification/mmdnn/bletchley/v3/reslayer"
    )
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section(
            "microsoft/model/classification/mmdnn/bletchley/v3/reslayer"
        )
        query_config_type = config.getoption("query_config_type", "0.8B")
        offer_config_type = config.getoption("offer_config_type", "0.8B")
        projection_dim = config.getoption("projection_dim", 288)
        num_ice = config.getoption("num_ice", 3181)
        num_seller = config.getoption("num_seller", 15020)
        num_brand = config.getoption("num_brand", 1000001)
        padding_idx = config.getoption("padding_idx", -1)
        hidden_dim = config.getoption("hidden_dim", 32)
        output_hidden_dim = config.getoption("output_hidden_dim", 64)
        hidden_dropout_prob = config.getoption("hidden_dropout_prob", 0.0)
        freeze_base_model = config.getoption("freeze_base_model", True)
        freeze_image_model = config.getoption("freeze_image_model", False)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)
        enable_quantization = config.getoption("enable_quantization", False)
        output_text_embed = config.getoption("output_text_embed", False)
        output_image_embed = config.getoption("output_image_embed", False)
        output_final_text_embed = config.getoption("output_final_text_embed", False)
        output_final_image_embed = config.getoption("output_final_image_embed", False)

        inst = cls(
            query_config_type=query_config_type,
            offer_config_type=offer_config_type,
            projection_dim=projection_dim,
            num_ice=num_ice,
            num_seller=num_seller,
            num_brand=num_brand,
            padding_idx=padding_idx,
            hidden_dim=hidden_dim,
            output_hidden_dim=output_hidden_dim,
            hidden_dropout_prob=hidden_dropout_prob,
            freeze_base_model=freeze_base_model,
            freeze_image_model=freeze_image_model,
            gradient_checkpointing=gradient_checkpointing,
            enable_quantization=enable_quantization,
            output_text_embed=output_text_embed,
            output_image_embed=output_image_embed,
            output_final_text_embed=output_final_text_embed,
            output_final_image_embed=output_final_image_embed,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        if pretrained_weight_path is not None:
            inst.from_pretrained(
                pretrained_weight_path,
                replace_keys={"query_encoder.": "text_encoder."},
            )

        return inst

    def get_text_embedding(self, input_ids, attention_mask):
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        text_embeds = text_outputs[:, 0]
        text_embeds = self.text_projection(text_embeds)
        text_embeds = self.text_layer_norm(quick_gelu(text_embeds))

        return text_embeds

    def get_image_embedding(self, images):
        image_outputs = self.image_encoder(
            images=images,
        )
        image_embeds = image_outputs[:, 0]
        image_embeds = self.image_projection(image_embeds)
        image_embeds = self.image_layer_norm(quick_gelu(image_embeds))

        return image_embeds

    def get_final_text_embedding(
        self,
        input_ids,
        attention_mask,
        ice_ids,
        do_norm=True,
    ):
        text_embeds = self.get_text_embedding(input_ids, attention_mask)
        zero_tensor = torch.tensor(0).to(input_ids)
        ice_mask = ice_ids.ne(self.padding_idx)
        ice_ids = ice_ids.maximum(zero_tensor)
        ice_embeds = self.ice_embedding(ice_ids)
        ice_embeds = ice_embeds * ice_mask.unsqueeze(-1)
        if ice_embeds.dim() == 3:
            ice_embeds = ice_embeds.sum(dim=1)
        ice_embeds = self.ice_layer_norm(ice_embeds)

        text_embeds = torch.cat([text_embeds, ice_embeds], dim=-1)
        text_embeds = self.final_text_projection(quick_gelu(text_embeds))
        if do_norm:
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        return text_embeds

    def get_final_image_embedding(self, images, seller_ids, brand_ids, do_norm=True):
        image_embeds = self.get_image_embedding(images)

        seller_embeds = self.seller_embedding(seller_ids)
        brand_embeds = self.brand_embedding(brand_ids)

        seller_embeds = self.seller_layer_norm(seller_embeds)
        brand_embeds = self.brand_layer_norm(brand_embeds)

        image_embeds = torch.cat([image_embeds, seller_embeds, brand_embeds], dim=-1)
        image_embeds = self.final_visual_projection(quick_gelu(image_embeds))
        if do_norm:
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        return image_embeds

    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        images: torch.Tensor = None,
        ice_ids: torch.Tensor = None,
        seller_ids: torch.Tensor = None,
        brand_ids: torch.Tensor = None,
    ):
        if not self.training and self.output_text_embed:
            text_embeds = self.get_text_embedding(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            return EmbeddingOutputs(embedding=text_embeds)

        if not self.training and self.output_image_embed:
            image_embeds = self.get_image_embedding(
                images=images,
            )
            return EmbeddingOutputs(embedding=image_embeds)

        if not self.training and self.output_final_text_embed:
            text_embeds = self.get_final_text_embedding(
                input_ids=input_ids,
                attention_mask=attention_mask,
                ice_ids=ice_ids,
                do_norm=False,
            )
            return EmbeddingOutputs(embedding=text_embeds)

        if not self.training and self.output_final_image_embed:
            image_embeds = self.get_final_image_embedding(
                images=images,
                seller_ids=seller_ids,
                brand_ids=brand_ids,
                do_norm=False,
            )
            return EmbeddingOutputs(embedding=image_embeds)

        text_embeds = self.get_final_text_embedding(
            input_ids=input_ids,
            attention_mask=attention_mask,
            ice_ids=ice_ids,
            do_norm=False,
        )
        image_embeds = self.get_final_image_embedding(
            images=images,
            seller_ids=seller_ids,
            brand_ids=brand_ids,
            do_norm=False,
        )

        mix_embeds = text_embeds * image_embeds
        mix_embeds = self.dropout(mix_embeds)
        mix_embeds = self.reslayer(mix_embeds)
        outputs = self.linear(mix_embeds)

        outputs = self.classifier(outputs)
        outputs = torch.sigmoid(outputs)

        return ClassificationOutputs(outputs=outputs)


@register_model("microsoft/model/distillation/mmdnn/bletchley/v3/reslayer")
class MMDNNBletchleyForDistillation(GenericModel):
    def __init__(
        self,
        config_type: str,
        new_config_type: str,
        gradient_checkpointing: Optional[bool] = False,
    ):
        super().__init__()
        self.text_encoder = BletchleyTextEncoder(
            config_type,
            add_projection_layer=False,
            gradient_checkpointing=gradient_checkpointing,
        )

        self.new_text_encoder = BletchleyTextEncoder(
            new_config_type,
            add_projection_layer=False,
            gradient_checkpointing=gradient_checkpointing,
        )

        self.init_weights()

        for p in self.text_encoder.parameters():
            p.requires_grad = False

    @classmethod
    @add_default_section_for_init(
        "microsoft/model/distillation/mmdnn/bletchley/v3/reslayer"
    )
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section(
            "microsoft/model/distillation/mmdnn/bletchley/v3/reslayer"
        )
        config_type = config.getoption("config_type", "0.8B")
        new_config_type = config.getoption("new_config_type", "0.8B")
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)

        inst = cls(
            config_type=config_type,
            new_config_type=new_config_type,
            gradient_checkpointing=gradient_checkpointing,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        if pretrained_weight_path is not None:
            inst.from_pretrained(
                pretrained_weight_path,
                replace_keys={"query_encoder.": "text_encoder."},
            )

        return inst

    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
    ):
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        text_embeds = text_outputs[:, 0]
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        new_text_outputs = self.new_text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        new_text_embeds = new_text_outputs[:, 0]
        new_text_embeds = new_text_embeds / new_text_embeds.norm(dim=-1, keepdim=True)
        loss = (
            nn.MSELoss(reduction="none")(new_text_embeds, text_embeds)
            .sum(dim=-1)
            .mean()
        )
        return LossOutputs(loss=loss)


@register_model("microsoft/model/classification/mmdnn/bletchley/v3/v2/reslayer")
class MMDNNBletchleyForClassificationV2(GenericModel):
    def __init__(
        self,
        query_config_type: str,
        offer_config_type: str,
        projection_dim: Optional[int] = 288,
        num_ice: Optional[int] = 3181,
        num_seller: Optional[int] = 15020,
        num_brand: Optional[int] = 1000001,
        padding_idx: Optional[int] = -1,
        hidden_dim: Optional[int] = 32,
        output_hidden_dim: Optional[int] = 64,
        hidden_dropout_prob: Optional[float] = 0.0,
        freeze_base_model: Optional[bool] = True,
        freeze_offer_model: Optional[bool] = False,
        gradient_checkpointing: Optional[bool] = False,
        enable_quantization: Optional[bool] = False,
        output_query_embed: Optional[bool] = False,
        output_offer_embed: Optional[bool] = False,
        output_image_embed: Optional[bool] = False,
        output_final_query_embed: Optional[bool] = False,
        output_final_offer_embed: Optional[bool] = False,
    ):
        super().__init__()
        self.padding_idx = padding_idx
        self.output_query_embed = output_query_embed
        self.output_offer_embed = output_offer_embed
        self.output_image_embed = output_image_embed
        self.output_final_query_embed = output_final_query_embed
        self.output_final_offer_embed = output_final_offer_embed

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

        self.query_projection = nn.Linear(self.query_embed_dim, self.projection_dim)
        self.offer_projection = nn.Linear(self.offer_embed_dim, self.projection_dim)
        self.image_projection = nn.Linear(self.image_embed_dim, self.projection_dim)

        self.query_layer_norm = nn.LayerNorm(projection_dim)
        self.image_layer_norm = nn.LayerNorm(projection_dim)
        self.offer_layer_norm = nn.LayerNorm(projection_dim)

        self.ice_embedding = nn.Embedding(num_ice, projection_dim)
        self.ice_layer_norm = nn.LayerNorm(projection_dim)
        self.seller_embedding = nn.Embedding(num_seller, hidden_dim)
        self.seller_layer_norm = nn.LayerNorm(hidden_dim)
        self.brand_embedding = nn.Embedding(num_brand, hidden_dim)
        self.brand_layer_norm = nn.LayerNorm(hidden_dim)

        self.final_offer_projection = nn.Linear(
            projection_dim * 2 + hidden_dim * 2,
            output_hidden_dim,
        )
        self.final_query_projection = nn.Linear(
            projection_dim + projection_dim,
            output_hidden_dim,
        )
        self.dropout = nn.Dropout(hidden_dropout_prob)

        self.reslayer = reslayer(
            output_hidden_dim,
            output_hidden_dim // 2,
            output_hidden_dim,
        )

        self.linear = nn.Linear(output_hidden_dim, 1)

        self.classifier = nn.Linear(1, 1)

        self.init_weights()
        self.classifier.weight.data.fill_(5.0)

        if enable_quantization:
            for __model__ in [
                self.query_encoder,
                self.query_layer_norm,
                self.ice_embedding,
                self.ice_layer_norm,
                self.final_query_projection,
            ]:
                __model__.qconfig = torch.quantization.get_default_qat_qconfig(
                    version=0
                )
                torch.quantization.prepare_qat(__model__, inplace=True)

        if freeze_base_model:
            for p in self.query_encoder.parameters():
                p.requires_grad = False

            for p in self.offer_encoder.parameters():
                p.requires_grad = False

            for p in self.image_encoder.parameters():
                p.requires_grad = False

        if freeze_offer_model:
            for p in self.offer_encoder.parameters():
                p.requires_grad = False

            for p in self.image_encoder.parameters():
                p.requires_grad = False

    @classmethod
    @add_default_section_for_init(
        "microsoft/model/classification/mmdnn/bletchley/v3/v2/reslayer"
    )
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section(
            "microsoft/model/classification/mmdnn/bletchley/v3/v2/reslayer"
        )
        query_config_type = config.getoption("query_config_type", "0.3B")
        offer_config_type = config.getoption("offer_config_type", "0.8B")

        projection_dim = config.getoption("projection_dim", 288)
        num_ice = config.getoption("num_ice", 3181)
        num_seller = config.getoption("num_seller", 15020)
        num_brand = config.getoption("num_brand", 1000001)
        padding_idx = config.getoption("padding_idx", -1)
        hidden_dim = config.getoption("hidden_dim", 32)
        output_hidden_dim = config.getoption("output_hidden_dim", 64)
        hidden_dropout_prob = config.getoption("hidden_dropout_prob", 0.0)
        freeze_base_model = config.getoption("freeze_base_model", True)
        freeze_offer_model = config.getoption("freeze_offer_model", False)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)
        enable_quantization = config.getoption("enable_quantization", False)
        output_query_embed = config.getoption("output_query_embed", False)
        output_offer_embed = config.getoption("output_offer_embed", False)
        output_image_embed = config.getoption("output_image_embed", False)
        output_final_query_embed = config.getoption("output_final_query_embed", False)
        output_final_offer_embed = config.getoption("output_final_offer_embed", False)

        inst = cls(
            query_config_type=query_config_type,
            offer_config_type=offer_config_type,
            projection_dim=projection_dim,
            num_ice=num_ice,
            num_seller=num_seller,
            num_brand=num_brand,
            padding_idx=padding_idx,
            hidden_dim=hidden_dim,
            output_hidden_dim=output_hidden_dim,
            hidden_dropout_prob=hidden_dropout_prob,
            freeze_base_model=freeze_base_model,
            freeze_offer_model=freeze_offer_model,
            gradient_checkpointing=gradient_checkpointing,
            enable_quantization=enable_quantization,
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

    def get_query_embedding(self, input_ids, attention_mask):
        query_outputs = self.query_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        query_embeds = query_outputs[:, 0]
        query_embeds = self.query_projection(query_embeds)
        query_embeds = self.query_layer_norm(quick_gelu(query_embeds))
        return query_embeds

    def get_offer_embedding(self, input_ids, attention_mask):
        offer_outputs = self.offer_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        offer_embeds = offer_outputs[:, 0]
        offer_embeds = self.offer_projection(offer_embeds)
        offer_embeds = self.offer_layer_norm(quick_gelu(offer_embeds))
        return offer_embeds

    def get_image_embedding(self, images):
        image_embeds = self.image_encoder(images)
        image_embeds = image_embeds[:, 0]
        image_embeds = self.image_projection(image_embeds)
        image_embeds = self.image_layer_norm(quick_gelu(image_embeds))
        return image_embeds

    def get_final_query_embedding(
        self,
        input_ids,
        attention_mask,
        ice_ids,
        do_norm=True,
    ):
        query_embeds = self.get_query_embedding(input_ids, attention_mask)
        zero_tensor = torch.tensor(0).to(input_ids)
        ice_mask = ice_ids.ne(self.padding_idx)
        ice_ids = ice_ids.maximum(zero_tensor)
        ice_embeds = self.ice_embedding(ice_ids)
        ice_embeds = ice_embeds * ice_mask.unsqueeze(-1)
        if ice_embeds.dim() == 3:
            ice_embeds = ice_embeds.sum(dim=1)
        ice_embeds = self.ice_layer_norm(ice_embeds)

        query_embeds = torch.cat([query_embeds, ice_embeds], dim=-1)
        query_embeds = self.final_query_projection(quick_gelu(query_embeds))
        if do_norm:
            query_embeds = query_embeds / query_embeds.norm(dim=-1, keepdim=True)
        return query_embeds

    def get_final_offer_embedding(
        self,
        input_ids,
        attention_mask,
        images,
        seller_ids,
        brand_ids,
        do_norm=True,
    ):
        offer_embeds = self.get_offer_embedding(input_ids, attention_mask)
        image_embeds = self.get_image_embedding(images)

        seller_embeds = self.seller_embedding(seller_ids)
        brand_embeds = self.brand_embedding(brand_ids)

        seller_embeds = self.seller_layer_norm(seller_embeds)
        brand_embeds = self.brand_layer_norm(brand_embeds)

        offer_embeds = torch.cat(
            [image_embeds, offer_embeds, seller_embeds, brand_embeds],
            dim=-1,
        )

        offer_embeds = self.final_offer_projection(quick_gelu(offer_embeds))
        if do_norm:
            offer_embeds = offer_embeds / offer_embeds.norm(dim=-1, keepdim=True)

        return offer_embeds

    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(
        self,
        query_input_ids: torch.Tensor = None,
        query_attention_mask: torch.Tensor = None,
        images: torch.Tensor = None,
        offer_input_ids: torch.Tensor = None,
        offer_attention_mask: torch.Tensor = None,
        ice_ids: torch.Tensor = None,
        seller_ids: torch.Tensor = None,
        brand_ids: torch.Tensor = None,
    ):
        if not self.training and self.output_query_embed:
            query_embeds = self.get_query_embedding(
                input_ids=query_input_ids,
                attention_mask=query_attention_mask,
            )

            return EmbeddingOutputs(embedding=query_embeds)

        if not self.training and self.output_offer_embed:
            offer_embeds = self.get_offer_embedding(
                input_ids=offer_input_ids,
                attention_mask=offer_attention_mask,
            )

            return EmbeddingOutputs(embedding=offer_embeds)

        if not self.training and self.output_image_embed:
            image_embeds = self.get_image_embedding(images=images)

            return EmbeddingOutputs(embedding=image_embeds)

        if not self.training and self.output_final_query_embed:
            query_embeds = self.get_final_query_embedding(
                input_ids=query_input_ids,
                attention_mask=query_attention_mask,
                ice_ids=ice_ids,
                do_norm=False,
            )
            return EmbeddingOutputs(embedding=query_embeds)

        if not self.training and self.output_final_offer_embed:
            offer_embeds = self.get_final_offer_embedding(
                input_ids=offer_input_ids,
                attention_mask=offer_attention_mask,
                images=images,
                seller_ids=seller_ids,
                brand_ids=brand_ids,
                do_norm=False,
            )
            return EmbeddingOutputs(embedding=offer_embeds)

        query_embeds = self.get_final_query_embedding(
            input_ids=query_input_ids,
            attention_mask=query_attention_mask,
            ice_ids=ice_ids,
            do_norm=False,
        )
        offer_embeds = self.get_final_offer_embedding(
            input_ids=offer_input_ids,
            attention_mask=offer_attention_mask,
            images=images,
            seller_ids=seller_ids,
            brand_ids=brand_ids,
            do_norm=False,
        )

        mix_embeds = query_embeds * offer_embeds
        mix_embeds = self.dropout(mix_embeds)
        mix_embeds = self.reslayer(mix_embeds)
        outputs = self.linear(mix_embeds)

        outputs = self.classifier(outputs)
        outputs = torch.sigmoid(outputs)

        return ClassificationOutputs(outputs=outputs)


@register_model("microsoft/model/classification/mmdnn/bletchley/v3/v2/text/reslayer")
class MMDNNBletchleyTextForClassificationV2(GenericModel):
    def __init__(
        self,
        query_config_type: str,
        offer_config_type: str,
        projection_dim: Optional[int] = 288,
        num_ice: Optional[int] = 3181,
        num_seller: Optional[int] = 15020,
        num_brand: Optional[int] = 1000001,
        padding_idx: Optional[int] = -1,
        hidden_dim: Optional[int] = 32,
        output_hidden_dim: Optional[int] = 64,
        hidden_dropout_prob: Optional[float] = 0.0,
        freeze_base_model: Optional[bool] = True,
        freeze_offer_model: Optional[bool] = False,
        gradient_checkpointing: Optional[bool] = False,
        enable_quantization: Optional[bool] = False,
        output_query_embed: Optional[bool] = False,
        output_offer_embed: Optional[bool] = False,
        output_final_query_embed: Optional[bool] = False,
        output_final_offer_embed: Optional[bool] = False,
    ):
        super().__init__()
        self.padding_idx = padding_idx
        self.output_query_embed = output_query_embed
        self.output_offer_embed = output_offer_embed
        self.output_final_query_embed = output_final_query_embed
        self.output_final_offer_embed = output_final_offer_embed

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

        self.query_projection = nn.Linear(self.query_embed_dim, self.projection_dim)
        self.offer_projection = nn.Linear(self.offer_embed_dim, self.projection_dim)

        self.query_layer_norm = nn.LayerNorm(projection_dim)
        self.offer_layer_norm = nn.LayerNorm(projection_dim)

        self.ice_embedding = nn.Embedding(num_ice, projection_dim)
        self.ice_layer_norm = nn.LayerNorm(projection_dim)
        self.seller_embedding = nn.Embedding(num_seller, hidden_dim)
        self.seller_layer_norm = nn.LayerNorm(hidden_dim)
        self.brand_embedding = nn.Embedding(num_brand, hidden_dim)
        self.brand_layer_norm = nn.LayerNorm(hidden_dim)

        self.final_offer_projection = nn.Linear(
            projection_dim + hidden_dim * 2,
            output_hidden_dim,
        )
        self.final_query_projection = nn.Linear(
            projection_dim + projection_dim,
            output_hidden_dim,
        )
        self.dropout = nn.Dropout(hidden_dropout_prob)

        self.reslayer = reslayer(
            output_hidden_dim,
            output_hidden_dim // 2,
            output_hidden_dim,
        )

        self.linear = nn.Linear(output_hidden_dim, 1)

        self.classifier = nn.Linear(1, 1)

        self.init_weights()
        self.classifier.weight.data.fill_(5.0)

        if enable_quantization:
            for __model__ in [
                self.query_encoder,
                self.query_layer_norm,
                self.ice_embedding,
                self.ice_layer_norm,
                self.final_query_projection,
            ]:
                __model__.qconfig = torch.quantization.get_default_qat_qconfig(
                    version=0
                )
                torch.quantization.prepare_qat(__model__, inplace=True)

        if freeze_base_model:
            for p in self.query_encoder.parameters():
                p.requires_grad = False

            for p in self.offer_encoder.parameters():
                p.requires_grad = False

        if freeze_offer_model:
            for p in self.offer_encoder.parameters():
                p.requires_grad = False

    @classmethod
    @add_default_section_for_init(
        "microsoft/model/classification/mmdnn/bletchley/v3/v2/text/reslayer"
    )
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section(
            "microsoft/model/classification/mmdnn/bletchley/v3/v2/text/reslayer"
        )
        query_config_type = config.getoption("query_config_type", "0.3B")
        offer_config_type = config.getoption("offer_config_type", "0.3B")
        projection_dim = config.getoption("projection_dim", 288)
        num_ice = config.getoption("num_ice", 3181)
        num_seller = config.getoption("num_seller", 15020)
        num_brand = config.getoption("num_brand", 1000001)
        padding_idx = config.getoption("padding_idx", -1)
        hidden_dim = config.getoption("hidden_dim", 32)
        output_hidden_dim = config.getoption("output_hidden_dim", 64)
        hidden_dropout_prob = config.getoption("hidden_dropout_prob", 0.0)
        freeze_base_model = config.getoption("freeze_base_model", True)
        freeze_offer_model = config.getoption("freeze_offer_model", False)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)
        enable_quantization = config.getoption("enable_quantization", False)
        output_query_embed = config.getoption("output_query_embed", False)
        output_offer_embed = config.getoption("output_offer_embed", False)
        output_final_query_embed = config.getoption("output_final_query_embed", False)
        output_final_offer_embed = config.getoption("output_final_offer_embed", False)

        inst = cls(
            query_config_type=query_config_type,
            offer_config_type=offer_config_type,
            projection_dim=projection_dim,
            num_ice=num_ice,
            num_seller=num_seller,
            num_brand=num_brand,
            padding_idx=padding_idx,
            hidden_dim=hidden_dim,
            output_hidden_dim=output_hidden_dim,
            hidden_dropout_prob=hidden_dropout_prob,
            freeze_base_model=freeze_base_model,
            freeze_offer_model=freeze_offer_model,
            gradient_checkpointing=gradient_checkpointing,
            enable_quantization=enable_quantization,
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

    def get_query_embedding(self, input_ids, attention_mask):
        query_outputs = self.query_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        query_embeds = query_outputs[:, 0]
        query_embeds = self.query_projection(query_embeds)
        query_embeds = self.query_layer_norm(quick_gelu(query_embeds))
        return query_embeds

    def get_offer_embedding(self, input_ids, attention_mask):
        offer_outputs = self.offer_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        offer_embeds = offer_outputs[:, 0]
        offer_embeds = self.offer_projection(offer_embeds)
        offer_embeds = self.offer_layer_norm(quick_gelu(offer_embeds))
        return offer_embeds

    def get_final_query_embedding(
        self,
        input_ids,
        attention_mask,
        ice_ids,
        do_norm=True,
    ):
        query_embeds = self.get_query_embedding(input_ids, attention_mask)
        zero_tensor = torch.tensor(0).to(input_ids)
        ice_mask = ice_ids.ne(self.padding_idx)
        ice_ids = ice_ids.maximum(zero_tensor)
        ice_embeds = self.ice_embedding(ice_ids)
        ice_embeds = ice_embeds * ice_mask.unsqueeze(-1)
        if ice_embeds.dim() == 3:
            ice_embeds = ice_embeds.sum(dim=1)
        ice_embeds = self.ice_layer_norm(ice_embeds)

        query_embeds = torch.cat([query_embeds, ice_embeds], dim=-1)
        query_embeds = self.final_query_projection(quick_gelu(query_embeds))
        if do_norm:
            query_embeds = query_embeds / query_embeds.norm(dim=-1, keepdim=True)
        return query_embeds

    def get_final_offer_embedding(
        self,
        input_ids,
        attention_mask,
        seller_ids,
        brand_ids,
        do_norm=True,
    ):
        offer_embeds = self.get_offer_embedding(input_ids, attention_mask)

        seller_embeds = self.seller_embedding(seller_ids)
        brand_embeds = self.brand_embedding(brand_ids)

        seller_embeds = self.seller_layer_norm(seller_embeds)
        brand_embeds = self.brand_layer_norm(brand_embeds)

        offer_embeds = torch.cat(
            [offer_embeds, seller_embeds, brand_embeds],
            dim=-1,
        )

        offer_embeds = self.final_offer_projection(quick_gelu(offer_embeds))
        if do_norm:
            offer_embeds = offer_embeds / offer_embeds.norm(dim=-1, keepdim=True)

        return offer_embeds

    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(
        self,
        query_input_ids: torch.Tensor = None,
        query_attention_mask: torch.Tensor = None,
        offer_input_ids: torch.Tensor = None,
        offer_attention_mask: torch.Tensor = None,
        ice_ids: torch.Tensor = None,
        seller_ids: torch.Tensor = None,
        brand_ids: torch.Tensor = None,
    ):
        if not self.training and self.output_query_embed:
            query_embeds = self.get_query_embedding(
                input_ids=query_input_ids,
                attention_mask=query_attention_mask,
            )
            return EmbeddingOutputs(embedding=query_embeds)

        if not self.training and self.output_offer_embed:
            offer_embeds = self.get_offer_embedding(
                input_ids=offer_input_ids,
                attention_mask=offer_attention_mask,
            )

            return EmbeddingOutputs(embedding=offer_embeds)

        if not self.training and self.output_final_query_embed:
            query_embeds = self.get_final_query_embedding(
                input_ids=query_input_ids,
                attention_mask=query_attention_mask,
                ice_ids=ice_ids,
                do_norm=False,
            )
            return EmbeddingOutputs(embedding=query_embeds)

        if not self.training and self.output_final_offer_embed:
            offer_embeds = self.get_final_offer_embedding(
                input_ids=offer_input_ids,
                attention_mask=offer_attention_mask,
                seller_ids=seller_ids,
                brand_ids=brand_ids,
                do_norm=False,
            )
            return EmbeddingOutputs(embedding=offer_embeds)

        query_embeds = self.get_final_query_embedding(
            input_ids=query_input_ids,
            attention_mask=query_attention_mask,
            ice_ids=ice_ids,
            do_norm=False,
        )
        offer_embeds = self.get_final_offer_embedding(
            input_ids=offer_input_ids,
            attention_mask=offer_attention_mask,
            seller_ids=seller_ids,
            brand_ids=brand_ids,
            do_norm=False,
        )

        mix_embeds = query_embeds * offer_embeds
        mix_embeds = self.dropout(mix_embeds)
        mix_embeds = self.reslayer(mix_embeds)
        outputs = self.linear(mix_embeds)

        outputs = self.classifier(outputs)
        outputs = torch.sigmoid(outputs)

        return ClassificationOutputs(outputs=outputs)


@register_model("microsoft/model/classification/mmdnn/bletchley/v3/v2/text/v2/reslayer")
class MMDNNBletchleyTextForClassificationV2_2(GenericModel):
    def __init__(
        self,
        query_config_type: str,
        offer_config_type: str,
        projection_dim: Optional[int] = 288,
        num_ice: Optional[int] = 3181,
        num_seller: Optional[int] = 15020,
        num_brand: Optional[int] = 1000001,
        padding_idx: Optional[int] = -1,
        hidden_dim: Optional[int] = 32,
        offer_hidden_dim: Optional[int] = 64,
        output_hidden_dim: Optional[int] = 512,
        hidden_dropout_prob: Optional[float] = 0.0,
        freeze_base_model: Optional[bool] = True,
        freeze_offer_model: Optional[bool] = False,
        gradient_checkpointing: Optional[bool] = False,
        enable_quantization: Optional[bool] = False,
        output_query_embed: Optional[bool] = False,
        output_offer_embed: Optional[bool] = False,
        output_final_query_embed: Optional[bool] = False,
        output_final_offer_embed: Optional[bool] = False,
    ):
        super().__init__()
        self.padding_idx = padding_idx
        self.output_query_embed = output_query_embed
        self.output_offer_embed = output_offer_embed
        self.output_final_query_embed = output_final_query_embed
        self.output_final_offer_embed = output_final_offer_embed

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

        self.query_projection = nn.Linear(self.query_embed_dim, self.projection_dim)
        self.offer_projection = nn.Linear(self.offer_embed_dim, self.projection_dim)

        self.query_layer_norm = nn.LayerNorm(projection_dim)
        self.offer_layer_norm = nn.LayerNorm(projection_dim)

        self.ice_embedding = nn.Embedding(num_ice, projection_dim)
        self.ice_layer_norm = nn.LayerNorm(projection_dim)
        self.seller_embedding = nn.Embedding(num_seller, hidden_dim)
        self.seller_layer_norm = nn.LayerNorm(hidden_dim)
        self.brand_embedding = nn.Embedding(num_brand, hidden_dim)
        self.brand_layer_norm = nn.LayerNorm(hidden_dim)

        self.offer_downscale_projection = nn.Linear(
            projection_dim + hidden_dim * 2,
            offer_hidden_dim,
        )
        self.final_offer_projection = nn.Linear(
            offer_hidden_dim,
            output_hidden_dim,
        )
        self.final_query_projection = nn.Linear(
            projection_dim + projection_dim,
            output_hidden_dim,
        )
        self.dropout = nn.Dropout(hidden_dropout_prob)

        self.reslayer = reslayer(
            output_hidden_dim,
            output_hidden_dim // 2,
            output_hidden_dim,
        )

        self.linear = nn.Linear(output_hidden_dim, 1)

        self.classifier = nn.Linear(1, 1)

        self.init_weights()
        self.classifier.weight.data.fill_(5.0)

        if enable_quantization:
            for __model__ in [
                self.query_encoder,
                self.query_layer_norm,
                self.ice_embedding,
                self.ice_layer_norm,
                self.final_query_projection,
            ]:
                __model__.qconfig = torch.quantization.get_default_qat_qconfig(
                    version=0
                )
                torch.quantization.prepare_qat(__model__, inplace=True)

        if freeze_base_model:
            for p in self.query_encoder.parameters():
                p.requires_grad = False

            for p in self.offer_encoder.parameters():
                p.requires_grad = False

        if freeze_offer_model:
            for p in self.offer_encoder.parameters():
                p.requires_grad = False

    @classmethod
    @add_default_section_for_init(
        "microsoft/model/classification/mmdnn/bletchley/v3/v2/text/v2/reslayer"
    )
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section(
            "microsoft/model/classification/mmdnn/bletchley/v3/v2/text/v2/reslayer"
        )
        query_config_type = config.getoption("query_config_type", "0.3B")
        offer_config_type = config.getoption("offer_config_type", "0.3B")
        projection_dim = config.getoption("projection_dim", 288)
        num_ice = config.getoption("num_ice", 3181)
        num_seller = config.getoption("num_seller", 15020)
        num_brand = config.getoption("num_brand", 1000001)
        padding_idx = config.getoption("padding_idx", -1)
        hidden_dim = config.getoption("hidden_dim", 32)
        offer_hidden_dim = config.getoption("offer_hidden_dim", 64)
        output_hidden_dim = config.getoption("output_hidden_dim", 512)
        freeze_base_model = config.getoption("freeze_base_model", True)
        freeze_offer_model = config.getoption("freeze_offer_model", False)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)
        enable_quantization = config.getoption("enable_quantization", False)
        output_query_embed = config.getoption("output_query_embed", False)
        output_offer_embed = config.getoption("output_offer_embed", False)
        output_final_query_embed = config.getoption("output_final_query_embed", False)
        output_final_offer_embed = config.getoption("output_final_offer_embed", False)

        inst = cls(
            query_config_type=query_config_type,
            offer_config_type=offer_config_type,
            projection_dim=projection_dim,
            num_ice=num_ice,
            num_seller=num_seller,
            num_brand=num_brand,
            padding_idx=padding_idx,
            hidden_dim=hidden_dim,
            offer_hidden_dim=offer_hidden_dim,
            output_hidden_dim=output_hidden_dim,
            freeze_base_model=freeze_base_model,
            freeze_offer_model=freeze_offer_model,
            gradient_checkpointing=gradient_checkpointing,
            enable_quantization=enable_quantization,
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

    def get_query_embedding(self, input_ids, attention_mask):
        query_outputs = self.query_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        query_embeds = query_outputs[:, 0]
        query_embeds = self.query_projection(query_embeds)
        query_embeds = self.query_layer_norm(quick_gelu(query_embeds))
        return query_embeds

    def get_offer_embedding(self, input_ids, attention_mask):
        offer_outputs = self.offer_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        offer_embeds = offer_outputs[:, 0]
        offer_embeds = self.offer_projection(offer_embeds)
        offer_embeds = self.offer_layer_norm(quick_gelu(offer_embeds))
        return offer_embeds

    def get_final_query_embedding(
        self,
        input_ids,
        attention_mask,
        ice_ids,
        do_norm=True,
    ):
        query_embeds = self.get_query_embedding(input_ids, attention_mask)
        zero_tensor = torch.tensor(0).to(input_ids)
        ice_mask = ice_ids.ne(self.padding_idx)
        ice_ids = ice_ids.maximum(zero_tensor)
        ice_embeds = self.ice_embedding(ice_ids)
        ice_embeds = ice_embeds * ice_mask.unsqueeze(-1)
        if ice_embeds.dim() == 3:
            ice_embeds = ice_embeds.sum(dim=1)
        ice_embeds = self.ice_layer_norm(ice_embeds)

        query_embeds = torch.cat([query_embeds, ice_embeds], dim=-1)
        query_embeds = self.final_query_projection(quick_gelu(query_embeds))
        if do_norm:
            query_embeds = query_embeds / query_embeds.norm(dim=-1, keepdim=True)
        return query_embeds

    def get_final_offer_embedding(
        self,
        input_ids,
        attention_mask,
        seller_ids,
        brand_ids,
        do_norm=True,
    ):
        offer_embeds = self.get_offer_embedding(input_ids, attention_mask)

        seller_embeds = self.seller_embedding(seller_ids)
        brand_embeds = self.brand_embedding(brand_ids)

        seller_embeds = self.seller_layer_norm(seller_embeds)
        brand_embeds = self.brand_layer_norm(brand_embeds)

        offer_embeds = torch.cat(
            [offer_embeds, seller_embeds, brand_embeds],
            dim=-1,
        )

        offer_embeds = self.offer_downscale_projection(quick_gelu(offer_embeds))
        offer_embeds = self.final_offer_projection(offer_embeds)
        if do_norm:
            offer_embeds = offer_embeds / offer_embeds.norm(dim=-1, keepdim=True)

        return offer_embeds

    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(
        self,
        query_input_ids: torch.Tensor = None,
        query_attention_mask: torch.Tensor = None,
        offer_input_ids: torch.Tensor = None,
        offer_attention_mask: torch.Tensor = None,
        ice_ids: torch.Tensor = None,
        seller_ids: torch.Tensor = None,
        brand_ids: torch.Tensor = None,
    ):
        if not self.training and self.output_query_embed:
            query_embeds = self.get_query_embedding(
                input_ids=query_input_ids,
                attention_mask=query_attention_mask,
            )
            return EmbeddingOutputs(embedding=query_embeds)

        if not self.training and self.output_offer_embed:
            offer_embeds = self.get_offer_embedding(
                input_ids=offer_input_ids,
                attention_mask=offer_attention_mask,
            )

            return EmbeddingOutputs(embedding=offer_embeds)

        if not self.training and self.output_final_query_embed:
            query_embeds = self.get_final_query_embedding(
                input_ids=query_input_ids,
                attention_mask=query_attention_mask,
                ice_ids=ice_ids,
                do_norm=False,
            )
            return EmbeddingOutputs(embedding=query_embeds)

        if not self.training and self.output_final_offer_embed:
            offer_embeds = self.get_final_offer_embedding(
                input_ids=offer_input_ids,
                attention_mask=offer_attention_mask,
                seller_ids=seller_ids,
                brand_ids=brand_ids,
                do_norm=False,
            )
            return EmbeddingOutputs(embedding=offer_embeds)

        query_embeds = self.get_final_query_embedding(
            input_ids=query_input_ids,
            attention_mask=query_attention_mask,
            ice_ids=ice_ids,
            do_norm=False,
        )
        offer_embeds = self.get_final_offer_embedding(
            input_ids=offer_input_ids,
            attention_mask=offer_attention_mask,
            seller_ids=seller_ids,
            brand_ids=brand_ids,
            do_norm=False,
        )

        mix_embeds = query_embeds * offer_embeds
        mix_embeds = self.dropout(mix_embeds)
        mix_embeds = self.reslayer(mix_embeds)
        outputs = self.linear(mix_embeds)

        outputs = self.classifier(outputs)
        outputs = torch.sigmoid(outputs)

        return ClassificationOutputs(outputs=outputs)
