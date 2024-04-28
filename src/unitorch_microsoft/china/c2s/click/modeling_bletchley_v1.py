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
from unitorch_microsoft import cached_path


@register_model("microsoft/china/msan/pretrain/bletchley/v1")
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
        self.ads_encoder = BletchleyTextEncoder(text_config, add_projection_layer=False)

        self.use_all_gather = use_all_gather
        self.attn = nn.Linear(text_config.hidden_size, 1, bias=False)
        self.user_projection = nn.Linear(
            self.text_embed_dim,
            self.projection_dim,
        )
        self.ads_projection = nn.Linear(
            self.text_embed_dim,
            self.projection_dim,
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init_value)

        self.init_weights()

    @classmethod
    @add_default_section_for_init("microsoft/china/msan/pretrain/bletchley/v1")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/china/msan/pretrain/bletchley/v1")
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
        user_outputs = self.user_encoder(
            user_input_ids.view(-1, seq_len), user_attention_mask.view(-1, seq_len)
        )
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


@register_model("microsoft/china/msan/classification/bletchley/v1")
class BletchleyForClassification(GenericModel):
    def __init__(
        self,
        config_type,
        projection_dim: int = 32,
        num_tgs: Optional[int] = 50,
        num_demands: Optional[int] = 10,
        num_markets: Optional[int] = 200,
        num_positions: Optional[int] = 13000,
        freeze_base_model: Optional[bool] = True,
        enable_tgs_features: Optional[bool] = True,
        enable_demands_features: Optional[bool] = True,
        enable_markets_features: Optional[bool] = True,
        enable_positions_features: Optional[bool] = True,
        output_user_embeds: Optional[bool] = False,
        output_ads_embeds: Optional[bool] = False,
        gradient_checkpointing: Optional[bool] = False,
    ):
        super().__init__()

        text_config = get_bletchley_text_config(config_type, gradient_checkpointing)

        self.projection_dim = projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.enable_tgs_features = enable_tgs_features
        self.enable_demands_features = enable_demands_features
        self.enable_markets_features = enable_markets_features
        self.enable_positions_features = enable_positions_features
        self.output_user_embeds = output_user_embeds
        self.output_ads_embeds = output_ads_embeds

        self.user_encoder = BletchleyTextEncoder(
            text_config, add_projection_layer=False
        )
        self.ads_encoder = BletchleyTextEncoder(text_config, add_projection_layer=False)

        self.click_attn = nn.Linear(text_config.hidden_size, 1, bias=False)
        self.user_click_projection = nn.Linear(
            self.text_embed_dim,
            self.projection_dim,
        )
        self.user_click_layer_norm = nn.LayerNorm(self.projection_dim)

        self.ads_click_projection = nn.Linear(
            self.text_embed_dim,
            self.projection_dim,
        )
        self.ads_click_layer_norm = nn.LayerNorm(self.projection_dim)

        if enable_tgs_features:
            self.click_tg_embedding = nn.Embedding(num_tgs, self.projection_dim)
            self.click_tg_layer_norm = nn.LayerNorm(self.projection_dim)

        if enable_demands_features:
            self.click_demand_embedding = nn.Embedding(num_demands, self.projection_dim)
            self.click_demand_layer_norm = nn.LayerNorm(self.projection_dim)

        if enable_markets_features:
            self.click_market_embedding = nn.Embedding(num_markets, self.projection_dim)
            self.click_market_layer_norm = nn.LayerNorm(self.projection_dim)

        if enable_positions_features:
            self.click_position_embedding = nn.Embedding(
                num_positions, self.projection_dim
            )
            self.click_position_layer_norm = nn.LayerNorm(self.projection_dim)

        num_features = 1 + int(
            sum(
                [
                    enable_tgs_features,
                    enable_demands_features,
                    enable_markets_features,
                    enable_positions_features,
                ]
            )
        )

        self.user_click_final_projection = nn.Linear(
            self.projection_dim * num_features,
            self.projection_dim,
        )
        self.ads_click_final_projection = nn.Linear(
            self.projection_dim,
            self.projection_dim,
        )

        self.click_classifier = nn.Linear(1, 1)

        self.init_weights()
        self.click_classifier.weight.data.fill_(5.0)

        if freeze_base_model:
            for param in self.user_encoder.parameters():
                param.requires_grad = False
            for param in self.ads_encoder.parameters():
                param.requires_grad = False

    @classmethod
    @add_default_section_for_init("microsoft/china/msan/classification/bletchley/v1")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/china/msan/classification/bletchley/v1")
        config_type = config.getoption("config_type", "0.15B")
        projection_dim = config.getoption("projection_dim", 32)
        num_tgs = config.getoption("num_tgs", 50)
        num_demands = config.getoption("num_demands", 10)
        num_markets = config.getoption("num_markets", 200)
        num_positions = config.getoption("num_positions", 13000)
        freeze_base_model = config.getoption("freeze_base_model", True)
        enable_tgs_features = config.getoption("enable_tgs_features", True)
        enable_demands_features = config.getoption("enable_demands_features", True)
        enable_markets_features = config.getoption("enable_markets_features", True)
        enable_positions_features = config.getoption("enable_positions_features", True)
        output_user_embeds = config.getoption("output_user_embeds", False)
        output_ads_embeds = config.getoption("output_ads_embeds", False)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)

        inst = cls(
            config_type=config_type,
            projection_dim=projection_dim,
            num_tgs=num_tgs,
            num_demands=num_demands,
            num_markets=num_markets,
            num_positions=num_positions,
            freeze_base_model=freeze_base_model,
            enable_tgs_features=enable_tgs_features,
            enable_demands_features=enable_demands_features,
            enable_markets_features=enable_markets_features,
            enable_positions_features=enable_positions_features,
            output_user_embeds=output_user_embeds,
            output_ads_embeds=output_ads_embeds,
            gradient_checkpointing=gradient_checkpointing,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        if pretrained_weight_path is not None:
            pretrained_weight_path = cached_path(pretrained_weight_path)
            inst.from_pretrained(pretrained_weight_path)

        return inst

    def get_user_base_embeds(
        self,
        user_input_ids,
        user_attention_mask=None,
        user_num_attention_mask=None,
    ):
        batch, num, seq_len = user_input_ids.shape
        user_outputs = self.user_encoder(
            user_input_ids.view(-1, seq_len), user_attention_mask.view(-1, seq_len)
        )
        pooled_output = user_outputs[:, 0]

        click_attn_score = self.click_attn(pooled_output).view(batch, num)
        click_attn_score = click_attn_score + (1 - user_num_attention_mask) * -10000.0
        click_attn_score = F.softmax(click_attn_score, dim=-1)
        user_click_attn_outputs = torch.bmm(
            click_attn_score.unsqueeze(1), pooled_output.view(batch, num, -1)
        ).squeeze(1)
        user_click_embeds = self.user_click_projection(user_click_attn_outputs)

        user_click_embeds = self.user_click_layer_norm(quick_gelu(user_click_embeds))

        return user_click_embeds

    def get_user_embeds(
        self,
        user_input_ids,
        user_attention_mask=None,
        user_num_attention_mask=None,
        tg_ids=None,
        demand_ids=None,
        market_ids=None,
        pos_ids=None,
        do_norm=True,
    ):
        user_click_embeds = self.get_user_base_embeds(
            user_input_ids=user_input_ids,
            user_attention_mask=user_attention_mask,
            user_num_attention_mask=user_num_attention_mask,
        )

        click_emb = [user_click_embeds]
        if self.enable_positions_features:
            click_pos_emb = self.click_position_layer_norm(
                self.click_position_embedding(pos_ids)
            )
            click_emb.append(click_pos_emb)

        if self.enable_tgs_features:
            click_tg_emb = self.click_tg_layer_norm(self.click_tg_embedding(tg_ids))
            click_emb.append(click_tg_emb)

        if self.enable_demands_features:
            click_demand_emb = self.click_demand_layer_norm(
                self.click_demand_embedding(demand_ids)
            )
            click_emb.append(click_demand_emb)

        if self.enable_markets_features:
            click_market_emb = self.click_market_layer_norm(
                self.click_market_embedding(market_ids)
            )
            click_emb.append(click_market_emb)

        user_click_embeds = torch.cat(
            click_emb,
            dim=-1,
        )
        user_click_embeds = self.user_click_final_projection(user_click_embeds)

        if do_norm:
            user_click_embeds = user_click_embeds / user_click_embeds.norm(
                dim=-1, keepdim=True
            )
        return user_click_embeds

    def get_ads_embeds(
        self,
        ads_input_ids=None,
        ads_attention_mask=None,
        do_norm=True,
    ):
        ads_outputs = self.ads_encoder(ads_input_ids, ads_attention_mask)
        ads_click_embeds = self.ads_click_projection(ads_outputs[:, 0])
        ads_click_embeds = self.ads_click_layer_norm(quick_gelu(ads_click_embeds))

        ads_click_embeds = self.ads_click_final_projection(ads_click_embeds)

        if do_norm:
            ads_click_embeds = ads_click_embeds / ads_click_embeds.norm(
                dim=-1, keepdim=True
            )

        return ads_click_embeds

    @autocast()
    def forward(
        self,
        user_input_ids=None,
        user_attention_mask=None,
        user_num_attention_mask=None,
        tg_ids=None,
        demand_ids=None,
        market_ids=None,
        pos_ids=None,
        ads_input_ids=None,
        ads_attention_mask=None,
    ):
        if not self.training and self.output_user_embeds:
            user_click_embeds = self.get_user_base_embeds(
                user_input_ids=user_input_ids,
                user_attention_mask=user_attention_mask,
                user_num_attention_mask=user_num_attention_mask,
            )
            return EmbeddingOutputs(
                embedding=user_click_embeds,
            )

        if not self.training and self.output_ads_embeds:
            ads_click_embeds = self.get_ads_embeds(
                ads_input_ids=ads_input_ids,
                ads_attention_mask=ads_attention_mask,
            )

            return EmbeddingOutputs(
                embedding=ads_click_embeds,
            )

        user_click_embeds = self.get_user_embeds(
            user_input_ids=user_input_ids,
            user_attention_mask=user_attention_mask,
            user_num_attention_mask=user_num_attention_mask,
            tg_ids=tg_ids,
            demand_ids=demand_ids,
            market_ids=market_ids,
            pos_ids=pos_ids,
        )

        ads_click_embeds = self.get_ads_embeds(
            ads_input_ids=ads_input_ids,
            ads_attention_mask=ads_attention_mask,
        )

        click_scores = torch.sum(
            user_click_embeds * ads_click_embeds, dim=-1, keepdim=True
        )

        click_scores = self.click_classifier(click_scores)
        # click_scores = torch.sigmoid(click_scores)

        return ClassificationOutputs(outputs=click_scores)
