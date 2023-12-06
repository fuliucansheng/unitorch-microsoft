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


@register_model("microsoft/msan/l1/classification/bletchley/v1/exp2")
class BletchleyForClassification(GenericModel):
    def __init__(
        self,
        config_type,
        projection_dim: int = 32,
        num_tgs: Optional[int] = 50,
        num_demands: Optional[int] = 10,
        num_publisher: Optional[int] = 10,
        num_positions: Optional[int] = 13000,
        freeze_base_model: Optional[bool] = True,
        gradient_checkpointing: Optional[bool] = False,
    ):
        super().__init__()

        text_config = get_bletchley_text_config(config_type, gradient_checkpointing)

        self.projection_dim = projection_dim
        self.text_embed_dim = text_config.hidden_size

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

        self.conv_attn = nn.Linear(text_config.hidden_size, 1, bias=False)
        self.user_conv_projection = nn.Linear(
            self.text_embed_dim,
            self.projection_dim,
        )
        self.user_conv_layer_norm = nn.LayerNorm(self.projection_dim)

        self.ads_click_projection = nn.Linear(
            self.text_embed_dim,
            self.projection_dim,
        )
        self.ads_click_layer_norm = nn.LayerNorm(self.projection_dim)
        self.ads_conv_projection = nn.Linear(
            self.text_embed_dim,
            self.projection_dim,
        )
        self.ads_conv_layer_norm = nn.LayerNorm(self.projection_dim)

        # self.click_position_embedding = nn.Embedding(num_positions, self.projection_dim)
        # self.click_position_layer_norm = nn.LayerNorm(self.projection_dim)
        self.click_tg_embedding = nn.Embedding(num_tgs, self.projection_dim)
        self.click_tg_layer_norm = nn.LayerNorm(self.projection_dim)
        self.click_demand_embedding = nn.Embedding(num_demands, self.projection_dim)
        self.click_demand_layer_norm = nn.LayerNorm(self.projection_dim)
        self.click_publisher_embedding = nn.Embedding(num_publisher, self.projection_dim)
        self.click_publisher_layer_norm = nn.LayerNorm(self.projection_dim)

        self.user_click_final_projection = nn.Linear(
            self.projection_dim * 4,
            self.projection_dim,
        )
        self.ads_click_final_projection = nn.Linear(
            self.projection_dim,
            self.projection_dim,
        )

        # self.conv_position_embedding = nn.Embedding(num_positions, self.projection_dim)
        # self.conv_position_layer_norm = nn.LayerNorm(self.projection_dim)
        self.conv_tg_embedding = nn.Embedding(num_tgs, self.projection_dim)
        self.conv_tg_layer_norm = nn.LayerNorm(self.projection_dim)
        self.conv_demand_embedding = nn.Embedding(num_demands, self.projection_dim)
        self.conv_demand_layer_norm = nn.LayerNorm(self.projection_dim)
        self.conv_publisher_embedding = nn.Embedding(num_publisher, self.projection_dim)
        self.conv_publisher_layer_norm = nn.LayerNorm(self.projection_dim)

        self.user_conv_final_projection = nn.Linear(
            self.projection_dim * 3,
            self.projection_dim,
        )
        self.ads_conv_final_projection = nn.Linear(
            self.projection_dim,
            self.projection_dim,
        )

        self.click_classifier = nn.Linear(1, 1)
        self.conv_classifier = nn.Linear(1, 1)

        self.init_weights()
        self.click_classifier.weight.data.fill_(5.0)
        self.conv_classifier.weight.data.fill_(5.0)

        if freeze_base_model:
            for param in self.user_encoder.parameters():
                param.requires_grad = False
            for param in self.ads_encoder.parameters():
                param.requires_grad = False

    @classmethod
    @add_default_section_for_init("microsoft/msan/l1/classification/bletchley/v1/exp2")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/msan/l1/classification/bletchley/v1/exp2")
        config_type = config.getoption("config_type", "0.15B")
        projection_dim = config.getoption("projection_dim", 32)
        num_tgs = config.getoption("num_tgs", 50)
        num_demands = config.getoption("num_demands", 10)
        num_publisher = config.getoption("num_publisher", 10)
        # num_positions = config.getoption("num_positions", 13000)
        freeze_base_model = config.getoption("freeze_base_model", True)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)

        inst = cls(
            config_type=config_type,
            projection_dim=projection_dim,
            num_tgs=num_tgs,
            num_demands=num_demands,
            num_publisher=num_publisher,
            # num_positions=num_positions,
            freeze_base_model=freeze_base_model,
            gradient_checkpointing=gradient_checkpointing,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        if pretrained_weight_path is not None:
            pretrained_weight_path = cached_path(pretrained_weight_path)
            inst.from_pretrained(pretrained_weight_path)

        return inst

    @autocast()
    def forward(
        self,
        task,
        user_input_ids,
        user_attention_mask,
        user_num_attention_mask,
        tg_ids,
        demand_ids,
        publisher_ids,
        # pos_ids,
        ads_input_ids,
        ads_attention_mask,
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

        conv_attn_score = self.conv_attn(pooled_output).view(batch, num)
        conv_attn_score = conv_attn_score + (1 - user_num_attention_mask) * -10000.0
        conv_attn_score = F.softmax(conv_attn_score, dim=-1)
        user_conv_attn_outputs = torch.bmm(
            conv_attn_score.unsqueeze(1), pooled_output.view(batch, num, -1)
        ).squeeze(1)
        user_conv_embeds = self.user_conv_projection(user_conv_attn_outputs)

        ads_outputs = self.ads_encoder(ads_input_ids, ads_attention_mask)
        ads_click_embeds = self.ads_click_projection(ads_outputs[:, 0])
        ads_conv_embeds = self.ads_conv_projection(ads_outputs[:, 0])

        user_click_embeds = self.user_click_layer_norm(quick_gelu(user_click_embeds))
        ads_click_embeds = self.ads_click_layer_norm(quick_gelu(ads_click_embeds))
        user_conv_embeds = self.user_conv_layer_norm(quick_gelu(user_conv_embeds))
        ads_conv_embeds = self.ads_conv_layer_norm(quick_gelu(ads_conv_embeds))

        # click_pos_emb = self.click_position_layer_norm(
        #     self.click_position_embedding(pos_ids)
        # )
        click_tg_emb = self.click_tg_layer_norm(self.click_tg_embedding(tg_ids))
        click_demand_emb = self.click_demand_layer_norm(
            self.click_demand_embedding(demand_ids)
        )
        click_publisher_emb = self.click_publisher_layer_norm(
            self.click_publisher_embedding(publisher_ids)
        )

        # conv_pos_emb = self.conv_position_layer_norm(
        #     self.conv_position_embedding(pos_ids)
        # )
        conv_tg_emb = self.conv_tg_layer_norm(self.conv_tg_embedding(tg_ids))
        conv_demand_emb = self.conv_demand_layer_norm(
            self.conv_demand_embedding(demand_ids)
        )
        conv_publisher_emb = self.conv_publisher_layer_norm(
            self.conv_publisher_embedding(publisher_ids)
        )

        user_click_embeds = torch.cat(
            [
                user_click_embeds,
                # click_pos_emb,
                click_publisher_emb,
                click_tg_emb,
                click_demand_emb,
            ],
            dim=-1,
        )
        user_conv_embeds = torch.cat(
            [
                user_conv_embeds,
                # conv_pos_emb,
                conv_publisher_emb,               
                conv_tg_emb,
                conv_demand_emb,
            ],
            dim=-1,
        )
        user_click_embeds = self.user_click_final_projection(user_click_embeds)
        user_conv_embeds = self.user_conv_final_projection(user_conv_embeds)

        ads_click_embeds = self.ads_click_final_projection(ads_click_embeds)
        ads_conv_embeds = self.ads_conv_final_projection(ads_conv_embeds)

        user_click_embeds = user_click_embeds / user_click_embeds.norm(
            dim=-1, keepdim=True
        )
        ads_click_embeds = ads_click_embeds / ads_click_embeds.norm(
            dim=-1, keepdim=True
        )

        user_conv_embeds = user_conv_embeds / user_conv_embeds.norm(
            dim=-1, keepdim=True
        )
        ads_conv_embeds = ads_conv_embeds / ads_conv_embeds.norm(dim=-1, keepdim=True)

        click_scores = torch.sum(
            user_click_embeds * ads_click_embeds, dim=-1, keepdim=True
        )
        conv_scores = torch.sum(
            user_conv_embeds * ads_conv_embeds, dim=-1, keepdim=True
        )

        click_scores = self.click_classifier(click_scores)
        # click_scores = torch.sigmoid(click_scores)
        conv_scores = self.conv_classifier(conv_scores)
        # conv_scores = torch.sigmoid(conv_scores)

        if task.dim() == 1:
            task = task.unsqueeze(-1)

        outputs = (task == 1) * click_scores + (task == 0) * conv_scores

        return ClassificationOutputs(outputs=outputs)
