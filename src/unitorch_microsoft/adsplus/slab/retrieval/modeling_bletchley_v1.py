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

from unitorch_microsoft.models.bletchley.modeling_v1 import (
    get_bletchley_text_config,
    get_bletchley_image_config,
    BletchleyTextEncoder,
)
from unitorch.models import GenericModel
from unitorch.cli import (
    add_default_section_for_init,
    add_default_section_for_function,
    register_model,
)
from unitorch.models.clip.modeling import AllGather, _clip_loss
from unitorch.cli.models import (
    ClassificationOutputs,
    LossOutputs,
    EmbeddingOutputs,
)


# note: uilr == bletchley is true
# and this retrieval model is similar with MMDNN bletchley model, but have no ice_ids
@register_model("microsoft/adsplus/slab/retrieval/text_bletchleyv1")
class BletchleyV1ForSLABRetrieval(GenericModel):
    def __init__(
        self,
        query_config_type: str,
        offer_config_type: str,
        max_position_embeddings_query: Optional[int] = 128,
        projection_dim: Optional[int] = 288,
        num_class: Optional[int] = 1,
        output_hidden_dim: Optional[int] = 64,
        freeze_base_model: Optional[bool] = True,
        gradient_checkpointing: Optional[bool] = False,
        output_query_embed: Optional[bool] = False,
        output_offer_embed: Optional[bool] = False,
    ):
        super().__init__()
        query_config = get_bletchley_text_config(
            query_config_type, gradient_checkpointing
        )
        text_config = get_bletchley_text_config(
            offer_config_type, gradient_checkpointing
        )
        query_config.max_position_embeddings = max_position_embeddings_query

        self.projection_dim = projection_dim
        self.query_embed_dim = query_config.hidden_size
        self.text_embed_dim = text_config.hidden_size

        self.output_query_embed = output_query_embed
        self.output_offer_embed = output_offer_embed

        self.query_encoder = BletchleyTextEncoder(
            query_config, add_projection_layer=False
        )
        self.offer_encoder = BletchleyTextEncoder(
            text_config, add_projection_layer=False
        )

        self.query_projection = nn.Linear(self.query_embed_dim, self.projection_dim)
        self.offer_projection = nn.Linear(self.text_embed_dim, self.projection_dim)

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
    @add_default_section_for_init("microsoft/adsplus/slab/retrieval/text_bletchleyv1")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/adsplus/slab/retrieval/text_bletchleyv1")
        query_config_type = config.getoption("query_config_type", "0.3B")
        offer_config_type = config.getoption("offer_config_type", "0.3B")
        max_position_embeddings_query = config.getoption(
            "max_position_embeddings_query", 128
        )
        projection_dim = config.getoption("projection_dim", 288)
        num_class = config.getoption("num_class", 1)
        output_hidden_dim = config.getoption("output_hidden_dim", 64)
        freeze_base_model = config.getoption("freeze_base_model", True)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)
        output_query_embed = config.getoption("output_query_embed", False)
        output_offer_embed = config.getoption("output_offer_embed", False)

        inst = cls(
            query_config_type=query_config_type,
            offer_config_type=offer_config_type,
            max_position_embeddings_query=max_position_embeddings_query,
            projection_dim=projection_dim,
            num_class=num_class,
            output_hidden_dim=output_hidden_dim,
            freeze_base_model=freeze_base_model,
            gradient_checkpointing=gradient_checkpointing,
            output_query_embed=output_query_embed,
            output_offer_embed=output_offer_embed,
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

        _self_state_dict = self.state_dict()
        state_dict = {
            k: v
            for k, v in state_dict.items()
            if k in _self_state_dict and v.shape == _self_state_dict[k].shape
        }

        _load_percent = len(state_dict) / len(_self_state_dict) * 100
        self.load_state_dict(state_dict, False)
        logging.info(
            f"{self.__class__.__name__} model load weight ({int(_load_percent)}%) from pretrain {weight_path}"
        )

    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(
        self,
        query_input_ids=None,
        query_attention_mask=None,
        offer_input_ids=None,
        offer_attention_mask=None,
    ):
        if not self.training and self.output_query_embed:
            query_outputs = self.query_encoder(
                input_ids=query_input_ids, attention_mask=query_attention_mask
            )
            query_embeds = query_outputs[:, 0]
            query_embeds = self.query_projection(query_embeds)
            query_embeds = self.query_layer_norm(quick_gelu(query_embeds))
            query_embeds = self.final_query_projection(quick_gelu(query_embeds))
            query_embeds = query_embeds / query_embeds.norm(dim=-1, keepdim=True)

            return EmbeddingOutputs(embedding=query_embeds)

        if not self.training and self.output_offer_embed:
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
        offer_outputs = self.offer_encoder(
            input_ids=offer_input_ids, attention_mask=offer_attention_mask
        )

        query_embeds = query_outputs[:, 0]
        query_embeds = self.query_projection(query_embeds)

        offer_embeds = offer_outputs[:, 0]
        offer_embeds = self.offer_projection(offer_embeds)

        query_embeds = self.query_layer_norm(quick_gelu(query_embeds))
        offer_embeds = self.offer_layer_norm(quick_gelu(offer_embeds))

        query_embeds = self.final_query_projection(quick_gelu(query_embeds))
        offer_embeds = self.final_offer_projection(quick_gelu(offer_embeds))

        query_embeds = query_embeds / query_embeds.norm(dim=-1, keepdim=True)
        offer_embeds = offer_embeds / offer_embeds.norm(dim=-1, keepdim=True)

        scores = torch.sum(query_embeds * offer_embeds, dim=-1, keepdim=True)

        outputs = self.classifier(scores)

        return ClassificationOutputs(outputs=outputs)


# loss is pairwise loss
@register_model("microsoft/adsplus/slab/retrieval/text_bletchleyv1/pairwise")
class BletchleyV1ForSLABRetrievalPairwise(GenericModel):
    def __init__(
        self,
        query_config_type: str,
        offer_config_type: str,
        max_position_embeddings_query: Optional[int] = 128,
        projection_dim: Optional[int] = 288,
        num_class: Optional[int] = 1,
        output_hidden_dim: Optional[int] = 64,
        freeze_base_model: Optional[bool] = True,
        gradient_checkpointing: Optional[bool] = False,
        output_query_embed: Optional[bool] = False,
        output_offer_embed: Optional[bool] = False,
        loss_type: Optional[str] = "bpr",
        hinge_margin: Optional[float] = 0.5,
        logit_scale_init_value: Optional[float] = 2.6592,
        use_all_gather: Optional[bool] = True,
        random_negative: Optional[int] = 3,
        tp: Optional[float] = 1.0,
    ):
        super().__init__()
        query_config = get_bletchley_text_config(
            query_config_type, gradient_checkpointing
        )
        text_config = get_bletchley_text_config(
            offer_config_type, gradient_checkpointing
        )
        query_config.max_position_embeddings = max_position_embeddings_query

        self.projection_dim = projection_dim
        self.query_embed_dim = query_config.hidden_size
        self.text_embed_dim = text_config.hidden_size

        self.output_query_embed = output_query_embed
        self.output_offer_embed = output_offer_embed

        self.loss_type = loss_type
        self.hinge_margin = hinge_margin
        self.use_all_gather = use_all_gather
        if self.loss_type == "bpr_1":
            self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init_value)
        self.random_negative = random_negative
        self.tp = tp

        self.query_encoder = BletchleyTextEncoder(
            query_config, add_projection_layer=False
        )
        self.offer_encoder = BletchleyTextEncoder(
            text_config, add_projection_layer=False
        )

        self.query_projection = nn.Linear(self.query_embed_dim, self.projection_dim)
        self.offer_projection = nn.Linear(self.text_embed_dim, self.projection_dim)

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

        if freeze_base_model:
            for p in self.query_encoder.parameters():
                p.requires_grad = False

            for p in self.offer_encoder.parameters():
                p.requires_grad = False

    @classmethod
    @add_default_section_for_init(
        "microsoft/adsplus/slab/retrieval/text_bletchleyv1/pairwise"
    )
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section(
            "microsoft/adsplus/slab/retrieval/text_bletchleyv1/pairwise"
        )
        query_config_type = config.getoption("query_config_type", "0.3B")
        offer_config_type = config.getoption("offer_config_type", "0.3B")
        max_position_embeddings_query = config.getoption(
            "max_position_embeddings_query", 128
        )
        projection_dim = config.getoption("projection_dim", 512)
        num_class = config.getoption("num_class", 1)
        output_hidden_dim = config.getoption("output_hidden_dim", 64)
        freeze_base_model = config.getoption("freeze_base_model", True)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)
        output_query_embed = config.getoption("output_query_embed", False)
        output_offer_embed = config.getoption("output_offer_embed", False)

        loss_type = config.getoption("loss_type", "bpr")
        hinge_margin = config.getoption("hinge_margin", 0.5)
        logit_scale_init_value = config.getoption("logit_scale_init_value", 2.6592)
        use_all_gather = config.getoption("use_all_gather", False)
        random_negative = config.getoption("random_negative", 32)
        tp = config.getoption("tp", 10.0)

        inst = cls(
            query_config_type=query_config_type,
            offer_config_type=offer_config_type,
            max_position_embeddings_query=max_position_embeddings_query,
            projection_dim=projection_dim,
            num_class=num_class,
            output_hidden_dim=output_hidden_dim,
            freeze_base_model=freeze_base_model,
            gradient_checkpointing=gradient_checkpointing,
            output_query_embed=output_query_embed,
            output_offer_embed=output_offer_embed,
            loss_type=loss_type,
            hinge_margin=hinge_margin,
            logit_scale_init_value=logit_scale_init_value,
            use_all_gather=use_all_gather,
            random_negative=random_negative,
            tp=tp,
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
        query_input_ids=None,
        query_attention_mask=None,
        offer_input_ids=None,
        offer_attention_mask=None,
    ):
        if not self.training and self.output_query_embed:
            query_outputs = self.query_encoder(
                input_ids=query_input_ids, attention_mask=query_attention_mask
            )
            query_embeds = query_outputs[:, 0]
            query_embeds = self.query_projection(query_embeds)
            query_embeds = self.query_layer_norm(quick_gelu(query_embeds))
            query_embeds = self.final_query_projection(quick_gelu(query_embeds))
            query_embeds = query_embeds / query_embeds.norm(dim=-1, keepdim=True)

            return EmbeddingOutputs(embedding=query_embeds)

        if not self.training and self.output_offer_embed:
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
        offer_outputs = self.offer_encoder(
            input_ids=offer_input_ids, attention_mask=offer_attention_mask
        )

        query_embeds = query_outputs[:, 0]
        query_embeds = self.query_projection(query_embeds)

        offer_embeds = offer_outputs[:, 0]
        offer_embeds = self.offer_projection(offer_embeds)

        query_embeds = self.query_layer_norm(quick_gelu(query_embeds))
        offer_embeds = self.offer_layer_norm(quick_gelu(offer_embeds))

        query_embeds = self.final_query_projection(quick_gelu(query_embeds))
        offer_embeds = self.final_offer_projection(quick_gelu(offer_embeds))

        query_embeds = query_embeds / query_embeds.norm(dim=-1, keepdim=True)
        offer_embeds = offer_embeds / offer_embeds.norm(dim=-1, keepdim=True)

        if self.loss_type == "bpr":
            if self.use_all_gather:
                query_embeds = self.all_gather(query_embeds)
                offer_embeds = self.all_gather(offer_embeds)
            y_batch_rn_ori = (
                torch.matmul(query_embeds, offer_embeds.t()) * self.tp
            )  # bs*bs

            # reorg
            bs = y_batch_rn_ori.shape[0]
            positive_eye = torch.eye(bs).to(
                y_batch_rn_ori.device
            )  # positive sample is on diagonal
            positive_eye_not = positive_eye == 0
            positive_eye = positive_eye > 0
            y_batch_rn_ori_positive = torch.masked_select(
                y_batch_rn_ori, positive_eye
            ).view(bs, -1)
            y_batch_rn_ori_negative = torch.masked_select(
                y_batch_rn_ori, positive_eye_not
            ).view(bs, -1)
            y_batch_rn_ori = torch.cat(
                [y_batch_rn_ori_positive, y_batch_rn_ori_negative], dim=-1
            )

            # sample
            if self.random_negative > bs - 1:
                self.random_negative = bs - 1
            mask = torch.zeros(bs, bs, dtype=torch.float).to(y_batch_rn_ori.device)
            mask[:, 0] = 1
            for i in range(bs):
                mask[i, random.sample(range(1, bs), self.random_negative)] = 1
            mask = mask > 0
            y_batch_rn_ori = torch.masked_select(y_batch_rn_ori, mask).view(bs, -1)

            loss = 0
            for i in range(1, y_batch_rn_ori.shape[1]):
                loss_i = torch.log(
                    1 + torch.exp(y_batch_rn_ori[:, i] - y_batch_rn_ori[:, 0])
                )
                loss += loss_i.sum()
            loss = loss / ((y_batch_rn_ori.shape[1] - 1) * y_batch_rn_ori.shape[0])
        elif self.loss_type == "pwbce":
            if self.use_all_gather:
                query_embeds = self.all_gather(query_embeds)
                offer_embeds = self.all_gather(offer_embeds)
            y_batch_rn_ori = (
                torch.matmul(query_embeds, offer_embeds.t()) * self.tp
            )  # bs*bs

            # reorg
            bs = y_batch_rn_ori.shape[0]
            positive_eye = torch.eye(bs).to(
                y_batch_rn_ori.device
            )  # positive sample is on diagonal
            positive_eye_not = positive_eye == 0
            positive_eye = positive_eye > 0
            y_batch_rn_ori_positive = torch.masked_select(
                y_batch_rn_ori, positive_eye
            ).view(bs, -1)
            y_batch_rn_ori_negative = torch.masked_select(
                y_batch_rn_ori, positive_eye_not
            ).view(bs, -1)
            y_batch_rn_ori = torch.cat(
                [y_batch_rn_ori_positive, y_batch_rn_ori_negative], dim=-1
            )

            # sample
            if self.random_negative > bs - 1:
                self.random_negative = bs - 1
            mask = torch.zeros(bs, bs, dtype=torch.float).to(y_batch_rn_ori.device)
            mask[:, 0] = 1
            for i in range(bs):
                mask[i, random.sample(range(1, bs), self.random_negative)] = 1
            mask = mask > 0
            y_batch_rn_ori = torch.masked_select(y_batch_rn_ori, mask).view(bs, -1)

            y_batch_final = F.softmax(
                y_batch_rn_ori, dim=1
            )  # this support random negative only, need input to be all positive
            target_final = torch.zeros_like(y_batch_final)
            target_final[:, 0] = 1
            loss = nn.BCEWithLogitsLoss()(
                y_batch_final.view(-1, 1), target_final.view(-1, 1)
            )
        elif self.loss_type == "hinge":
            if self.use_all_gather:
                query_embeds = self.all_gather(query_embeds)
                offer_embeds = self.all_gather(offer_embeds)
            y_batch_rn_ori = (
                torch.matmul(query_embeds, offer_embeds.t()) * self.tp
            )  # bs*bs

            # reorg
            bs = y_batch_rn_ori.shape[0]
            positive_eye = torch.eye(bs).to(
                y_batch_rn_ori.device
            )  # positive sample is on diagonal
            positive_eye_not = positive_eye == 0
            positive_eye = positive_eye > 0
            y_batch_rn_ori_positive = torch.masked_select(
                y_batch_rn_ori, positive_eye
            ).view(bs, -1)
            y_batch_rn_ori_negative = torch.masked_select(
                y_batch_rn_ori, positive_eye_not
            ).view(bs, -1)
            y_batch_rn_ori = torch.cat(
                [y_batch_rn_ori_positive, y_batch_rn_ori_negative], dim=-1
            )

            # sample
            if self.random_negative > bs - 1:
                self.random_negative = bs - 1
            mask = torch.zeros(bs, bs, dtype=torch.float).to(y_batch_rn_ori.device)
            mask[:, 0] = 1
            for i in range(bs):
                mask[i, random.sample(range(1, bs), self.random_negative)] = 1
            mask = mask > 0
            y_batch_rn_ori = torch.masked_select(y_batch_rn_ori, mask).view(bs, -1)

            loss = 0
            for i in range(1, y_batch_rn_ori.shape[1]):
                loss_i = self.hinge_margin - y_batch_rn_ori[:, 0] + y_batch_rn_ori[:, i]
                loss_less = loss_i < 0
                loss_i[loss_less] = 0
                loss += loss_i.sum()
            loss = loss / ((y_batch_rn_ori.shape[1] - 1) * y_batch_rn_ori.shape[0])

        elif self.loss_type == "bpr_1":
            # bpr loss from decu
            logit_scale = self.logit_scale.exp()
            if self.use_all_gather:
                query_embeds = self.all_gather(query_embeds)
                offer_embeds = self.all_gather(offer_embeds)

            logits1 = torch.sum(query_embeds * offer_embeds, dim=-1) * logit_scale
            logits2 = torch.matmul(query_embeds, offer_embeds.t()) * logit_scale

            mask = torch.eye(logits1.size(0)).to(logits2.device)
            bs = logits2.shape[0]
            # sample
            if self.random_negative > bs - 1:
                self.random_negative = bs - 1
            elif self.random_negative <= 0:
                self.random_negative = 1
            for i in range(bs):
                mask[i, random.sample(range(bs), bs - self.random_negative)] = 1
            mask = 1 - mask
            loss = torch.log(torch.exp(logits2.t() - logits1).t() + 1) * mask
            loss = loss.sum() / mask.sum()

        return LossOutputs(loss=loss)


@register_model("microsoft/adsplus/slab/retrieval/pretrain/text_bletchleyv1")
class BletchleyV1ForSLABRetrievalPretrain(GenericModel):
    def __init__(
        self,
        query_config_type: str,
        offer_config_type: str,
        query_max_position_embeddings: Optional[int] = 128,
        projection_dim: Optional[int] = 512,
        output_hidden_dim: Optional[int] = 64,
        freeze_base_model: Optional[bool] = True,
        logit_scale_init_value: Optional[float] = 2.6592,
        gradient_checkpointing: Optional[bool] = False,
        use_all_gather: Optional[bool] = True,
    ):
        super().__init__()
        query_config = get_bletchley_text_config(
            query_config_type, gradient_checkpointing
        )
        text_config = get_bletchley_text_config(
            offer_config_type, gradient_checkpointing
        )
        query_config.max_position_embeddings = query_max_position_embeddings

        self.projection_dim = projection_dim
        self.query_embed_dim = query_config.hidden_size
        self.text_embed_dim = text_config.hidden_size

        self.use_all_gather = use_all_gather

        self.query_encoder = BletchleyTextEncoder(
            query_config, add_projection_layer=False
        )
        self.offer_encoder = BletchleyTextEncoder(
            text_config, add_projection_layer=False
        )

        self.query_projection = nn.Linear(self.query_embed_dim, self.projection_dim)
        self.offer_projection = nn.Linear(self.text_embed_dim, self.projection_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init_value)

        self.init_weights()

        if freeze_base_model:
            for p in self.query_encoder.parameters():
                p.requires_grad = False

            for p in self.offer_encoder.parameters():
                p.requires_grad = False

    @classmethod
    @add_default_section_for_init(
        "microsoft/adsplus/slab/retrieval/pretrain/text_bletchleyv1"
    )
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section(
            "microsoft/adsplus/slab/retrieval/pretrain/text_bletchleyv1"
        )
        query_config_type = config.getoption("query_config_type", "0.3B")
        offer_config_type = config.getoption("offer_config_type", "0.3B")
        query_max_position_embeddings = config.getoption(
            "query_max_position_embeddings", 128
        )
        projection_dim = config.getoption("projection_dim", 512)
        output_hidden_dim = config.getoption("output_hidden_dim", 64)
        freeze_base_model = config.getoption("freeze_base_model", True)
        logit_scale_init_value = config.getoption("logit_scale_init_value", 2.6592)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)
        use_all_gather = config.getoption("use_all_gather", False)

        inst = cls(
            query_config_type=query_config_type,
            offer_config_type=offer_config_type,
            query_max_position_embeddings=query_max_position_embeddings,
            projection_dim=projection_dim,
            output_hidden_dim=output_hidden_dim,
            freeze_base_model=freeze_base_model,
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
        query_input_ids=None,
        query_attention_mask=None,
        offer_input_ids=None,
        offer_attention_mask=None,
    ):
        query_outputs = self.query_encoder(
            input_ids=query_input_ids, attention_mask=query_attention_mask
        )
        offer_outputs = self.offer_encoder(
            input_ids=offer_input_ids, attention_mask=offer_attention_mask
        )

        query_embeds = query_outputs[:, 0]
        query_embeds = self.query_projection(query_embeds)

        offer_embeds = offer_outputs[:, 0]
        offer_embeds = self.offer_projection(offer_embeds)

        query_embeds = query_embeds / query_embeds.norm(dim=-1, keepdim=True)
        offer_embeds = offer_embeds / offer_embeds.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        if self.use_all_gather:
            query_embeds = self.all_gather(query_embeds)
            offer_embeds = self.all_gather(offer_embeds)
        logits = torch.matmul(query_embeds, offer_embeds.t()) * logit_scale
        loss = _clip_loss(logits)

        return LossOutputs(loss=loss)
