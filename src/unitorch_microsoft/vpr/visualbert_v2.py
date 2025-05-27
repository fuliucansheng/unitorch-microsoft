# Copyright (c) MICROSOFT.
# Licensed under the MIT License.
import os
import torch
import logging
import torch.nn as nn
import torch.distributed as dist
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from torch import autocast
from transformers.models.visual_bert.modeling_visual_bert import (
    VisualBertConfig,
    VisualBertModel,
    VisualBertPreTrainingHeads,
)
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models import GenericModel
from unitorch.models.clip.modeling import AllGather, _clip_loss
from unitorch.cli import (
    add_default_section_for_init,
    add_default_section_for_function,
    register_model,
)
from unitorch.cli.models import ClassificationOutputs, EmbeddingOutputs, LossOutputs
from unitorch.cli.models.visualbert import pretrained_visualbert_infos
from unitorch_microsoft import cached_path


@register_model("microsoft/vpr/pretrain/visualbert/v2")
class VisualBertForPretrainV2(GenericModel):
    def __init__(
        self,
        config_path: str,
        image_embed_dim: Optional[int] = 100,
        projection_dim: Optional[int] = 64,
        logit_scale_init_value: Optional[float] = 2.6592,
        gradient_checkpointing: Optional[bool] = False,
        use_all_gather: Optional[bool] = True,
    ):
        super().__init__()
        self.config = VisualBertConfig.from_json_file(config_path)
        self.config.gradient_checkpointing = gradient_checkpointing
        self.config.visual_embedding_dim = self.config.hidden_size
        self.image_embed_dim = image_embed_dim
        self.image_conv = nn.Linear(image_embed_dim, self.config.hidden_size)
        self.query_bert = VisualBertModel(self.config, add_pooling_layer=False)
        self.projection_dim = projection_dim
        self.query_embed_dim = self.config.hidden_size
        self.offer_embed_dim = image_embed_dim
        self.use_all_gather = use_all_gather
        self.query_projection = nn.Linear(self.query_embed_dim, self.projection_dim)
        self.offer_projection = nn.Linear(
            self.offer_embed_dim,
            self.projection_dim,
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init_value)
        self.init_weights()

    def from_pretrained(self, weight_path):
        weight_path = cached_path(weight_path)
        state_dict = torch.load(weight_path, map_location="cpu")
        _keys = [key for key in state_dict.keys() if key.startswith("visual_bert")]
        for _key in _keys:
            _value = state_dict.pop(_key)
            state_dict["query_bert" + _key[11:]] = _value

        super().from_pretrained(state_dict=state_dict)

    @classmethod
    @add_default_section_for_init("microsoft/vpr/pretrain/visualbert/v2")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/vpr/pretrain/visualbert/v2")
        pretrained_name = config.getoption("pretrained_name", "visualbert-vqa-coco-pre")
        config_path = config.getoption("config_path", None)

        config_path = pop_value(
            config_path,
            nested_dict_value(
                pretrained_visualbert_infos,
                pretrained_name,
                "config",
            ),
        )

        config_path = cached_path(config_path)
        image_embed_dim = config.getoption("image_embed_dim", 64)
        projection_dim = config.getoption("projection_dim", 64)
        logit_scale_init_value = config.getoption("logit_scale_init_value", 2.6592)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)
        use_all_gather = config.getoption("use_all_gather", True)

        inst = cls(
            config_path,
            image_embed_dim=image_embed_dim,
            projection_dim=projection_dim,
            logit_scale_init_value=logit_scale_init_value,
            gradient_checkpointing=gradient_checkpointing,
            use_all_gather=use_all_gather,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(
                pretrained_visualbert_infos,
                pretrained_name,
                "weight",
            ),
            check_none=False,
        )
        if weight_path is not None:
            inst.from_pretrained(weight_path)

        return inst

    def all_gather(self, input):
        output = AllGather.apply(input)
        output = output.view(-1, *(output.shape[2:]))
        return output

    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(
        self,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        query_token_type_ids: torch.Tensor,
        query_position_ids: Optional[torch.Tensor] = None,
        query_image_embeds: Optional[torch.Tensor] = None,
        query_image_attention_mask: Optional[torch.Tensor] = None,
        query_image_token_type_ids: Optional[torch.Tensor] = None,
        offer_image_embeds: Optional[torch.Tensor] = None,
    ):
        query_image_embeds = self.image_conv(query_image_embeds)
        query_outputs = self.query_bert(
            query_input_ids,
            attention_mask=query_attention_mask,
            token_type_ids=query_token_type_ids,
            position_ids=query_position_ids,
            visual_embeds=query_image_embeds,
            visual_attention_mask=query_image_attention_mask,
            visual_token_type_ids=query_image_token_type_ids,
        )
        query_embeds = self.query_projection(query_outputs[0][:, 0])

        offer_image_embeds = offer_image_embeds[:, 0]
        offer_embeds = self.offer_projection(offer_image_embeds)

        query_embeds = query_embeds / query_embeds.norm(dim=-1, keepdim=True)
        offer_embeds = offer_embeds / offer_embeds.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        if self.use_all_gather and dist.is_initialized():
            query_embeds = self.all_gather(query_embeds)
            offer_embeds = self.all_gather(offer_embeds)
        logits = torch.matmul(offer_embeds, query_embeds.t()) * logit_scale

        loss = _clip_loss(logits)

        return LossOutputs(loss=loss)


@register_model("microsoft/vpr/classification/visualbert/v2")
class VisualBertForClassificationV2(GenericModel):
    def __init__(
        self,
        config_path: str,
        image_embed_dim: Optional[int] = 100,
        projection_dim: Optional[int] = 64,
        freeze_base_model: Optional[bool] = False,
        gradient_checkpointing: Optional[bool] = False,
        output_query_embed: Optional[bool] = False,
        output_offer_embed: Optional[bool] = False,
    ):
        super().__init__()
        self.config = VisualBertConfig.from_json_file(config_path)
        self.config.gradient_checkpointing = gradient_checkpointing
        self.config.visual_embedding_dim = self.config.hidden_size
        self.image_embed_dim = image_embed_dim

        self.output_query_embed = output_query_embed
        self.output_offer_embed = output_offer_embed

        self.image_conv = nn.Linear(image_embed_dim, self.config.hidden_size)
        self.query_bert = VisualBertModel(self.config, add_pooling_layer=False)
        self.projection_dim = projection_dim
        self.query_embed_dim = self.config.hidden_size
        self.offer_embed_dim = image_embed_dim
        self.query_projection = nn.Linear(self.query_embed_dim, self.projection_dim)
        self.offer_projection = nn.Linear(
            self.offer_embed_dim,
            self.projection_dim,
        )
        self.classifier = nn.Linear(1, 1)

        self.init_weights()
        self.classifier.weight.data.fill_(5.0)

        if freeze_base_model:
            for p in self.image_conv.parameters():
                p.requires_grad = False

            for p in self.query_bert.parameters():
                p.requires_grad = False

    def from_pretrained(self, weight_path):
        weight_path = cached_path(weight_path)
        state_dict = torch.load(weight_path, map_location="cpu")
        _keys = [key for key in state_dict.keys() if key.startswith("visual_bert")]
        for _key in _keys:
            _value = state_dict.pop(_key)
            state_dict["query_bert" + _key[11:]] = _value

        super().from_pretrained(state_dict=state_dict)

    @classmethod
    @add_default_section_for_init("microsoft/vpr/classification/visualbert/v2")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/vpr/classification/visualbert/v2")
        pretrained_name = config.getoption("pretrained_name", "visualbert-vqa-coco-pre")
        config_path = config.getoption("config_path", None)

        config_path = pop_value(
            config_path,
            nested_dict_value(
                pretrained_visualbert_infos,
                pretrained_name,
                "config",
            ),
        )

        config_path = cached_path(config_path)
        image_embed_dim = config.getoption("image_embed_dim", 64)
        projection_dim = config.getoption("projection_dim", 64)
        freeze_base_model = config.getoption("freeze_base_model", False)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)

        output_query_embed = config.getoption("output_query_embed", False)
        output_offer_embed = config.getoption("output_offer_embed", False)

        inst = cls(
            config_path,
            image_embed_dim=image_embed_dim,
            projection_dim=projection_dim,
            freeze_base_model=freeze_base_model,
            gradient_checkpointing=gradient_checkpointing,
            output_query_embed=output_query_embed,
            output_offer_embed=output_offer_embed,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(
                pretrained_visualbert_infos,
                pretrained_name,
                "weight",
            ),
            check_none=False,
        )
        if weight_path is not None:
            inst.from_pretrained(weight_path)

        return inst

    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(
        self,
        query_input_ids: Optional[torch.Tensor] = None,
        query_attention_mask: Optional[torch.Tensor] = None,
        query_token_type_ids: Optional[torch.Tensor] = None,
        query_position_ids: Optional[torch.Tensor] = None,
        query_image_embeds: Optional[torch.Tensor] = None,
        query_image_attention_mask: Optional[torch.Tensor] = None,
        query_image_token_type_ids: Optional[torch.Tensor] = None,
        offer_image_embeds: Optional[torch.Tensor] = None,
    ):
        if self.output_query_embed:
            query_image_embeds = self.image_conv(query_image_embeds)
            query_outputs = self.query_bert(
                query_input_ids,
                attention_mask=query_attention_mask,
                token_type_ids=query_token_type_ids,
                position_ids=query_position_ids,
                visual_embeds=query_image_embeds,
                visual_attention_mask=query_image_attention_mask,
                visual_token_type_ids=query_image_token_type_ids,
            )
            query_embeds = self.query_projection(query_outputs[0][:, 0])
            query_embeds = query_embeds / query_embeds.norm(dim=-1, keepdim=True)

            return EmbeddingOutputs(embedding=query_embeds)

        if self.output_offer_embed:
            offer_image_embeds = offer_image_embeds[:, 0]
            offer_embeds = self.offer_projection(offer_image_embeds)
            offer_embeds = offer_embeds / offer_embeds.norm(dim=-1, keepdim=True)

            return EmbeddingOutputs(embedding=offer_embeds)

        query_image_embeds = self.image_conv(query_image_embeds)
        query_outputs = self.query_bert(
            query_input_ids,
            attention_mask=query_attention_mask,
            token_type_ids=query_token_type_ids,
            position_ids=query_position_ids,
            visual_embeds=query_image_embeds,
            visual_attention_mask=query_image_attention_mask,
            visual_token_type_ids=query_image_token_type_ids,
        )
        query_embeds = self.query_projection(query_outputs[0][:, 0])
        query_embeds = query_embeds / query_embeds.norm(dim=-1, keepdim=True)

        offer_image_embeds = offer_image_embeds[:, 0]
        offer_embeds = self.offer_projection(offer_image_embeds)
        offer_embeds = offer_embeds / offer_embeds.norm(dim=-1, keepdim=True)

        scores = torch.sum(query_embeds * offer_embeds, dim=-1, keepdim=True)

        outputs = self.classifier(scores)
        # outputs = torch.sigmoid(outputs)

        return ClassificationOutputs(outputs=outputs)
