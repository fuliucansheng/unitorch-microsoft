# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import torch
import logging
import torch.nn as nn
import torch.distributed as dist
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from torch.cuda.amp import autocast

from unitorch.utils import pop_value, nested_dict_value
from unitorch.models import GenericModel
from unitorch.models.clip.modeling import AllGather, _clip_loss
from unitorch.cli import (
    add_default_section_for_init,
    add_default_section_for_function,
    register_model,
)
from unitorch.cli.models import ClassificationOutputs, EmbeddingOutputs, LossOutputs
from unitorch.cli.models.bert import pretrained_bert_infos
from unitorch_microsoft import cached_path

from transformers.models.bert.modeling_bert import BertModel, BertConfig


@register_model("microsoft/vpr/pretrain/bert")
class BertForPretrain(GenericModel):
    replace_keys_in_state_dict = {"gamma": "weight", "beta": "bias"}

    def __init__(
        self,
        config_path: str,
        num_hidden_layers: Optional[int] = 3,
        image_embed_dim: Optional[int] = 100,
        logit_scale_init_value: Optional[float] = 2.6592,
        gradient_checkpointing: Optional[bool] = False,
        use_all_gather: Optional[bool] = True,
        use_clip_loss: Optional[bool] = True,
        output_query_embed: Optional[bool] = False,
    ):
        super().__init__()
        self.config = BertConfig.from_json_file(config_path)
        self.config.num_hidden_layers = num_hidden_layers
        self.config.gradient_checkpointing = gradient_checkpointing
        self.image_embed_dim = image_embed_dim
        self.bert = BertModel(self.config, add_pooling_layer=False)
        self.query_projection = nn.Linear(self.config.hidden_size, image_embed_dim)

        self.use_all_gather = use_all_gather
        self.use_clip_loss = use_clip_loss
        self.output_query_embed = output_query_embed
        if self.use_clip_loss:
            self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init_value)
        self.init_weights()

    @classmethod
    @add_default_section_for_init("microsoft/vpr/pretrain/bert")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/vpr/pretrain/bert")
        pretrained_name = config.getoption("pretrained_name", "default-bert")
        config_path = config.getoption("config_path", None)
        num_hidden_layers = config.getoption("num_hidden_layers", 3)
        output_query_embed = config.getoption("output_query_embed", False)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_bert_infos, pretrained_name, "config"),
        )
        config_path = cached_path(config_path)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)
        use_clip_loss = config.getoption("use_clip_loss", True)
        inst = cls(
            config_path=config_path,
            num_hidden_layers=num_hidden_layers,
            gradient_checkpointing=gradient_checkpointing,
            use_clip_loss=use_clip_loss,
            output_query_embed=output_query_embed,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_bert_infos, pretrained_name, "weight"),
            check_none=False,
        )
        if weight_path is not None:
            weight_path = cached_path(weight_path)
            inst.from_pretrained(weight_path)

        return inst

    def all_gather(self, input):
        output = AllGather.apply(input)
        output = output.view(-1, *(output.shape[2:]))
        return output

    @autocast()
    def forward(
        self,
        input_ids: torch.Tensor,
        image_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        if self.output_query_embed:
            query_outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
            )
            query_embeds = self.query_projection(query_outputs[0][:, 0])

            query_embeds = query_embeds / query_embeds.norm(dim=-1, keepdim=True)
            return EmbeddingOutputs(embedding=query_embeds)

        query_outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        query_embeds = self.query_projection(query_outputs[0][:, 0])

        query_embeds = query_embeds / query_embeds.norm(dim=-1, keepdim=True)
        # image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        if self.use_clip_loss:
            logit_scale = self.logit_scale.exp()
            if self.use_all_gather and dist.is_initialized():
                query_embeds = self.all_gather(query_embeds)
                image_embeds = self.all_gather(image_embeds)
            logits = torch.matmul(image_embeds, query_embeds.t()) * logit_scale

            loss = _clip_loss(logits)
        else:
            # mse
            loss = (
                nn.MSELoss(reduction="none")(query_embeds, image_embeds)
                .sum(dim=-1)
                .mean()
            )

        return LossOutputs(loss=loss)


@register_model("microsoft/vpr/pretrain/bert/v2")
class BertForPretrainV2(GenericModel):
    replace_keys_in_state_dict = {"gamma": "weight", "beta": "bias"}

    def __init__(
        self,
        config_path: str,
        num_hidden_layers: Optional[int] = 3,
        image_embed_dim: Optional[int] = 100,
        logit_scale_init_value: Optional[float] = 2.6592,
        gradient_checkpointing: Optional[bool] = False,
        use_all_gather: Optional[bool] = True,
        use_clip_loss: Optional[bool] = True,
        output_query_embed: Optional[bool] = False,
    ):
        super().__init__()
        self.config = BertConfig.from_json_file(config_path)
        self.config.num_hidden_layers = num_hidden_layers
        self.config.gradient_checkpointing = gradient_checkpointing
        self.image_embed_dim = image_embed_dim
        self.bert = BertModel(self.config, add_pooling_layer=False)
        self.query_projection = nn.Linear(self.config.hidden_size, image_embed_dim)
        self.image_projection = nn.Linear(
            self.image_embed_dim,
            self.image_embed_dim,
        )

        self.use_all_gather = use_all_gather
        self.use_clip_loss = use_clip_loss
        self.output_query_embed = output_query_embed
        if self.use_clip_loss:
            self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init_value)
        self.init_weights()

    @classmethod
    @add_default_section_for_init("microsoft/vpr/pretrain/bert/v2")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/vpr/pretrain/bert/v2")
        pretrained_name = config.getoption("pretrained_name", "default-bert")
        config_path = config.getoption("config_path", None)
        num_hidden_layers = config.getoption("num_hidden_layers", 3)
        output_query_embed = config.getoption("output_query_embed", False)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_bert_infos, pretrained_name, "config"),
        )
        config_path = cached_path(config_path)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)
        use_clip_loss = config.getoption("use_clip_loss", True)
        inst = cls(
            config_path=config_path,
            num_hidden_layers=num_hidden_layers,
            gradient_checkpointing=gradient_checkpointing,
            use_clip_loss=use_clip_loss,
            output_query_embed=output_query_embed,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_bert_infos, pretrained_name, "weight"),
            check_none=False,
        )
        if weight_path is not None:
            weight_path = cached_path(weight_path)
            inst.from_pretrained(weight_path)

        return inst

    def all_gather(self, input):
        output = AllGather.apply(input)
        output = output.view(-1, *(output.shape[2:]))
        return output

    @autocast()
    def forward(
        self,
        input_ids: torch.Tensor,
        image_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        if self.output_query_embed:
            query_outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
            )
            query_embeds = self.query_projection(query_outputs[0][:, 0])

            query_embeds = query_embeds / query_embeds.norm(dim=-1, keepdim=True)
            return EmbeddingOutputs(embedding=query_embeds)

        query_outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        query_embeds = self.query_projection(query_outputs[0][:, 0])
        image_embeds = self.image_projection(image_embeds)

        query_embeds = query_embeds / query_embeds.norm(dim=-1, keepdim=True)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        if self.use_clip_loss:
            logit_scale = self.logit_scale.exp()
            if self.use_all_gather and dist.is_initialized():
                query_embeds = self.all_gather(query_embeds)
                image_embeds = self.all_gather(image_embeds)
            logits = torch.matmul(image_embeds, query_embeds.t()) * logit_scale

            loss = _clip_loss(logits)
        else:
            # mse
            loss = (
                nn.MSELoss(reduction="none")(query_embeds, image_embeds)
                .sum(dim=-1)
                .mean()
            )

        return LossOutputs(loss=loss)


@register_model("microsoft/vpr/classification/bert")
class BertForClassification(GenericModel):
    replace_keys_in_state_dict = {"gamma": "weight", "beta": "bias"}

    def __init__(
        self,
        config_path: str,
        num_hidden_layers: Optional[int] = 3,
        image_embed_dim: Optional[int] = 100,
        freeze_base_model: Optional[bool] = False,
        gradient_checkpointing: Optional[bool] = False,
        output_query_embed: Optional[bool] = False,
    ):
        super().__init__()
        self.config = BertConfig.from_json_file(config_path)
        self.config.num_hidden_layers = num_hidden_layers
        self.config.gradient_checkpointing = gradient_checkpointing
        self.image_embed_dim = image_embed_dim
        self.bert = BertModel(self.config, add_pooling_layer=False)
        self.query_projection = nn.Linear(self.config.hidden_size, image_embed_dim)

        self.output_query_embed = output_query_embed
        self.classifier = nn.Linear(1, 1)

        self.init_weights()
        self.classifier.weight.data.fill_(5.0)

        if freeze_base_model:
            for param in self.bert.parameters():
                param.requires_grad = False

    @classmethod
    @add_default_section_for_init("microsoft/vpr/classification/bert")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/vpr/classification/bert")
        pretrained_name = config.getoption("pretrained_name", "default-bert")
        config_path = config.getoption("config_path", None)
        num_hidden_layers = config.getoption("num_hidden_layers", 12)
        output_query_embed = config.getoption("output_query_embed", False)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_bert_infos, pretrained_name, "config"),
        )
        config_path = cached_path(config_path)
        freeze_base_model = config.getoption("freeze_base_model", False)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)
        inst = cls(
            config_path=config_path,
            num_hidden_layers=num_hidden_layers,
            freeze_base_model=freeze_base_model,
            gradient_checkpointing=gradient_checkpointing,
            output_query_embed=output_query_embed,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_bert_infos, pretrained_name, "weight"),
            check_none=False,
        )
        if weight_path is not None:
            weight_path = cached_path(weight_path)
            inst.from_pretrained(weight_path)

        return inst

    @autocast()
    def forward(
        self,
        input_ids: torch.Tensor,
        image_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        if self.output_query_embed:
            query_outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
            )
            query_embeds = self.query_projection(query_outputs[0][:, 0])

            query_embeds = query_embeds / query_embeds.norm(dim=-1, keepdim=True)
            return EmbeddingOutputs(embedding=query_embeds)

        query_outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        query_embeds = self.query_projection(query_outputs[0][:, 0])

        query_embeds = query_embeds / query_embeds.norm(dim=-1, keepdim=True)
        # image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

        scores = torch.sum(query_embeds * image_embeds, dim=-1, keepdim=True)
        outputs = self.classifier(scores)

        return ClassificationOutputs(outputs=outputs)


@register_model("microsoft/vpr/classification/bert/v2")
class BertForClassificationV2(GenericModel):
    replace_keys_in_state_dict = {"gamma": "weight", "beta": "bias"}

    def __init__(
        self,
        config_path: str,
        num_hidden_layers: Optional[int] = 3,
        image_embed_dim: Optional[int] = 100,
        freeze_base_model: Optional[bool] = False,
        gradient_checkpointing: Optional[bool] = False,
        output_query_embed: Optional[bool] = False,
        output_image_embed: Optional[bool] = False,
    ):
        super().__init__()
        self.config = BertConfig.from_json_file(config_path)
        self.config.num_hidden_layers = num_hidden_layers
        self.config.gradient_checkpointing = gradient_checkpointing
        self.image_embed_dim = image_embed_dim
        self.bert = BertModel(self.config, add_pooling_layer=False)
        self.query_projection = nn.Linear(self.config.hidden_size, image_embed_dim)
        self.image_projection = nn.Linear(
            self.image_embed_dim,
            self.image_embed_dim,
        )

        self.output_query_embed = output_query_embed
        self.output_image_embed = output_image_embed
        self.classifier = nn.Linear(1, 1)

        self.init_weights()
        self.classifier.weight.data.fill_(5.0)

        if freeze_base_model:
            for param in self.bert.parameters():
                param.requires_grad = False

    @classmethod
    @add_default_section_for_init("microsoft/vpr/classification/bert/v2")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/vpr/classification/bert/v2")
        pretrained_name = config.getoption("pretrained_name", "default-bert")
        config_path = config.getoption("config_path", None)
        num_hidden_layers = config.getoption("num_hidden_layers", 12)
        output_query_embed = config.getoption("output_query_embed", False)
        output_image_embed = config.getoption("output_image_embed", False)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_bert_infos, pretrained_name, "config"),
        )
        config_path = cached_path(config_path)
        freeze_base_model = config.getoption("freeze_base_model", False)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)
        inst = cls(
            config_path=config_path,
            num_hidden_layers=num_hidden_layers,
            freeze_base_model=freeze_base_model,
            gradient_checkpointing=gradient_checkpointing,
            output_query_embed=output_query_embed,
            output_image_embed=output_image_embed,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_bert_infos, pretrained_name, "weight"),
            check_none=False,
        )
        if weight_path is not None:
            weight_path = cached_path(weight_path)
            inst.from_pretrained(weight_path)

        return inst

    @autocast()
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        if self.output_query_embed:
            query_outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
            )
            query_embeds = self.query_projection(query_outputs[0][:, 0])

            query_embeds = query_embeds / query_embeds.norm(dim=-1, keepdim=True)
            return EmbeddingOutputs(embedding=query_embeds)

        if self.output_image_embed:
            image_embeds = self.image_projection(image_embeds)
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            return EmbeddingOutputs(embedding=image_embeds)

        query_outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        query_embeds = self.query_projection(query_outputs[0][:, 0])
        image_embeds = self.image_projection(image_embeds)

        query_embeds = query_embeds / query_embeds.norm(dim=-1, keepdim=True)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

        scores = torch.sum(query_embeds * image_embeds, dim=-1, keepdim=True)
        outputs = self.classifier(scores)

        return ClassificationOutputs(outputs=outputs)
