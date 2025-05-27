# Copyright (c) MICROSOFT.
# Licensed under the MIT License.
import os
import random
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torch import autocast
from transformers import PreTrainedModel, XLMRobertaConfig
from transformers.activations import quick_gelu
from transformers.models.roberta.modeling_roberta import (
    RobertaEmbeddings,
    RobertaEncoder as RobertaPostLNEncoder,
)
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
    cached_path,
    register_model,
)
from unitorch_microsoft.models.bletchley.roberta import (
    RobertaEncoder as RobertaPreLNEncoder,
)

from unitorch_microsoft.models.bletchley.modeling_v1 import (
    get_bletchley_text_config,
    BletchleyTextEncoder,
)
from pytorch_metric_learning.miners import TripletMarginMiner
from pytorch_metric_learning.losses import TripletMarginLoss
from unitorch.losses import BCELoss, MSELoss


class TripLoss(nn.Module):
    def __init__(self, weight, margin):
        super().__init__()
        self.miner = TripletMarginMiner(margin, "semihard")
        self.loss_func = TripletMarginLoss(margin)
        self.weight = weight

    def forward(self, visual_embedding, text_embedding):
        label = torch.ones(visual_embedding.size()[0], dtype=torch.long).to(
            visual_embedding.device
        )
        idx = torch.arange(0, visual_embedding.size()[0]).to(visual_embedding.device)
        label = torch.cat([label, label])
        embed = torch.cat([text_embedding, visual_embedding], dim=0)[label != 0]
        trip_label = torch.cat([idx, idx], dim=0).view(-1)[label != 0]
        if embed.shape[0] == 0:
            return 0.0 * torch.sum(visual_embedding)
        miner_output = self.miner(embed, trip_label)
        loss = self.loss_func(embed, trip_label, miner_output)
        return self.weight * loss


class MatchLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def get_pair(self, logits):
        # reorg
        batch_size = logits.shape[0]
        masks = torch.eye(batch_size).to(logits.device).bool()
        positive_samples = torch.masked_select(logits, masks).view(batch_size, -1)
        positive_labels = torch.ones(positive_samples.size()[0], dtype=torch.float).to(
            logits.device
        )
        negative_samples = torch.masked_select(logits, ~masks).view(batch_size, -1)
        negative_probs = F.softmax(negative_samples + 1e-4, dim=1)
        masks = torch.zeros(batch_size, batch_size - 1, dtype=torch.float).to(
            logits.device
        )
        # sample
        if torch.max(negative_probs) != 0:
            for b in range(batch_size):
                if torch.max(negative_probs[b]):
                    neg_idx = torch.multinomial(negative_probs[b], 1).item()
                else:
                    neg_idx = 0
                masks[b, neg_idx] = 1
        negative_samples = torch.masked_select(negative_samples, masks.bool()).view(
            batch_size, -1
        )
        negative_labels = torch.zeros(negative_samples.size()[0], dtype=torch.float).to(
            logits.device
        )
        # negative_labels = negative_labels * -1
        data = torch.cat([positive_samples, negative_samples], dim=0)
        label = torch.cat([positive_labels, negative_labels], dim=0)
        return data, label

    def forward(self, logits):
        data, label = self.get_pair(logits)
        data2, label2 = self.get_pair(logits.t())
        data = torch.cat([data, data2], dim=0)
        label = torch.cat([label, label2], dim=0)
        return data, label


@register_model("microsoft/china/selection/pretrain/v2/text")
# support hard negative
class BletchleyForTextPretrainV2(GenericModel):
    def __init__(
        self,
        query_config_type: str,
        doc_config_type: str,
        projection_dim: Optional[int] = 1024,
        freeze_base_model: Optional[bool] = True,
        logit_scale_init_value: Optional[float] = 2.6592,
        gradient_checkpointing: Optional[bool] = False,
        use_all_gather: Optional[bool] = True,
        output_query_embed: Optional[bool] = False,
        output_doc_embed: Optional[bool] = False,
        enable_quantization: Optional[bool] = False,
        enable_triploss: Optional[bool] = False,
        enable_matchloss: Optional[bool] = False,
        triploss_weight: Optional[float] = 1,
        triploss_margin: Optional[float] = 0.1,
        matchloss_weight: Optional[float] = 0.1,
        matchloss_type: Optional[str] = "MSE",
    ):
        super().__init__()
        query_config = get_bletchley_text_config(
            query_config_type, gradient_checkpointing
        )
        doc_config = get_bletchley_text_config(doc_config_type, gradient_checkpointing)

        self.use_all_gather = use_all_gather
        self.output_query_embed = output_query_embed
        self.output_doc_embed = output_doc_embed

        self.query_embed_dim = query_config.hidden_size
        self.doc_embed_dim = doc_config.hidden_size

        self.query_encoder = BletchleyTextEncoder(
            query_config, add_projection_layer=False
        )
        self.doc_encoder = BletchleyTextEncoder(doc_config, add_projection_layer=False)

        self.query_projection = nn.Linear(
            self.query_embed_dim,
            projection_dim,
        )
        self.doc_projection = nn.Linear(
            self.doc_embed_dim,
            projection_dim,
        )

        self.classifier = nn.Linear(1, 1)
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init_value)
        self.init_weights()
        self.classifier.weight.data.fill_(5.0)

        self.triploss = None
        self.matchloss = None
        if enable_triploss:
            self.triploss = TripLoss(weight=triploss_weight, margin=triploss_margin)
        if enable_matchloss:
            self.matchloss = MatchLoss()
            self.matchloss_weight = matchloss_weight
            self.matchloss_type = matchloss_type

        if enable_quantization:
            for __model__ in [self.query_encoder, self.query_projection]:
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
    @add_default_section_for_init("microsoft/china/selection/pretrain/v2/text")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/china/selection/pretrain/v2/text")
        query_config_type = config.getoption("query_config_type", "0.8B")
        doc_config_type = config.getoption("doc_config_type", "0.8B")

        projection_dim = config.getoption("projection_dim", 1024)
        freeze_base_model = config.getoption("freeze_base_model", True)
        logit_scale_init_value = config.getoption("logit_scale_init_value", 2.6592)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)
        use_all_gather = config.getoption("use_all_gather", True)
        output_query_embed = config.getoption("output_query_embed", False)
        output_doc_embed = config.getoption("output_doc_embed", False)
        enable_quantization = config.getoption("enable_quantization", False)
        enable_triploss = config.getoption("enable_triploss", False)
        enable_matchloss = config.getoption("enable_matchloss", False)
        triploss_weight = config.getoption("triploss_weight", 1)
        triploss_margin = config.getoption("triploss_margin", 0.1)
        matchloss_weight = config.getoption("matchloss_weight", 0.1)
        matchloss_type = config.getoption("matchloss_type", "MSE")
        inst = cls(
            query_config_type=query_config_type,
            doc_config_type=doc_config_type,
            projection_dim=projection_dim,
            freeze_base_model=freeze_base_model,
            logit_scale_init_value=logit_scale_init_value,
            gradient_checkpointing=gradient_checkpointing,
            use_all_gather=use_all_gather,
            output_query_embed=output_query_embed,
            output_doc_embed=output_doc_embed,
            enable_quantization=enable_quantization,
            enable_triploss=enable_triploss,
            enable_matchloss=enable_matchloss,
            triploss_weight=triploss_weight,
            triploss_margin=triploss_margin,
            matchloss_weight=matchloss_weight,
            matchloss_type=matchloss_type,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        if pretrained_weight_path is not None:
            inst.from_pretrained(pretrained_weight_path)

        return inst

    def from_pretrained(self, weight_path):
        weight_path = cached_path(weight_path)
        if not os.path.exists(weight_path):
            return
        state_dict = torch.load(weight_path, map_location="cpu")
        _keys = [key for key in state_dict.keys() if key.startswith("text_encoder")]
        for _key in _keys:
            _value = state_dict.pop(_key)
            state_dict["query_encoder" + _key[12:]] = _value
            state_dict["doc_encoder" + _key[12:]] = _value

        super().from_pretrained(state_dict=state_dict)

    def all_gather(self, input):
        output = AllGather.apply(input)
        output = output.view(-1, *(output.shape[2:]))
        return output

    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(
        self,
        query_input_ids=None,
        query_attention_mask=None,
        doc_input_ids=None,
        doc_attention_mask=None,
    ):
        if not self.training and self.output_query_embed:
            query_outputs = self.query_encoder(
                input_ids=query_input_ids, attention_mask=query_attention_mask
            )
            query_embeds = query_outputs[:, 0]
            query_embeds = self.query_projection(query_embeds)
            query_embeds = query_embeds / query_embeds.norm(dim=-1, keepdim=True)

            return EmbeddingOutputs(embedding=query_embeds)

        if not self.training and self.output_doc_embed:
            doc_outputs = self.doc_encoder(
                input_ids=doc_input_ids, attention_mask=doc_attention_mask
            )
            doc_embeds = doc_outputs[:, 0]
            doc_embeds = self.doc_projection(doc_embeds)
            doc_embeds = doc_embeds / doc_embeds.norm(dim=-1, keepdim=True)

            return EmbeddingOutputs(embedding=doc_embeds)
        query_outputs = self.query_encoder(
            input_ids=query_input_ids, attention_mask=query_attention_mask
        )
        doc_outputs = self.doc_encoder(
            input_ids=doc_input_ids, attention_mask=doc_attention_mask
        )

        query_embeds = query_outputs[:, 0]
        query_embeds = self.query_projection(query_embeds)

        doc_embeds = doc_outputs[:, 0]
        doc_embeds = self.doc_projection(doc_embeds)

        query_embeds = query_embeds / query_embeds.norm(dim=-1, keepdim=True)
        doc_embeds = doc_embeds / doc_embeds.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        if self.use_all_gather and dist.is_initialized():
            query_embeds = self.all_gather(query_embeds)
            doc_embeds = self.all_gather(doc_embeds)

        # use multiple loss
        logits_raw = torch.matmul(query_embeds, doc_embeds.t())
        logits_per_text = logits_raw * logit_scale
        loss_clip = _clip_loss(logits_per_text)
        # tiplet loss
        if self.triploss != None:
            loss_trip = self.triploss(query_embeds, doc_embeds)
        else:
            loss_trip = 0.0 * loss_clip
        # match loss
        if self.matchloss != None:
            data, label = self.matchloss(logits_raw)
            data = self.classifier(data)
            if self.matchloss_type == "BCE":
                loss_func = BCELoss()
            else:
                loss_func = MSELoss()
            loss_match = self.matchloss_weight * loss_func(data, label)
        else:
            loss_match = 0.0 * loss_clip

        loss = loss_clip + loss_trip + loss_match

        return LossOutputs(loss=loss)


@register_model("microsoft/china/selection/pretrain/v1/text")
class BletchleyForTextPretrain(GenericModel):
    def __init__(
        self,
        query_config_type: str,
        doc_config_type: str,
        projection_dim: Optional[int] = 1024,
        freeze_base_model: Optional[bool] = True,
        logit_scale_init_value: Optional[float] = 2.6592,
        gradient_checkpointing: Optional[bool] = False,
        use_all_gather: Optional[bool] = True,
        output_query_embed: Optional[bool] = False,
        output_doc_embed: Optional[bool] = False,
        enable_quantization: Optional[bool] = False,
    ):
        super().__init__()
        query_config = get_bletchley_text_config(
            query_config_type, gradient_checkpointing
        )
        doc_config = get_bletchley_text_config(doc_config_type, gradient_checkpointing)

        self.use_all_gather = use_all_gather
        self.output_query_embed = output_query_embed
        self.output_doc_embed = output_doc_embed

        self.query_embed_dim = query_config.hidden_size
        self.doc_embed_dim = doc_config.hidden_size

        self.query_encoder = BletchleyTextEncoder(
            query_config, add_projection_layer=False
        )
        self.doc_encoder = BletchleyTextEncoder(doc_config, add_projection_layer=False)

        self.query_projection = nn.Linear(
            self.query_embed_dim,
            projection_dim,
        )
        self.doc_projection = nn.Linear(
            self.doc_embed_dim,
            projection_dim,
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init_value)
        self.init_weights()

        if enable_quantization:
            for __model__ in [self.query_encoder, self.query_projection]:
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
    @add_default_section_for_init("microsoft/china/selection/pretrain/v1/text")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/china/selection/pretrain/v1/text")
        query_config_type = config.getoption("query_config_type", "0.8B")
        doc_config_type = config.getoption("doc_config_type", "0.8B")

        projection_dim = config.getoption("projection_dim", 1024)
        freeze_base_model = config.getoption("freeze_base_model", True)
        logit_scale_init_value = config.getoption("logit_scale_init_value", 2.6592)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)
        use_all_gather = config.getoption("use_all_gather", True)
        output_query_embed = config.getoption("output_query_embed", False)
        output_doc_embed = config.getoption("output_doc_embed", False)
        enable_quantization = config.getoption("enable_quantization", False)

        inst = cls(
            query_config_type=query_config_type,
            doc_config_type=doc_config_type,
            projection_dim=projection_dim,
            freeze_base_model=freeze_base_model,
            logit_scale_init_value=logit_scale_init_value,
            gradient_checkpointing=gradient_checkpointing,
            use_all_gather=use_all_gather,
            output_query_embed=output_query_embed,
            output_doc_embed=output_doc_embed,
            enable_quantization=enable_quantization,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        if pretrained_weight_path is not None:
            inst.from_pretrained(pretrained_weight_path)

        return inst

    def from_pretrained(self, weight_path):
        weight_path = cached_path(weight_path)
        if not os.path.exists(weight_path):
            return
        state_dict = torch.load(weight_path, map_location="cpu")
        _keys = [key for key in state_dict.keys() if key.startswith("text_encoder")]
        for _key in _keys:
            _value = state_dict.pop(_key)
            state_dict["query_encoder" + _key[12:]] = _value
            state_dict["doc_encoder" + _key[12:]] = _value

        super().from_pretrained(state_dict=state_dict)

    def all_gather(self, input):
        output = AllGather.apply(input)
        output = output.view(-1, *(output.shape[2:]))
        return output

    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(
        self,
        query_input_ids=None,
        query_attention_mask=None,
        doc_input_ids=None,
        doc_attention_mask=None,
    ):
        if not self.training and self.output_query_embed:
            query_outputs = self.query_encoder(
                input_ids=query_input_ids, attention_mask=query_attention_mask
            )
            query_embeds = query_outputs[:, 0]
            query_embeds = self.query_projection(query_embeds)
            query_embeds = query_embeds / query_embeds.norm(dim=-1, keepdim=True)

            return EmbeddingOutputs(embedding=query_embeds)

        if not self.training and self.output_doc_embed:
            doc_outputs = self.doc_encoder(
                input_ids=doc_input_ids, attention_mask=doc_attention_mask
            )
            doc_embeds = doc_outputs[:, 0]
            doc_embeds = self.doc_projection(doc_embeds)
            doc_embeds = doc_embeds / doc_embeds.norm(dim=-1, keepdim=True)

            return EmbeddingOutputs(embedding=doc_embeds)
        query_outputs = self.query_encoder(
            input_ids=query_input_ids, attention_mask=query_attention_mask
        )
        doc_outputs = self.doc_encoder(
            input_ids=doc_input_ids, attention_mask=doc_attention_mask
        )

        query_embeds = query_outputs[:, 0]
        query_embeds = self.query_projection(query_embeds)

        doc_embeds = doc_outputs[:, 0]
        doc_embeds = self.doc_projection(doc_embeds)

        query_embeds = query_embeds / query_embeds.norm(dim=-1, keepdim=True)
        doc_embeds = doc_embeds / doc_embeds.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        if self.use_all_gather and dist.is_initialized():
            query_embeds = self.all_gather(query_embeds)
            doc_embeds = self.all_gather(doc_embeds)
        logits_per_text = torch.matmul(query_embeds, doc_embeds.t()) * logit_scale

        loss = _clip_loss(logits_per_text)

        return LossOutputs(loss=loss)


@register_model("microsoft/china/selection/retrieval/v1/text")
class BletchleyForTextRetrieval(GenericModel):
    def __init__(
        self,
        query_config_type: str,
        doc_config_type: str,
        projection_dim: Optional[int] = 1024,
        output_hidden_dim: Optional[int] = 64,
        freeze_base_model: Optional[bool] = True,
        gradient_checkpointing: Optional[bool] = False,
        use_all_gather: Optional[bool] = True,
        output_query_embed: Optional[bool] = False,
        output_doc_embed: Optional[bool] = False,
        random_negative_samples: Optional[int] = 10,
        temperature: Optional[float] = 1.0,
    ):
        super().__init__()
        query_config = get_bletchley_text_config(
            query_config_type, gradient_checkpointing
        )
        doc_config = get_bletchley_text_config(doc_config_type, gradient_checkpointing)

        self.use_all_gather = use_all_gather
        self.output_query_embed = output_query_embed
        self.output_doc_embed = output_doc_embed

        self.query_embed_dim = query_config.hidden_size
        self.doc_embed_dim = doc_config.hidden_size
        self.random_negative_samples = random_negative_samples
        self.temperature = temperature

        self.query_encoder = BletchleyTextEncoder(
            query_config, add_projection_layer=False
        )
        self.doc_encoder = BletchleyTextEncoder(doc_config, add_projection_layer=False)

        self.query_projection = nn.Linear(
            self.query_embed_dim,
            projection_dim,
        )  # text_encoder.projection.weight, text_encoder.projection.bias
        self.doc_projection = nn.Linear(
            self.doc_embed_dim,
            projection_dim,
        )  # image_encoder.projection.weight,  image_encoder.projection.bias

        self.query_layer_norm = nn.LayerNorm(projection_dim)
        self.doc_layer_norm = nn.LayerNorm(projection_dim)

        self.final_doc_projection = nn.Linear(
            projection_dim,
            output_hidden_dim,
        )
        self.final_query_projection = nn.Linear(
            projection_dim,
            output_hidden_dim,
        )

        self.init_weights()

        if freeze_base_model:
            for p in self.query_encoder.parameters():
                p.requires_grad = False

            for p in self.doc_encoder.parameters():
                p.requires_grad = False

    @classmethod
    @add_default_section_for_init("microsoft/china/selection/retrieval/v1/text")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/china/selection/retrieval/v1/text")
        query_config_type = config.getoption("query_config_type", "0.8B")
        doc_config_type = config.getoption("doc_config_type", "0.8B")

        projection_dim = config.getoption("projection_dim", 1024)
        output_hidden_dim = config.getoption("output_hidden_dim", 64)
        freeze_base_model = config.getoption("freeze_base_model", True)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)
        use_all_gather = config.getoption("use_all_gather", True)
        output_query_embed = config.getoption("output_query_embed", False)
        output_doc_embed = config.getoption("output_doc_embed", False)

        inst = cls(
            query_config_type=query_config_type,
            doc_config_type=doc_config_type,
            projection_dim=projection_dim,
            output_hidden_dim=output_hidden_dim,
            freeze_base_model=freeze_base_model,
            gradient_checkpointing=gradient_checkpointing,
            use_all_gather=use_all_gather,
            output_query_embed=output_query_embed,
            output_doc_embed=output_doc_embed,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        if pretrained_weight_path is not None:
            inst.from_pretrained(pretrained_weight_path)

        return inst

    def from_pretrained(self, weight_path):
        weight_path = cached_path(weight_path)
        if not os.path.exists(weight_path):
            return
        state_dict = torch.load(weight_path, map_location="cpu")
        _keys = [key for key in state_dict.keys() if key.startswith("text_encoder")]
        for _key in _keys:
            _value = state_dict.pop(_key)
            state_dict["query_encoder" + _key[12:]] = _value
            state_dict["doc_encoder" + _key[12:]] = _value

        super().from_pretrained(state_dict=state_dict)

    def all_gather(self, input):
        output = AllGather.apply(input)
        output = output.view(-1, *(output.shape[2:]))
        return output

    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(
        self,
        query_input_ids=None,
        query_attention_mask=None,
        doc_input_ids=None,
        doc_attention_mask=None,
        neg_doc_input_ids=None,
        neg_doc_attention_mask=None,
        neg_doc_num_attention_mask=None,
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

        if not self.training and self.output_doc_embed:
            doc_outputs = self.doc_encoder(
                input_ids=doc_input_ids, attention_mask=doc_attention_mask
            )
            doc_embeds = doc_outputs[:, 0]
            doc_embeds = self.doc_projection(doc_embeds)
            doc_embeds = self.doc_layer_norm(quick_gelu(doc_embeds))
            doc_embeds = self.final_doc_projection(quick_gelu(doc_embeds))
            doc_embeds = doc_embeds / doc_embeds.norm(dim=-1, keepdim=True)

            return EmbeddingOutputs(embedding=doc_embeds)
        query_outputs = self.query_encoder(
            input_ids=query_input_ids, attention_mask=query_attention_mask
        )
        doc_outputs = self.doc_encoder(
            input_ids=doc_input_ids, attention_mask=doc_attention_mask
        )
        neg_doc_input_ids = neg_doc_input_ids.view(-1, neg_doc_input_ids.shape[2])
        neg_doc_attention_mask = neg_doc_attention_mask.view(
            -1, neg_doc_attention_mask.shape[2]
        )
        neg_doc_outputs = self.doc_encoder(
            input_ids=neg_doc_input_ids, attention_mask=neg_doc_attention_mask
        )

        query_embeds = query_outputs[:, 0]
        query_embeds = self.query_projection(query_embeds)

        doc_embeds = doc_outputs[:, 0]
        doc_embeds = self.doc_projection(doc_embeds)

        neg_doc_embeds = neg_doc_outputs[:, 0]
        neg_doc_embeds = self.doc_projection(neg_doc_embeds)

        query_embeds = self.query_layer_norm(quick_gelu(query_embeds))
        doc_embeds = self.doc_layer_norm(quick_gelu(doc_embeds))
        neg_doc_embeds = self.doc_layer_norm(quick_gelu(neg_doc_embeds))

        query_embeds = self.final_query_projection(quick_gelu(query_embeds))
        doc_embeds = self.final_doc_projection(quick_gelu(doc_embeds))
        neg_doc_embeds = self.final_doc_projection(quick_gelu(neg_doc_embeds))

        query_embeds = query_embeds / query_embeds.norm(dim=-1, keepdim=True)
        doc_embeds = doc_embeds / doc_embeds.norm(dim=-1, keepdim=True)
        neg_doc_embeds = neg_doc_embeds / neg_doc_embeds.norm(dim=-1, keepdim=True)

        if self.use_all_gather and dist.is_initialized():
            query_embeds = self.all_gather(query_embeds)
            doc_embeds = self.all_gather(doc_embeds)
            neg_doc_embeds = self.all_gather(neg_doc_embeds)
            neg_doc_num_attention_mask = self.all_gather(neg_doc_num_attention_mask)
        logits = torch.matmul(query_embeds, doc_embeds.t()) * self.temperature  # bs*bs
        neg_doc_embeds = neg_doc_embeds.view(
            logits.shape[0], -1, neg_doc_embeds.shape[-1]
        )  # bs*k*dim
        neg_logits = (
            torch.matmul(
                query_embeds.unsqueeze(1), neg_doc_embeds.transpose(1, 2)
            ).squeeze(1)
            * self.temperature
        )  # bs*k
        # reorg
        batch_size = logits.shape[0]
        masks = torch.eye(batch_size).to(logits.device).bool()
        positive_samples = torch.masked_select(logits, masks).view(batch_size, -1)
        negative_samples = torch.masked_select(logits, ~masks).view(batch_size, -1)
        samples = torch.cat(
            [positive_samples, negative_samples, neg_logits], dim=-1
        )  # bs*(bs+k)
        # sample
        if self.random_negative_samples > batch_size - 1:
            self.random_negative_samples = batch_size - 1
        masks = torch.zeros(batch_size, batch_size, dtype=torch.float).to(
            samples.device
        )
        masks[:, 0] = 1
        masks = torch.cat([masks, neg_doc_num_attention_mask], dim=-1)
        for i in range(batch_size):
            masks[
                i,
                random.sample(
                    range(1, batch_size),
                    self.random_negative_samples
                    - torch.count_nonzero(neg_doc_num_attention_mask[i]),
                ),
            ] = 1
        samples = torch.masked_select(samples, masks.bool()).view(batch_size, -1)

        loss = 0
        for i in range(1, samples.shape[1]):
            loss_i = torch.log(1 + torch.exp(samples[:, i] - samples[:, 0]))
            loss += loss_i.sum()
        loss = loss / ((samples.shape[1] - 1) * samples.shape[0])

        return LossOutputs(loss=loss)


@register_model("microsoft/china/selection/matching/bletchley/v1")
class BletchleyForMatching(GenericModel):
    replace_keys_in_state_dict = {
        "text_encoder.projection": "text_projection",
        "image_encoder.projection": "image_projection",
    }

    def __init__(
        self,
        query_config_type: str,
        doc_config_type: str,
        projection_dim: Optional[int] = 1024,
        freeze_base_model: Optional[bool] = True,
        gradient_checkpointing: Optional[bool] = False,
        output_query_embed: Optional[bool] = False,
        output_doc_embed: Optional[bool] = False,
    ):
        super().__init__()
        query_config = get_bletchley_text_config(
            query_config_type, gradient_checkpointing
        )
        doc_config = get_bletchley_text_config(doc_config_type, gradient_checkpointing)

        self.output_query_embed = output_query_embed
        self.output_doc_embed = output_doc_embed

        self.query_embed_dim = query_config.hidden_size
        self.doc_embed_dim = doc_config.hidden_size

        self.query_encoder = BletchleyTextEncoder(
            query_config, add_projection_layer=False
        )
        self.doc_encoder = BletchleyTextEncoder(doc_config, add_projection_layer=False)

        self.query_projection = nn.Linear(
            self.query_embed_dim,
            projection_dim,
        )  # text_encoder.projection.weight, text_encoder.projection.bias
        self.doc_projection = nn.Linear(
            self.doc_embed_dim,
            projection_dim,
        )  # image_encoder.projection.weight,  image_encoder.projection.bias

        self.classifier = nn.Linear(1, 1)

        self.init_weights()
        self.classifier.weight.data.fill_(5.0)

        if freeze_base_model:
            for p in self.query_encoder.parameters():
                p.requires_grad = False

            for p in self.doc_encoder.parameters():
                p.requires_grad = False

    @classmethod
    @add_default_section_for_init("microsoft/china/selection/matching/bletchley/v1")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/china/selection/matching/bletchley/v1")
        query_config_type = config.getoption("query_config_type", "0.8B")
        doc_config_type = config.getoption("doc_config_type", "0.8B")

        projection_dim = config.getoption("projection_dim", 1024)
        freeze_base_model = config.getoption("freeze_base_model", True)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)
        output_query_embed = config.getoption("output_query_embed", False)
        output_doc_embed = config.getoption("output_doc_embed", False)

        inst = cls(
            query_config_type=query_config_type,
            doc_config_type=doc_config_type,
            projection_dim=projection_dim,
            freeze_base_model=freeze_base_model,
            gradient_checkpointing=gradient_checkpointing,
            output_query_embed=output_query_embed,
            output_doc_embed=output_doc_embed,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        if pretrained_weight_path is not None:
            inst.from_pretrained(pretrained_weight_path)

        return inst

    def from_pretrained(self, weight_path):
        weight_path = cached_path(weight_path)
        if not os.path.exists(weight_path):
            return
        state_dict = torch.load(weight_path, map_location="cpu")
        _keys = [key for key in state_dict.keys() if key.startswith("text_encoder")]
        for _key in _keys:
            _value = state_dict.pop(_key)
            state_dict["query_encoder" + _key[12:]] = _value
            state_dict["doc_encoder" + _key[12:]] = _value

        super().from_pretrained(state_dict=state_dict)

    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(
        self,
        query_input_ids=None,
        query_attention_mask=None,
        doc_input_ids=None,
        doc_attention_mask=None,
    ):
        if not self.training and self.output_query_embed:
            query_outputs = self.query_encoder(
                input_ids=query_input_ids, attention_mask=query_attention_mask
            )
            query_embeds = query_outputs[:, 0]
            query_embeds = self.query_projection(query_embeds)
            query_embeds = query_embeds / query_embeds.norm(dim=-1, keepdim=True)
            return EmbeddingOutputs(embedding=query_embeds)

        if not self.training and self.output_doc_embed:
            doc_outputs = self.doc_encoder(
                input_ids=doc_input_ids,
                attention_mask=doc_attention_mask,
            )
            doc_embeds = doc_outputs[:, 0]
            doc_embeds = self.doc_projection(doc_embeds)
            doc_embeds = doc_embeds / doc_embeds.norm(dim=-1, keepdim=True)
            return EmbeddingOutputs(embedding=doc_embeds)
        query_outputs = self.query_encoder(
            input_ids=query_input_ids, attention_mask=query_attention_mask
        )
        doc_outputs = self.doc_encoder(
            input_ids=doc_input_ids, attention_mask=doc_attention_mask
        )

        query_embeds = query_outputs[:, 0]
        query_embeds = self.query_projection(query_embeds)

        doc_embeds = doc_outputs[:, 0]
        doc_embeds = self.doc_projection(doc_embeds)

        query_embeds = query_embeds / query_embeds.norm(dim=-1, keepdim=True)
        doc_embeds = doc_embeds / doc_embeds.norm(dim=-1, keepdim=True)
        scores = torch.sum(query_embeds * doc_embeds, dim=-1, keepdim=True)

        outputs = self.classifier(scores)
        return ClassificationOutputs(outputs=outputs)
