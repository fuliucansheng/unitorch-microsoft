# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import torch.distributed as dist
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from transformers.models.bert.modeling_bert import (
    BertConfig,
    BertEmbeddings,
    BertEncoder,
    BertPooler,
    BertModel,
    BertPreTrainedModel,
)
from unitorch_microsoft.adsplus.relevance.tribert.modeling_v8 import MultiTriPostLayer
from unitorch_microsoft.adsplus.relevance.tribert.modeling_v9 import TriTwinBertEncoder
from unitorch.models.clip.modeling import AllGather, _clip_loss
from unitorch.cli import cached_path
from unitorch.models import GenericModel
from unitorch.utils import pop_value, nested_dict_value
from unitorch.cli import (
    add_default_section_for_init,
    add_default_section_for_function,
    register_model,
)
from unitorch.cli.models import (
    EmbeddingOutputs,
    LossOutputs,
    ClassificationOutputs,
)
from unitorch_microsoft import cached_path
from unitorch_microsoft.adsplus.relevance.tribert import pretrained_bert_infos
    
class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim, projection_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1, bias=False)
        self.projection = nn.Linear(hidden_dim, projection_dim, bias=False)

    def forward(self, hidden_states, attention_mask):
        attention_scores = self.attn(hidden_states).squeeze(dim=-1)
        attention_mask = attention_mask.to(attention_scores)
        attention_scores = attention_scores + (1 - attention_mask) * -10000
        attention_probs = F.softmax(attention_scores, dim=-1)
        output = torch.bmm(
            hidden_states.transpose(1, 2), attention_probs.unsqueeze(dim=-1)
        ).squeeze(dim=-1)
        return self.projection(output)
    

@register_model("microsoft/adsplus/relevance/pretrain/tribert/v9/clip")
class TribertClipForPretrain(GenericModel):
    replace_keys_in_state_dict = {"gamma": "weight", "beta": "bias"}
    prefix_keys_in_state_dict = {}

    def __init__(
        self,
        config_path,
        pooltype: str = "bert",
        projection_dim: Optional[int] = 512,
        gradient_checkpointing: Optional[bool] = False,
        use_all_gather: Optional[bool] = True,
        alpha1: Optional[float] = 1.0,
        alpha2: Optional[float] = 1.0,
        alpha3: Optional[float] = 1.0,
    ):
        super().__init__()
        self.config = BertConfig.from_json_file(config_path)
        self.config.gradient_checkpointing = gradient_checkpointing
        self.config.logit_scale_init_value = 2.6592
        self.projection_dim = projection_dim
        self.text_embed_dim = self.config.hidden_size
        self.use_all_gather = use_all_gather
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3

        self.query_encoder = TriTwinBertEncoder(
            self.config,
            add_pooling_layer=pooltype == "bert",
        )

        self.doc_encoder = TriTwinBertEncoder(
            self.config,
            add_pooling_layer=pooltype == "bert",
        )

        self.ads_encoder = TriTwinBertEncoder(
            self.config,
            add_pooling_layer=pooltype == "bert",
        )

        self.init_weights()
        self.query_doc_projection = AttentionLayer(
            self.text_embed_dim,
            self.projection_dim,
        )
        self.query_ads_projection = AttentionLayer(
            self.text_embed_dim,
            self.projection_dim,
        )

        self.doc_query_projection = AttentionLayer(
            self.text_embed_dim,
            self.projection_dim,
        )
        self.doc_ads_projection = AttentionLayer(
            self.text_embed_dim,
            self.projection_dim,
        )

        self.ads_query_projection = AttentionLayer(
            self.text_embed_dim,
            self.projection_dim,
        )
        self.ads_doc_projection = AttentionLayer(
            self.text_embed_dim,
            self.projection_dim,
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * self.config.logit_scale_init_value)
        self.init_weights()

        self.query_encoder.bert.gradient_checkpointing = gradient_checkpointing
        self.doc_encoder.bert.gradient_checkpointing = gradient_checkpointing
        self.ads_encoder.bert.gradient_checkpointing = gradient_checkpointing


    def from_pretrained(self, weight_path):
        if not os.path.exists(weight_path):
            return
        state_dict = torch.load(weight_path, map_location="cpu")
        _keys = [key for key in state_dict.keys() if key.startswith("bert")]
        for _key in _keys:
            _value = state_dict.pop(_key)
            state_dict["query_encoder." + _key] = _value
            state_dict["doc_encoder." + _key] = _value
            state_dict["ads_encoder." + _key] = _value

        super().from_pretrained(state_dict=state_dict)

    @classmethod
    @add_default_section_for_init(
        "microsoft/adsplus/relevance/pretrain/tribert/v9/clip"
    )
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section(
            "microsoft/adsplus/relevance/pretrain/tribert/v9/clip"
        )
        pretrained_name = config.getoption("pretrained_name", "default-bert")
        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_bert_infos, pretrained_name, "config"),
        )
        config_path = cached_path(config_path)
        pooltype = config.getoption("pooltype", "bert")
        projection_dim = config.getoption("projection_dim", 512)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)
        use_all_gather = config.getoption("use_all_gather", False)
        inst = cls(
            config_path=config_path,
            pooltype=pooltype,
            projection_dim=projection_dim,
            gradient_checkpointing=gradient_checkpointing,
            use_all_gather=use_all_gather,
        )

        weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            weight_path,
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
        task,
        query_input_ids,
        query_attention_mask,
        query_token_type_ids,
        query_position_ids,
        doc_input_ids,
        doc_attention_mask,
        doc_token_type_ids,
        doc_position_ids,
        ads_input_ids,
        ads_attention_mask,
        ads_token_type_ids,
        ads_position_ids,
    ):
        query_outputs = self.query_encoder(
            query_input_ids,
            query_attention_mask,
            query_token_type_ids,
            query_position_ids,
        )
        doc_outputs = self.doc_encoder(
            doc_input_ids, doc_attention_mask, doc_token_type_ids, doc_position_ids
        )
        ads_outputs = self.ads_encoder(
            ads_input_ids, ads_attention_mask, ads_token_type_ids, ads_position_ids
        )
        query_doc_embeds = self.query_doc_projection(query_outputs.last_hidden_state, query_attention_mask)
        query_ads_embeds = self.query_ads_projection(query_outputs.last_hidden_state, query_attention_mask)
        doc_query_embeds = self.doc_query_projection(doc_outputs.last_hidden_state, doc_attention_mask)
        doc_ads_embeds = self.doc_ads_projection(doc_outputs.last_hidden_state, doc_attention_mask)
        ads_query_embeds = self.ads_query_projection(ads_outputs.last_hidden_state, ads_attention_mask)
        ads_doc_embeds = self.ads_doc_projection(ads_outputs.last_hidden_state, ads_attention_mask)
        
        # normalized features
        query_doc_embeds = query_doc_embeds / query_doc_embeds.norm(
            dim=-1, keepdim=True
        )
        query_ads_embeds = query_ads_embeds / query_ads_embeds.norm(
            dim=-1, keepdim=True
        )
        doc_query_embeds = doc_query_embeds / doc_query_embeds.norm(
            dim=-1, keepdim=True
        )
        doc_ads_embeds = doc_ads_embeds / doc_ads_embeds.norm(dim=-1, keepdim=True)
        ads_query_embeds = ads_query_embeds / ads_query_embeds.norm(
            dim=-1, keepdim=True
        )
        ads_doc_embeds = ads_doc_embeds / ads_doc_embeds.norm(dim=-1, keepdim=True)

        if self.use_all_gather and dist.is_initialized():
            query_doc_embeds = self.all_gather(query_doc_embeds)
            query_ads_embeds = self.all_gather(query_ads_embeds)
            doc_query_embeds = self.all_gather(doc_query_embeds)
            doc_ads_embeds = self.all_gather(doc_ads_embeds)
            ads_query_embeds = self.all_gather(ads_query_embeds)
            ads_doc_embeds = self.all_gather(ads_doc_embeds)

        logit_scale = self.logit_scale.exp()
        logits1 = torch.matmul(query_doc_embeds, doc_query_embeds.t()) * logit_scale
        logits2 = torch.matmul(query_ads_embeds, ads_query_embeds.t()) * logit_scale
        logits3 = torch.matmul(doc_ads_embeds, ads_doc_embeds.t()) * logit_scale

        outputs = (
            self.alpha1 * _clip_loss(logits1)
            + self.alpha2 * _clip_loss(logits2)
            + self.alpha3 * _clip_loss(logits3)
        )
        outputs = outputs / (self.alpha1 + self.alpha2 + self.alpha3)

        return LossOutputs(loss=outputs)
