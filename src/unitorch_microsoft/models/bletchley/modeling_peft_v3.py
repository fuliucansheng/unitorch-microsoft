# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import random
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torch.cuda.amp import autocast
from peft import LoraConfig
from unitorch.models import GenericModel
from unitorch.models.peft import GenericPeftModel, PeftModelForSequenceClassification
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
from unitorch_microsoft.models.bletchley.modeling_v3 import (
    BletchleyTextEncoder,
    BletchleyImageEncoder,
)


class BletchleyForMatching(GenericModel):
    replace_keys_in_state_dict = {
        "text_encoder.projection": "text_projection",
        "image_encoder.projection": "image_projection",
    }

    def __init__(
        self,
        config_type: str,
        projection_dim: Optional[int] = 1024,
        gradient_checkpointing: Optional[bool] = False,
    ):
        super().__init__()

        self.text_encoder = BletchleyTextEncoder(
            config_type,
            add_projection_layer=False,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.image_encoder = BletchleyImageEncoder(
            config_type,
            add_projection_layer=False,
            gradient_checkpointing=gradient_checkpointing,
        )

        self.text_embed_dim = self.text_encoder.hidden_size
        self.image_embed_dim = self.image_encoder.hidden_size

        self.image_projection = nn.Linear(
            self.image_embed_dim,
            projection_dim,
        )
        self.text_projection = nn.Linear(
            self.text_embed_dim,
            projection_dim,
        )

        self.init_weights()

    @autocast()
    def forward(
        self,
        input_ids=None,
        images=None,
        attention_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        image_outputs = self.image_encoder(
            images=images,
        )
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        image_embeds = image_outputs[:, 0]
        image_embeds = self.image_projection(image_embeds)
        text_embeds = text_outputs[:, 0]
        text_embeds = self.text_projection(text_embeds)

        return (text_embeds, image_embeds)


class BletchleyForTextMatching(GenericModel):
    replace_keys_in_state_dict = {
        "query_encoder.projection": "query_projection",
        "doc_encoder.projection": "doc_projection",
    }

    def __init__(
        self,
        query_config_type: str,
        doc_config_type: str,
        projection_dim: Optional[int] = 1024,
        gradient_checkpointing: Optional[bool] = False,
    ):
        super().__init__()
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

        self.query_embed_dim = self.query_encoder.hidden_size
        self.doc_embed_dim = self.doc_encoder.hidden_size

        self.query_projection = nn.Linear(
            self.query_embed_dim,
            projection_dim,
        )
        self.doc_projection = nn.Linear(
            self.doc_embed_dim,
            projection_dim,
        )

        self.init_weights()

    @autocast()
    def forward(
        self,
        query_input_ids=None,
        query_attention_mask=None,
        doc_input_ids=None,
        doc_attention_mask=None,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        query_outputs = self.query_encoder(
            input_ids=query_input_ids, attention_mask=query_attention_mask
        )
        doc_outputs = self.doc_encoder(
            input_ids=doc_input_ids,
            attention_mask=doc_attention_mask,
        )

        query_embeds = query_outputs[:, 0]
        query_embeds = self.query_projection(query_embeds)
        doc_embeds = doc_outputs[:, 0]
        doc_embeds = self.doc_projection(doc_embeds)

        return (query_embeds, doc_embeds)


@register_model("microsoft/model/pretrain/peft/lora/bletchley/v3")
class BletchleyLoraForPretrain(GenericPeftModel):
    prefix_keys_in_state_dict = {
        "^(?!peft_model\.base_model\.model\.).*": "peft_model.base_model.model."
    }
    replace_keys_in_state_dict = {
        "attn.proj.weight": "attn.proj.base_layer.weight",
        "attn.proj.bias": "attn.proj.base_layer.bias",
        "text_encoder.projection": "text_projection",
        "image_encoder.projection": "image_projection",
    }

    def __init__(
        self,
        config_type: str,
        projection_dim: Optional[int] = 1024,
        lora_r: Optional[int] = 16,
        lora_alpha: Optional[int] = 32,
        lora_dropout: Optional[float] = 0.05,
        fan_in_fan_out: Optional[bool] = True,
        target_modules: Optional[Union[List[str], str]] = ["attn.proj"],
        use_all_gather: Optional[bool] = True,
        logit_scale_init_value: Optional[float] = 2.6592,
        gradient_checkpointing: Optional[bool] = False,
    ):
        super().__init__()
        self.peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            target_modules=target_modules,
        )
        self.peft_model = PeftModelForSequenceClassification(
            BletchleyForMatching(
                config_type,
                projection_dim=projection_dim,
                gradient_checkpointing=gradient_checkpointing,
            ),
            self.peft_config,
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init_value)
        self.use_all_gather = use_all_gather
        self.init_weights()

    @classmethod
    @add_default_section_for_init("microsoft/model/pretrain/peft/lora/bletchley/v3")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/model/pretrain/peft/lora/bletchley/v3")
        config_type = config.getoption("config_type", "0.8B")

        projection_dim = config.getoption("projection_dim", 1024)
        lora_r = config.getoption("lora_r", 16)
        lora_alpha = config.getoption("lora_alpha", 32)
        lora_dropout = config.getoption("lora_dropout", 0.05)
        fan_in_fan_out = config.getoption("fan_in_fan_out", True)
        target_modules = config.getoption("target_modules", ["attn.proj"])
        use_all_gather = config.getoption("use_all_gather", True)
        logit_scale_init_value = config.getoption("logit_scale_init_value", 2.6592)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)

        inst = cls(
            config_type=config_type,
            projection_dim=projection_dim,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            target_modules=target_modules,
            use_all_gather=use_all_gather,
            logit_scale_init_value=logit_scale_init_value,
            gradient_checkpointing=gradient_checkpointing,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        if pretrained_weight_path is not None:
            inst.from_pretrained(pretrained_weight_path)

        return inst

    def all_gather(self, input):
        output = AllGather.apply(input)
        output = output.view(-1, *(output.shape[2:]))
        return output

    @autocast()
    def forward(
        self,
        input_ids: torch.Tensor = None,
        images: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
    ):
        text_embeds, image_embeds = self.peft_model(
            input_ids=input_ids,
            images=images,
            attention_mask=attention_mask,
            return_dict=False,
        )

        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        if self.use_all_gather and dist.is_initialized():
            text_embeds = self.all_gather(text_embeds)
            image_embeds = self.all_gather(image_embeds)
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.T

        loss = _clip_loss(logits_per_text)
        return LossOutputs(loss=loss)


@register_model("microsoft/model/matching/peft/lora/bletchley/v3")
class BletchleyLoraForMatching(GenericPeftModel):
    prefix_keys_in_state_dict = {
        "^(?!peft_model\.base_model\.model\.).*": "peft_model.base_model.model."
    }
    replace_keys_in_state_dict = {
        "attn.proj.weight": "attn.proj.base_layer.weight",
        "attn.proj.bias": "attn.proj.base_layer.bias",
        "text_encoder.projection": "text_projection",
        "image_encoder.projection": "image_projection",
    }
    modules_to_save_checkpoints = ["lora", "classifier"]

    def __init__(
        self,
        config_type: str,
        projection_dim: Optional[int] = 1024,
        lora_r: Optional[int] = 16,
        lora_alpha: Optional[int] = 32,
        lora_dropout: Optional[float] = 0.05,
        fan_in_fan_out: Optional[bool] = True,
        target_modules: Optional[Union[List[str], str]] = ["attn.proj"],
    ):
        super().__init__()
        self.peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            target_modules=target_modules,
        )
        self.peft_model = PeftModelForSequenceClassification(
            BletchleyForMatching(config_type, projection_dim=projection_dim),
            self.peft_config,
        )
        self.classifier = nn.Linear(1, 1)

        self.init_weights()
        self.classifier.weight.data.fill_(5.0)

    @classmethod
    @add_default_section_for_init("microsoft/model/matching/peft/lora/bletchley/v3")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/model/matching/peft/lora/bletchley/v3")
        config_type = config.getoption("config_type", "0.8B")

        projection_dim = config.getoption("projection_dim", 1024)
        lora_r = config.getoption("lora_r", 16)
        lora_alpha = config.getoption("lora_alpha", 32)
        lora_dropout = config.getoption("lora_dropout", 0.05)
        fan_in_fan_out = config.getoption("fan_in_fan_out", True)
        target_modules = config.getoption("target_modules", ["attn.proj"])

        inst = cls(
            config_type=config_type,
            projection_dim=projection_dim,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            target_modules=target_modules,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        if pretrained_weight_path is not None:
            inst.from_pretrained(pretrained_weight_path)

        return inst

    @autocast()
    def forward(
        self,
        input_ids: torch.Tensor = None,
        images: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
    ):
        text_embeds, image_embeds = self.peft_model(
            input_ids=input_ids,
            images=images,
            attention_mask=attention_mask,
            return_dict=False,
        )

        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        scores = torch.sum(text_embeds * image_embeds, dim=-1, keepdim=True)

        outputs = self.classifier(scores)
        return ClassificationOutputs(outputs=outputs)


@register_model("microsoft/model/pretrain/peft/lora/bletchley/v3/text")
class BletchleyLoraForTextPretrain(GenericPeftModel):
    prefix_keys_in_state_dict = {
        "^(?!peft_model\.base_model\.model\.).*": "peft_model.base_model.model."
    }
    replace_keys_in_state_dict = {
        "attn.proj.weight": "attn.proj.base_layer.weight",
        "attn.proj.bias": "attn.proj.base_layer.bias",
        "query_encoder.projection": "query_projection",
        "doc_encoder.projection": "doc_projection",
    }

    def __init__(
        self,
        query_config_type: str,
        doc_config_type: str,
        projection_dim: Optional[int] = 1024,
        lora_r: Optional[int] = 16,
        lora_alpha: Optional[int] = 32,
        lora_dropout: Optional[float] = 0.05,
        fan_in_fan_out: Optional[bool] = True,
        target_modules: Optional[Union[List[str], str]] = ["attn.proj"],
        use_all_gather: Optional[bool] = True,
        logit_scale_init_value: Optional[float] = 2.6592,
        gradient_checkpointing: Optional[bool] = False,
    ):
        super().__init__()
        self.peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            target_modules=target_modules,
        )
        self.peft_model = PeftModelForSequenceClassification(
            BletchleyForTextMatching(
                query_config_type,
                doc_config_type,
                projection_dim=projection_dim,
                gradient_checkpointing=gradient_checkpointing,
            ),
            self.peft_config,
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init_value)
        self.use_all_gather = use_all_gather
        self.init_weights()

    @classmethod
    @add_default_section_for_init(
        "microsoft/model/pretrain/peft/lora/bletchley/v3/text"
    )
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section(
            "microsoft/model/pretrain/peft/lora/bletchley/v3/text"
        )
        query_config_type = config.getoption("query_config_type", "0.8B")
        doc_config_type = config.getoption("doc_config_type", "0.8B")
        projection_dim = config.getoption("projection_dim", 1024)
        lora_r = config.getoption("lora_r", 16)
        lora_alpha = config.getoption("lora_alpha", 32)
        lora_dropout = config.getoption("lora_dropout", 0.05)
        fan_in_fan_out = config.getoption("fan_in_fan_out", True)
        target_modules = config.getoption("target_modules", ["attn.proj"])
        use_all_gather = config.getoption("use_all_gather", True)
        logit_scale_init_value = config.getoption("logit_scale_init_value", 2.6592)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)

        inst = cls(
            query_config_type=query_config_type,
            doc_config_type=doc_config_type,
            projection_dim=projection_dim,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            target_modules=target_modules,
            use_all_gather=use_all_gather,
            logit_scale_init_value=logit_scale_init_value,
            gradient_checkpointing=gradient_checkpointing,
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

    @autocast()
    def forward(
        self,
        query_input_ids=None,
        query_attention_mask=None,
        doc_input_ids=None,
        doc_attention_mask=None,
    ):
        query_embeds, doc_embeds = self.peft_model(
            query_input_ids=query_input_ids,
            query_attention_mask=query_attention_mask,
            doc_input_ids=doc_input_ids,
            doc_attention_mask=doc_attention_mask,
            return_dict=False,
        )

        query_embeds = query_embeds / query_embeds.norm(dim=-1, keepdim=True)
        doc_embeds = doc_embeds / doc_embeds.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        if self.use_all_gather and dist.is_initialized():
            query_embeds = self.all_gather(query_embeds)
            doc_embeds = self.all_gather(doc_embeds)
        logits_per_query = torch.matmul(query_embeds, doc_embeds.t()) * logit_scale
        logits_per_doc = logits_per_query.T

        loss = _clip_loss(logits_per_query)
        return LossOutputs(loss=loss)


@register_model("microsoft/model/matching/peft/lora/bletchley/v3/text")
class BletchleyLoraForTextMatching(GenericPeftModel):
    prefix_keys_in_state_dict = {
        "^(?!peft_model\.base_model\.model\.).*": "peft_model.base_model.model."
    }
    replace_keys_in_state_dict = {
        "attn.proj.weight": "attn.proj.base_layer.weight",
        "attn.proj.bias": "attn.proj.base_layer.bias",
        "query_encoder.projection": "query_projection",
        "doc_encoder.projection": "doc_projection",
    }
    modules_to_save_checkpoints = ["lora", "classifier"]

    def __init__(
        self,
        query_config_type: str,
        doc_config_type: str,
        projection_dim: Optional[int] = 1024,
        lora_r: Optional[int] = 16,
        lora_alpha: Optional[int] = 32,
        lora_dropout: Optional[float] = 0.05,
        fan_in_fan_out: Optional[bool] = True,
        target_modules: Optional[Union[List[str], str]] = ["attn.proj"],
    ):
        super().__init__()
        self.peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            target_modules=target_modules,
        )
        self.peft_model = PeftModelForSequenceClassification(
            BletchleyForTextMatching(
                query_config_type, doc_config_type, projection_dim=projection_dim
            ),
            self.peft_config,
        )
        self.classifier = nn.Linear(1, 1)

        self.init_weights()
        self.classifier.weight.data.fill_(5.0)

    @classmethod
    @add_default_section_for_init(
        "microsoft/model/matching/peft/lora/bletchley/v3/text"
    )
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section(
            "microsoft/model/matching/peft/lora/bletchley/v3/text"
        )
        query_config_type = config.getoption("query_config_type", "0.8B")
        doc_config_type = config.getoption("doc_config_type", "0.8B")
        projection_dim = config.getoption("projection_dim", 1024)
        lora_r = config.getoption("lora_r", 16)
        lora_alpha = config.getoption("lora_alpha", 32)
        lora_dropout = config.getoption("lora_dropout", 0.05)
        fan_in_fan_out = config.getoption("fan_in_fan_out", True)
        target_modules = config.getoption("target_modules", ["attn.proj"])

        inst = cls(
            query_config_type=query_config_type,
            doc_config_type=doc_config_type,
            projection_dim=projection_dim,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            target_modules=target_modules,
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

    @autocast()
    def forward(
        self,
        query_input_ids=None,
        query_attention_mask=None,
        doc_input_ids=None,
        doc_attention_mask=None,
    ):
        query_embeds, doc_embeds = self.peft_model(
            query_input_ids=query_input_ids,
            query_attention_mask=query_attention_mask,
            doc_input_ids=doc_input_ids,
            doc_attention_mask=doc_attention_mask,
            return_dict=False,
        )

        query_embeds = query_embeds / query_embeds.norm(dim=-1, keepdim=True)
        doc_embeds = doc_embeds / doc_embeds.norm(dim=-1, keepdim=True)
        scores = torch.sum(query_embeds * doc_embeds, dim=-1, keepdim=True)

        outputs = self.classifier(scores)
        return ClassificationOutputs(outputs=outputs)
