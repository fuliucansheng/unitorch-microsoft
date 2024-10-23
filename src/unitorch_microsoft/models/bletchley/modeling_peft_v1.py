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
from unitorch.models.peft import (
    GenericPeftModel,
    PeftModelForSequenceClassification,
    PeftWeightLoaderMixin,
)
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
from unitorch_microsoft.models.bletchley.modeling_v1 import (
    BletchleyTextEncoder,
    BletchleyImageEncoder,
    get_bletchley_text_config,
    get_bletchley_image_config,
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
        text_config = get_bletchley_text_config(config_type, gradient_checkpointing)
        image_config = get_bletchley_image_config(config_type, gradient_checkpointing)

        self.text_embed_dim = text_config.hidden_size
        self.image_embed_dim = image_config.hidden_size

        self.text_encoder = BletchleyTextEncoder(
            text_config, add_projection_layer=False
        )
        self.image_encoder = BletchleyImageEncoder(
            image_config, add_projection_layer=False
        )

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
        query_config = get_bletchley_text_config(
            query_config_type, gradient_checkpointing
        )
        doc_config = get_bletchley_text_config(doc_config_type, gradient_checkpointing)

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


@register_model("microsoft/model/pretrain/peft/lora/bletchley/v1")
class BletchleyLoraForPretrain(GenericPeftModel, PeftWeightLoaderMixin):
    prefix_keys_in_state_dict = {
        "^(?!peft_model\.base_model\.model\.).*": "peft_model.base_model.model."
    }
    replace_keys_in_state_dict = {
        "query.weight": "query.base_layer.weight",
        "query.bias": "query.base_layer.bias",
        "value.weight": "value.base_layer.weight",
        "value.bias": "value.base_layer.bias",
        "text_encoder.projection": "text_projection",
        "image_encoder.projection": "image_projection",
    }
    replace_keys_in_peft_state_dict = {
        ".weight": ".base_layer.weight",
        ".bias": ".base_layer.bias",
    }

    modules_to_save_checkpoints = ["lora", "output_projection"]

    def __init__(
        self,
        config_type: str,
        projection_dim: Optional[int] = 1024,
        lora_r: Optional[int] = 16,
        lora_alpha: Optional[int] = 32,
        lora_dropout: Optional[float] = 0.05,
        fan_in_fan_out: Optional[bool] = True,
        target_modules: Optional[Union[List[str], str]] = ["query", "value"],
        output_embed_dim: Optional[int] = None,
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
        self.output_embed_dim = output_embed_dim
        if output_embed_dim is not None:
            self.output_projection = nn.Linear(
                projection_dim,
                output_embed_dim,
            )
        else:
            self.output_projection = None
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init_value)
        self.use_all_gather = use_all_gather
        self.init_weights()

    @classmethod
    @add_default_section_for_init("microsoft/model/pretrain/peft/lora/bletchley/v1")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/model/pretrain/peft/lora/bletchley/v1")
        config_type = config.getoption("config_type", "0.8B")

        projection_dim = config.getoption("projection_dim", 1024)
        lora_r = config.getoption("lora_r", 16)
        lora_alpha = config.getoption("lora_alpha", 32)
        lora_dropout = config.getoption("lora_dropout", 0.05)
        fan_in_fan_out = config.getoption("fan_in_fan_out", True)
        target_modules = config.getoption("target_modules", ["query", "value"])
        output_embed_dim = config.getoption("output_embed_dim", None)
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
            output_embed_dim=output_embed_dim,
            use_all_gather=use_all_gather,
            logit_scale_init_value=logit_scale_init_value,
            gradient_checkpointing=gradient_checkpointing,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        if pretrained_weight_path is not None:
            inst.from_pretrained(pretrained_weight_path)

        pretrained_lora_weight_path = config.getoption(
            "pretrained_lora_weight_path", None
        )
        pretrained_lora_weight = config.getoption("pretrained_lora_weight", 1.0)
        pretrained_lora_alpha = config.getoption("pretrained_lora_alpha", 32.0)
        if pretrained_lora_weight_path is not None:
            inst.load_lora_weights(
                pretrained_lora_weight_path,
                lora_weights=pretrained_lora_weight,
                lora_alphas=pretrained_lora_alpha,
            )

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
        if self.output_projection is not None:
            text_embeds = self.output_projection(text_embeds)
            image_embeds = self.output_projection(image_embeds)

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


@register_model("microsoft/model/matching/peft/lora/bletchley/v1")
class BletchleyLoraForMatching(GenericPeftModel, PeftWeightLoaderMixin):
    prefix_keys_in_state_dict = {
        "^(?!peft_model\.base_model\.model\.).*": "peft_model.base_model.model."
    }
    replace_keys_in_state_dict = {
        "query.weight": "query.base_layer.weight",
        "query.bias": "query.base_layer.bias",
        "value.weight": "value.base_layer.weight",
        "value.bias": "value.base_layer.bias",
        "text_encoder.projection": "text_projection",
        "image_encoder.projection": "image_projection",
    }
    modules_to_save_checkpoints = ["lora", "output_projection", "classifier"]
    replace_keys_in_peft_state_dict = {
        ".weight": ".base_layer.weight",
        ".bias": ".base_layer.bias",
    }

    def __init__(
        self,
        config_type: str,
        projection_dim: Optional[int] = 1024,
        lora_r: Optional[int] = 16,
        lora_alpha: Optional[int] = 32,
        lora_dropout: Optional[float] = 0.05,
        fan_in_fan_out: Optional[bool] = True,
        target_modules: Optional[Union[List[str], str]] = ["query", "value"],
        output_embed_dim: Optional[int] = None,
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
        self.output_embed_dim = output_embed_dim
        if output_embed_dim is not None:
            self.output_projection = nn.Linear(
                projection_dim,
                output_embed_dim,
            )
        else:
            self.output_projection = None
        self.classifier = nn.Linear(1, 1)

        self.init_weights()
        self.classifier.weight.data.fill_(5.0)

    @classmethod
    @add_default_section_for_init("microsoft/model/matching/peft/lora/bletchley/v1")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/model/matching/peft/lora/bletchley/v1")
        config_type = config.getoption("config_type", "0.8B")

        projection_dim = config.getoption("projection_dim", 1024)
        lora_r = config.getoption("lora_r", 16)
        lora_alpha = config.getoption("lora_alpha", 32)
        lora_dropout = config.getoption("lora_dropout", 0.05)
        fan_in_fan_out = config.getoption("fan_in_fan_out", True)
        target_modules = config.getoption("target_modules", ["query", "value"])
        output_embed_dim = config.getoption("output_embed_dim", None)

        inst = cls(
            config_type=config_type,
            projection_dim=projection_dim,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            target_modules=target_modules,
            output_embed_dim=output_embed_dim,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        if pretrained_weight_path is not None:
            inst.from_pretrained(pretrained_weight_path)
        pretrained_lora_weight_path = config.getoption(
            "pretrained_lora_weight_path", None
        )
        pretrained_lora_weight = config.getoption("pretrained_lora_weight", 1.0)
        pretrained_lora_alpha = config.getoption("pretrained_lora_alpha", 32.0)
        if pretrained_lora_weight_path is not None:
            inst.load_lora_weights(
                pretrained_lora_weight_path,
                lora_weights=pretrained_lora_weight,
                lora_alphas=pretrained_lora_alpha,
            )

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
        if self.output_projection is not None:
            text_embeds = self.output_projection(text_embeds)
            image_embeds = self.output_projection(image_embeds)

        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        scores = torch.sum(text_embeds * image_embeds, dim=-1, keepdim=True)

        outputs = self.classifier(scores)
        return ClassificationOutputs(outputs=outputs)


@register_model("microsoft/model/pretrain/peft/lora/bletchley/v1/text")
class BletchleyLoraForTextPretrain(GenericPeftModel, PeftWeightLoaderMixin):
    prefix_keys_in_state_dict = {
        "^(?!peft_model\.base_model\.model\.).*": "peft_model.base_model.model."
    }
    replace_keys_in_state_dict = {
        "query.weight": "query.base_layer.weight",
        "query.bias": "query.base_layer.bias",
        "value.weight": "value.base_layer.weight",
        "value.bias": "value.base_layer.bias",
        "query_encoder.projection": "query_projection",
        "doc_encoder.projection": "doc_projection",
    }
    replace_keys_in_peft_state_dict = {
        ".weight": ".base_layer.weight",
        ".bias": ".base_layer.bias",
    }
    modules_to_save_checkpoints = ["lora", "output_projection"]

    def __init__(
        self,
        query_config_type: str,
        doc_config_type: str,
        projection_dim: Optional[int] = 1024,
        lora_r: Optional[int] = 16,
        lora_alpha: Optional[int] = 32,
        lora_dropout: Optional[float] = 0.05,
        fan_in_fan_out: Optional[bool] = True,
        target_modules: Optional[Union[List[str], str]] = ["query", "value"],
        output_embed_dim: Optional[int] = None,
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
        self.output_embed_dim = output_embed_dim
        if output_embed_dim is not None:
            self.output_projection = nn.Linear(
                projection_dim,
                output_embed_dim,
            )
        else:
            self.output_projection = None
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init_value)
        self.use_all_gather = use_all_gather

        self.init_weights()

    @classmethod
    @add_default_section_for_init(
        "microsoft/model/pretrain/peft/lora/bletchley/v1/text"
    )
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section(
            "microsoft/model/pretrain/peft/lora/bletchley/v1/text"
        )
        query_config_type = config.getoption("query_config_type", "0.8B")
        doc_config_type = config.getoption("doc_config_type", "0.8B")
        projection_dim = config.getoption("projection_dim", 1024)
        lora_r = config.getoption("lora_r", 16)
        lora_alpha = config.getoption("lora_alpha", 32)
        lora_dropout = config.getoption("lora_dropout", 0.05)
        fan_in_fan_out = config.getoption("fan_in_fan_out", True)
        target_modules = config.getoption("target_modules", ["query", "value"])
        output_embed_dim = config.getoption("output_embed_dim", None)
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
            output_embed_dim=output_embed_dim,
            use_all_gather=use_all_gather,
            logit_scale_init_value=logit_scale_init_value,
            gradient_checkpointing=gradient_checkpointing,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        if pretrained_weight_path is not None:
            inst.from_pretrained(pretrained_weight_path)

        pretrained_lora_weight_path = config.getoption(
            "pretrained_lora_weight_path", None
        )
        pretrained_lora_weight = config.getoption("pretrained_lora_weight", 1.0)
        pretrained_lora_alpha = config.getoption("pretrained_lora_alpha", 32.0)
        if pretrained_lora_weight_path is not None:
            inst.load_lora_weights(
                pretrained_lora_weight_path,
                lora_weights=pretrained_lora_weight,
                lora_alphas=pretrained_lora_alpha,
            )

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
        if self.output_projection is not None:
            query_embeds = self.output_projection(query_embeds)
            doc_embeds = self.output_projection(doc_embeds)

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


@register_model("microsoft/model/matching/peft/lora/bletchley/v1/text")
class BletchleyLoraForTextMatching(GenericPeftModel, PeftWeightLoaderMixin):
    prefix_keys_in_state_dict = {
        "^(?!peft_model\.base_model\.model\.).*": "peft_model.base_model.model."
    }
    replace_keys_in_state_dict = {
        "query.weight": "query.base_layer.weight",
        "query.bias": "query.base_layer.bias",
        "value.weight": "value.base_layer.weight",
        "value.bias": "value.base_layer.bias",
        "query_encoder.projection": "query_projection",
        "doc_encoder.projection": "doc_projection",
    }
    modules_to_save_checkpoints = ["lora", "output_projection", "classifier"]
    replace_keys_in_peft_state_dict = {
        ".weight": ".base_layer.weight",
        ".bias": ".base_layer.bias",
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
        target_modules: Optional[Union[List[str], str]] = ["query", "value"],
        output_embed_dim: Optional[int] = None,
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
        self.output_embed_dim = output_embed_dim
        if output_embed_dim is not None:
            self.output_projection = nn.Linear(
                projection_dim,
                output_embed_dim,
            )
        else:
            self.output_projection = None
        self.classifier = nn.Linear(1, 1)

        self.init_weights()
        self.classifier.weight.data.fill_(5.0)

    @classmethod
    @add_default_section_for_init(
        "microsoft/model/matching/peft/lora/bletchley/v1/text"
    )
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section(
            "microsoft/model/matching/peft/lora/bletchley/v1/text"
        )
        query_config_type = config.getoption("query_config_type", "0.8B")
        doc_config_type = config.getoption("doc_config_type", "0.8B")
        projection_dim = config.getoption("projection_dim", 1024)
        lora_r = config.getoption("lora_r", 16)
        lora_alpha = config.getoption("lora_alpha", 32)
        lora_dropout = config.getoption("lora_dropout", 0.05)
        fan_in_fan_out = config.getoption("fan_in_fan_out", True)
        target_modules = config.getoption("target_modules", ["query", "value"])
        output_embed_dim = config.getoption("output_embed_dim", None)

        inst = cls(
            query_config_type=query_config_type,
            doc_config_type=doc_config_type,
            projection_dim=projection_dim,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            target_modules=target_modules,
            output_embed_dim=output_embed_dim,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        if pretrained_weight_path is not None:
            inst.from_pretrained(pretrained_weight_path)
        pretrained_lora_weight_path = config.getoption(
            "pretrained_lora_weight_path", None
        )
        pretrained_lora_weight = config.getoption("pretrained_lora_weight", 1.0)
        pretrained_lora_alpha = config.getoption("pretrained_lora_alpha", 32.0)
        if pretrained_lora_weight_path is not None:
            inst.load_lora_weights(
                pretrained_lora_weight_path,
                lora_weights=pretrained_lora_weight,
                lora_alphas=pretrained_lora_alpha,
            )

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
        if self.output_projection is not None:
            query_embeds = self.output_projection(query_embeds)
            doc_embeds = self.output_projection(doc_embeds)

        query_embeds = query_embeds / query_embeds.norm(dim=-1, keepdim=True)
        doc_embeds = doc_embeds / doc_embeds.norm(dim=-1, keepdim=True)
        scores = torch.sum(query_embeds * doc_embeds, dim=-1, keepdim=True)

        outputs = self.classifier(scores)
        return ClassificationOutputs(outputs=outputs)

@register_model("microsoft/model/matching/peft/lora/bletchley/v1/text/distill")
class BletchleyLoraForTextMatchingDistill(GenericPeftModel, PeftWeightLoaderMixin):
    prefix_keys_in_state_dict = {
        "^(?!peft_model\.base_model\.model\.|teacher).*": "peft_model.base_model.model."
    }
    replace_keys_in_state_dict = {
        r"^(peft_model.*?)query\.weight": r"\1query.base_layer.weight",
        r"^(peft_model.*?)query\.bias": r"\1query.base_layer.bias",
        r"^(peft_model.*?)value\.weight": r"\1value.base_layer.weight",
        r"^(peft_model.*?)value\.bias": r"\1value.base_layer.bias",
        "query_encoder.projection": "query_projection",
        "doc_encoder.projection": "doc_projection",
        "teacher_query_encoder.projection": "teacher_query_projection",
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
        target_modules: Optional[Union[List[str], str]] = ["query", "value"],
        momentum: Optional[float] = 1.0,
        distill_weight: Optional[float] = 0.5,
    ):
        super().__init__()
        self.momentum = momentum
        self.distill_weight = distill_weight

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

        query_config = get_bletchley_text_config(
            query_config_type
        )
        self.teacher_query_encoder = BletchleyTextEncoder(
            query_config, add_projection_layer=False
        )
        self.teacher_query_projection = nn.Linear(
            query_config.hidden_size,
            projection_dim,
        )

        self.init_weights()
        self.classifier.weight.data.fill_(5.0)

        for p in self.teacher_query_encoder.parameters():
            p.requires_grad = False
        for p in self.teacher_query_projection.parameters():
            p.requires_grad = False


    @classmethod
    @add_default_section_for_init(
        "microsoft/model/matching/peft/lora/bletchley/v1/text/distill"
    )
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section(
            "microsoft/model/matching/peft/lora/bletchley/v1/text/distill"
        )
        query_config_type = config.getoption("query_config_type", "0.8B")
        doc_config_type = config.getoption("doc_config_type", "0.8B")
        projection_dim = config.getoption("projection_dim", 1024)
        lora_r = config.getoption("lora_r", 16)
        lora_alpha = config.getoption("lora_alpha", 32)
        lora_dropout = config.getoption("lora_dropout", 0.05)
        fan_in_fan_out = config.getoption("fan_in_fan_out", True)
        target_modules = config.getoption("target_modules", ["query", "value"])
        momentum = config.getoption("momentum", 1.0)
        distill_weight = config.getoption("distill_weight", 0.5)

        inst = cls(
            query_config_type=query_config_type,
            doc_config_type=doc_config_type,
            projection_dim=projection_dim,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            target_modules=target_modules,
            momentum=momentum,
            distill_weight=distill_weight
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        if pretrained_weight_path is not None:
            inst.from_pretrained(pretrained_weight_path)

        pretrained_lora_weight_path = config.getoption(
            "pretrained_lora_weight_path", None
        )
        pretrained_lora_weight = config.getoption("pretrained_lora_weight", 1.0)
        pretrained_lora_alpha = config.getoption("pretrained_lora_alpha", 32.0)
        if pretrained_lora_weight_path is not None:
            inst.load_lora_weights(
                pretrained_lora_weight_path,
                lora_weights=pretrained_lora_weight,
                lora_alphas=pretrained_lora_alpha,
            )

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

        _keys = [key for key in state_dict.keys() if key.startswith("query_encoder")]
        for _key in _keys:
            state_dict["teacher_"+_key] = state_dict[_key].clone()

        super().from_pretrained(state_dict=state_dict)
    
    @torch.no_grad()        
    def _momentum_update(self):          
        for param, param_m in zip(self.peft_model.query_encoder.parameters(), self.teacher_query_encoder.parameters()):
            print(param)
            print(param_m)
            param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
        for param, param_m in zip(self.peft_model.query_projection.parameters(), self.teacher_query_projection.parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)        


    @autocast()
    def forward(
        self,
        query_input_ids=None,
        query_attention_mask=None,
        doc_input_ids=None,
        doc_attention_mask=None,
        labels=None,
    ):
        query_embeds, doc_embeds = self.peft_model(
            query_input_ids=query_input_ids,
            query_attention_mask=query_attention_mask,
            doc_input_ids=doc_input_ids,
            doc_attention_mask=doc_attention_mask,
            return_dict=False,
        )

        #self._momentum_update()
        query_embeds = query_embeds / query_embeds.norm(dim=-1, keepdim=True)
        doc_embeds = doc_embeds / doc_embeds.norm(dim=-1, keepdim=True)
        scores = torch.sum(query_embeds * doc_embeds, dim=-1, keepdim=True)
        outputs = self.classifier(scores)

        teacher_query_outputs = self.teacher_query_encoder(
            input_ids=query_input_ids, attention_mask=query_attention_mask
        )

        teacher_query_embeds = teacher_query_outputs[:, 0]
        teacher_query_embeds = self.teacher_query_projection(teacher_query_embeds)
        teacher_query_embeds = teacher_query_embeds / teacher_query_embeds.norm(dim=-1, keepdim=True)

        if self.training:
            loss1 = (            
                nn.MSELoss(reduction="none")(teacher_query_embeds, query_embeds)
                .sum(dim=-1)
                .mean()
            )
            loss2 = (            
                nn.MSELoss(reduction="none")(outputs, labels.unsqueeze(1))
                .sum(dim=-1)
                .mean()
            )
            loss = self.distill_weight*loss1 + (1-self.distill_weight)*loss2
            return LossOutputs(loss = loss)

        return ClassificationOutputs(outputs=outputs)
