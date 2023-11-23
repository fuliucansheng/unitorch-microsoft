# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torch.cuda.amp import autocast
from transformers import PreTrainedModel, XLMRobertaConfig
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
    register_model,
)
from unitorch_microsoft.models.bletchley.roberta import (
    RobertaEncoder as RobertaPreLNEncoder,
)


def get_bletchley_text_config(
    config_type: str,
    gradient_checkpointing: Optional[bool] = False,
):
    assert config_type in [
        "0.15B",
        "0.3B",
        "0.8B",
        "2.5B",
        None,
    ], "Invalid config passed"
    config = XLMRobertaConfig.from_pretrained("xlm-roberta-large")
    config.num_hidden_layers = 12
    config.max_position_embeddings = 128
    config.hidden_dropout_prob = 0.0
    config.attention_probs_dropout_prob = 0.0
    config.global_vector_size = 2048 if config_type == "2.5B" else 1024
    config.add_postln_encoder = True
    config.add_preln_encoder = True
    if config_type == "0.3B":
        config.hidden_size = 768
        config.intermediate_size = config.hidden_size * 4
        config.num_hidden_layers = 6
        config.num_attention_heads = 12
        config.add_preln_encoder = False
        config.global_vector_size = 768

    if config_type == "0.15B":
        config.hidden_size = 768
        config.intermediate_size = config.hidden_size * 4
        config.num_hidden_layers = 3
        config.num_attention_heads = 12
        config.add_preln_encoder = False
        config.global_vector_size = 768
    config.gradient_checkpointing = gradient_checkpointing
    return config


def get_bletchley_image_config(
    config_type: str,
    gradient_checkpointing: Optional[bool] = False,
):
    assert config_type in ["0.3B", "0.8B", "2.5B", None], "Invalid config passed"
    config = XLMRobertaConfig.from_pretrained("xlm-roberta-large")
    config.hidden_dropout_prob = 0.0
    config.attention_probs_dropout_prob = 0.0
    config.max_image_position_embeddings = 256
    config.global_vector_size = 1024
    config.patch_size = 14

    if config_type == "2.5B":
        config.hidden_size = 2048
        config.intermediate_size = config.hidden_size * 4
        config.num_hidden_layers = 36
        config.num_attention_heads = 32
        config.global_vector_size = 2048

    if config_type == "0.3B":
        config.hidden_size = 768
        config.intermediate_size = config.hidden_size * 4
        config.num_hidden_layers = 12
        config.num_attention_heads = 12
        config.global_vector_size = 768
        config.patch_size = 16
    config.gradient_checkpointing = gradient_checkpointing
    return config


class BletchleyTextEncoder(PreTrainedModel):
    def __init__(
        self,
        config,
        add_projection_layer: Optional[bool] = True,
    ):
        super().__init__(config)
        self.pad_token_id = config.pad_token_id
        self.add_postln_encoder = config.add_postln_encoder
        self.add_preln_encoder = config.add_preln_encoder
        self.embeddings = RobertaEmbeddings(config)

        if self.add_postln_encoder:
            self.postln_encoder = RobertaPostLNEncoder(config)
        else:
            self.postln_encoder = None

        if self.add_preln_encoder:
            self.preln_encoder = RobertaPreLNEncoder(config)
        else:
            self.preln_encoder = None

        if add_projection_layer:
            self.projection = nn.Linear(config.hidden_size, config.global_vector_size)
        else:
            self.projection = None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ):
        if attention_mask is None:
            attention_mask = input_ids.ne(self.pad_token_id)
        embedding_output = self.embeddings(input_ids=input_ids)
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_ids.size(), input_ids.device
        )

        if self.add_postln_encoder:
            outputs = self.postln_encoder(
                embedding_output, attention_mask=extended_attention_mask
            ).last_hidden_state
        else:
            outputs = embedding_output

        if self.add_preln_encoder:
            outputs = self.preln_encoder(
                outputs, attention_mask=extended_attention_mask
            ).last_hidden_state

        if self.projection is not None:
            return self.projection(outputs[:, 0].float())
        return outputs


class BletchleyImageEncoder(PreTrainedModel):
    def __init__(
        self,
        config,
        add_projection_layer: Optional[bool] = True,
    ):
        super().__init__(config)

        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.cls = nn.Parameter(torch.Tensor(config.hidden_size))
        self.position_embeddings = nn.Embedding(
            config.max_image_position_embeddings, config.hidden_size
        )
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_image_position_embeddings).expand((1, -1)),
        )

        self.encoder = RobertaPreLNEncoder(self.config)
        if add_projection_layer:
            self.projection = nn.Linear(
                self.config.hidden_size, self.config.global_vector_size
            )
        else:
            self.projection = None

    def forward(self, images: torch.Tensor):
        images = self.conv(images).permute([0, 2, 3, 1])
        images = images.view([images.size(0), -1, images.size(-1)])

        position_embeddings = (
            self.position_embeddings.weight
        )  # Handles constant folding error when exporting onnx model
        seq_length = images.size(1)

        images = images + position_embeddings[:seq_length]
        images = torch.cat(
            [self.cls.view(1, 1, -1).repeat(images.size(0), 1, 1), images], dim=1
        )

        attention_mask = torch.ones(images.size()[:-1], device=images.device)
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, images.size()[:-1], images.device
        )

        outputs = self.encoder(
            images, attention_mask=extended_attention_mask
        ).last_hidden_state
        if self.projection is not None:
            return self.projection(outputs[:, 0].float())
        return outputs


@register_model("microsoft/model/pretrain/bletchley/v1")
class BletchleyForPretrain(GenericModel):
    replace_keys_in_state_dict = {
        "text_encoder.projection": "text_projection",
        "image_encoder.projection": "image_projection",
    }

    def __init__(
        self,
        config_type: str,
        projection_dim: Optional[int] = 1024,
        freeze_base_model: Optional[bool] = True,
        logit_scale_init_value: Optional[float] = 2.6592,
        gradient_checkpointing: Optional[bool] = False,
        use_all_gather: Optional[bool] = True,
    ):
        super().__init__()
        text_config = get_bletchley_text_config(config_type, gradient_checkpointing)
        image_config = get_bletchley_image_config(config_type, gradient_checkpointing)
        self.text_encoder = BletchleyTextEncoder(
            text_config, add_projection_layer=False
        )
        self.image_encoder = BletchleyImageEncoder(
            image_config, add_projection_layer=False
        )

        self.projection_dim = projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.image_embed_dim = image_config.hidden_size
        self.use_all_gather = use_all_gather
        self.text_projection = nn.Linear(
            self.text_embed_dim,
            self.projection_dim,
        )
        self.image_projection = nn.Linear(
            self.image_embed_dim,
            self.projection_dim,
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init_value)

        self.init_weights()

        if freeze_base_model:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

            for p in self.image_encoder.parameters():
                p.requires_grad = False

    @classmethod
    @add_default_section_for_init("microsoft/model/pretrain/bletchley/v1")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/model/pretrain/bletchley/v1")
        config_type = config.getoption("config_type", "0.8B")
        projection_dim = config.getoption("projection_dim", 1024)
        freeze_base_model = config.getoption("freeze_base_model", True)
        logit_scale_init_value = config.getoption("logit_scale_init_value", 2.6592)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)
        use_all_gather = config.getoption("use_all_gather", True)

        inst = cls(
            config_type=config_type,
            projection_dim=projection_dim,
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

    @autocast()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        images: torch.Tensor,
    ):
        text_outputs = self.text_encoder(input_ids, attention_mask)
        text_embeds = self.text_projection(text_outputs[:, 0])

        image_outputs = self.image_encoder(images)
        image_embeds = self.image_projection(image_outputs[:, 0])

        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        if self.use_all_gather and dist.is_initialized():
            text_embeds = self.all_gather(text_embeds)
            image_embeds = self.all_gather(image_embeds)
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.T

        loss = _clip_loss(logits_per_text)
        return LossOutputs(loss=loss)


@register_model("microsoft/model/classification/bletchley/v1")
class BletchleyForClassification(GenericModel):
    replace_keys_in_state_dict = {
        "text_encoder.projection": "text_projection",
        "image_encoder.projection": "image_projection",
    }

    def __init__(
        self,
        config_type: str,
        projection_dim: Optional[int] = 1024,
        num_classes: Optional[int] = 1,
        freeze_base_model: Optional[bool] = True,
        gradient_checkpointing: Optional[bool] = False,
    ):
        super().__init__()
        text_config = get_bletchley_text_config(config_type, gradient_checkpointing)
        image_config = get_bletchley_image_config(config_type, gradient_checkpointing)
        self.text_encoder = BletchleyTextEncoder(
            text_config, add_projection_layer=False
        )
        self.image_encoder = BletchleyImageEncoder(
            image_config, add_projection_layer=False
        )

        self.projection_dim = projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.image_embed_dim = image_config.hidden_size
        self.text_projection = nn.Linear(
            self.text_embed_dim,
            self.projection_dim,
        )
        self.image_projection = nn.Linear(
            self.image_embed_dim,
            self.projection_dim,
        )

        self.classifier = nn.Linear(self.projection_dim * 2, num_classes)
        self.init_weights()

        if freeze_base_model:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

            for p in self.image_encoder.parameters():
                p.requires_grad = False

    @classmethod
    @add_default_section_for_init("microsoft/model/classification/bletchley/v1")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/model/classification/bletchley/v1")
        config_type = config.getoption("config_type", "0.8B")
        projection_dim = config.getoption("projection_dim", 1024)
        num_classes = config.getoption("num_classes", 1)
        freeze_base_model = config.getoption("freeze_base_model", True)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)

        inst = cls(
            config_type=config_type,
            projection_dim=projection_dim,
            num_classes=num_classes,
            freeze_base_model=freeze_base_model,
            gradient_checkpointing=gradient_checkpointing,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        if pretrained_weight_path is not None:
            inst.from_pretrained(pretrained_weight_path)

        return inst

    @autocast()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        images: torch.Tensor,
    ):
        text_outputs = self.text_encoder(input_ids, attention_mask)
        text_embeds = self.text_projection(text_outputs[:, 0])

        image_outputs = self.image_encoder(images)
        image_embeds = self.image_projection(image_outputs[:, 0])
        outputs = self.classifier(
            F.relu(torch.cat([image_embeds, text_embeds], axis=1))
        )
        return ClassificationOutputs(outputs=outputs)


@register_model("microsoft/model/classification/bletchley/v1/text")
class BletchleyForTextClassification(GenericModel):
    replace_keys_in_state_dict = {
        "text_encoder.projection": "text_projection",
    }

    def __init__(
        self,
        config_type: str,
        projection_dim: Optional[int] = 1024,
        num_classes: Optional[int] = 1,
        freeze_base_model: Optional[bool] = True,
        gradient_checkpointing: Optional[bool] = False,
    ):
        super().__init__()
        config = get_bletchley_text_config(config_type, gradient_checkpointing)
        self.text_encoder = BletchleyTextEncoder(config, add_projection_layer=False)

        self.projection_dim = projection_dim
        self.text_embed_dim = config.hidden_size
        self.text_projection = nn.Linear(
            self.text_embed_dim,
            self.projection_dim,
        )

        self.classifier = nn.Linear(self.projection_dim, num_classes)
        self.init_weights()

        if freeze_base_model:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

    @classmethod
    @add_default_section_for_init("microsoft/model/classification/bletchley/v1/text")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/model/classification/bletchley/v1/text")
        config_type = config.getoption("config_type", "0.8B")
        projection_dim = config.getoption("projection_dim", 1024)
        num_classes = config.getoption("num_classes", 1)
        freeze_base_model = config.getoption("freeze_base_model", True)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)

        inst = cls(
            config_type=config_type,
            projection_dim=projection_dim,
            num_classes=num_classes,
            freeze_base_model=freeze_base_model,
            gradient_checkpointing=gradient_checkpointing,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        if pretrained_weight_path is not None:
            inst.from_pretrained(pretrained_weight_path)

        return inst

    @autocast()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        text_outputs = self.text_encoder(input_ids, attention_mask)
        text_embeds = self.text_projection(text_outputs[:, 0])
        outputs = self.classifier(F.relu(text_embeds))
        return ClassificationOutputs(outputs=outputs)


@register_model("microsoft/model/classification/bletchley/v1/image")
class BletchleyForImageClassification(GenericModel):
    replace_keys_in_state_dict = {
        "image_encoder.projection": "image_projection",
    }

    def __init__(
        self,
        config_type: str,
        projection_dim: Optional[int] = 1024,
        num_classes: Optional[int] = 1,
        freeze_base_model: Optional[bool] = True,
        gradient_checkpointing: Optional[bool] = False,
    ):
        super().__init__()
        config = get_bletchley_image_config(config_type, gradient_checkpointing)
        self.image_encoder = BletchleyImageEncoder(config, add_projection_layer=False)

        self.projection_dim = projection_dim
        self.image_embed_dim = config.hidden_size
        self.image_projection = nn.Linear(
            self.image_embed_dim,
            self.projection_dim,
        )

        self.classifier = nn.Linear(self.projection_dim, num_classes)
        self.init_weights()

        if freeze_base_model:
            for p in self.image_encoder.parameters():
                p.requires_grad = False

    @classmethod
    @add_default_section_for_init("microsoft/model/classification/bletchley/v1/image")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/model/classification/bletchley/v1/image")
        config_type = config.getoption("config_type", "0.8B")
        projection_dim = config.getoption("projection_dim", 1024)
        num_classes = config.getoption("num_classes", 1)
        freeze_base_model = config.getoption("freeze_base_model", True)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)

        inst = cls(
            config_type=config_type,
            projection_dim=projection_dim,
            num_classes=num_classes,
            freeze_base_model=freeze_base_model,
            gradient_checkpointing=gradient_checkpointing,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        if pretrained_weight_path is not None:
            inst.from_pretrained(pretrained_weight_path)

        return inst

    @autocast()
    def forward(self, images: torch.Tensor):
        image_outputs = self.image_encoder(images)
        image_embeds = self.image_projection(image_outputs[:, 0])
        outputs = self.classifier(F.relu(image_embeds))
        return ClassificationOutputs(outputs=outputs)


@register_model("microsoft/model/matching/bletchley/v1")
class BletchleyForMatching(GenericModel):
    replace_keys_in_state_dict = {
        "text_encoder.projection": "text_projection",
        "image_encoder.projection": "image_projection",
    }

    def __init__(
        self,
        config_type: str,
        projection_dim: Optional[int] = 1024,
        freeze_base_model: Optional[bool] = True,
        gradient_checkpointing: Optional[bool] = False,
        output_text_embed: Optional[bool] = False,
        output_image_embed: Optional[bool] = False,
    ):
        super().__init__()
        text_config = get_bletchley_text_config(config_type, gradient_checkpointing)
        image_config = get_bletchley_image_config(config_type, gradient_checkpointing)

        self.output_text_embed = output_text_embed
        self.output_image_embed = output_image_embed

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
        )  # text_encoder.projection.weight, text_encoder.projection.bias
        self.text_projection = nn.Linear(
            self.text_embed_dim,
            projection_dim,
        )  # image_encoder.projection.weight,  image_encoder.projection.bias

        self.classifier = nn.Linear(1, 1)

        self.init_weights()
        self.classifier.weight.data.fill_(5.0)

        if freeze_base_model:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

            for p in self.image_encoder.parameters():
                p.requires_grad = False

    @classmethod
    @add_default_section_for_init("microsoft/model/matching/bletchley/v1")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/model/matching/bletchley/v1")
        config_type = config.getoption("config_type", "0.8B")

        projection_dim = config.getoption("projection_dim", 1024)
        freeze_base_model = config.getoption("freeze_base_model", True)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)
        output_text_embed = config.getoption("output_text_embed", False)
        output_image_embed = config.getoption("output_image_embed", False)

        inst = cls(
            config_type=config_type,
            projection_dim=projection_dim,
            freeze_base_model=freeze_base_model,
            gradient_checkpointing=gradient_checkpointing,
            output_text_embed=output_text_embed,
            output_image_embed=output_image_embed,
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
        if not self.training and self.output_image_embed:
            image_outputs = self.image_encoder(
                images=images,
            )
            image_embeds = image_outputs[:, 0]
            image_embeds = self.image_projection(image_embeds)
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            return EmbeddingOutputs(embedding=image_embeds)

        if not self.training and self.output_text_embed:
            text_outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            text_embeds = text_outputs[:, 0]
            text_embeds = self.text_projection(text_embeds)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            return EmbeddingOutputs(embedding=text_embeds)

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

        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        scores = torch.sum(text_embeds * image_embeds, dim=-1, keepdim=True)

        outputs = self.classifier(scores)
        return ClassificationOutputs(outputs=outputs)
