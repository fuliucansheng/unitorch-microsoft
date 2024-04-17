# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torch.cuda.amp import autocast
from unitorch.models import GenericModel
from unitorch.models.clip.modeling import AllGather, _clip_loss
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


class Mlp(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU
    ):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)

        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim=768,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_head_dim=None,
    ):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim

        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)

        self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
        self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        self.proj = nn.Linear(all_head_dim, dim)

    def forward(self, x, attention_mask=None):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (
                    self.q_bias,
                    torch.zeros_like(self.v_bias, requires_grad=False),
                    self.v_bias,
                )
            )

        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if attention_mask is not None:
            attention_mask = attention_mask.bool()
            attn = attn.masked_fill(~attention_mask[:, None, None, :], float("-inf"))
        attn = attn.softmax(dim=-1).type_as(x)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)

        return x


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
    ):
        super().__init__()

        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)

        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        init_values=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_head_dim=None,
    ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_head_dim=attn_head_dim,
        )

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer
        )
        self.norm = norm_layer(dim)

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.norm1(x), attention_mask=attention_mask)

        return x + self.mlp(self.norm(x))


class BletchleyImageEncoder(nn.Module):
    def __init__(
        self,
        config_type,
        add_projection_layer=True,
        gradient_checkpointing=False,
    ):
        super().__init__()
        if config_type == "0.3B":
            self.hidden_layers = 12
            self.hidden_size = 768
            self.num_heads = 12
        elif config_type == "0.8B":
            self.hidden_layers = 12
            self.hidden_size = 768
            self.num_heads = 12
        elif config_type == "2.5B":
            self.hidden_layers = 24
            self.hidden_size = 1024
            self.num_heads = 16
        else:
            raise Exception(f"Not Supported Config Tpye, {config_type}")

        self.gradient_checkpointing = gradient_checkpointing
        self.patch_embed = PatchEmbed(embed_dim=self.hidden_size)
        self.num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, self.hidden_size)
        )
        self.encoder = nn.ModuleList(
            [
                Block(dim=self.hidden_size, num_heads=self.num_heads)
                for i in range(self.hidden_layers)
            ]
        )
        self.projection = (
            nn.Linear(self.hidden_size, self.hidden_size)
            if add_projection_layer
            else None
        )

    def image_features(self, x):
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        return x

    def forward(self, images):
        outputs = self.image_features(images)
        for blk in self.encoder:
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(blk),
                    outputs,
                )
            else:
                outputs = blk(outputs)

        if self.projection:
            outputs = self.projection(outputs[:, 0])

        return outputs


class BletchleyTextEncoder(nn.Module):
    def __init__(
        self,
        config_type,
        add_projection_layer=True,
        gradient_checkpointing=False,
    ):
        super().__init__()
        if config_type == "0.15B":
            self.hidden_layers = 3
            self.hidden_size = 768
            self.num_heads = 12
        elif config_type == "0.3B":
            self.hidden_layers = 6
            self.hidden_size = 768
            self.num_heads = 12
        elif config_type == "0.8B":
            self.hidden_layers = 12
            self.hidden_size = 768
            self.num_heads = 12
        elif config_type == "2.5B":
            self.hidden_layers = 24
            self.hidden_size = 1024
            self.num_heads = 16
        else:
            raise Exception(f"Not Supported Config Tpye, {config_type}")

        self.gradient_checkpointing = gradient_checkpointing
        self.pad_token_id = 1
        self.vocab_size = 250002
        self.num_text_pos_embed = 128
        self.word_embeddings = nn.Embedding(
            self.vocab_size, self.hidden_size, padding_idx=0
        )
        self.text_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_text_pos_embed, self.hidden_size)
        )
        self.encoder = nn.ModuleList(
            [
                Block(dim=self.hidden_size, num_heads=self.num_heads)
                for i in range(self.hidden_layers)
            ]
        )
        self.projection = (
            nn.Linear(self.hidden_size, self.hidden_size)
            if add_projection_layer
            else None
        )

    def forward(self, input_ids, attention_mask):
        _, seq_len = input_ids.size()
        outputs = self.word_embeddings(input_ids)
        outputs += self.text_pos_embed[:, :seq_len, :]
        for blk in self.encoder:
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(blk),
                    outputs,
                    attention_mask,
                )
            else:
                outputs = blk(
                    outputs,
                    attention_mask=attention_mask,
                )

        if self.projection:
            outputs = self.projection(outputs[:, 0])

        return outputs


@register_model("microsoft/model/pretrain/bletchley/v3")
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

        self.projection_dim = projection_dim
        self.text_embed_dim = self.text_encoder.hidden_size
        self.image_embed_dim = self.image_encoder.hidden_size
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
    @add_default_section_for_init("microsoft/model/pretrain/bletchley/v3")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/model/pretrain/bletchley/v3")
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


@register_model("microsoft/model/matching/bletchley/v3")
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

        self.output_text_embed = output_text_embed
        self.output_image_embed = output_image_embed

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

        self.projection_dim = projection_dim
        self.text_embed_dim = self.text_encoder.hidden_size
        self.image_embed_dim = self.image_encoder.hidden_size
        self.text_projection = nn.Linear(
            self.text_embed_dim,
            self.projection_dim,
        )
        self.image_projection = nn.Linear(
            self.image_embed_dim,
            self.projection_dim,
        )

        self.classifier = nn.Linear(1, 1)

        self.init_weights()
        self.classifier.weight.data.fill_(5.0)

        if freeze_base_model:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

            for p in self.image_encoder.parameters():
                p.requires_grad = False

    @classmethod
    @add_default_section_for_init("microsoft/model/matching/bletchley/v3")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/model/matching/bletchley/v3")
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
            image_outputs = self.image_encoder(images)
            image_embeds = self.image_projection(image_outputs[:, 0])
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            return EmbeddingOutputs(embedding=image_embeds)

        if not self.training and self.output_text_embed:
            text_outputs = self.text_encoder(input_ids, attention_mask)
            text_embeds = self.text_projection(text_outputs[:, 0])
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            return EmbeddingOutputs(embedding=text_embeds)

        text_outputs = self.text_encoder(input_ids, attention_mask)
        text_embeds = self.text_projection(text_outputs[:, 0])

        image_outputs = self.image_encoder(images)
        image_embeds = self.image_projection(image_outputs[:, 0])

        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        scores = torch.sum(text_embeds * image_embeds, dim=-1, keepdim=True)

        outputs = self.classifier(scores)
        return ClassificationOutputs(outputs=outputs)


def sce_loss(target, pred, weights=None):
    logits = torch.nn.LogSoftmax(dim=1)(pred)
    # targets = torch.softmax(label, 1)
    loss = -(logits * target).sum(dim=1)
    inf_mask = ~torch.isinf(loss).to(loss.device)
    loss = torch.masked_select(loss, inf_mask)
    if weights is not None:
        assert loss.shape == weights.shape
        loss = loss * weights
    # print('--sce loss: ')
    # print(loss.detach().cpu().numpy())
    loss = loss.mean()
    # print('--mean sce loss: %s' % loss.detach().cpu().numpy())
    return loss


def kl_divergence(alpha, num_classes, device):
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl


def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step, device=None):
    y = y.to(
        device
    )  # y is of shape [batch_size, num_classes], with each row being a one-hot embedding
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)

    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return A + kl_div


def edl_digamma_loss(output, target, num_classes, annealing_coef, device, edl_act):
    if edl_act == "relu":
        alpha = torch.relu(output) + 1.0
    elif edl_act == "elu":
        alpha = torch.nn.ELU()(output) + 2.0
    S = torch.sum(alpha, dim=1, keepdim=True)
    A = torch.sum(
        target * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True
    )
    kl_alpha = (alpha - 1) * (1 - target) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device)
    loss = torch.mean(A + kl_div)
    return loss


def edl_mse_loss(
    output, target, annealing_coef, device, edl_act="elu", num_classes=1, weights=None
):
    if edl_act == "relu":
        alpha = torch.relu(output) + 1.0
    elif edl_act == "elu":
        alpha = torch.nn.ELU()(output) + 2.0
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum((target - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
    )
    loglikelihood = loglikelihood_err + loglikelihood_var
    kl_alpha = (alpha - 1) * (1 - target) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device)
    loss = loglikelihood + kl_div
    inf_mask = ~torch.isinf(loss).to(loss.device)
    loss = torch.masked_select(loss, inf_mask)
    if weights is not None:
        assert loss.shape == weights.shape
        loss = loss * weights
    # print("--edl loss")
    # print(loss.reshape(1, -1).detach().cpu().numpy())
    loss = torch.mean(loss)
    # print('--mean edl loss: %s' % loss.detach().cpu().numpy())
    return loss


def edl_log_loss(output, target, num_classes, annealing_coef, device, edl_act):
    if edl_act == "relu":
        alpha = torch.relu(output) + 1.0
    elif edl_act == "elu":
        alpha = torch.nn.ELU()(output) + 2.0
    S = torch.sum(alpha, dim=1, keepdim=True)
    A = torch.sum(target * (torch.log(S) - torch.log(alpha)), dim=1, keepdim=True)
    kl_alpha = (alpha - 1) * (1 - target) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device)
    loss = torch.mean(A + kl_div)
    return loss


@register_model("microsoft/model/matching/bletchley/uncertainty/v3")
class bletchleyformatchingUncertainty(GenericModel):
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
        num_classes: Optional[int] = 2,
        annealing_coef: Optional[float] = 1.0,
        edl_act: Optional[str] = "elu",
    ):
        super().__init__()

        self.output_text_embed = output_text_embed
        self.output_image_embed = output_image_embed
        self.num_classes = num_classes
        self.annealing_coef = annealing_coef
        self.edl_act = edl_act

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

        self.projection_dim = projection_dim
        self.text_embed_dim = self.text_encoder.hidden_size
        self.image_embed_dim = self.image_encoder.hidden_size
        self.text_projection = nn.Linear(
            self.text_embed_dim,
            self.projection_dim,
        )
        self.image_projection = nn.Linear(
            self.image_embed_dim,
            self.projection_dim,
        )

        self.classifier = nn.Linear(1, 1)

        self.init_weights()
        self.classifier.weight.data.fill_(5.0)

        if freeze_base_model:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

            for p in self.image_encoder.parameters():
                p.requires_grad = False

    @classmethod
    @add_default_section_for_init("microsoft/model/matching/bletchley/uncertainty/v3")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/model/matching/bletchley/uncertainty/v3")
        config_type = config.getoption("config_type", "0.8B")

        projection_dim = config.getoption("projection_dim", 1024)
        freeze_base_model = config.getoption("freeze_base_model", True)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)
        output_text_embed = config.getoption("output_text_embed", False)
        output_image_embed = config.getoption("output_image_embed", False)
        num_classes = config.getoption("num_classes", 2)
        annealing_coef = config.getoption("annealing_coef", 1.0)
        edl_act = config.getoption("edl_act", "elu")

        inst = cls(
            config_type=config_type,
            projection_dim=projection_dim,
            freeze_base_model=freeze_base_model,
            gradient_checkpointing=gradient_checkpointing,
            output_text_embed=output_text_embed,
            output_image_embed=output_image_embed,
            num_classes=num_classes,
            annealing_coef=annealing_coef,
            edl_act=edl_act,
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
        label: Optional[torch.Tensor] = None,
    ):
        if not self.training and self.output_image_embed:
            image_outputs = self.image_encoder(images)
            image_embeds = self.image_projection(image_outputs[:, 0])
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            return EmbeddingOutputs(embedding=image_embeds)

        if not self.training and self.output_text_embed:
            text_outputs = self.text_encoder(input_ids, attention_mask)
            text_embeds = self.text_projection(text_outputs[:, 0])
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            return EmbeddingOutputs(embedding=text_embeds)

        text_outputs = self.text_encoder(input_ids, attention_mask)
        text_embeds = self.text_projection(text_outputs[:, 0])

        image_outputs = self.image_encoder(images)
        image_embeds = self.image_projection(image_outputs[:, 0])

        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        scores = torch.sum(text_embeds * image_embeds, dim=-1, keepdim=True)

        outputs = self.classifier(scores)
        if label is None:
            return ClassificationOutputs(outputs=outputs)

        label = label.unsqueeze(1)
        loss = edl_log_loss(
            outputs,
            label,
            self.num_classes,
            self.annealing_coef,
            device=input_ids.device,
            edl_act=self.edl_act,
        )
        return LossOutputs(loss=loss)


@register_model("microsoft/model/matching/bletchley/3towers")
class BletchleyForMatching3towers(GenericModel):
    replace_keys_in_state_dict = {
        "text_encoder.projection": "text_projection",
        "image_encoder.projection": "image_projection",
        "query_encoder.projection": "query_projection",
    }

    def __init__(
        self,
        config_type: str,
        projection_dim: Optional[int] = 1024,
        freeze_base_model: Optional[bool] = True,
        gradient_checkpointing: Optional[bool] = False,
        output_query_embed: Optional[bool] = False,
        output_text_embed: Optional[bool] = False,
        output_image_embed: Optional[bool] = False,
    ):
        super().__init__()

        self.output_query_embed = output_query_embed
        self.output_text_embed = output_text_embed
        self.output_image_embed = output_image_embed

        self.query_encoder = BletchleyTextEncoder(
            config_type,
            add_projection_layer=False,
            gradient_checkpointing=gradient_checkpointing,
        )
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

        self.projection_dim = projection_dim
        self.query_embed_dim = self.query_encoder.hidden_size
        self.text_embed_dim = self.text_encoder.hidden_size
        self.image_embed_dim = self.image_encoder.hidden_size
        self.query_projection = nn.Linear(
            self.query_embed_dim,
            self.projection_dim,
        )
        self.text_projection = nn.Linear(
            self.text_embed_dim,
            self.projection_dim,
        )
        self.image_projection = nn.Linear(
            self.image_embed_dim,
            self.projection_dim,
        )

        self.classifier1 = nn.Linear(1, 1)
        self.classifier2 = nn.Linear(1, 1)

        self.init_weights()
        self.classifier1.weight.data.fill_(5.0)
        self.classifier2.weight.data.fill_(5.0)

        if freeze_base_model:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

            for p in self.image_encoder.parameters():
                p.requires_grad = False

            for p in self.query_encoder.parameters():
                p.requires_grad = False

    @classmethod
    @add_default_section_for_init("microsoft/model/matching/bletchley/3towers")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/model/matching/bletchley/3towers")
        config_type = config.getoption("config_type", "0.8B")

        projection_dim = config.getoption("projection_dim", 1024)
        freeze_base_model = config.getoption("freeze_base_model", True)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)

        output_query_embed = config.getoption("output_query_embed", False)
        output_text_embed = config.getoption("output_text_embed", False)
        output_image_embed = config.getoption("output_image_embed", False)

        inst = cls(
            config_type=config_type,
            projection_dim=projection_dim,
            freeze_base_model=freeze_base_model,
            gradient_checkpointing=gradient_checkpointing,
            output_query_embed=output_query_embed,
            output_text_embed=output_text_embed,
            output_image_embed=output_image_embed,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        if pretrained_weight_path is not None:
            inst.from_pretrained(pretrained_weight_path)

        return inst

    def get_query_embedding(self, input_ids, attention_mask):
        query_outputs = self.query_encoder(input_ids, attention_mask)
        query_embeds = self.query_projection(query_outputs[:, 0])
        query_embeds = query_embeds / query_embeds.norm(dim=-1, keepdim=True)
        return query_embeds

    def get_text_embedding(self, input_ids, attention_mask):
        text_outputs = self.text_encoder(input_ids, attention_mask)
        text_embeds = self.text_projection(text_outputs[:, 0])
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        return text_embeds

    def get_image_embedding(self, images):
        image_outputs = self.image_encoder(images)
        image_embeds = self.image_projection(image_outputs[:, 0])
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        return image_embeds

    @autocast()
    def forward(
        self,
        query_input_ids: torch.Tensor = None,
        input_ids: torch.Tensor = None,
        images: torch.Tensor = None,
        query_attention_mask: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
    ):
        if not self.training:
            if (
                self.output_image_embed
                and not self.output_text_embed
                and not self.output_query_embed
            ):
                image_embeds = self.get_image_embedding(images)
                return EmbeddingOutputs(embedding=image_embeds)

            if (
                self.output_text_embed
                and not self.output_image_embed
                and not self.output_query_embed
            ):
                text_embeds = self.get_text_embedding(input_ids, attention_mask)
                return EmbeddingOutputs(embedding=text_embeds)
            if (
                self.output_query_embed
                and not self.output_text_embed
                and not self.output_image_embed
            ):
                query_embeds = self.get_query_embedding(
                    query_input_ids, query_attention_mask
                )
                return EmbeddingOutputs(embedding=query_embeds)

            if (
                self.output_image_embed
                and self.output_text_embed
                and self.output_query_embed
            ):
                image_embeds = self.get_image_embedding(images)
                text_embeds = self.get_text_embedding(input_ids, attention_mask)
                query_embeds = self.get_query_embedding(
                    query_input_ids, query_attention_mask
                )
                return EmbeddingOutputs(
                    embedding1=image_embeds,
                    embedding2=text_embeds,
                    embedding3=query_embeds,
                )

        query_embeds = self.get_query_embedding(query_input_ids, query_attention_mask)
        text_embeds = self.get_text_embedding(input_ids, attention_mask)
        image_embeds = self.get_image_embedding(images)

        scores1 = torch.sum(query_embeds * image_embeds, dim=-1, keepdim=True)
        scores2 = torch.sum(query_embeds * text_embeds, dim=-1, keepdim=True)

        outputs = self.classifier1(scores1) + self.classifier2(scores2)
        return ClassificationOutputs(outputs=outputs)
