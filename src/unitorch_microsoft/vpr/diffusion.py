# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import json
import torch
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torch.cuda.amp import autocast
import diffusers.schedulers as schedulers
from transformers import DistilBertConfig, DistilBertModel
from diffusers.schedulers import SchedulerMixin
from diffusers.utils.torch_utils import randn_tensor
from diffusers.models.attention_processor import LoRAAttnProcessor, LoRAAttnProcessor2_0
from diffusers.models import (
    UNet2DModel,
    UNet2DConditionModel,
    AutoencoderKL,
)
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models import GenericModel
from unitorch.models.clip.modeling import AllGather, _clip_loss
from unitorch.cli import (
    add_default_section_for_init,
    add_default_section_for_function,
    register_model,
)
from unitorch.cli.models import diffusion_model_decorator
from unitorch.cli.models import EmbeddingOutputs, LossOutputs
from unitorch.cli.models.diffusers import pretrained_diffusers_infos
from unitorch_microsoft import cached_path


@register_model("microsoft/vpr/diffusion/stable/argus", diffusion_model_decorator)
class StableForArgusGeneration(GenericModel):
    prefix_keys_in_state_dict = {
        # unet weights
        "^conv_in.*": "unet.",
        "^conv_norm_out.*": "unet.",
        "^conv_out.*": "unet.",
        "^time_embedding.*": "unet.",
        "^up_blocks.*": "unet.",
        "^mid_block.*": "unet.",
        "^down_blocks.*": "unet.",
        # vae weights
        "^encoder.*": "vae.",
        "^decoder.*": "vae.",
        "^post_quant_conv.*": "vae.",
        "^quant_conv.*": "vae.",
    }

    replace_keys_in_state_dict = {
        "\.query\.": ".to_q.",
        "\.key\.": ".to_k.",
        "\.value\.": ".to_v.",
        "\.proj_attn\.": ".to_out.0.",
    }

    def __init__(
        self,
        config_path: str,
        text_config_path: str,
        vae_config_path: str,
        scheduler_config_path: str,
        num_train_timesteps: Optional[int] = 1000,
        num_infer_timesteps: Optional[int] = 50,
        freeze_vae_encoder: Optional[bool] = False,
        freeze_text_encoder: Optional[bool] = True,
        seed: Optional[int] = 1123,
    ):
        super().__init__()
        self.seed = seed
        self.num_train_timesteps = num_train_timesteps
        self.num_infer_timesteps = num_infer_timesteps

        config_dict = json.load(open(config_path))
        self.unet = UNet2DConditionModel.from_config(config_dict)

        text_config = DistilBertConfig.from_json_file(text_config_path)
        self.text = DistilBertModel(text_config)

        vae_config_dict = json.load(open(vae_config_path))
        self.vae = AutoencoderKL.from_config(vae_config_dict)

        scheduler_config_dict = json.load(open(scheduler_config_path))
        scheduler_class_name = scheduler_config_dict.get("_class_name", "DDPMScheduler")
        assert hasattr(schedulers, scheduler_class_name)
        scheduler_class = getattr(schedulers, scheduler_class_name)
        assert issubclass(scheduler_class, SchedulerMixin)
        scheduler_config_dict["num_train_timesteps"] = num_train_timesteps
        self.scheduler = scheduler_class.from_config(scheduler_config_dict)

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.num_channels_latents = self.unet.config.in_channels

        if freeze_vae_encoder:
            for param in self.vae.parameters():
                param.requires_grad = False

        if freeze_text_encoder:
            for param in self.text.parameters():
                param.requires_grad = False

        self.scheduler.set_timesteps(num_inference_steps=self.num_infer_timesteps)

    @classmethod
    @add_default_section_for_init("microsoft/vpr/diffusion/stable/argus")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/vpr/diffusion/stable/argus")
        pretrained_name = config.getoption("pretrained_name", "stable-v2")
        pretrain_infos = nested_dict_value(pretrained_diffusers_infos, pretrained_name)

        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrain_infos, "unet", "config"),
        )
        config_path = cached_path(config_path)

        text_config_path = config.getoption("text_config_path", None)
        text_config_path = pop_value(
            text_config_path,
            nested_dict_value(pretrain_infos, "text", "config"),
        )
        text_config_path = cached_path(text_config_path)

        vae_config_path = config.getoption("vae_config_path", None)
        vae_config_path = pop_value(
            vae_config_path,
            nested_dict_value(pretrain_infos, "vae", "config"),
        )
        vae_config_path = cached_path(vae_config_path)

        scheduler_config_path = config.getoption("scheduler_config_path", None)
        scheduler_config_path = pop_value(
            scheduler_config_path,
            nested_dict_value(pretrain_infos, "scheduler"),
        )
        scheduler_config_path = cached_path(scheduler_config_path)

        num_train_timesteps = config.getoption("num_train_timesteps", 1000)
        num_infer_timesteps = config.getoption("num_infer_timesteps", 50)
        freeze_vae_encoder = config.getoption("freeze_vae_encoder", False)
        freeze_text_encoder = config.getoption("freeze_text_encoder", True)
        seed = config.getoption("seed", 1123)

        inst = cls(
            config_path=config_path,
            text_config_path=text_config_path,
            vae_config_path=vae_config_path,
            scheduler_config_path=scheduler_config_path,
            num_train_timesteps=num_train_timesteps,
            num_infer_timesteps=num_infer_timesteps,
            freeze_vae_encoder=freeze_vae_encoder,
            freeze_text_encoder=freeze_text_encoder,
            seed=seed,
        )

        weight_path = config.getoption("pretrained_weight_path", None)

        if weight_path is None and pretrain_infos is not None:
            weight_path = [
                cached_path(nested_dict_value(pretrain_infos, "unet", "weight")),
                cached_path(nested_dict_value(pretrain_infos, "text", "weight")),
                cached_path(nested_dict_value(pretrain_infos, "vae", "weight")),
            ]

        inst.from_pretrained(weight_path)
        return inst

    @autocast()
    def forward(
        self,
        argus_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        latents = self.vae.encode(argus_embeds).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        noise = torch.randn(latents.shape).to(latents.device)
        batch = latents.size(0)

        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (batch,),
            device=argus_embeds.device,
        ).long()

        noise_latents = self.scheduler.add_noise(
            latents,
            noise,
            timesteps,
        )

        encoder_hidden_states = self.text(input_ids, attention_mask)[0]
        outputs = self.unet(
            noise_latents,
            timesteps,
            encoder_hidden_states,
        ).sample
        if self.scheduler.config.prediction_type == "v_prediction":
            noise = self.scheduler.get_velocity(latents, noise, timesteps)

        loss = F.mse_loss(outputs, noise, reduction="mean")
        return LossOutputs(loss=loss)

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        height: Optional[int] = 10,
        width: Optional[int] = 10,
    ):
        prompt_embeds = self.text(
            input_ids,
            attention_mask,
        )[0]
        shape = (
            input_ids.size(0),
            self.num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        latents = randn_tensor(
            shape,
            generator=torch.Generator(device=prompt_embeds.device).manual_seed(
                self.seed
            ),
            device=prompt_embeds.device,
            dtype=prompt_embeds.dtype,
        )
        latents = latents * self.scheduler.init_noise_sigma
        for t in self.scheduler.timesteps:
            latents = self.scheduler.scale_model_input(latents, t)
            noise_pred = self.unet(
                latents,
                t,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        latents = 1 / self.vae.config.scaling_factor * latents
        argus_embeds = self.vae.decode(latents).sample
        return EmbeddingOutputs(
            embedding=argus_embeds.reshape(argus_embeds.size(0), -1),
        )
