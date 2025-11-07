# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import glob
import json
import torch
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import diffusers.schedulers as schedulers
from transformers import (
    PretrainedConfig,
    UMT5Config,
    UMT5EncoderModel,
    CLIPVisionConfig,
    CLIPVisionModel,
    CLIPVisionModelWithProjection,
)
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import T5EncoderModel
from diffusers.schedulers import SchedulerMixin, FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import (
    compute_loss_weighting_for_sd3,
    compute_density_for_timestep_sampling,
)
from diffusers.models import (
    AutoencoderKLWan,
    WanTransformer3DModel,
)
from diffusers.pipelines import (
    WanPipeline,
    WanImageToVideoPipeline,
)
from unitorch.models import (
    GenericModel,
    GenericOutputs,
    QuantizationConfig,
    QuantizationMixin,
)
from unitorch.models.peft import PeftWeightLoaderMixin
from unitorch.models.diffusers import compute_snr

from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
    register_model,
)
from unitorch.utils import (
    pop_value,
    nested_dict_value,
    is_bfloat16_available,
    is_cuda_available,
)
from unitorch.cli.models.diffusers import (
    pretrained_stable_infos,
    pretrained_stable_extensions_infos,
    load_weight,
)
from torch import autocast
from unitorch.cli.models import diffusion_model_decorator
from unitorch.cli.models import DiffusionOutputs, LossOutputs


class GenericWanModel(GenericModel, QuantizationMixin, PeftWeightLoaderMixin):
    prefix_keys_in_state_dict = {
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
        config2_path: Optional[str] = None,
        quant_config_path: Optional[str] = None,
        num_train_timesteps: Optional[int] = 1000,
        num_infer_timesteps: Optional[int] = 50,
        freeze_vae_encoder: Optional[bool] = True,
        freeze_text_encoder: Optional[bool] = True,
        freeze_transformer_encoder: Optional[bool] = False,
        snr_gamma: Optional[float] = 5.0,
        seed: Optional[int] = 1123,
    ):
        super().__init__()
        self.seed = seed
        self.num_train_timesteps = num_train_timesteps
        self.num_infer_timesteps = num_infer_timesteps
        self.snr_gamma = snr_gamma

        config_dict = json.load(open(config_path))
        self.transformer = WanTransformer3DModel.from_config(config_dict).to(
            torch.bfloat16
        )

        if config2_path is not None:
            config2_dict = json.load(open(config2_path))
            self.transformer2 = WanTransformer3DModel.from_config(config2_dict).to(
                torch.bfloat16
            )

        text_config = UMT5Config.from_json_file(text_config_path)
        self.text = UMT5EncoderModel(text_config).to(torch.bfloat16)

        vae_config_dict = json.load(open(vae_config_path))
        self.vae = AutoencoderKLWan.from_config(vae_config_dict).to(torch.bfloat16)

        scheduler_config_dict = json.load(open(scheduler_config_path))
        scheduler_class_name = scheduler_config_dict.get(
            "_class_name", "FlowMatchEulerDiscreteScheduler"
        )
        assert hasattr(schedulers, scheduler_class_name)
        scheduler_class = getattr(schedulers, scheduler_class_name)
        assert issubclass(scheduler_class, SchedulerMixin)
        scheduler_config_dict["num_train_timesteps"] = num_train_timesteps
        self.scheduler = scheduler_class.from_config(scheduler_config_dict)

        if freeze_vae_encoder:
            for param in self.vae.parameters():
                param.requires_grad = False

        if freeze_text_encoder:
            for param in self.text.parameters():
                param.requires_grad = False

        if freeze_transformer_encoder:
            for param in self.transformer.parameters():
                param.requires_grad = False
            if hasattr(self, "transformer2"):
                for param in self.transformer2.parameters():
                    param.requires_grad = False

        if quant_config_path is not None:
            self.quant_config = QuantizationConfig.from_json_file(quant_config_path)
            self.quantize(
                self.quant_config, ignore_modules=["lm_head", "transformer", "vae"]
            )

    def get_sigmas(self, timesteps, n_dim=4, dtype=torch.float32):
        sigmas = self.scheduler.sigmas.to(device=self.device, dtype=dtype)
        schedule_timesteps = self.scheduler.timesteps.to(self.device)
        timesteps = timesteps.to(self.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def get_prompt_outputs(
        self,
        input_ids: torch.Tensor,
        negative_input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        enable_cpu_offload: Optional[bool] = False,
        cpu_offload_device: Optional[str] = "cpu",
    ):
        if enable_cpu_offload:
            self.text.to(cpu_offload_device)
            input_ids = input_ids.to(cpu_offload_device)
            attention_mask = attention_mask.to(cpu_offload_device)
            negative_input_ids = negative_input_ids.to(cpu_offload_device)
            negative_attention_mask = negative_attention_mask.to(cpu_offload_device)
        prompt_embeds = self.text(
            input_ids,
            attention_mask,
        )[0]
        prompt_embeds = prompt_embeds * attention_mask.unsqueeze(-1)
        negative_prompt_embeds = self.text(
            negative_input_ids,
            negative_attention_mask,
        )[0]
        negative_prompt_embeds = (
            negative_prompt_embeds * negative_attention_mask.unsqueeze(-1)
        )
        if enable_cpu_offload:
            self.text.to("cpu")
        return GenericOutputs(
            prompt_embeds=(
                prompt_embeds.to("cpu") if enable_cpu_offload else prompt_embeds
            ),
            negative_prompt_embeds=(
                negative_prompt_embeds.to("cpu")
                if enable_cpu_offload
                else negative_prompt_embeds
            ),
        )


@register_model("microsoft/model/diffusers/text2video/wan", diffusion_model_decorator)
class WanForText2VideoGeneration(GenericWanModel):
    def __init__(
        self,
        config_path: str,
        text_config_path: str,
        vae_config_path: str,
        scheduler_config_path: str,
        config2_path: Optional[str] = None,
        quant_config_path: Optional[str] = None,
        num_train_timesteps: Optional[int] = 1000,
        num_infer_timesteps: Optional[int] = 50,
        freeze_vae_encoder: Optional[bool] = True,
        freeze_text_encoder: Optional[bool] = True,
        snr_gamma: Optional[float] = 5.0,
        seed: Optional[int] = 1123,
        gradient_checkpointing: Optional[bool] = True,
    ):
        super().__init__(
            config_path=config_path,
            text_config_path=text_config_path,
            vae_config_path=vae_config_path,
            scheduler_config_path=scheduler_config_path,
            config2_path=config2_path,
            quant_config_path=quant_config_path,
            num_train_timesteps=num_train_timesteps,
            num_infer_timesteps=num_infer_timesteps,
            freeze_vae_encoder=freeze_vae_encoder,
            freeze_text_encoder=freeze_text_encoder,
            snr_gamma=snr_gamma,
            seed=seed,
        )
        if gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()

        self.pipeline = WanPipeline(
            vae=self.vae,
            text_encoder=self.text,
            transformer=self.transformer,
            transformer_2=getattr(self, "transformer2", None),
            scheduler=self.scheduler,
            tokenizer=None,
        )
        self.pipeline.set_progress_bar_config(disable=True)

    @classmethod
    @add_default_section_for_init("microsoft/model/diffusers/text2video/wan")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/model/diffusers/text2video/wan")
        pretrained_name = config.getoption("pretrained_name", "wan-v2.2-t2v-14b")
        pretrained_infos = nested_dict_value(pretrained_stable_infos, pretrained_name)

        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_infos, "transformer", "config"),
        )
        config_path = cached_path(config_path)

        config2_path = config.getoption("config2_path", None)
        config2_path = pop_value(
            config2_path,
            nested_dict_value(pretrained_infos, "transformer2", "config"),
        )

        if config2_path is not None:
            config2_path = cached_path(config2_path)

        text_config_path = config.getoption("text_config_path", None)
        text_config_path = pop_value(
            text_config_path,
            nested_dict_value(pretrained_infos, "text", "config"),
        )
        text_config_path = cached_path(text_config_path)

        vae_config_path = config.getoption("vae_config_path", None)
        vae_config_path = pop_value(
            vae_config_path,
            nested_dict_value(pretrained_infos, "vae", "config"),
        )
        vae_config_path = cached_path(vae_config_path)

        scheduler_config_path = config.getoption("scheduler_config_path", None)
        scheduler_config_path = pop_value(
            scheduler_config_path,
            nested_dict_value(pretrained_infos, "scheduler"),
        )
        scheduler_config_path = cached_path(scheduler_config_path)

        quant_config_path = config.getoption("quant_config_path", None)
        if quant_config_path is not None:
            quant_config_path = cached_path(quant_config_path)

        num_train_timesteps = config.getoption("num_train_timesteps", 1000)
        num_infer_timesteps = config.getoption("num_infer_timesteps", 50)
        freeze_vae_encoder = config.getoption("freeze_vae_encoder", True)
        freeze_text_encoder = config.getoption("freeze_text_encoder", True)
        snr_gamma = config.getoption("snr_gamma", 5.0)
        seed = config.getoption("seed", 1123)
        gradient_checkpointing = config.getoption("gradient_checkpointing", True)

        inst = cls(
            config_path=config_path,
            text_config_path=text_config_path,
            vae_config_path=vae_config_path,
            scheduler_config_path=scheduler_config_path,
            config2_path=config2_path,
            quant_config_path=quant_config_path,
            num_train_timesteps=num_train_timesteps,
            num_infer_timesteps=num_infer_timesteps,
            freeze_vae_encoder=freeze_vae_encoder,
            freeze_text_encoder=freeze_text_encoder,
            snr_gamma=snr_gamma,
            seed=seed,
            gradient_checkpointing=gradient_checkpointing,
        )

        weight_path = config.getoption("pretrained_weight_path", None)
        pretrained_weight_folder = config.getoption("pretrained_weight_folder", None)

        state_dict = None
        if weight_path is None and pretrained_weight_folder is not None:
            state_dict = [
                load_weight(
                    glob.glob(f"{pretrained_weight_folder}/transformer/*.safetensors"),
                    prefix_keys={"": "transformer."},
                ),
                load_weight(
                    glob.glob(
                        f"{pretrained_weight_folder}/transformer_2/*.safetensors"
                    ),
                    prefix_keys={"": "transformer2."},
                ),
                load_weight(
                    glob.glob(f"{pretrained_weight_folder}/text_encoder/*.safetensors"),
                    prefix_keys={"": "text."},
                ),
                load_weight(
                    glob.glob(f"{pretrained_weight_folder}/vae/*.safetensors"),
                    prefix_keys={"": "vae."},
                ),
            ]
        elif weight_path is None and pretrained_infos is not None:
            state_dict = [
                load_weight(
                    nested_dict_value(pretrained_infos, "transformer", "weight"),
                    prefix_keys={"": "transformer."},
                ),
                load_weight(
                    nested_dict_value(pretrained_infos, "transformer2", "weight"),
                    prefix_keys={"": "transformer2."},
                ),
                load_weight(
                    nested_dict_value(pretrained_infos, "text", "weight"),
                    prefix_keys={"": "text."},
                ),
                load_weight(
                    nested_dict_value(pretrained_infos, "vae", "weight"),
                    prefix_keys={"": "vae."},
                ),
            ]

        inst.from_pretrained(weight_path, state_dict=state_dict)

        pretrained_lora_names = config.getoption("pretrained_lora_names", None)
        pretrained_lora_weights = config.getoption("pretrained_lora_weights", 1.0)

        if isinstance(pretrained_lora_names, str):
            pretrained_lora_weights_path = nested_dict_value(
                pretrained_stable_extensions_infos,
                pretrained_lora_names,
                "lora",
                "weight",
            )
        elif isinstance(pretrained_lora_names, list):
            pretrained_lora_weights_path = [
                nested_dict_value(
                    pretrained_stable_extensions_infos, name, "lora", "weight"
                )
                for name in pretrained_lora_names
            ]
        else:
            pretrained_lora_weights_path = None

        lora_weights_path = config.getoption(
            "pretrained_lora_weights_path", pretrained_lora_weights_path
        )
        if lora_weights_path is not None:
            inst.load_lora_weights(
                lora_weights_path,
                pretrained_lora_weights,
                replace_keys={},
                save_base_state=False,
            )
        return inst

    @autocast(
        device_type=("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=(torch.bfloat16 if is_bfloat16_available() else torch.float32),
    )
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        latents = self.vae.encode(pixel_values).latent_dist.mode()
        noise = torch.randn(latents.shape).to(latents.device)
        batch = latents.shape[0]
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
            1, self.vae.config.z_dim, 1, 1, 1
        ).to(latents.device, latents.dtype)
        latents = (latents - latents_mean) * latents_std

        # u = compute_density_for_timestep_sampling(
        #     weighting_scheme="none",
        #     batch_size=batch,
        #     logit_mean=0.0,
        #     logit_std=1.0,
        #     mode_scale=1.29,
        # )
        # indices = (u * self.scheduler.config.num_train_timesteps).long()
        # timesteps = self.scheduler.timesteps[indices].to(device=self.device)

        # sigmas = self.get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
        # noise_latents = (1.0 - sigmas) * latents + sigmas * noise

        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (batch,),
            device=pixel_values.device,
        ).long()

        noise_latents = self.scheduler.add_noise(
            latents,
            noise,
            timesteps,
        )

        encoder_hidden_states = self.text(input_ids, attention_mask)[0]
        outputs = self.transformer(
            noise_latents,
            timesteps,
            encoder_hidden_states,
        ).sample
        # weighting = compute_loss_weighting_for_sd3(
        #     weighting_scheme="none", sigmas=sigmas
        # )
        # target = noise - latents
        # loss = torch.mean(
        #     (weighting.float() * (outputs.float() - target.float()) ** 2).reshape(
        #         target.shape[0], -1
        #     ),
        #     1,
        # )
        # loss = loss.mean()
        loss = F.mse_loss(outputs, noise, reduction="mean")
        return LossOutputs(loss=loss)

    @add_default_section_for_function("microsoft/model/diffusers/text2video/wan")
    @autocast(
        device_type=("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=(torch.bfloat16 if is_bfloat16_available() else torch.float32),
    )
    def generate(
        self,
        input_ids: torch.Tensor,
        negative_input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        height: Optional[int] = 480,
        width: Optional[int] = 832,
        num_frames: Optional[int] = 81,
        guidance_scale: Optional[float] = 5.0,
        enable_cpu_offload: Optional[bool] = False,
        cpu_offload_device: Optional[str] = None,
    ):
        if cpu_offload_device is None:
            cpu_offload_device = torch.cuda.current_device()

        if enable_cpu_offload and cpu_offload_device != "cpu":
            self.pipeline.enable_model_cpu_offload(cpu_offload_device)

        outputs = self.get_prompt_outputs(
            input_ids=input_ids,
            negative_input_ids=negative_input_ids,
            attention_mask=attention_mask,
            negative_attention_mask=negative_attention_mask,
            enable_cpu_offload=enable_cpu_offload,
            cpu_offload_device=cpu_offload_device,
        )

        frames = self.pipeline(
            prompt_embeds=outputs.prompt_embeds,
            negative_prompt_embeds=outputs.negative_prompt_embeds,
            generator=torch.Generator(device=self.pipeline.device).manual_seed(
                self.seed
            ),
            num_inference_steps=self.num_infer_timesteps,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            output_type="pt",
        ).frames

        return DiffusionOutputs(outputs=frames.float())


@register_model("microsoft/model/diffusers/image2video/wan", diffusion_model_decorator)
class WanForImage2VideoGeneration(GenericWanModel):
    def __init__(
        self,
        config_path: str,
        text_config_path: str,
        vae_config_path: str,
        scheduler_config_path: str,
        config2_path: Optional[str] = None,
        quant_config_path: Optional[str] = None,
        num_train_timesteps: Optional[int] = 1000,
        num_infer_timesteps: Optional[int] = 50,
        freeze_vae_encoder: Optional[bool] = True,
        freeze_text_encoder: Optional[bool] = True,
        snr_gamma: Optional[float] = 5.0,
        seed: Optional[int] = 1123,
        gradient_checkpointing: Optional[bool] = True,
    ):
        super().__init__(
            config_path=config_path,
            text_config_path=text_config_path,
            vae_config_path=vae_config_path,
            scheduler_config_path=scheduler_config_path,
            config2_path=config2_path,
            quant_config_path=quant_config_path,
            num_train_timesteps=num_train_timesteps,
            num_infer_timesteps=num_infer_timesteps,
            freeze_vae_encoder=freeze_vae_encoder,
            freeze_text_encoder=freeze_text_encoder,
            snr_gamma=snr_gamma,
            seed=seed,
        )
        if gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()
            if hasattr(self, "transformer2"):
                self.transformer2.enable_gradient_checkpointing()

        self.pipeline = WanImageToVideoPipeline(
            vae=self.vae,
            text_encoder=self.text,
            transformer=self.transformer,
            transformer_2=getattr(self, "transformer2", None),
            scheduler=self.scheduler,
            tokenizer=None,
            image_processor=None,
        )
        self.pipeline.set_progress_bar_config(disable=True)

    @classmethod
    @add_default_section_for_init("microsoft/model/diffusers/image2video/wan")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/model/diffusers/image2video/wan")
        pretrained_name = config.getoption("pretrained_name", "wan-v2.2-i2v-14b")
        pretrained_infos = nested_dict_value(pretrained_stable_infos, pretrained_name)

        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_infos, "transformer", "config"),
        )
        config_path = cached_path(config_path)

        config2_path = config.getoption("config2_path", None)
        config2_path = pop_value(
            config2_path,
            nested_dict_value(pretrained_infos, "transformer2", "config"),
        )

        if config2_path is not None:
            config2_path = cached_path(config2_path)

        text_config_path = config.getoption("text_config_path", None)
        text_config_path = pop_value(
            text_config_path,
            nested_dict_value(pretrained_infos, "text", "config"),
        )
        text_config_path = cached_path(text_config_path)

        vae_config_path = config.getoption("vae_config_path", None)
        vae_config_path = pop_value(
            vae_config_path,
            nested_dict_value(pretrained_infos, "vae", "config"),
        )
        vae_config_path = cached_path(vae_config_path)

        scheduler_config_path = config.getoption("scheduler_config_path", None)
        scheduler_config_path = pop_value(
            scheduler_config_path,
            nested_dict_value(pretrained_infos, "scheduler"),
        )
        scheduler_config_path = cached_path(scheduler_config_path)

        quant_config_path = config.getoption("quant_config_path", None)
        if quant_config_path is not None:
            quant_config_path = cached_path(quant_config_path)

        num_train_timesteps = config.getoption("num_train_timesteps", 1000)
        num_infer_timesteps = config.getoption("num_infer_timesteps", 50)
        freeze_vae_encoder = config.getoption("freeze_vae_encoder", True)
        freeze_text_encoder = config.getoption("freeze_text_encoder", True)
        snr_gamma = config.getoption("snr_gamma", 5.0)
        seed = config.getoption("seed", 1123)
        gradient_checkpointing = config.getoption("gradient_checkpointing", True)

        inst = cls(
            config_path=config_path,
            text_config_path=text_config_path,
            vae_config_path=vae_config_path,
            scheduler_config_path=scheduler_config_path,
            config2_path=config2_path,
            quant_config_path=quant_config_path,
            num_train_timesteps=num_train_timesteps,
            num_infer_timesteps=num_infer_timesteps,
            freeze_vae_encoder=freeze_vae_encoder,
            freeze_text_encoder=freeze_text_encoder,
            snr_gamma=snr_gamma,
            seed=seed,
            gradient_checkpointing=gradient_checkpointing,
        )

        weight_path = config.getoption("pretrained_weight_path", None)

        pretrained_weight_folder = config.getoption("pretrained_weight_folder", None)

        state_dict = None
        if weight_path is None and pretrained_weight_folder is not None:
            state_dict = [
                load_weight(
                    glob.glob(f"{pretrained_weight_folder}/transformer/*.safetensors"),
                    prefix_keys={"": "transformer."},
                ),
                load_weight(
                    glob.glob(f"{pretrained_weight_folder}/text_encoder/*.safetensors"),
                    prefix_keys={"": "text."},
                ),
                load_weight(
                    glob.glob(
                        f"{pretrained_weight_folder}/image_encoder/*.safetensors"
                    ),
                    prefix_keys={"": "image."},
                ),
                load_weight(
                    glob.glob(f"{pretrained_weight_folder}/vae/*.safetensors"),
                    prefix_keys={"": "vae."},
                ),
            ]
        elif weight_path is None and pretrained_infos is not None:
            state_dict = [
                load_weight(
                    nested_dict_value(pretrained_infos, "transformer", "weight"),
                    prefix_keys={"": "transformer."},
                ),
                load_weight(
                    nested_dict_value(pretrained_infos, "text", "weight"),
                    prefix_keys={"": "text."},
                ),
                load_weight(
                    nested_dict_value(pretrained_infos, "image", "weight"),
                    prefix_keys={"": "image."},
                ),
                load_weight(
                    nested_dict_value(pretrained_infos, "vae", "weight"),
                    prefix_keys={"": "vae."},
                ),
            ]

        elif weight_path is not None:
            state_dict = load_weight(weight_path)

        if state_dict is not None:
            inst.from_pretrained(state_dict=state_dict)

        pretrained_lora_names = config.getoption("pretrained_lora_names", None)
        pretrained_lora_weights = config.getoption("pretrained_lora_weights", 1.0)

        if isinstance(pretrained_lora_names, str):
            pretrained_lora_weights_path = nested_dict_value(
                pretrained_stable_extensions_infos,
                pretrained_lora_names,
                "lora",
                "weight",
            )
        elif isinstance(pretrained_lora_names, list):
            pretrained_lora_weights_path = [
                nested_dict_value(
                    pretrained_stable_extensions_infos, name, "lora", "weight"
                )
                for name in pretrained_lora_names
            ]
        else:
            pretrained_lora_weights_path = None

        lora_weights_path = config.getoption(
            "pretrained_lora_weights_path", pretrained_lora_weights_path
        )
        if lora_weights_path is not None:
            inst.load_lora_weights(
                lora_weights_path,
                pretrained_lora_weights,
                replace_keys={},
                save_base_state=False,
            )
        return inst

    @autocast(
        device_type=("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=(torch.bfloat16 if is_bfloat16_available() else torch.float32),
    )
    def forward(
        self,
        pixel_values: torch.Tensor,
        vae_pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        latents = self.vae.encode(pixel_values).latent_dist.mode()
        noise = torch.randn(latents.shape).to(latents.device)
        batch = latents.shape[0]
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
            1, self.vae.config.z_dim, 1, 1, 1
        ).to(latents.device, latents.dtype)
        latents = (latents - latents_mean) * latents_std

        # u = compute_density_for_timestep_sampling(
        #     weighting_scheme="none",
        #     batch_size=batch,
        #     logit_mean=0.0,
        #     logit_std=1.0,
        #     mode_scale=1.29,
        # )
        # indices = (u * self.scheduler.config.num_train_timesteps).long()
        # timesteps = self.scheduler.timesteps[indices].to(device=self.device)

        # sigmas = self.get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
        # noise_latents = (1.0 - sigmas) * latents + sigmas * noise

        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (batch,),
            device=pixel_values.device,
        ).long()

        noise_latents = self.scheduler.add_noise(
            latents,
            noise,
            timesteps,
        )

        num_frames = pixel_values.shape[-3]

        video_condition = torch.cat(
            [
                vae_pixel_values.unsqueeze(2),
                vae_pixel_values.new_zeros(
                    vae_pixel_values.shape[0],
                    vae_pixel_values.shape[1],
                    num_frames - 1,
                    vae_pixel_values.shape[-2],
                    vae_pixel_values.shape[-1],
                    device=vae_pixel_values.device,
                ),
            ],
            dim=2,
        )
        latent_condition = self.vae.encode(video_condition).latent_dist.mode()
        latent_condition = latent_condition.to(latents.dtype)
        latent_condition = (latent_condition - latents_mean) * latents_std

        mask_lat_size = torch.ones(
            latents.shape[0], 1, num_frames, latents.shape[-2], latents.shape[-1]
        )
        mask_lat_size[:, :, list(range(1, num_frames))] = 0
        first_frame_mask = mask_lat_size[:, :, 0:1]
        first_frame_mask = torch.repeat_interleave(
            first_frame_mask, dim=2, repeats=self.pipeline.vae_scale_factor_temporal
        )
        mask_lat_size = torch.concat(
            [first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2
        )
        mask_lat_size = mask_lat_size.view(
            latents.shape[0],
            -1,
            self.pipeline.vae_scale_factor_temporal,
            latents.shape[-2],
            latents.shape[-1],
        )
        mask_lat_size = mask_lat_size.transpose(1, 2)
        mask_lat_size = mask_lat_size.to(latent_condition.device)
        condition_latents = torch.concat([mask_lat_size, latent_condition], dim=1)
        latent_model_input = torch.cat([noise_latents, condition_latents], dim=1)

        encoder_hidden_states = self.text(input_ids, attention_mask)[0]
        outputs = self.transformer(
            latent_model_input,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
        ).sample
        # weighting = compute_loss_weighting_for_sd3(
        #     weighting_scheme="none", sigmas=sigmas
        # )
        # target = noise - latents
        # loss = torch.mean(
        #     (weighting.float() * (outputs.float() - target.float()) ** 2).reshape(
        #         target.shape[0], -1
        #     ),
        #     1,
        # )
        # loss = loss.mean()
        loss = F.mse_loss(outputs, noise, reduction="mean")
        return LossOutputs(loss=loss)

    @add_default_section_for_function("microsoft/model/diffusers/image2video/wan")
    @autocast(
        device_type=("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=(torch.bfloat16 if is_bfloat16_available() else torch.float32),
    )
    def generate(
        self,
        input_ids: torch.Tensor,
        negative_input_ids: torch.Tensor,
        vae_pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        num_frames: Optional[int] = 81,
        guidance_scale: Optional[float] = 5.0,
        enable_cpu_offload: Optional[bool] = False,
        cpu_offload_device: Optional[str] = None,
    ):
        if cpu_offload_device is None:
            cpu_offload_device = torch.cuda.current_device()

        if enable_cpu_offload and cpu_offload_device != "cpu":
            self.pipeline.enable_model_cpu_offload(cpu_offload_device)

        outputs = self.get_prompt_outputs(
            input_ids=input_ids,
            negative_input_ids=negative_input_ids,
            attention_mask=attention_mask,
            negative_attention_mask=negative_attention_mask,
            enable_cpu_offload=enable_cpu_offload,
            cpu_offload_device=cpu_offload_device,
        )

        if enable_cpu_offload:
            self.image.to(device=cpu_offload_device)

        if enable_cpu_offload:
            self.image.to(device="cpu")
            condition_hidden_states = condition_hidden_states.to("cpu")

        frames = self.pipeline(
            image=vae_pixel_values,
            prompt_embeds=outputs.prompt_embeds,
            negative_prompt_embeds=outputs.negative_prompt_embeds,
            generator=torch.Generator(device=self.pipeline.device).manual_seed(
                self.seed
            ),
            num_inference_steps=self.num_infer_timesteps,
            height=vae_pixel_values.size(-2),
            width=vae_pixel_values.size(-1),
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            output_type="pt",
        ).frames

        return DiffusionOutputs(outputs=frames.float())
