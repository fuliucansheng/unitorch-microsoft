# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import json
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import diffusers.schedulers as schedulers
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection
from diffusers.schedulers import SchedulerMixin
from diffusers.models import (
    ControlNetModel,
    UNet2DModel,
    UNet2DConditionModel,
    AutoencoderKL,
)
from diffusers.pipelines import (
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLControlNetImg2ImgPipeline,
    StableDiffusionXLControlNetInpaintPipeline,
)
from unitorch.models import (
    GenericModel,
    GenericOutputs,
    QuantizationConfig,
    QuantizationMixin,
)
from unitorch.models.diffusers import GenericStableXLModel
from unitorch.utils import pop_value, nested_dict_value
from unitorch.cli import (
    add_default_section_for_init,
    add_default_section_for_function,
    register_model,
)
from unitorch.cli.models import DiffusionOutputs, LossOutputs
from unitorch.cli.models import diffusion_model_decorator
from unitorch.cli.models.diffusers import (
    pretrained_stable_infos,
    pretrained_stable_extensions_infos,
    load_weight,
)
from unitorch_microsoft import cached_path


@register_model(
    "microsoft/model/diffusers/image2image/controlnet_xl", diffusion_model_decorator
)
class ControlNetXLForImage2ImageGeneration(GenericStableXLModel):
    def __init__(
        self,
        config_path: str,
        text_config_path: str,
        text2_config_path: str,
        vae_config_path: str,
        controlnet_configs_path: Union[str, List[str]],
        scheduler_config_path: str,
        quant_config_path: Optional[str] = None,
        image_size: Optional[int] = None,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_train_timesteps: Optional[int] = 1000,
        num_infer_timesteps: Optional[int] = 50,
        freeze_vae_encoder: Optional[bool] = True,
        freeze_text_encoder: Optional[bool] = True,
        freeze_unet_encoder: Optional[bool] = False,
        seed: Optional[int] = 1123,
    ):
        super().__init__(
            config_path=config_path,
            text_config_path=text_config_path,
            text2_config_path=text2_config_path,
            vae_config_path=vae_config_path,
            controlnet_configs_path=controlnet_configs_path,
            scheduler_config_path=scheduler_config_path,
            quant_config_path=quant_config_path,
            image_size=image_size,
            in_channels=in_channels,
            out_channels=out_channels,
            num_train_timesteps=num_train_timesteps,
            num_infer_timesteps=num_infer_timesteps,
            freeze_vae_encoder=freeze_vae_encoder,
            freeze_text_encoder=freeze_text_encoder,
            freeze_unet_encoder=freeze_unet_encoder,
            seed=seed,
        )
        self.pipeline = StableDiffusionXLControlNetImg2ImgPipeline(
            vae=self.vae,
            text_encoder=self.text,
            text_encoder_2=self.text2,
            unet=self.unet,
            controlnet=self.controlnet,
            scheduler=self.scheduler,
            tokenizer=None,
            tokenizer_2=None,
        )
        self.pipeline.set_progress_bar_config(disable=True)

    @classmethod
    @add_default_section_for_init("microsoft/model/diffusers/image2image/controlnet_xl")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section(
            "microsoft/model/diffusers/image2image/controlnet_xl"
        )
        pretrained_name = config.getoption("pretrained_name", "stable-xl-base")
        pretrained_infos = nested_dict_value(pretrained_stable_infos, pretrained_name)

        pretrained_controlnet_names = config.getoption(
            "pretrained_controlnet_names", "stable-xl-controlnet-canny"
        )
        if isinstance(pretrained_controlnet_names, str):
            pretrained_controlnet_names = [pretrained_controlnet_names]
        pretrained_controlnet_infos = [
            nested_dict_value(
                pretrained_stable_extensions_infos, pretrained_controlnet_name
            )
            for pretrained_controlnet_name in pretrained_controlnet_names
        ]

        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_infos, "unet", "config"),
        )
        config_path = cached_path(config_path)

        text_config_path = config.getoption("text_config_path", None)
        text_config_path = pop_value(
            text_config_path,
            nested_dict_value(pretrained_infos, "text", "config"),
        )
        text_config_path = cached_path(text_config_path)

        text2_config_path = config.getoption("text2_config_path", None)
        text2_config_path = pop_value(
            text2_config_path,
            nested_dict_value(pretrained_infos, "text2", "config"),
        )
        text2_config_path = cached_path(text2_config_path)

        vae_config_path = config.getoption("vae_config_path", None)
        vae_config_path = pop_value(
            vae_config_path,
            nested_dict_value(pretrained_infos, "vae", "config"),
        )
        vae_config_path = cached_path(vae_config_path)

        controlnet_configs_path = config.getoption("controlnet_configs_path", None)
        if isinstance(controlnet_configs_path, str):
            controlnet_configs_path = [controlnet_configs_path]
        controlnet_configs_path = pop_value(
            controlnet_configs_path,
            [
                nested_dict_value(pretrained_controlnet_info, "controlnet", "config")
                for pretrained_controlnet_info in pretrained_controlnet_infos
            ],
        )
        controlnet_configs_path = [
            cached_path(controlnet_config_path)
            for controlnet_config_path in controlnet_configs_path
        ]

        scheduler_config_path = config.getoption("scheduler_config_path", None)
        scheduler_config_path = pop_value(
            scheduler_config_path,
            nested_dict_value(pretrained_infos, "scheduler"),
        )
        scheduler_config_path = cached_path(scheduler_config_path)

        quant_config_path = config.getoption("quant_config_path", None)
        if quant_config_path is not None:
            quant_config_path = cached_path(quant_config_path)

        image_size = config.getoption("image_size", None)
        in_channels = config.getoption("in_channels", None)
        out_channels = config.getoption("out_channels", None)
        num_train_timesteps = config.getoption("num_train_timesteps", 1000)
        num_infer_timesteps = config.getoption("num_infer_timesteps", 50)
        freeze_vae_encoder = config.getoption("freeze_vae_encoder", True)
        freeze_text_encoder = config.getoption("freeze_text_encoder", True)
        freeze_unet_encoder = config.getoption("freeze_unet_encoder", True)
        seed = config.getoption("seed", 1123)

        inst = cls(
            config_path=config_path,
            text_config_path=text_config_path,
            text2_config_path=text2_config_path,
            vae_config_path=vae_config_path,
            controlnet_configs_path=controlnet_configs_path,
            scheduler_config_path=scheduler_config_path,
            quant_config_path=quant_config_path,
            image_size=image_size,
            in_channels=in_channels,
            out_channels=out_channels,
            num_train_timesteps=num_train_timesteps,
            num_infer_timesteps=num_infer_timesteps,
            freeze_vae_encoder=freeze_vae_encoder,
            freeze_text_encoder=freeze_text_encoder,
            freeze_unet_encoder=freeze_unet_encoder,
            seed=seed,
        )

        weight_path = config.getoption("pretrained_weight_path", None)

        state_dict = None
        if weight_path is None and pretrained_infos is not None:
            state_dict = [
                load_weight(
                    nested_dict_value(pretrained_infos, "unet", "weight"),
                    prefix_keys={"": "unet."},
                ),
                load_weight(
                    nested_dict_value(pretrained_infos, "text", "weight"),
                    prefix_keys={"": "text."},
                ),
                load_weight(
                    nested_dict_value(pretrained_infos, "text2", "weight"),
                    prefix_keys={"": "text2."},
                ),
                load_weight(
                    nested_dict_value(pretrained_infos, "vae", "weight"),
                    prefix_keys={"": "vae."},
                ),
            ]
            if len(pretrained_controlnet_infos) > 1:
                for i, pretrained_controlnet_info in enumerate(
                    pretrained_controlnet_infos
                ):
                    state_dict.append(
                        load_weight(
                            nested_dict_value(
                                pretrained_controlnet_info, "controlnet", "weight"
                            ),
                            prefix_keys={"": f"controlnet.{i}."},
                        )
                    )
            else:
                state_dict.append(
                    load_weight(
                        nested_dict_value(
                            pretrained_controlnet_infos[0], "controlnet", "weight"
                        ),
                        prefix_keys={"": "controlnet."},
                    )
                )
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
                lora_weights_path, pretrained_lora_weights, replace_keys={}
            )

        return inst

    @autocast()
    def forward(
        self,
        input_ids: torch.Tensor,
        input2_ids: torch.Tensor,
        add_time_ids: torch.Tensor,
        input_pixel_values: torch.Tensor,
        pixel_values: torch.Tensor,
        condition_pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention2_mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input IDs.
            input2_ids (torch.Tensor): Second input IDs.
            add_time_ids (torch.Tensor): Additional time IDs.
            pixel_values (torch.Tensor): Pixel values.
            condition_pixel_values (torch.Tensor): Condition pixel values.
            attention_mask (Optional[torch.Tensor]): Attention mask (default: None).
            attention2_mask (Optional[torch.Tensor]): Second attention mask (default: None).

        Returns:
            torch.Tensor: Loss value.
        """
        prompt_outputs = self.text(
            input_ids,
            # attention_mask,
            output_hidden_states=True,
        )
        prompt_embeds = prompt_outputs.hidden_states[-2]
        prompt2_outputs = self.text2(
            input2_ids,
            # attention2_mask,
            output_hidden_states=True,
        )
        prompt2_embeds = prompt2_outputs.hidden_states[-2]
        prompt_embeds = torch.concat([prompt_embeds, prompt2_embeds], dim=-1)
        pooled_prompt_embeds = prompt2_outputs[0]

        latents = self.vae.encode(pixel_values).latent_dist.sample()
        input_latents = self.vae.encode(input_pixel_values).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        input_latents = input_latents * self.vae.config.scaling_factor

        noise = torch.randn(latents.shape).to(latents.device)
        batch = latents.size(0)

        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (batch,),
            device=pixel_values.device,
        ).long()

        noise_latents = self.scheduler.add_noise(
            input_latents,
            noise,
            timesteps,
        )

        down_block_res_samples, mid_block_res_sample = self.controlnet(
            noise_latents,
            timesteps,
            encoder_hidden_states=prompt_embeds,
            controlnet_cond=condition_pixel_values,
            added_cond_kwargs={
                "time_ids": add_time_ids,
                "text_embeds": pooled_prompt_embeds,
            },
            return_dict=False,
        )
        outputs = self.unet(
            noise_latents,
            timesteps,
            prompt_embeds,
            added_cond_kwargs={
                "time_ids": add_time_ids,
                "text_embeds": pooled_prompt_embeds,
            },
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
        ).sample

        noise = self.scheduler.get_velocity(latents, noise, timesteps)

        loss = F.mse_loss(outputs, noise, reduction="mean")
        return LossOutputs(loss=loss)

    @add_default_section_for_function("core/model/diffusers/image2image/controlnet_xl")
    @autocast()
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        input2_ids: torch.Tensor,
        negative_input_ids: torch.Tensor,
        negative_input2_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        condition_pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention2_mask: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        negative_attention2_mask: Optional[torch.Tensor] = None,
        strength: Optional[float] = 0.99,
        guidance_scale: Optional[float] = 7.5,
        controlnet_conditioning_scale: Optional[Union[float, List[float]]] = None,
    ):
        outputs = self.get_prompt_outputs(
            input_ids=input_ids,
            input2_ids=input2_ids,
            negative_input_ids=negative_input_ids,
            negative_input2_ids=negative_input2_ids,
            attention_mask=attention_mask,
            attention2_mask=attention2_mask,
            negative_attention_mask=negative_attention_mask,
            negative_attention2_mask=negative_attention2_mask,
        )
        if controlnet_conditioning_scale is None:
            if self.num_controlnets == 1:
                controlnet_conditioning_scale = 1.0
            else:
                controlnet_conditioning_scale = [1.0] * self.num_controlnets
        elif (
            not isinstance(controlnet_conditioning_scale, list)
            and self.num_controlnets > 1
        ):
            controlnet_conditioning_scale = [
                controlnet_conditioning_scale
            ] * self.num_controlnets
        images = self.pipeline(
            image=pixel_values,
            control_image=condition_pixel_values
            if self.num_controlnets == 1
            else list(condition_pixel_values.transpose(0, 1)),
            prompt_embeds=outputs.prompt_embeds,
            negative_prompt_embeds=outputs.negative_prompt_embeds,
            pooled_prompt_embeds=outputs.pooled_prompt_embeds,
            negative_pooled_prompt_embeds=outputs.negative_pooled_prompt_embeds,
            generator=torch.Generator(device=self.pipeline.device).manual_seed(
                self.seed
            ),
            strength=strength,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            output_type="np.array",
        ).images

        return DiffusionOutputs(outputs=torch.from_numpy(images))
