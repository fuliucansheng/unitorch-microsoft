# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import diffusers.schedulers as schedulers
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import T5EncoderModel
from diffusers.schedulers import SchedulerMixin
from diffusers.models import (
    FluxControlNetModel,
    FluxMultiControlNetModel,
    FluxTransformer2DModel,
    AutoencoderKL,
)
from diffusers.pipelines import (
    FluxPipeline,
    FluxControlNetPipeline,
)
from unitorch.utils import (
    pop_value,
    nested_dict_value,
    is_bfloat16_available,
    is_cuda_available,
)
from unitorch.models import (
    GenericModel,
    GenericOutputs,
    QuantizationConfig,
    QuantizationMixin,
)
from unitorch.models.peft import PeftWeightLoaderMixin
from unitorch.models.diffusers import compute_snr
from unitorch.models.diffusers import GenericStableFluxModel
from unitorch.cli import (
    add_default_section_for_init,
    add_default_section_for_function,
    register_model,
)
from unitorch.cli.models import DiffusionOutputs, LossOutputs
from unitorch.cli.models.diffusers import (
    pretrained_stable_infos,
    pretrained_stable_extensions_infos,
    load_weight,
)
from unitorch_microsoft import cached_path
from unitorch_microsoft.models.diffusers.modeling_flux_utils import (
    FluxInpaintPipelineV2,
    FluxControlNetInpaintPipelineV2,
)


@register_model("microsoft/picasso/diffusion/text2image/controlnet_flux")
class ControlNetFluxForText2ImageGeneration(GenericStableFluxModel):
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
        freeze_transformer_encoder: Optional[bool] = False,
        snr_gamma: Optional[float] = 5.0,
        seed: Optional[int] = 1123,
        guidance_scale: Optional[float] = 3.5,
        controlnet_conditioning_mode: Optional[Union[int, List[int]]] = None,
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
            freeze_transformer_encoder=freeze_transformer_encoder,
            snr_gamma=snr_gamma,
            seed=seed,
        )

        self.pipeline = FluxControlNetPipeline(
            vae=self.vae,
            text_encoder=self.text,
            text_encoder_2=self.text2,
            transformer=self.transformer,
            controlnet=FluxMultiControlNetModel([self.controlnet]),
            scheduler=self.scheduler,
            tokenizer=None,
            tokenizer_2=None,
        )
        self.guidance_scale = guidance_scale
        self.pipeline.set_progress_bar_config(disable=True)
        self.controlnet_conditioning_mode = controlnet_conditioning_mode

    @classmethod
    @add_default_section_for_init(
        "microsoft/picasso/diffusion/text2image/controlnet_flux"
    )
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section(
            "microsoft/picasso/diffusion/text2image/controlnet_flux"
        )
        pretrained_name = config.getoption("pretrained_name", "stable-flux-schnell")
        pretrained_infos = nested_dict_value(pretrained_stable_infos, pretrained_name)

        pretrained_controlnet_names = config.getoption(
            "pretrained_controlnet_names", "stable-v3-controlnet-canny"
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
            nested_dict_value(pretrained_infos, "transformer", "config"),
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
        freeze_transformer_encoder = config.getoption(
            "freeze_transformer_encoder", True
        )
        snr_gamma = config.getoption("snr_gamma", 5.0)
        seed = config.getoption("seed", 1123)
        guidance_scale = config.getoption("guidance_scale", 3.5)
        controlnet_conditioning_mode = config.getoption(
            "controlnet_conditioning_mode", None
        )

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
            freeze_transformer_encoder=freeze_transformer_encoder,
            snr_gamma=snr_gamma,
            seed=seed,
            guidance_scale=guidance_scale,
            controlnet_conditioning_mode=controlnet_conditioning_mode,
        )

        weight_path = config.getoption("pretrained_weight_path", None)

        state_dict = None
        if weight_path is None and pretrained_infos is not None:
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
                    nested_dict_value(pretrained_infos, "text2", "weight"),
                    prefix_keys={"": "text2."},
                ),
                load_weight(
                    nested_dict_value(pretrained_infos, "vae", "weight"),
                    prefix_keys={"": "vae."},
                ),
            ]
            state_dict.append(
                load_weight(
                    nested_dict_value(
                        pretrained_controlnet_infos[0], "controlnet", "weight"
                    ),
                    prefix_keys={"": "controlnet."},
                    replace_keys={"controlnet\.controlnet\.": "controlnet."},
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
                lora_weights_path,
                pretrained_lora_weights,
                replace_keys={},
                save_base_state=False,
            )

        return inst

    def cuda(self):
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            self.pipeline.enable_model_cpu_offload(device)
            # self.pipeline.to(device)
        return self

    def cpu(self):
        self.pipeline.cpu()
        return self

    @add_default_section_for_function(
        "microsoft/picasso/diffusion/text2image/controlnet_flux"
    )
    @autocast(
        device_type=("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=(torch.bfloat16 if is_bfloat16_available() else torch.float32),
    )
    def forward(
        self,
        input_ids: torch.Tensor,
        input2_ids: torch.Tensor,
        condition_pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention2_mask: Optional[torch.Tensor] = None,
        height: Optional[int] = 1024,
        width: Optional[int] = 1024,
        guidance_scale: Optional[float] = 7.5,
        controlnet_conditioning_scale: Optional[Union[float, List[float]]] = None,
        controlnet_conditioning_mode: Optional[Union[int, List[int]]] = None,
    ):
        controlnet_conditioning_mode = (
            controlnet_conditioning_mode or self.controlnet_conditioning_mode
        )
        outputs = self.get_prompt_outputs(
            input_ids=input_ids,
            input2_ids=input2_ids,
            attention_mask=attention_mask,
            attention2_mask=attention2_mask,
            enable_cpu_offload=True if torch.cuda.is_available() else False,
            cpu_offload_device=torch.cuda.current_device()
            if torch.cuda.is_available()
            else "cpu",
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

        if controlnet_conditioning_mode is None:
            if self.num_controlnets == 1:
                controlnet_conditioning_mode = None
            else:
                controlnet_conditioning_mode = [None] * self.num_controlnets

        images = self.pipeline(
            control_image=condition_pixel_values
            if condition_pixel_values.dim() == 4
            else list(condition_pixel_values.transpose(0, 1)),
            control_mode=controlnet_conditioning_mode,
            prompt_embeds=outputs.prompt_embeds,
            pooled_prompt_embeds=outputs.pooled_prompt_embeds,
            generator=torch.Generator(device=self.pipeline.device).manual_seed(
                self.seed
            ),
            num_inference_steps=self.num_infer_timesteps,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            output_type="np.array",
        ).images

        return DiffusionOutputs(outputs=torch.from_numpy(images))
