# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import json
import torch
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import diffusers.schedulers as schedulers
from transformers import (
    PretrainedConfig,
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import T5EncoderModel
from diffusers.schedulers import SchedulerMixin, FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import (
    compute_loss_weighting_for_sd3,
    compute_density_for_timestep_sampling,
)
from diffusers.models import (
    FluxTransformer2DModel,
    AutoencoderKL,
)
from diffusers.pipelines import (
    FluxPipeline,
    FluxImg2ImgPipeline,
    FluxInpaintPipeline,
    FluxControlPipeline,
    FluxFillPipeline,
)
from diffusers.pipelines.flux.modeling_flux import ReduxImageEncoder
from unitorch.models import (
    GenericModel,
    GenericOutputs,
    QuantizationConfig,
    QuantizationMixin,
)
from unitorch.models.peft import PeftWeightLoaderMixin
from unitorch.models.diffusers import compute_snr


class GenericStableFluxModel(GenericModel, QuantizationMixin, PeftWeightLoaderMixin):
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
        text2_config_path: str,
        vae_config_path: str,
        scheduler_config_path: str,
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
        self.transformer = FluxTransformer2DModel.from_config(config_dict).to(
            torch.float16
        )

        text_config = CLIPTextConfig.from_json_file(text_config_path)
        self.text = CLIPTextModel(text_config).to(torch.float16)

        text_config2 = T5Config.from_json_file(text2_config_path)
        self.text2 = T5EncoderModel(text_config2).to(torch.float16)

        vae_config_dict = json.load(open(vae_config_path))
        self.vae = AutoencoderKL.from_config(vae_config_dict).to(torch.float16)

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
            for param in self.text2.parameters():
                param.requires_grad = False

        if freeze_transformer_encoder:
            for param in self.transformer.parameters():
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
        input2_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention2_mask: Optional[torch.Tensor] = None,
        enable_cpu_offload: Optional[bool] = False,
        cpu_offload_device: Optional[str] = "cpu",
    ):
        if enable_cpu_offload:
            self.text = self.text.to(cpu_offload_device)
            self.text2 = self.text2.to(cpu_offload_device)
            input_ids = input_ids.to(cpu_offload_device)
            input2_ids = input2_ids.to(cpu_offload_device)

        prompt_outputs = self.text(
            input_ids,
            # attention_mask,
            output_hidden_states=False,
        )
        pooled_prompt_embeds = prompt_outputs.pooler_output

        prompt_embeds = self.text2(
            input2_ids,
            # attention2_mask,
            output_hidden_states=False,
        )[0]
        if enable_cpu_offload:
            self.text = self.text.to("cpu")
            self.text2 = self.text2.to("cpu")

        return GenericOutputs(
            prompt_embeds=(
                prompt_embeds.to("cpu") if enable_cpu_offload else prompt_embeds
            ),
            pooled_prompt_embeds=(
                pooled_prompt_embeds.to("cpu")
                if enable_cpu_offload
                else pooled_prompt_embeds
            ),
        )
