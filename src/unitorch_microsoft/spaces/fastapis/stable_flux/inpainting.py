# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import io
import re
import gc
import json
import logging
import torch
import hashlib
import asyncio
import pandas as pd
import diffusers
from PIL import Image
import numpy as np
from torch import autocast
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import StreamingResponse
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from transformers import PretrainedConfig, SiglipVisionConfig, SiglipVisionModel
from diffusers.utils import numpy_to_pil
from diffusers.pipelines.flux.pipeline_flux_fill import (
    FluxFillPipeline,
    FluxPipelineOutput,
    retrieve_timesteps,
    calculate_shift,
    randn_tensor,
    retrieve_latents,
)
from diffusers.models import ControlNetModel
from diffusers.pipelines import (
    FluxPipeline,
    FluxControlNetPipeline,
    FluxImg2ImgPipeline,
    FluxControlNetImg2ImgPipeline,
    FluxInpaintPipeline,
    FluxControlNetInpaintPipeline,
    FluxFillPipeline,
)
from diffusers.pipelines.flux.modeling_flux import ReduxImageEncoder
from unitorch import is_xformers_available
from unitorch.utils import is_remote_url
from unitorch.models.diffusers import GenericStableFluxModel
from unitorch.models.diffusers import StableFluxProcessor
from unitorch.utils import (
    pop_value,
    nested_dict_value,
    is_bfloat16_available,
    is_cuda_available,
)
from unitorch.cli import cached_path
from unitorch.cli.models.diffusers import (
    pretrained_stable_infos,
    pretrained_stable_extensions_infos,
    load_weight,
)
from unitorch.cli import CoreConfigureParser, GenericFastAPI
from unitorch.cli import register_fastapi


class FluxFillPipelineV2(FluxFillPipeline):
    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
        image=None,
        timestep=None,
    ):
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, num_channels_latents, height, width)

        if latents is not None:
            latent_image_ids = self._prepare_latent_image_ids(
                batch_size, height // 2, width // 2, device, dtype
            )
            return latents.to(device=device, dtype=dtype), latent_image_ids

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        if image is not None:
            image_latents = retrieve_latents(
                self.vae.encode(image.to(device=device, dtype=dtype)),
                generator=generator,
            )
            image_latents = (
                image_latents - self.vae.config.shift_factor
            ) * self.vae.config.scaling_factor
            latents = self.scheduler.scale_noise(image_latents, timestep, latents)
        latents = self._pack_latents(
            latents, batch_size, num_channels_latents, height, width
        )

        latent_image_ids = self._prepare_latent_image_ids(
            batch_size, height // 2, width // 2, device, dtype
        )

        return latents, latent_image_ids

    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(num_inference_steps * strength, num_inference_steps)

        t_start = int(max(num_inference_steps - init_timestep, 0))
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(t_start * self.scheduler.order)

        return timesteps, num_inference_steps - t_start

    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        image: Optional[torch.FloatTensor] = None,
        mask_image: Optional[torch.FloatTensor] = None,
        masked_image_latents: Optional[torch.FloatTensor] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 30.0,
        strength: float = 1.0,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
            image=image,
            mask_image=mask_image,
            masked_image_latents=masked_image_latents,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Prepare prompt embeddings
        lora_scale = (
            self.joint_attention_kwargs.get("scale", None)
            if self.joint_attention_kwargs is not None
            else None
        )
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        sigmas = (
            np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            if sigmas is None
            else sigmas
        )
        image_seq_len = (int(height) // self.vae_scale_factor // 2) * (
            int(width) // self.vae_scale_factor // 2
        )
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )

        # 4. Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels
        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps, strength, device
        )
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
            image=image,
            timestep=latent_timestep,
        )

        # 5. Prepare mask and masked image latents
        if masked_image_latents is not None:
            masked_image_latents = masked_image_latents.to(latents.device)
        else:
            image = self.image_processor.preprocess(image, height=height, width=width)
            mask_image = self.mask_processor.preprocess(
                mask_image, height=height, width=width
            )

            masked_image = image * (1 - mask_image)
            masked_image = masked_image.to(device=device, dtype=prompt_embeds.dtype)

            height, width = image.shape[-2:]
            mask, masked_image_latents = self.prepare_mask_latents(
                mask_image,
                masked_image,
                batch_size,
                num_channels_latents,
                num_images_per_prompt,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
            )
            masked_image_latents = torch.cat((masked_image_latents, mask), dim=-1)

        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full(
                [1], guidance_scale, device=device, dtype=torch.float32
            )
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        # 7. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                noise_pred = self.transformer(
                    hidden_states=torch.cat((latents, masked_image_latents), dim=2),
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        # 8. Post-process the image
        if output_type == "latent":
            image = latents

        else:
            latents = self._unpack_latents(
                latents, height, width, self.vae_scale_factor
            )
            latents = (
                latents / self.vae.config.scaling_factor
            ) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)


class StableFluxForReduxInpaintingFastAPIPipeline(GenericStableFluxModel):
    def __init__(
        self,
        config_path: str,
        text_config_path: str,
        text2_config_path: str,
        vae_config_path: str,
        scheduler_config_path: str,
        vocab_path: str,
        merge_path: str,
        vocab2_path: str,
        image_config_path: str,
        redux_image_config_path: str,
        redux_process_config_path: str,
        quant_config_path: Optional[str] = None,
        max_seq_length: Optional[int] = 77,
        max_seq_length2: Optional[int] = 256,
        pad_token: Optional[str] = "<|endoftext|>",
        weight_path: Optional[Union[str, List[str]]] = None,
        state_dict: Optional[Dict[str, Any]] = None,
        lora_checkpoints: Optional[Union[str, List[str]]] = None,
        lora_weights: Optional[Union[float, List[float]]] = 1.0,
        lora_alphas: Optional[Union[float, List[float]]] = 32,
        device: Optional[Union[str, int]] = "cpu",
        enable_cpu_offload: Optional[bool] = False,
        enable_xformers: Optional[bool] = False,
    ):
        super().__init__(
            config_path=config_path,
            text_config_path=text_config_path,
            text2_config_path=text2_config_path,
            vae_config_path=vae_config_path,
            scheduler_config_path=scheduler_config_path,
            quant_config_path=quant_config_path,
        )
        image_config = SiglipVisionConfig.from_json_file(image_config_path)
        self.image = SiglipVisionModel(image_config)
        redux_image_config = PretrainedConfig.from_json_file(redux_image_config_path)
        self.redux_image = ReduxImageEncoder(
            redux_dim=redux_image_config.redux_dim,
            txt_in_features=redux_image_config.txt_in_features,
        )
        self.processor = StableFluxProcessor(
            vocab_path=vocab_path,
            merge_path=merge_path,
            vocab2_path=vocab2_path,
            vae_config_path=vae_config_path,
            redux_config_path=redux_process_config_path,
            max_seq_length=max_seq_length,
            max_seq_length2=max_seq_length2,
            pad_token=pad_token,
        )
        self._device = "cpu" if device == "cpu" else int(device)

        self.from_pretrained(weight_path, state_dict=state_dict)

        self.eval()

        self.prompt_embeds_scale = 1.0
        self.pooled_prompt_embeds_scale = 1.0

        self.pipeline = FluxFillPipelineV2(
            vae=self.vae,
            text_encoder=self.text,
            text_encoder_2=self.text2,
            transformer=self.transformer,
            scheduler=self.scheduler,
            tokenizer=None,
            tokenizer_2=None,
        )
        self.pipeline.set_progress_bar_config(disable=True)

        if lora_checkpoints is not None:
            self.load_lora_weights(
                lora_checkpoints,
                lora_weights=lora_weights,
                lora_alphas=lora_alphas,
                save_base_state=False,
            )

        self._enable_cpu_offload = enable_cpu_offload
        self._enable_xformers = enable_xformers

        if not self._enable_cpu_offload:
            self.image.to(device=self._device)
            self.redux_image.to(device=self._device)

        if self._enable_cpu_offload and self._device != "cpu":
            self.pipeline.enable_model_cpu_offload(self._device)
        else:
            self.to(device=self._device)

        if self._enable_xformers and self._device != "cpu":
            assert is_xformers_available(), "Please install xformers first."
            self.pipeline.enable_xformers_memory_efficient_attention()

    @classmethod
    def from_core_configure(
        cls,
        config,
        pretrained_name: Optional[str] = None,
        config_path: Optional[str] = None,
        text_config_path: Optional[str] = None,
        text2_config_path: Optional[str] = None,
        vae_config_path: Optional[str] = None,
        scheduler_config_path: Optional[str] = None,
        vocab_path: Optional[str] = None,
        merge_path: Optional[str] = None,
        vocab2_path: Optional[str] = None,
        image_config_path: Optional[str] = None,
        redux_image_config_path: Optional[str] = None,
        redux_process_config_path: Optional[str] = None,
        quant_config_path: Optional[str] = None,
        pretrained_weight_path: Optional[str] = None,
        device: Optional[str] = None,
        pretrained_lora_names: Optional[Union[str, List[str]]] = None,
        pretrained_lora_weights_path: Optional[Union[str, List[str]]] = None,
        pretrained_lora_weights: Optional[Union[float, List[float]]] = None,
        pretrained_lora_alphas: Optional[Union[float, List[float]]] = None,
        **kwargs,
    ):
        config.set_default_section(
            "microsoft/spaces/fastapi/pipeline/stable_flux/inpainting"
        )
        pretrained_name = pretrained_name or config.getoption(
            "pretrained_name", "stable-flux-dev-redux-fill"
        )
        pretrained_infos = nested_dict_value(pretrained_stable_infos, pretrained_name)

        config_path = config_path or config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_infos, "transformer", "config"),
        )
        config_path = cached_path(config_path)

        text_config_path = text_config_path or config.getoption(
            "text_config_path", None
        )
        text_config_path = pop_value(
            text_config_path,
            nested_dict_value(pretrained_infos, "text", "config"),
        )
        text_config_path = cached_path(text_config_path)

        text2_config_path = text2_config_path or config.getoption(
            "text2_config_path", None
        )
        text2_config_path = pop_value(
            text2_config_path,
            nested_dict_value(pretrained_infos, "text2", "config"),
        )
        text2_config_path = cached_path(text2_config_path)

        vae_config_path = vae_config_path or config.getoption("vae_config_path", None)
        vae_config_path = pop_value(
            vae_config_path,
            nested_dict_value(pretrained_infos, "vae", "config"),
        )
        vae_config_path = cached_path(vae_config_path)

        scheduler_config_path = scheduler_config_path or config.getoption(
            "scheduler_config_path", None
        )
        scheduler_config_path = pop_value(
            scheduler_config_path,
            nested_dict_value(pretrained_infos, "scheduler"),
        )
        scheduler_config_path = cached_path(scheduler_config_path)

        vocab_path = vocab_path or config.getoption("vocab_path", None)
        vocab_path = pop_value(
            vocab_path,
            nested_dict_value(pretrained_infos, "text", "vocab"),
        )
        vocab_path = cached_path(vocab_path)

        merge_path = merge_path or config.getoption("merge_path", None)
        merge_path = pop_value(
            merge_path,
            nested_dict_value(pretrained_infos, "text", "merge"),
        )
        merge_path = cached_path(merge_path)

        vocab2_path = vocab2_path or config.getoption("vocab2_path", None)
        vocab2_path = pop_value(
            vocab2_path,
            nested_dict_value(pretrained_infos, "text2", "vocab"),
        )
        vocab2_path = cached_path(vocab2_path)

        image_config_path = image_config_path or config.getoption(
            "image_config_path", None
        )
        image_config_path = pop_value(
            image_config_path,
            nested_dict_value(pretrained_infos, "image", "config"),
        )
        image_config_path = cached_path(image_config_path)

        redux_image_config_path = redux_image_config_path or config.getoption(
            "redux_image_config_path", None
        )
        redux_image_config_path = pop_value(
            redux_image_config_path,
            nested_dict_value(pretrained_infos, "redux_image", "config"),
        )
        redux_image_config_path = cached_path(redux_image_config_path)

        redux_process_config_path = redux_process_config_path or config.getoption(
            "redux_process_config_path", None
        )
        redux_process_config_path = pop_value(
            redux_process_config_path,
            nested_dict_value(pretrained_infos, "image", "vision_config"),
        )
        redux_process_config_path = cached_path(redux_process_config_path)

        quant_config_path = quant_config_path or config.getoption(
            "quant_config_path", None
        )
        if quant_config_path is not None:
            quant_config_path = cached_path(quant_config_path)

        max_seq_length = config.getoption("max_seq_length", 77)
        max_seq_length2 = config.getoption("max_seq_length2", 256)
        pad_token = config.getoption("pad_token", "<|endoftext|>")
        weight_path = pretrained_weight_path or config.getoption(
            "pretrained_weight_path", None
        )
        device = config.getoption("device", "cpu") if device is None else device
        enable_cpu_offload = config.getoption("enable_cpu_offload", True)
        enable_xformers = config.getoption("enable_xformers", False)

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
                load_weight(
                    nested_dict_value(pretrained_infos, "image", "weight"),
                    prefix_keys={"": "image."},
                ),
                load_weight(
                    nested_dict_value(pretrained_infos, "redux_image", "weight"),
                    prefix_keys={"": "redux_image."},
                ),
            ]

        pretrained_lora_names = pretrained_lora_names or config.getoption(
            "pretrained_lora_names", None
        )
        pretrained_lora_weights = pretrained_lora_weights or config.getoption(
            "pretrained_lora_weights", 1.0
        )
        pretrained_lora_alphas = pretrained_lora_alphas or config.getoption(
            "pretrained_lora_alphas", 32.0
        )

        if (
            isinstance(pretrained_lora_names, str)
            and pretrained_lora_weights_path is None
        ):
            pretrained_lora_weights_path = nested_dict_value(
                pretrained_stable_extensions_infos,
                pretrained_lora_names,
                "lora",
                "weight",
            )
        elif (
            isinstance(pretrained_lora_names, list)
            and pretrained_lora_weights_path is None
        ):
            pretrained_lora_weights_path = [
                nested_dict_value(
                    pretrained_stable_extensions_infos, name, "lora", "weight"
                )
                for name in pretrained_lora_names
            ]
            assert len(pretrained_lora_weights_path) == len(pretrained_lora_weights)
            assert len(pretrained_lora_weights_path) == len(pretrained_lora_alphas)

        lora_weights_path = pretrained_lora_weights_path or config.getoption(
            "pretrained_lora_weights_path", None
        )

        inst = cls(
            config_path=config_path,
            text_config_path=text_config_path,
            text2_config_path=text2_config_path,
            vae_config_path=vae_config_path,
            scheduler_config_path=scheduler_config_path,
            vocab_path=vocab_path,
            merge_path=merge_path,
            vocab2_path=vocab2_path,
            image_config_path=image_config_path,
            redux_image_config_path=redux_image_config_path,
            redux_process_config_path=redux_process_config_path,
            quant_config_path=quant_config_path,
            pad_token=pad_token,
            max_seq_length=max_seq_length,
            max_seq_length2=max_seq_length2,
            weight_path=weight_path,
            state_dict=state_dict,
            lora_checkpoints=lora_weights_path,
            lora_weights=pretrained_lora_weights,
            lora_alphas=pretrained_lora_alphas,
            device=device,
            enable_cpu_offload=enable_cpu_offload,
            enable_xformers=enable_xformers,
        )
        return inst

    @torch.no_grad()
    @autocast(
        device_type=("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=(torch.bfloat16 if is_bfloat16_available() else torch.float32),
    )
    def __call__(
        self,
        text: str,
        image: Image.Image,
        mask_image: Image.Image,
        redux_image: Optional[Image.Image] = None,
        redux_image_2: Optional[Image.Image] = None,
        neg_text: Optional[str] = "",
        width: Optional[int] = None,
        height: Optional[int] = None,
        guidance_scale: Optional[float] = 30.0,
        strength: Optional[float] = 1.0,
        num_timesteps: Optional[int] = 50,
        prompt_embeds_scale: Optional[float] = 1.0,
        prompt_embeds_scale_2: Optional[float] = 1.0,
        pooled_prompt_embeds_scale: Optional[float] = 1.0,
        seed: Optional[int] = 1123,
    ):
        if width is None or height is None:
            width, height = image.size
        width = width // 8 * 8
        height = height // 8 * 8
        image = image.resize((width, height))
        mask_image = mask_image.resize((width, height))

        text_inputs = self.processor.text2image_inputs(
            text,
            negative_prompt=neg_text,
        )
        image_inputs = self.processor.inpainting_inputs(image, mask_image)
        inputs = {
            **text_inputs,
            **image_inputs,
        }
        if redux_image is not None:
            redux_image_inputs = self.processor.redux_image_inputs(redux_image)
            inputs = {
                **inputs,
                **{"redux_pixel_values": redux_image_inputs["pixel_values"]},
            }

        if redux_image_2 is not None:
            redux_image_inputs_2 = self.processor.redux_image_inputs(redux_image_2)
            inputs = {
                **inputs,
                **{"redux_pixel_values_2": redux_image_inputs_2["pixel_values"]},
            }

        self.seed = seed

        inputs = {k: v.unsqueeze(0) if v is not None else v for k, v in inputs.items()}
        inputs = {
            k: v.to(device=self.device) if v is not None else v
            for k, v in inputs.items()
        }

        prompt_outputs = self.get_prompt_outputs(
            input_ids=inputs.get("input_ids"),
            input2_ids=inputs.get("input2_ids"),
            attention_mask=inputs.get("attention_mask"),
            attention2_mask=inputs.get("attention2_mask"),
            enable_cpu_offload=self._enable_cpu_offload,
            cpu_offload_device=self._device,
        )

        if self._enable_cpu_offload and redux_image is not None:
            self.image.to(device=self._device)
            self.redux_image.to(device=self._device)

        if redux_image is not None:
            redux_pixel_values = inputs["redux_pixel_values"].to(self._device)
            redux_image_embeds = self.image(redux_pixel_values).last_hidden_state
            redux_image_embeds = self.redux_image(redux_image_embeds).image_embeds

        if redux_image_2 is not None:
            redux_pixel_values_2 = inputs["redux_pixel_values_2"].to(self._device)
            redux_image_embeds_2 = self.image(redux_pixel_values_2).last_hidden_state
            redux_image_embeds_2 = self.redux_image(redux_image_embeds_2).image_embeds

        if self._enable_cpu_offload and (
            redux_image is not None or redux_image_2 is not None
        ):
            self.image.to(device="cpu")
            self.redux_image.to(device="cpu")
            if redux_image is not None:
                redux_image_embeds = redux_image_embeds.to("cpu")
            if redux_image_2 is not None:
                redux_image_embeds_2 = redux_image_embeds_2.to("cpu")

        if redux_image_2 is not None:
            redux_image_embeds = torch.cat(
                [
                    redux_image_embeds,
                    redux_image_embeds_2 * prompt_embeds_scale_2 / prompt_embeds_scale,
                ],
                dim=1,
            )

        if redux_image is not None:
            prompt_embeds = (
                torch.cat([prompt_outputs.prompt_embeds, redux_image_embeds], dim=1)
                * prompt_embeds_scale
            )
            pooled_prompt_embeds = (
                prompt_outputs.pooled_prompt_embeds * pooled_prompt_embeds_scale
            )
        else:
            prompt_embeds = prompt_outputs.prompt_embeds * prompt_embeds_scale
            pooled_prompt_embeds = (
                prompt_outputs.pooled_prompt_embeds * pooled_prompt_embeds_scale
            )

        outputs = self.pipeline(
            image=inputs["pixel_values"],
            mask_image=inputs["pixel_masks"],
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            width=inputs["pixel_values"].size(-1),
            height=inputs["pixel_values"].size(-2),
            generator=torch.Generator(device=self.pipeline.device).manual_seed(
                self.seed
            ),
            num_inference_steps=num_timesteps,
            guidance_scale=guidance_scale,
            strength=strength,
            output_type="np.array",
        )

        images = torch.from_numpy(outputs.images)
        images = numpy_to_pil(images.cpu().numpy())
        return images[0]


@register_fastapi("microsoft/spaces/fastapi/stable_flux/inpainting")
class StableFluxReduxInpaintingFastAPI(GenericFastAPI):
    def __init__(self, config: CoreConfigureParser):
        self.config = config
        config.set_default_section(f"microsoft/spaces/fastapi/stable_flux/inpainting")
        router = config.getoption(
            "router", "/microsoft/spaces/fastapi/stable_flux/inpainting"
        )
        self._pipe = None
        self._router = APIRouter(prefix=router)
        self._router.add_api_route("/generate1", self.serve1, methods=["POST"])
        self._router.add_api_route("/generate2", self.serve2, methods=["POST"])
        self._router.add_api_route("/generate3", self.serve3, methods=["POST"])
        self._router.add_api_route("/status", self.status, methods=["GET"])
        self._router.add_api_route("/start", self.start, methods=["GET"])
        self._router.add_api_route("/stop", self.stop, methods=["GET"])
        self._lock = asyncio.Lock()

    @property
    def router(self):
        return self._router

    def start(self):
        if self.status() == "running":
            return "already started"
        self._pipe = StableFluxForReduxInpaintingFastAPIPipeline.from_core_configure(
            self.config
        )
        return "start success"

    def stop(self):
        if self.status() == "stopped":
            return "already stopped"
        self._pipe.to("cpu")
        del self._pipe
        gc.collect()
        torch.cuda.empty_cache()
        self._pipe = None
        return "stop success"

    def status(self):
        return "running" if self._pipe is not None else "stopped"

    async def serve1(
        self,
        text: str,
        image: UploadFile,
        mask_image: UploadFile,
        guidance_scale: Optional[float] = 30.0,
        strength: Optional[float] = 1.0,
        num_timesteps: Optional[int] = 50,
        prompt_embeds_scale: Optional[float] = 1.0,
        pooled_prompt_embeds_scale: Optional[float] = 1.0,
        seed: Optional[int] = 1123,
    ):
        assert self._pipe is not None
        image_bytes = await image.read()
        image = Image.open(io.BytesIO(image_bytes))
        mask_image_bytes = await mask_image.read()
        mask_image = Image.open(io.BytesIO(mask_image_bytes))
        async with self._lock:
            image = self._pipe(
                text,
                image,
                mask_image,
                guidance_scale=guidance_scale,
                strength=strength,
                num_timesteps=num_timesteps,
                prompt_embeds_scale=prompt_embeds_scale,
                pooled_prompt_embeds_scale=pooled_prompt_embeds_scale,
                seed=seed,
            )
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        return StreamingResponse(
            io.BytesIO(buffer.getvalue()),
            media_type="image/png",
        )

    async def serve2(
        self,
        text: str,
        image: UploadFile,
        mask_image: UploadFile,
        redux_image: UploadFile,
        guidance_scale: Optional[float] = 30.0,
        strength: Optional[float] = 1.0,
        num_timesteps: Optional[int] = 50,
        prompt_embeds_scale: Optional[float] = 1.0,
        pooled_prompt_embeds_scale: Optional[float] = 1.0,
        seed: Optional[int] = 1123,
    ):
        assert self._pipe is not None
        image_bytes = await image.read()
        image = Image.open(io.BytesIO(image_bytes))
        mask_image_bytes = await mask_image.read()
        mask_image = Image.open(io.BytesIO(mask_image_bytes))
        redux_image_bytes = await redux_image.read()
        redux_image = Image.open(io.BytesIO(redux_image_bytes))
        async with self._lock:
            image = self._pipe(
                text,
                image,
                mask_image,
                redux_image=redux_image,
                guidance_scale=guidance_scale,
                strength=strength,
                num_timesteps=num_timesteps,
                prompt_embeds_scale=prompt_embeds_scale,
                pooled_prompt_embeds_scale=pooled_prompt_embeds_scale,
                seed=seed,
            )
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        return StreamingResponse(
            io.BytesIO(buffer.getvalue()),
            media_type="image/png",
        )

    async def serve3(
        self,
        text: str,
        image: UploadFile,
        mask_image: UploadFile,
        redux_image: UploadFile,
        redux_image_2: UploadFile,
        guidance_scale: Optional[float] = 30.0,
        strength: Optional[float] = 1.0,
        num_timesteps: Optional[int] = 50,
        prompt_embeds_scale: Optional[float] = 1.0,
        prompt_embeds_scale_2: Optional[float] = 1.0,
        pooled_prompt_embeds_scale: Optional[float] = 1.0,
        seed: Optional[int] = 1123,
    ):
        assert self._pipe is not None
        image_bytes = await image.read()
        image = Image.open(io.BytesIO(image_bytes))
        mask_image_bytes = await mask_image.read()
        mask_image = Image.open(io.BytesIO(mask_image_bytes))
        redux_image_bytes = await redux_image.read()
        redux_image = Image.open(io.BytesIO(redux_image_bytes))
        redux_image_2_bytes = await redux_image_2.read()
        redux_image_2 = Image.open(io.BytesIO(redux_image_2_bytes))
        async with self._lock:
            image = self._pipe(
                text,
                image,
                mask_image,
                redux_image=redux_image,
                redux_image_2=redux_image_2,
                guidance_scale=guidance_scale,
                strength=strength,
                num_timesteps=num_timesteps,
                prompt_embeds_scale=prompt_embeds_scale,
                prompt_embeds_scale_2=prompt_embeds_scale,
                pooled_prompt_embeds_scale=pooled_prompt_embeds_scale,
                seed=seed,
            )
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        return StreamingResponse(
            io.BytesIO(buffer.getvalue()),
            media_type="image/png",
        )
