# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
import diffusers
from torch import autocast
from itertools import accumulate
from collections import UserDict
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from diffusers.utils import numpy_to_pil
from diffusers.pipelines.flux.pipeline_flux_fill import (
    FluxFillPipeline,
    FluxPipelineOutput,
    retrieve_timesteps,
    calculate_shift,
    randn_tensor,
    retrieve_latents,
)
import unitorch
from unitorch.utils import is_bfloat16_available, is_fastapi_available
from unitorch.utils.decorators import replace
from unitorch.cli import add_default_section_for_function

@replace(diffusers.pipelines.flux.pipeline_flux_fill.FluxFillPipeline)
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


if is_fastapi_available():
    from unitorch.cli.fastapis.stable_flux import (
        StableFluxForImageInpaintingFastAPIPipeline,
        StableFluxForReduxInpaintingFastAPIPipeline,
    )
    
    @replace(unitorch.cli.fastapis.stable_flux.StableFluxForImageInpaintingFastAPIPipeline)
    class StableFluxForImageInpaintingFastAPIPipelineV2(
        StableFluxForImageInpaintingFastAPIPipeline
    ):
        @torch.no_grad()
        @autocast(
            device_type=("cuda" if torch.cuda.is_available() else "cpu"),
            dtype=(torch.bfloat16 if is_bfloat16_available() else torch.float32),
        )
        @add_default_section_for_function("core/fastapi/pipeline/stable_flux/inpainting")
        def __call__(
            self,
            text: str,
            image: Image.Image,
            mask_image: Image.Image,
            neg_text: Optional[str] = "",
            width: Optional[int] = None,
            height: Optional[int] = None,
            guidance_scale: Optional[float] = 30.0,
            strength: Optional[float] = 1.0,
            num_timesteps: Optional[int] = 50,
            seed: Optional[int] = 1123,
        ):
            if width is None or height is None:
                width, height = image.size
            width = width // 16 * 16
            height = height // 16 * 16
            image = image.resize((width, height))
            mask_image = mask_image.resize((width, height))

            text_inputs = self.processor.text2image_inputs(
                text,
                negative_prompt=neg_text,
            )
            image_inputs = self.processor.inpainting_inputs(image, mask_image)
            inputs = {**text_inputs, **image_inputs}
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

            prompt_embeds = prompt_outputs.prompt_embeds
            pooled_prompt_embeds = prompt_outputs.pooled_prompt_embeds

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


    @replace(unitorch.cli.fastapis.stable_flux.StableFluxForReduxInpaintingFastAPIPipeline)
    class StableFluxForReduxInpaintingFastAPIPipelineV2(
        StableFluxForReduxInpaintingFastAPIPipeline
    ):
        @torch.no_grad()
        @autocast(
            device_type=("cuda" if torch.cuda.is_available() else "cpu"),
            dtype=(torch.bfloat16 if is_bfloat16_available() else torch.float32),
        )
        @add_default_section_for_function(
            "core/fastapi/pipeline/stable_flux/redux_inpainting"
        )
        def __call__(
            self,
            text: str,
            image: Image.Image,
            mask_image: Image.Image,
            redux_image: Image.Image,
            neg_text: Optional[str] = "",
            width: Optional[int] = None,
            height: Optional[int] = None,
            guidance_scale: Optional[float] = 30.0,
            strength: Optional[float] = 1.0,
            num_timesteps: Optional[int] = 50,
            prompt_embeds_scale: Optional[float] = 1.0,
            pooled_prompt_embeds_scale: Optional[float] = 1.0,
            seed: Optional[int] = 1123,
        ):
            if width is None or height is None:
                width, height = image.size
            width = width // 16 * 16
            height = height // 16 * 16
            image = image.resize((width, height))
            mask_image = mask_image.resize((width, height))

            text_inputs = self.processor.text2image_inputs(
                text,
                negative_prompt=neg_text,
            )
            image_inputs = self.processor.inpainting_inputs(image, mask_image)
            redux_image_inputs = self.processor.redux_image_inputs(redux_image)
            inputs = {
                **text_inputs,
                **image_inputs,
                **{"redux_pixel_values": redux_image_inputs["pixel_values"]},
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

            if self._enable_cpu_offload:
                self.image.to(device=self._device)
                self.redux_image.to(device=self._device)

            redux_pixel_values = inputs["redux_pixel_values"].to(self._device)
            redux_image_embeds = self.image(redux_pixel_values).last_hidden_state
            redux_image_embeds = self.redux_image(redux_image_embeds).image_embeds

            if self._enable_cpu_offload:
                self.image.to(device="cpu")
                self.redux_image.to(device="cpu")
                redux_image_embeds = redux_image_embeds.to("cpu")

            prompt_embeds = (
                torch.cat([prompt_outputs.prompt_embeds, redux_image_embeds], dim=1)
                * prompt_embeds_scale
            )
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
