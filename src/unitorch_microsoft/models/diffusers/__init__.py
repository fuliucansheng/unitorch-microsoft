# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

from unitorch.cli.models.diffusers import (
    __hf_hub_stable_v1_5_dict__,
    __hf_hub_vae_dict,
    __hf_hub_stable_v1_5_safetensors_dict__,
    __hf_hub_vae_safetensors_dict__,
    hf_endpoint_url,
    pretrained_stable_infos,
    pretrained_stable_extensions_infos,
)

pretrained_stable_infos.update(
    {
        # "stable-v1.5-realistic-v5.1-sd2-vae": {
        #     **__hf_hub_stable_v1_5_dict__("stablediffusionapi/realistic-vision-v51"),
        #     **__hf_hub_vae_dict("stabilityai/stable-diffusion-2"),
        # },
        # "stable-v1.5-realistic-v5.1-inpainting-sd2-vae": {
        #     **__hf_hub_stable_v1_5_safetensors_dict__("Uminosachi/realisticVisionV51_v51VAE-inpainting"),
        #     **__hf_hub_vae_dict("stabilityai/stable-diffusion-2"),
        # },
    }
)

import unitorch_microsoft.models.diffusers.modeling_flux_utils
import unitorch_microsoft.models.diffusers.modeling_stable_flux
import unitorch_microsoft.models.diffusers.modeling_vae
