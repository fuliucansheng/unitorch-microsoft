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
        "stable-v1.5-x4-upscaler-ms-logo-32x32": {
            **__hf_hub_stable_v1_5_dict__("stabilityai/stable-diffusion-x4-upscaler"),
            **__hf_hub_vae_safetensors_dict__(
                "stabilityai/stable-diffusion-x4-upscaler"
            ),
            **{
                "unet": {
                    "config": hf_endpoint_url(
                        "/stabilityai/stable-diffusion-x4-upscaler/resolve/main/unet/config.json"
                    ),
                    "weight": "",
                }
            },
        }
    }
)

pretrained_stable_extensions_infos.update(
    {
        "stable-flux-lora-dev-fill-obj-removal": {
            "lora": {
                "weight": "https://huggingface.co/lrzjason/ObjectRemovalFluxFill/resolve/main/objectRemovalBeta_AlphaRegR20-3600.safetensors"
            }
        },
        "stable-flux-lora-ms-dev-recraft": {
            "lora": {
                "weight": "https://unitorchazureblob.blob.core.windows.net/shares/models/picasso/pytorch_model.flux.dev.ms.lora.recraft.bin"
            }
        },
        "stable-flux-lora-ms-dev-fill-simple": {
            "lora": {
                "weight": "https://unitorchazureblob.blob.core.windows.net/shares/models/picasso/pytorch_model.flux.dev.fill.ms.lora.simple.bin"
            }
        },
    }
)

import unitorch_microsoft.models.diffusers.pipeline_stable_flux
import unitorch_microsoft.models.diffusers.modeling_stable_flux
import unitorch_microsoft.models.diffusers.modeling_stable
import unitorch_microsoft.models.diffusers.modeling_vae
