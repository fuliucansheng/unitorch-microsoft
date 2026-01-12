# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

from unitorch.cli.models.diffusers import (
    __hf_hub_stable_v1_5_dict__,
    __hf_hub_vae_dict,
    __hf_hub_stable_v1_5_safetensors_dict__,
    __hf_hub_vae_safetensors_dict__,
    __hf_hub_stable_flux_safetensors_dict__,
    hf_endpoint_url,
    pretrained_stable_infos,
    pretrained_stable_extensions_infos,
)

pretrained_ms_stable_infos = {
    "stable-v1.5-x4-upscaler-ms-logo-32x32": {
        **__hf_hub_stable_v1_5_dict__("stabilityai/stable-diffusion-x4-upscaler"),
        **__hf_hub_vae_safetensors_dict__("stabilityai/stable-diffusion-x4-upscaler"),
        **{
            "unet": {
                "config": hf_endpoint_url(
                    "/stabilityai/stable-diffusion-x4-upscaler/resolve/main/unet/config.json"
                ),
                "weight": "https://huggingface.co/datasets/fuliucansheng/unitorchblobfuse/resolve/main/models/picasso/pytorch_model.sd1.5.4x-upscaler.logo.bin",
            }
        },
    },
    "stable-flux-schnell-ms-fill": {
        **__hf_hub_stable_flux_safetensors_dict__("lzyvegetable/FLUX.1-schnell"),
        **__hf_hub_vae_safetensors_dict__("lzyvegetable/FLUX.1-schnell"),
        **{
            "transformer": {
                "config": "omnipixel/experiments/stable-flux-schnell-fill-config.json",
                "weight": [
                    hf_endpoint_url(
                        f"/lzyvegetable/FLUX.1-schnell/resolve/main/transformer/diffusion_pytorch_model-{str(i).rjust(5, '0')}-of-{str(3).rjust(5, '0')}.safetensors"
                    )
                    for i in range(1, 3 + 1)
                ],
            }
        },
    },
}

pretrained_ms_stable_extensions_infos = {
    "stable-flux-lora-dev-fill-obj-removal": {
        "lora": {
            "weight": "https://huggingface.co/lrzjason/ObjectRemovalFluxFill/resolve/main/objectRemovalBeta_AlphaRegR20-3600.safetensors"
        }
    },
    "stable-flux-lora-ms-dev-recraft": {
        "lora": {
            "weight": "https://huggingface.co/datasets/fuliucansheng/unitorchblobfuse/resolve/main/models/picasso/pytorch_model.flux.dev.ms.lora.recraft.bin"
        }
    },
    "stable-flux-lora-ms-dev-fill-simple": {
        "lora": {
            "weight": "https://huggingface.co/datasets/fuliucansheng/unitorchblobfuse/resolve/main/models/picasso/pytorch_model.flux.dev.fill.ms.lora.simple.bin"
        }
    },
}

pretrained_stable_infos.update(pretrained_ms_stable_infos)
pretrained_stable_extensions_infos.update(pretrained_ms_stable_extensions_infos)

import unitorch_microsoft.models.diffusers.pipeline_stable_flux
import unitorch_microsoft.models.diffusers.modeling_stable_flux
import unitorch_microsoft.models.diffusers.modeling_stable
import unitorch_microsoft.models.diffusers.modeling_vae
import unitorch_microsoft.models.diffusers.modeling_wan
import unitorch_microsoft.models.diffusers.modeling_peft_wan
import unitorch_microsoft.models.diffusers.processing_wan
import unitorch_microsoft.models.diffusers.video_utils
