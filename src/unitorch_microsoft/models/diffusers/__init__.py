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

pretrained_stable_infos.update({})

pretrained_stable_extensions_infos.update(
    {
        "stable-flux-lora-ms-dev-recraft": {"weight": ""},
        "stable-flux-lora-ms-dev-fill-simple": {"weight": ""},
    }
)

import unitorch_microsoft.models.diffusers.pipeline_stable_flux
import unitorch_microsoft.models.diffusers.modeling_stable_flux
import unitorch_microsoft.models.diffusers.modeling_vae
