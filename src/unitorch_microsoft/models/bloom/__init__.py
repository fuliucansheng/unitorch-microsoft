# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

from unitorch.cli.models.bloom import pretrained_bloom_extensions_infos

pretrained_bloom_extensions_infos.update(
    {
        "bloom-lora-3b-slab-cn": {
            "text": "Generate a short title and description for the given landing page. # Input: {Input} # Output:",
            "weight": "https://unitorchazureblob.blob.core.windows.net/shares/models/adsplus/lora/bloom/pytorch_model.bloom.slab.cn.lora16.bin",
        },
    }
)

import unitorch_microsoft.models.bloom.modeling
import unitorch_microsoft.models.bloom.processing
