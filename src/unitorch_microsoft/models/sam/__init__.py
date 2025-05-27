# Copyright (c) MICROSOFT.
# Licensed under the MIT License.
from unitorch.cli.models.sam import (
    pretrained_sam_infos,
    pretrained_sam_extensions_infos,
)

pretrained_sam_extensions_infos.update(
    {
        "sam-lora-dis5k": {
            "text": "use box as prompt",
            "weight": "https://huggingface.co/datasets/fuliucansheng/unitorchblobfuse/resolve/main/models/adsplus/lora/pytorch_model.sam.large.lora4.2409.bin",
        },
    }
)

import unitorch_microsoft.models.sam.modeling_peft
import unitorch_microsoft.models.sam.processing
