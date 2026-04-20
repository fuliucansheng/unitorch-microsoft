# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

from unitorch.cli.models.mask2former import pretrained_mask2former_infos

pretrained_mask2former_infos.update(
    {
        "picasso-mask2former-swin-base-ade-semantic": {
            "config": "https://huggingface.co/datasets/fuliucansheng/unitorchblobfuse/resolve/main/models/picasso/mask2former/config.json",
            "vision_config": "https://huggingface.co/datasets/fuliucansheng/unitorchblobfuse/resolve/main/models/picasso/mask2former/preprocessor_config.json",
            "weight": "https://huggingface.co/datasets/fuliucansheng/unitorchblobfuse/resolve/main/models/picasso/mask2former/pytorch_model.bin",
        }
    }
)

import unitorch_microsoft.models.mask2former.modeling
import unitorch_microsoft.models.mask2former.processing

from unitorch_microsoft.models.mask2former.modeling import Mask2FormerForSegmentation
from unitorch_microsoft.models.mask2former.processing import Mask2FormerProcessor

import unitorch_microsoft.models.mask2former.pipeline