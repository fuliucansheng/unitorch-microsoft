# Copyright (c) MICROSOFT.
# Licensed under the MIT License.
from unitorch.cli.models.detr import (
    pretrained_detr_infos,
)
from unitorch.cli import hf_endpoint_url

pretrained_detr_infos.update(
    {
        "detr-resnet-50-picasso-roi-v2": {
            "config": "https://huggingface.co/datasets/fuliucansheng/unitorchblobfuse/resolve/main/models/picasso/pytorch_model.detr.roi.config.v2.2505.json",
            "vision_config": hf_endpoint_url(
                "/facebook/detr-resnet-50/resolve/main/preprocessor_config.json"
            ),
            "weight": "https://huggingface.co/datasets/fuliucansheng/unitorchblobfuse/resolve/main/models/picasso/pytorch_model.detr.roi.v2.2505.bin",
        },
    }
)
