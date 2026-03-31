# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

from unitorch.cli import hf_endpoint_url
from unitorch.cli.models.qwen import (
    pretrained_qwen_infos,
    pretrained_qwen_extensions_infos,
)

pretrained_qwen_infos.update(
    {
        "qwen3-vl-2b-instruct-img-lp-relevance": {
            "config": hf_endpoint_url(
                "/Qwen/Qwen3-VL-2B-Instruct/resolve/main/config.json"
            ),
            "tokenizer": hf_endpoint_url(
                "/Qwen/Qwen3-VL-2B-Instruct/resolve/main/tokenizer.json"
            ),
            "vision_config": hf_endpoint_url(
                "/Qwen/Qwen3-VL-2B-Instruct/resolve/main/preprocessor_config.json"
            ),
            "tokenizer_config": hf_endpoint_url(
                "/Qwen/Qwen3-VL-2B-Instruct/resolve/main/tokenizer_config.json"
            ),
            "chat_template": hf_endpoint_url(
                "/Qwen/Qwen3-VL-2B-Instruct/resolve/main/chat_template.json"
            ),
            "weight": "/home/decu/model.safetensors",
        },
    }
)
