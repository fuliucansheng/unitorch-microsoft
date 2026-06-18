# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

from unitorch.cli.models.vllm import hf_endpoint_url, pretrained_vllm_infos

pretrained_vllm_infos.update(**{
    "qwen3-vl-2b-instruct-lp-image-relevance": {
        "hf_pretrained_name": "fuliucansheng/Qwen3-VL-2B-Instruct-LP-Image-Relevance",
        "tokenizer": hf_endpoint_url(
            "/Qwen/Qwen3-VL-2B-Instruct/resolve/main/tokenizer.json"
        ),
        "tokenizer_config": hf_endpoint_url(
            "/Qwen/Qwen3-VL-2B-Instruct/resolve/main/tokenizer_config.json"
        ),
        "chat_template": hf_endpoint_url(
            "/Qwen/Qwen3-VL-2B-Instruct/resolve/main/chat_template.json"
        ),
    },
})