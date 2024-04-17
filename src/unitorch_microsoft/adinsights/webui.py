# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

from unitorch.cli import register_webui
from unitorch.cli.models.bloom import pretrained_bloom_infos
from unitorch.cli.models.mistral import pretrained_mistral_infos
from unitorch.cli.models.peft import pretrained_peft_infos
from unitorch.cli.webuis import matched_pretrained_names
from unitorch.cli.webuis.bloom import BloomWebUI
from unitorch.cli.webuis.mistral import PeftMistralLoraWebUI

pretrained_peft_infos.update(
    {
        "peft-lora-mistral-7b-q2k-eem-cn": {
            "config": "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/raw/main/config.json",
            "vocab": "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/resolve/main/tokenizer.model",
            "weight": [
                f"https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/resolve/main/pytorch_model-{str(i).rjust(5, '0')}-of-00002.bin"
                for i in range(1, 3)
            ],
            "lora": {
                "rank": 128,
                "weight": "https://unitorchazureblob.blob.core.windows.net/shares/models/adinsights/generation/pytorch_model.mistral.q2k.eem.cn.lora128.bin",
            },
        },
        "peft-lora-mistral-7b-q2k-epm-cn": {
            "config": "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/raw/main/config.json",
            "vocab": "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/resolve/main/tokenizer.model",
            "weight": [
                f"https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/resolve/main/pytorch_model-{str(i).rjust(5, '0')}-of-00002.bin"
                for i in range(1, 3)
            ],
            "lora": {
                "rank": 128,
                "weight": "https://unitorchazureblob.blob.core.windows.net/shares/models/adinsights/generation/pytorch_model.mistral.q2k.epm.cn.lora128.bin",
            },
        },
        "peft-lora-mistral-7b-k2q-epm-cn": {
            "config": "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/raw/main/config.json",
            "vocab": "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/resolve/main/tokenizer.model",
            "weight": [
                f"https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/resolve/main/pytorch_model-{str(i).rjust(5, '0')}-of-00002.bin"
                for i in range(1, 3)
            ],
            "lora": {
                "rank": 128,
                "weight": "https://unitorchazureblob.blob.core.windows.net/shares/models/adinsights/generation/pytorch_model.mistral.k2q.epm.cn.lora128.bin",
            },
        },
        "peft-lora-mistral-7b-perf-top1-sahara": {
            "config": "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/raw/main/config.json",
            "vocab": "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/resolve/main/tokenizer.model",
            "weight": [
                f"https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/resolve/main/pytorch_model-{str(i).rjust(5, '0')}-of-00002.bin"
                for i in range(1, 3)
            ],
            "lora": {
                "rank": 64,
                "weight": "https://unitorchazureblob.blob.core.windows.net/shares/models/adinsights/generation/pytorch_model.mistral.perf.top1.sahara.instruction.lora64.bin",
            },
        },
        "peft-lora-mistral-7b-perf-top5-sahara": {
            "config": "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/raw/main/config.json",
            "vocab": "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/resolve/main/tokenizer.model",
            "weight": [
                f"https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/resolve/main/pytorch_model-{str(i).rjust(5, '0')}-of-00002.bin"
                for i in range(1, 3)
            ],
            "lora": {
                "rank": 64,
                "weight": "https://unitorchazureblob.blob.core.windows.net/shares/models/adinsights/generation/pytorch_model.mistral.perf.top5.sahara.instruction.lora64.bin",
            },
        },
        "peft-lora-mistral-7b-perf-top1-sahara-temu": {
            "config": "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/raw/main/config.json",
            "vocab": "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/resolve/main/tokenizer.model",
            "weight": [
                f"https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/resolve/main/pytorch_model-{str(i).rjust(5, '0')}-of-00002.bin"
                for i in range(1, 3)
            ],
            "lora": {
                "rank": 64,
                "weight": "https://unitorchazureblob.blob.core.windows.net/shares/models/adinsights/generation/pytorch_model.mistral.perf.top1.sahara.instruction.temu.lora64.bin",
            },
        },
    }
)


@register_webui("microsoft/adinsights/webui/bloom")
class BloomWebUIV2(BloomWebUI):
    match_patterns = [
        "^bloom",
    ]
    pretrained_names = list(pretrained_bloom_infos.keys())
    supported_pretrained_names = matched_pretrained_names(
        pretrained_names,
        match_patterns,
    )

    @property
    def name(self):
        return "AdInsights-Bloom"


@register_webui("microsoft/adinsights/webui/peft/mistral/lora")
class PeftMistralLoraWebUIV2(PeftMistralLoraWebUI):
    match_patterns = [
        "^peft-lora-mistral",
    ]
    pretrained_names = list(pretrained_peft_infos.keys())
    supported_pretrained_names = matched_pretrained_names(
        pretrained_names,
        match_patterns,
    )

    @property
    def name(self):
        return "AdInsights-Peft-Mistral"
