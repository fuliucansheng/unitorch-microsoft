# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

from unitorch.cli import register_webui
from unitorch.cli.models.bloom import pretrained_bloom_infos
from unitorch.cli.models.mistral import pretrained_mistral_infos
from unitorch.cli.models.peft import pretrained_peft_infos
from unitorch.cli.webuis import matched_pretrained_names
from unitorch.cli.webuis.bloom import BloomWebUI, PeftBloomLoraWebUI
from unitorch.cli.webuis.mistral import PeftMistralLoraWebUI

pretrained_peft_infos.update(
    {
        "peft-lora-bloomz-560m-slab-cn": {
            "config": "https://huggingface.co/bigscience/bloomz-560m/resolve/main/config.json",
            "tokenizer": "https://huggingface.co/bigscience/bloomz-560m/resolve/main/tokenizer.json",
            "weight": "https://huggingface.co/bigscience/bloomz-560m/resolve/main/pytorch_model.bin",
            "lora": {
                "rank": 16,
                "weight": "https://unitorchazureblob.blob.core.windows.net/shares/models/china/slab/pytorch_model.bloom.slab.cn.lora16.bin",
            },
        },
    }
)


@register_webui("microsoft/china/slab/webui/bloom")
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
        return "China-Bloom"


@register_webui("microsoft/china/slab/webui/peft/bloom/lora")
class PeftBloomLoraWebUIV2(PeftBloomLoraWebUI):
    match_patterns = [
        "^peft-lora-bloom",
    ]
    pretrained_names = list(pretrained_peft_infos.keys())
    supported_pretrained_names = matched_pretrained_names(
        pretrained_names,
        match_patterns,
    )

    @property
    def name(self):
        return "China-Peft-Bloom"


# @register_webui("microsoft/china/slab/webui/peft/mistral/lora")
# class PeftMistralLoraWebUIV2(PeftMistralLoraWebUI):
#     match_patterns = [
#         "^peft-lora-mistral",
#     ]
#     pretrained_names = list(pretrained_peft_infos.keys())
#     supported_pretrained_names = matched_pretrained_names(
#         pretrained_names,
#         match_patterns,
#     )

#     @property
#     def name(self):
#         return "China-Peft-Mistral"
