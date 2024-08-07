# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

from unitorch.cli.models.llava import (
    pretrained_llava_infos,
    pretrained_llava_extensions_infos,
)

pretrained_llava_infos = {
    **pretrained_llava_infos,
    **{
        "llava-v1.6-mistral-7b-bletchley-v1-2.5b": {
            "config": "https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf/resolve/main/config.json",
            "vocab": "https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf/resolve/main/tokenizer.model",
            "weight": [
                "https://unitorchazureblob.blob.core.windows.net/shares/models/bletchley/v1/pytorch_model.2.5B.bin"
            ]
            + [
                f"https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf/resolve/main/model-{str(i).rjust(5, '0')}-of-00004.safetensors"
                for i in range(1, 5)
            ],
        },
        "llava-v1.6-mistral-7b-bletchley-v3-2.5b": {
            "config": "https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf/resolve/main/config.json",
            "vocab": "https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf/resolve/main/tokenizer.model",
            "weight": [
                "https://unitorchazureblob.blob.core.windows.net/shares/models/bletchley/v3/pytorch_model.large.bin"
            ]
            + [
                f"https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf/resolve/main/model-{str(i).rjust(5, '0')}-of-00004.safetensors"
                for i in range(1, 5)
            ],
        },
        "llava-v1.6-mistral-7b-MMAExtension-bletchley-v3-2.5b": {
            "config": "https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf/resolve/main/config.json",
            "vocab": "https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf/resolve/main/tokenizer.model",
            "weight": [
                "https://unitorchazureblob.blob.core.windows.net/shares/models/bletchley/v3/MMAExtensionFinetune/pytorch_model.bin"
            ]
            + [
                f"https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf/resolve/main/model-{str(i).rjust(5, '0')}-of-00004.safetensors"
                for i in range(1, 5)
            ],
        },
    },
}

import unitorch_microsoft.models.llava.modeling_bletchley_v3
import unitorch_microsoft.models.llava.modeling_peft_bletchley_v3
import unitorch_microsoft.models.llava.processing_bletchley_v3
