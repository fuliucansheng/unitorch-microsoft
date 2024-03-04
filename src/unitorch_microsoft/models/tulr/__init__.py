# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

# pretrained infos
pretrained_tulr_infos = {
    "default-tulrv6": {
        "config": "https://unitorchazureblob.blob.core.windows.net/shares/models/tulrv6/base/config.json",
        "vocab": "https://unitorchazureblob.blob.core.windows.net/shares/models/tulrv6/base/sentencepiece.bpe.model",
    },
    "tulrv6-base": {
        "config": "https://unitorchazureblob.blob.core.windows.net/shares/models/tulrv6/base/config.json",
        "vocab": "https://unitorchazureblob.blob.core.windows.net/shares/models/tulrv6/base/sentencepiece.bpe.model",
        "weight": "https://unitorchazureblob.blob.core.windows.net/shares/models/tulrv6/base/pytorch_model.bin",
    },
    "tulrv6-large": {
        "config": "https://unitorchazureblob.blob.core.windows.net/shares/models/tulrv6/large/config.json",
        "vocab": "https://unitorchazureblob.blob.core.windows.net/shares/models/tulrv6/large/sentencepiece.bpe.model",
        "weight": "https://unitorchazureblob.blob.core.windows.net/shares/models/tulrv6/large/pytorch_model.bin",
    },
    "tulrv6-xlarge": {
        "config": "https://unitorchazureblob.blob.core.windows.net/shares/models/tulrv6/xlarge/config.json",
        "vocab": "https://unitorchazureblob.blob.core.windows.net/shares/models/tulrv6/xlarge/sentencepiece.bpe.model",
        "weight": "https://unitorchazureblob.blob.core.windows.net/shares/models/tulrv6/xlarge/pytorch_model.bin",
    },
}

import unitorch_microsoft.models.tulr.modeling_v6
import unitorch_microsoft.models.tulr.processing_v6
from unitorch_microsoft.models.tulr.modeling_v6 import TULRV6Config, TULRV6Model
