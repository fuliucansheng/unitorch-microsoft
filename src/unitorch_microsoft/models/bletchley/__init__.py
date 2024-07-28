# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

pretrained_bletchley_v1_infos = {
    "0.3B": "https://unitorchazureblob.blob.core.windows.net/shares/models/bletchley/v1/pytorch_model.0.3B.bin",
    "0.8B": "https://unitorchazureblob.blob.core.windows.net/shares/models/bletchley/v1/pytorch_model.0.8B.bin",
    "2.5B": "https://unitorchazureblob.blob.core.windows.net/shares/models/bletchley/v1/pytorch_model.2.5B.bin",
}

pretrained_bletchley_v3_infos = {
    "base": "https://unitorchazureblob.blob.core.windows.net/shares/models/bletchley/v3/pytorch_model.base.bin",
    "large": "https://unitorchazureblob.blob.core.windows.net/shares/models/bletchley/v3/pytorch_model.large.bin",
}

import unitorch_microsoft.models.bletchley.modeling_v1
import unitorch_microsoft.models.bletchley.modeling_v3
import unitorch_microsoft.models.bletchley.processing_v1
import unitorch_microsoft.models.bletchley.processing_v3
