# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

from unitorch.cli.models.bert import pretrained_bert_infos

pretrained_bert_infos.update(
    {
        "bert-base-multilingual-cased": {
            "config": "https://huggingface.co/bert-base-multilingual-cased/resolve/main/config.json",
            "vocab": "https://huggingface.co/bert-base-multilingual-cased/resolve/main/vocab.txt",
            "weight": "https://huggingface.co/bert-base-multilingual-cased/resolve/main/pytorch_model.bin",
        },
    }
)

import unitorch_microsoft.adsplus.relevance.tribert.modeling_v8
import unitorch_microsoft.adsplus.relevance.tribert.modeling_v9
import unitorch_microsoft.adsplus.relevance.tribert.processing_v8
import unitorch_microsoft.adsplus.relevance.tribert.processing_v9
import unitorch_microsoft.adsplus.relevance.tribert.modeling_v9_clip
