# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

pretrained_bletchley_v1_infos = {
    "0.3B": "https://unitorchazureblob.blob.core.windows.net/shares/models/bletchley/v1/pytorch_model.0.3B.bin",
    "0.8B": "https://unitorchazureblob.blob.core.windows.net/shares/models/bletchley/v1/pytorch_model.0.8B.bin",
    "2.5B": "https://unitorchazureblob.blob.core.windows.net/shares/models/bletchley/v1/pytorch_model.2.5B.bin",
}

pretrained_bletchley_v1_extensions_infos = {
    "lora-2.5B-picasso-blurry-image-quality": {
        "text": "#### Prompt: blurry \n #### Max Sequence: 36",
        "weight": "https://unitorchazureblob.blob.core.windows.net/shares/models/adsplus/lora/bletchley/pytorch_model.v1.lora4.blurry.2409.bin",
    },
    "lora-0.3B-picasso-watermark": {
        "text": "#### Prompt: watermarked, no watermark signature, brand logo \n #### Max Sequence: 36",
        "weight": "https://unitorchazureblob.blob.core.windows.net/shares/models/adsplus/lora/bletchley/pytorch_model.v1.0.3B.lora4.watermark.2410.bin",
    },
    "lora-2.5B-picasso-watermark": {
        "text": "#### Prompt: watermarked, no watermark signature, brand logo \n #### Max Sequence: 36",
        "weight": "https://unitorchazureblob.blob.core.windows.net/shares/models/adsplus/lora/bletchley/pytorch_model.v1.2.5B.lora4.watermark.2410.bin",
    },
}

pretrained_bletchley_v3_infos = {
    "0.8B": "https://unitorchazureblob.blob.core.windows.net/shares/models/bletchley/v3/pytorch_model.base.bin",
    "2.5B": "https://unitorchazureblob.blob.core.windows.net/shares/models/bletchley/v3/pytorch_model.large.bin",
}

pretrained_bletchley_v3_extensions_infos = {
    "lora-2.5B-flux-image-quality": {
        "text": "#### Prompt: worst quality, normal quality, low quality, low res, blurry, distortion, text, watermark, logo, banner, extra digits, cropped, jpeg artifacts, signature, username, error, sketch, duplicate, ugly, monochrome, horror, geometry, mutation, disgusting, bad anatomy, bad proportions, bad quality, deformed, disconnected limbs, out of frame, out of focus, dehydrated, disfigured, extra arms, extra limbs, extra hands, fused fingers, gross proportions, long neck, jpeg, malformed limbs, mutated, mutated hands, mutated limbs, missing arms, missing fingers, picture frame, poorly drawn hands, poorly drawn face, collage, pixel, pixelated, grainy, color aberration, amputee, autograph, bad illustration, beyond the borders, blank background, body out of frame, boring background, branding, cut off, dismembered, disproportioned, distorted, draft, duplicated features, extra fingers, extra legs, fault, flaw, grains, hazy, identifying mark, improper scale, incorrect physiology, incorrect ratio, indistinct, kitsch, low resolution, macabre, malformed, mark, misshapen, missing hands, missing legs, mistake, morbid, mutilated, off-screen, outside the picture, poorly drawn feet, printed words, render, repellent, replicate, reproduce, revolting dimensions, script, shortened, sign, split image, squint, storyboard, tiling, trimmed, unfocused, unattractive, unnatural pose, unreal engine, unsightly, written language \n #### Max Sequence: 36",
        "weight": "https://unitorchazureblob.blob.core.windows.net/shares/models/adsplus/lora/bletchley/pytorch_model.v3.lora4.imagequality.flux.2409.bin",
    },
    "lora-2.5B-picasso-blurry-image-quality": {
        "text": "#### Prompt: blurry \n #### Max Sequence: 36",
        "weight": "https://unitorchazureblob.blob.core.windows.net/shares/models/adsplus/lora/bletchley/pytorch_model.v3.lora4.blurry.2409.bin",
    },
    "lora-0.8B-picasso-watermark": {
        "text": "#### Prompt: watermarked, no watermark signature, brand logo \n #### Max Sequence: 36",
        "weight": "https://unitorchazureblob.blob.core.windows.net/shares/models/adsplus/lora/bletchley/pytorch_model.v3.0.8B.lora4.watermark.2410.bin",
    },
    "lora-2.5B-picasso-watermark": {
        "text": "#### Prompt: watermarked, no watermark signature, brand logo \n #### Max Sequence: 36",
        "weight": "https://unitorchazureblob.blob.core.windows.net/shares/models/adsplus/lora/bletchley/pytorch_model.v3.2.5B.lora4.watermark.2410.bin",
    },
    "lora-2.5B-query-image-relevance": {
        "text": "#### Prompt: query text \n #### Max Sequence: 36",
        "weight": "https://unitorchazureblob.blob.core.windows.net/shares/models/adsplus/lora/bletchley/pytorch_model.v3.lora4.image.relevance.0.9367.2409.bin",
    },
}

import unitorch_microsoft.models.bletchley.modeling_v1
import unitorch_microsoft.models.bletchley.modeling_peft_v1
import unitorch_microsoft.models.bletchley.modeling_v3
import unitorch_microsoft.models.bletchley.modeling_peft_v3
import unitorch_microsoft.models.bletchley.processing_v1
import unitorch_microsoft.models.bletchley.processing_v3
