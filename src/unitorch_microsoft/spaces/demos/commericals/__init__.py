# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

from unitorch_microsoft.spaces.demos.commericals.recraft import (
    RecraftCreateImgWebUI,
    RecraftChangeBGWebUI,
    RecraftExpandBGWebUI,
)
from unitorch_microsoft.spaces.demos.commericals.gpt import (
    GPT5WebUI,
    GPT5VisonWebUI,
    GPTCreateImgWebUI,
    SORA2CreateVideoWebUI,
)
from unitorch_microsoft.spaces.demos.commericals.gemini import (
    NanoBananaWebUI,
)
from unitorch_microsoft.spaces.demos.commericals.keling import (
    KelingImage2VideoWebUI,
)

commerical_pages = [
    GPT5WebUI,
    GPT5VisonWebUI,
    GPTCreateImgWebUI,
    SORA2CreateVideoWebUI,
    NanoBananaWebUI,
    KelingImage2VideoWebUI,
    RecraftCreateImgWebUI,
    RecraftChangeBGWebUI,
    RecraftExpandBGWebUI,
]
