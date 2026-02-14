# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

from unitorch_microsoft.spaces.demos.commericals.recraft import (
    RecraftCreateImgWebUI,
    RecraftChangeBGWebUI,
    RecraftExpandBGWebUI,
    RecraftRemoveBGWebUI,
)
from unitorch_microsoft.spaces.demos.commericals.gpt import (
    GPT5WebUI,
    SORA2CreateVideoWebUI,
)
from unitorch_microsoft.spaces.demos.commericals.gemini import (
    NanoBananaWebUI,
    Gemini3WebUI,
)
from unitorch_microsoft.spaces.demos.commericals.keling import (
    KelingImage2VideoWebUI,
)
from unitorch_microsoft.spaces.demos.commericals.seedream import (
    SeedreamWebUI,
)

commerical_pages = [
    GPT5WebUI,
    SORA2CreateVideoWebUI,
    NanoBananaWebUI,
    Gemini3WebUI,
    SeedreamWebUI,
    KelingImage2VideoWebUI,
    RecraftCreateImgWebUI,
    RecraftChangeBGWebUI,
    RecraftExpandBGWebUI,
    RecraftRemoveBGWebUI,
]
