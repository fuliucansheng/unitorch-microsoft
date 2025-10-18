# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

from unitorch_microsoft.spaces.demos.contribs.stable.bg_change import (
    ChangeBGWebUI as StableChangeBGWebUI,
)
from unitorch_microsoft.spaces.demos.contribs.stable.bg_expand import (
    ExpandBGWebUI as StableExpandBGWebUI,
)
from unitorch_microsoft.spaces.demos.contribs.stable.img_create import (
    CreateImgWebUI as StableCreateImgWebUI,
)

stable_pages = [
    StableCreateImgWebUI,
    StableChangeBGWebUI,
    StableExpandBGWebUI,
]
