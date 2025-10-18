# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

from unitorch_microsoft.spaces.demos.contribs.flux.bg_change import (
    ChangeBGWebUI as FluxChangeBGWebUI,
)
from unitorch_microsoft.spaces.demos.contribs.flux.bg_expand import (
    ExpandBGWebUI as FluxExpandBGWebUI,
)
from unitorch_microsoft.spaces.demos.contribs.flux.img_create import (
    CreateImgWebUI as FluxCreateImgWebUI,
)

flux_pages = [
    FluxCreateImgWebUI,
    FluxChangeBGWebUI,
    FluxExpandBGWebUI,
]
