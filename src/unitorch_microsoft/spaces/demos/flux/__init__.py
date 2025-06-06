# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

from unitorch_microsoft.spaces.demos.flux.bg_change import (
    ChangeBGWebUI as FluxChangeBGWebUI,
)
from unitorch_microsoft.spaces.demos.flux.bg_expand import (
    ExpandBGWebUI as FluxExpandBGWebUI,
)
from unitorch_microsoft.spaces.demos.flux.obj_add import AddObjWebUI as FluxAddObjWebUI
from unitorch_microsoft.spaces.demos.flux.obj_remove import (
    RemoveObjWebUI as FluxRemoveObjWebUI,
)
from unitorch_microsoft.spaces.demos.flux.img_create import (
    CreateImgWebUI as FluxCreateImgWebUI,
)

flux_pages = [
    FluxCreateImgWebUI,
    FluxChangeBGWebUI,
    FluxExpandBGWebUI,
    FluxAddObjWebUI,
    FluxRemoveObjWebUI,
]
