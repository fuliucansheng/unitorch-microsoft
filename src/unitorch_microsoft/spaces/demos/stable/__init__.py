# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

from unitorch_microsoft.spaces.demos.stable.bg_change import (
    ChangeBGWebUI as StableChangeBGWebUI,
)
from unitorch_microsoft.spaces.demos.stable.bg_expand import (
    ExpandBGWebUI as StableExpandBGWebUI,
)
from unitorch_microsoft.spaces.demos.stable.obj_add import (
    AddObjWebUI as StableAddObjWebUI,
)
from unitorch_microsoft.spaces.demos.stable.obj_remove import (
    RemoveObjWebUI as StableRemoveObjWebUI,
)
from unitorch_microsoft.spaces.demos.stable.img_create import (
    CreateImgWebUI as StableCreateImgWebUI,
)

stable_pages = [
    StableCreateImgWebUI,
    StableChangeBGWebUI,
    StableExpandBGWebUI,
    StableAddObjWebUI,
    StableRemoveObjWebUI,
]
