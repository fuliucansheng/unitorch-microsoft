# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
from unitorch.utils import read_file
from unitorch.models import GenericOutputs
from unitorch.cli import CoreConfigureParser
from unitorch_microsoft import cached_path
from unitorch_microsoft.spaces import (
    spaces_settings,
    create_element,
    create_row,
    create_column,
    create_flex_layout,
    create_blocks,
    create_toper_menus,
    create_footer,
    create_dashboard_card,
    create_card,
    create_dashboard_cards_group,
    create_cards_group,
    hashed_link,
)
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
from unitorch_microsoft.spaces.demos.recraft.img_create import (
    CreateImgWebUI as RecraftCreateImgWebUI,
)
from unitorch_microsoft.spaces.demos.video.opencv import ZoomInWebUI
from unitorch_microsoft.spaces.demos.tools.bg_remove import RemoveBGWebUI
from unitorch_microsoft.spaces.demos.tools.img_caption import CaptionImgWebUI
from unitorch_microsoft.spaces.demos.tools.joycaption2 import JoyCaption2WebUI

# image
stable_create_img_page = StableCreateImgWebUI(spaces_settings).iface
stable_change_bg_page = StableChangeBGWebUI(spaces_settings).iface
stable_expand_bg_page = StableExpandBGWebUI(spaces_settings).iface
stable_add_obj_page = StableAddObjWebUI(spaces_settings).iface
stable_remove_obj_page = StableRemoveObjWebUI(spaces_settings).iface

# flux
flux_create_img_page = FluxCreateImgWebUI(spaces_settings).iface
flux_change_bg_page = FluxChangeBGWebUI(spaces_settings).iface
flux_expand_bg_page = FluxExpandBGWebUI(spaces_settings).iface
flux_add_obj_page = FluxAddObjWebUI(spaces_settings).iface
flux_remove_obj_page = FluxRemoveObjWebUI(spaces_settings).iface

# recraft
recraft_create_img_page = RecraftCreateImgWebUI(spaces_settings).iface

# video
zoom_in_page = ZoomInWebUI(spaces_settings).iface

# tools
remove_bg_page = RemoveBGWebUI(spaces_settings).iface
caption_img_page = CaptionImgWebUI(spaces_settings).iface
joycaption2_page = JoyCaption2WebUI(spaces_settings).iface

stable_pages = [
    stable_create_img_page,
    stable_change_bg_page,
    stable_expand_bg_page,
    stable_add_obj_page,
    stable_remove_obj_page,
]

flux_pages = [
    flux_create_img_page,
    flux_change_bg_page,
    flux_expand_bg_page,
    flux_add_obj_page,
    flux_remove_obj_page,
]

recraft_pages = [
    recraft_create_img_page,
]

video_pages = [
    zoom_in_page,
]

tools_pages = [
    remove_bg_page,
    caption_img_page,
    joycaption2_page,
]

all_pages = stable_pages + flux_pages + recraft_pages + video_pages + tools_pages
for page in all_pages:
    page._link = f"/demos/{hashed_link(page._title + page._description, 6)}"
    page.title = f"Ads Spaces | Demos - {page._title}"


def create_demos_page():
    toper_menus = create_toper_menus()

    stable_cards = [
        GenericOutputs(
            title=p._title,
            desc=p._description,
            link=p._link,
        )
        for p in stable_pages
    ]
    stable_group = create_cards_group(
        "✨ Stable V1.5",
        stable_cards,
    )

    flux_cards = [
        GenericOutputs(
            title=p._title,
            desc=p._description,
            link=p._link,
        )
        for p in flux_pages
    ]
    flux_group = create_cards_group(
        "🎨 FLUX",
        flux_cards,
    )

    recraft_cards = [
        GenericOutputs(
            title=p._title,
            desc=p._description,
            link=p._link,
        )
        for p in recraft_pages
    ]
    recraft_group = create_cards_group(
        "🖼️ Recraft",
        recraft_cards,
    )

    video_cards = [
        GenericOutputs(
            title=p._title,
            desc=p._description,
            link=p._link,
        )
        for p in video_pages
    ]
    video_group = create_cards_group(
        "🎬 Video",
        video_cards,
    )

    tools_cards = [
        GenericOutputs(
            title=p._title,
            desc=p._description,
            link=p._link,
        )
        for p in tools_pages
    ]
    tools_group = create_cards_group(
        "🛠️ Tools",
        tools_cards,
    )
    footer = create_footer()
    return create_blocks(
        toper_menus,
        stable_group,
        flux_group,
        recraft_group,
        video_group,
        tools_group,
        footer,
    )


demos_page = create_demos_page()
demos_page.title = "Ads Spaces | Demos Home"

demos_routers = {
    "/demos": demos_page,
    **{p._link: p for p in all_pages},
}
