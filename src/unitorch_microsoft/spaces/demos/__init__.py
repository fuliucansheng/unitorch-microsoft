# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
from unitorch.utils import read_file
from unitorch.models import GenericOutputs
from unitorch.cli import CoreConfigureParser
from unitorch_microsoft import cached_path
from unitorch_microsoft.spaces import (
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
from unitorch_microsoft.spaces.demos.stable.bg_change import ChangeBGWebUI
from unitorch_microsoft.spaces.demos.stable.bg_expand import ExpandBGWebUI
from unitorch_microsoft.spaces.demos.stable.bg_remove import RemoveBGWebUI
from unitorch_microsoft.spaces.demos.stable.obj_add import AddObjWebUI
from unitorch_microsoft.spaces.demos.stable.obj_remove import RemoveObjWebUI
from unitorch_microsoft.spaces.demos.video.opencv import ZoomInWebUI
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
from unitorch_microsoft.spaces.demos.flux.t2i import T2IWebUI


config_path = cached_path("spaces/config.ini")
config = CoreConfigureParser(config_path)

# image
change_bg_page = ChangeBGWebUI(config).iface
change_bg_page.infos = {
    "title": "Change Background",
    "description": "This is a demo for changing background.",
}

expand_bg_page = ExpandBGWebUI(config).iface
expand_bg_page.infos = {
    "title": "Expand Background",
    "description": "This is a demo for expanding background.",
}

remove_bg_page = RemoveBGWebUI(config).iface
remove_bg_page.infos = {
    "title": "Remove Background",
    "description": "This is a demo for removing background.",
}

add_obj_page = AddObjWebUI(config).iface
add_obj_page.infos = {
    "title": "Add Object",
    "description": "This is a demo for adding object.",
}

remove_obj_page = RemoveObjWebUI(config).iface
remove_obj_page.infos = {
    "title": "Remove Object",
    "description": "This is a demo for removing object.",
}

# flux
flux_t2i_page = T2IWebUI(config).iface
flux_t2i_page.infos = {
    "title": "Text to Image",
    "description": "This is a demo for text to image with FLUX.",
}
flux_change_bg_page = FluxChangeBGWebUI(config).iface
flux_change_bg_page.infos = {
    "title": "Change Background",
    "description": "This is a demo for changing background with FLUX.",
}

flux_expand_bg_page = FluxExpandBGWebUI(config).iface
flux_expand_bg_page.infos = {
    "title": "Expand Background",
    "description": "This is a demo for expanding background with FLUX.",
}

flux_add_obj_page = FluxAddObjWebUI(config).iface
flux_add_obj_page.infos = {
    "title": "Add Object",
    "description": "This is a demo for adding object with FLUX.",
}

flux_remove_obj_page = FluxRemoveObjWebUI(config).iface
flux_remove_obj_page.infos = {
    "title": "Remove Object",
    "description": "This is a demo for removing object with FLUX.",
}

# video
zoom_in_page = ZoomInWebUI(config).iface
zoom_in_page.infos = {
    "title": "Zoom In",
    "description": "This is a demo for zooming in video.",
}

image_pages = [
    change_bg_page,
    expand_bg_page,
    remove_bg_page,
    add_obj_page,
    remove_obj_page,
]

flux_pages = [
    flux_t2i_page,
    flux_change_bg_page,
    flux_expand_bg_page,
    flux_add_obj_page,
    flux_remove_obj_page,
]

video_pages = [
    zoom_in_page,
]

all_pages = image_pages + flux_pages + video_pages
for page in all_pages:
    page.infos[
        "link"
    ] = f"/demos/{hashed_link(page.infos['title'] + page.infos['description'], 6)}"
    page.title = f"Ads Spaces | Demos - {page.infos['title']}"


def create_demos_page():
    toper_menus = create_toper_menus()

    image_cards = [
        GenericOutputs(
            title=p.infos["title"],
            desc=p.infos["description"],
            link=p.infos["link"],
        )
        for p in image_pages
    ]
    image_group = create_cards_group(
        "✨ Stable V1.5",
        image_cards,
    )

    flux_cards = [
        GenericOutputs(
            title=p.infos["title"],
            desc=p.infos["description"],
            link=p.infos["link"],
        )
        for p in flux_pages
    ]
    flux_group = create_cards_group(
        "🎨 FLUX",
        flux_cards,
    )

    video_cards = [
        GenericOutputs(
            title=p.infos["title"],
            desc=p.infos["description"],
            link=p.infos["link"],
        )
        for p in video_pages
    ]
    video_group = create_cards_group(
        "🎬 Video",
        video_cards,
    )
    footer = create_footer()
    return create_blocks(toper_menus, image_group, flux_group, video_group, footer)


demos_page = create_demos_page()
demos_page.title = "Ads Spaces | Demos Home"

demos_routers = {
    "/demos": demos_page,
    **{p.infos["link"]: p for p in all_pages},
}
