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
from unitorch_microsoft.spaces.demos.stable import stable_pages
from unitorch_microsoft.spaces.demos.flux import flux_pages
from unitorch_microsoft.spaces.demos.recraft import recraft_pages
from unitorch_microsoft.spaces.demos.video import video_pages
from unitorch_microsoft.spaces.demos.tools import tools_pages

stable_pages = [_webui(spaces_settings).iface for _webui in stable_pages]
flux_pages = [_webui(spaces_settings).iface for _webui in flux_pages]
recraft_pages = [_webui(spaces_settings).iface for _webui in recraft_pages]
video_pages = [_webui(spaces_settings).iface for _webui in video_pages]
tools_pages = [_webui(spaces_settings).iface for _webui in tools_pages]

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
