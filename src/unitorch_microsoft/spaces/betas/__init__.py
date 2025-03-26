# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import hashlib
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
    http_url as _http_url,
)
from unitorch_microsoft.spaces.betas.bg_expand import ExpandBGWebUI
from unitorch_microsoft.spaces.betas.bg_expand2 import ExpandBG2WebUI

# beta examples
expand_bg_page = ExpandBGWebUI(spaces_settings).iface
expand_bg_page2 = ExpandBG2WebUI(spaces_settings).iface

image_pages = [
    expand_bg_page,
    expand_bg_page2,
]

all_pages = image_pages

for page in all_pages:
    page._link = f"/betas/{hashed_link(page._title + page._description, 6)}"
    page.title = f"Ads Spaces | Beta - {page._title}"


def create_betas_page():
    toper_menus = create_toper_menus()
    image_cards = [
        GenericOutputs(
            title=p._title,
            desc=p._description,
            link=p._link,
        )
        for p in image_pages
    ]
    image_group = create_cards_group(
        "✨ Image",
        image_cards,
    )

    footer = create_footer()
    return create_blocks(
        toper_menus,
        image_group,
        footer,
    )


betas_page = create_betas_page()
betas_page.title = "Ads Spaces | Beta Home"

betas_routers = {
    "/betas": betas_page,
    **{p._link: p for p in all_pages},
}
