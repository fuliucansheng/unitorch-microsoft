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
)
from unitorch_microsoft.spaces.picasso.dashboard import DashboardWebUI
from unitorch_microsoft.spaces.picasso.dr_checks import DRChecksWebUI
from unitorch_microsoft.spaces.picasso.roi_detect import ROIDetectWebUI
from unitorch_microsoft.spaces.picasso.roi_detect_v2 import ROIDetectV2WebUI
from unitorch_microsoft.spaces.picasso.bg_expand import ExpandBGWebUI
from unitorch_microsoft.spaces.picasso.bg_expand_v2 import ExpandBGV2WebUI

# Picasso dashboard
dashboards = [
    GenericOutputs(
        title="Dashboard",
        desc="This Dashboard provides a comprehensive view of the Picasso image generative models performance and progress. It serves as a centralized hub for tracking key metrics related to the models powering Picasso, ensuring stakeholders and developers can monitor model health, accuracy, and efficiency.",
        link="/picasso/dashboard",
    )
]

# Picasso examples
examples = [
    DRChecksWebUI,
    ExpandBGWebUI,
    ExpandBGV2WebUI,
    ROIDetectWebUI,
    ROIDetectV2WebUI,
]

examples_pages = [_webui(spaces_settings).iface for _webui in examples]

for page in examples_pages:
    page._link = f"/picasso/{hashed_link(page._title + page._description, 6)}"
    page.title = f"Ads Spaces | Picasso - {page._title}"

examples = [
    GenericOutputs(
        title=p._title,
        desc=p._description,
        link=p._link,
    )
    for p in examples_pages
]


def create_picasso_page():
    toper_menus = create_toper_menus()
    dashboard_group = create_dashboard_cards_group(
        "🌐 Overview",
        dashboards,
    )

    examples_group = create_cards_group(
        "🎢 Examples",
        examples,
        elem_classes="ut-ms-min-50-height",
    )

    footer = create_footer()
    return create_blocks(
        toper_menus,
        dashboard_group,
        examples_group,
        footer,
    )


picasso_page = create_picasso_page()
picasso_page.title = "Ads Spaces | Picasso Home"

dashboard_page = DashboardWebUI(spaces_settings).iface

picasso_routers = {
    "/picasso": picasso_page,
    "/picasso/dashboard": dashboard_page,
    **{p._link: p for p in examples_pages},
}
