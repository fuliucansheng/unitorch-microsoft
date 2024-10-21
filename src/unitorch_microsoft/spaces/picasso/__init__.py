# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import hashlib
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
    http_url as _http_url,
)
from unitorch_microsoft.spaces.picasso.dashboard import DashboardWebUI
from unitorch_microsoft.spaces.picasso.example import ExampleWebUI

config_path = cached_path("spaces/config.ini")
config = CoreConfigureParser(config_path)

# Picasso dashboard
dashboards = [
    GenericOutputs(
        title="Dashboard",
        desc="This Dashboard provides a comprehensive view of the Picasso image generative models performance and progress. It serves as a centralized hub for tracking key metrics related to the models powering Picasso, ensuring stakeholders and developers can monitor model health, accuracy, and efficiency.",
        link="/picasso/dashboard",
    )
]

# Picasso examples
config.set_default_section("microsoft/spaces/picasso/examples")
_sections = config.getoption("sections", [])
_titles = config.getoption("titles", [])
_descriptions = config.getoption("descriptions", [])
_links = [
    "/picasso/" + hashed_link(title + desc, 6)
    for title, desc in zip(_titles, _descriptions)
]
examples = [
    GenericOutputs(section=section, title=title, desc=desc, link=link)
    for section, title, desc, link in zip(_sections, _titles, _descriptions, _links)
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

dashboard_page = DashboardWebUI(config).iface
dashboard_page.title = "Ads Spaces | Picasso Dashboard"

examples_pages = {}
for example in examples:
    page = ExampleWebUI(
        config, example.section, example.title, example.desc, _http_url
    ).iface
    page.title = "Ads Spaces | Picasso - " + example.title
    examples_pages[example.link] = page

picasso_routers = {
    "/picasso": picasso_page,
    "/picasso/dashboard": dashboard_page,
    **examples_pages,
}
