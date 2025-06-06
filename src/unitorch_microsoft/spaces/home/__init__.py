# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
from unitorch.utils import read_file
from unitorch.models import GenericOutputs
from unitorch_microsoft import cached_path
from unitorch_microsoft.spaces import (
    create_element,
    create_row,
    create_column,
    create_blocks,
    create_toper_menus,
    create_footer,
    create_card,
)


def create_home_page():
    toper_menus = create_toper_menus()
    intro_page = create_element(
        "markdown",
        read_file(cached_path("spaces/home/intro.md")),
        elem_classes="ut-ms-home-intro",
    )
    news_page = create_element(
        "markdown",
        read_file(cached_path("spaces/home/news.md")),
        elem_classes="ut-ms-home-news",
    )
    footer = create_footer()
    return create_blocks(toper_menus, intro_page, news_page, footer)


home_page = create_home_page()
home_page.title = "Ads Spaces | AI Cube for Ads Team"

home_routers = {
    "/": home_page,
}
