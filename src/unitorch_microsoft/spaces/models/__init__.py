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
)

config_path = cached_path("spaces/config.ini")
config = CoreConfigureParser(config_path)


def create_models_page():
    toper_menus = create_toper_menus()
    footer = create_footer()
    return create_blocks(
        toper_menus,
        footer,
    )


models_page = create_models_page()
models_page.title = "Ads Spaces | Models Home"

models_routers = {
    "/models": models_page,
}
