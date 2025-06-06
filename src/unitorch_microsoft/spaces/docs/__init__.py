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
)


def create_docs_page():
    toper_menus = create_toper_menus()
    footer = create_footer()
    return create_blocks(toper_menus, footer)


docs_page = create_docs_page()
docs_page.title = "Ads Spaces | Docs Home"


docs_routers = {
    "/docs": docs_page,
}
