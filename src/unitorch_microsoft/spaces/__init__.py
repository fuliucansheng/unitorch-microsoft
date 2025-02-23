# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import importlib_resources
from unitorch.cli import CoreConfigureParser

spaces_folder = os.path.join(importlib_resources.files("unitorch_microsoft"), "spaces")

local_config_path = os.path.join(spaces_folder, "config.local.ini")
if os.path.exists(local_config_path):
    config_path = local_config_path
else:
    config_path = os.path.join(spaces_folder, "config.ini")

spaces_settings = CoreConfigureParser(config_path)

from unitorch_microsoft.spaces.utils import (
    create_element,
    create_row,
    create_column,
    create_tab,
    create_tabs,
    create_flex_layout,
    create_blocks,
    create_toper_menus,
    create_footer,
    create_dashboard_card,
    create_card,
    create_dashboard_cards_group,
    create_cards_group,
    hashed_link,
    start_http_server,
)

http_url = start_http_server()
