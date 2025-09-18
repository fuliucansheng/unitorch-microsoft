# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import importlib_resources
from unitorch.cli import CoreConfigureParser

spaces_folder = os.path.join(importlib_resources.files("unitorch_microsoft"), "spaces")
config_path = os.path.join(spaces_folder, "config.local.ini")
if not os.path.exists(config_path):
    config_path = os.path.join(spaces_folder, "config.ini")

spaces_settings = CoreConfigureParser(config_path)

fastapi_endpoint = spaces_settings.getdefault("core/cli", "fastapi_endpoint", None)
custom_environ = spaces_settings.getdefault("core/cli", "environ", {})
for k, v in custom_environ.items():
    os.environ[k] = v

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
    call_fastapi,
)
