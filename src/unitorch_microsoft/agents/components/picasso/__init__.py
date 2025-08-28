# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os

_PICASSO_TEMP_DIR = "/tmp/picasso"

if not os.path.exists(_PICASSO_TEMP_DIR):
    os.makedirs(_PICASSO_TEMP_DIR)


def get_picasso_temp_dir() -> str:
    """
    Get the temporary directory for Picasso images.

    Returns:
        str: The path to the Picasso temporary directory.
    """
    return _PICASSO_TEMP_DIR


from unitorch_microsoft.agents.components.picasso.image import PicassoImageTool
from unitorch_microsoft.agents.components.picasso.internal import PicassoInternalTool
from unitorch_microsoft.agents.components.picasso.html import (
    PicassoHtmlTool,
    PicassoLayoutTool,
)
