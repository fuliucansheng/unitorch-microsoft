# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import io
import requests
import hashlib
from PIL import Image
from unitorch.cli.webuis import (
    create_element,
    create_accordion,
    create_row,
    create_column,
    create_tab,
    create_tabs,
    create_flex_layout,
    create_group,
    create_tab,
    create_tabs,
    create_blocks,
)
from unitorch_microsoft.spaces.utils.tools import (
    start_http_server,
    random_bg_color,
    hashed_link,
)

from unitorch_microsoft.spaces.utils.layouts import (
    create_toper_menus,
    create_footer,
    create_dashboard_card,
    create_card,
    create_dashboard_cards_group,
    create_cards_group,
)


def call_fastapi(url, params={}, images=None, req_type="POST", resp_type="json"):
    assert resp_type in ["json", "image"], f"Unsupported response type: {resp_type}"

    def process_image(image):
        image = image.convert("RGB")
        byte_arr = io.BytesIO()
        image.save(byte_arr, format="JPEG")
        byte_arr.seek(0)
        return byte_arr

    if images is None:
        files = {}
    else:
        files = {
            k: (f"{k}.jpg", process_image(v), "image/jpeg") for k, v in images.items()
        }
    if req_type == "POST" or images is not None:
        resp = (
            requests.post(url, params=params, files=files)
            if images is not None
            else requests.post(url, params=params)
        )
    else:
        resp = requests.get(url, params=params)
    if resp_type == "json":
        result = resp.json()
    elif resp_type == "image":
        result = Image.open(io.BytesIO(resp.content)).convert("RGB")
    return result
