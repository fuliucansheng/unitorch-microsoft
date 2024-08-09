# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import io
import torch
import gc
import numpy as np
import gradio as gr
from PIL import Image, ImageFilter, ImageOps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.utils import is_opencv_available
from unitorch.cli import CoreConfigureParser, GenericWebUI
from unitorch.cli import register_webui
from unitorch.cli.webuis import (
    create_element,
    create_accordion,
    create_row,
    create_column,
    create_group,
    create_tab,
    create_tabs,
    create_blocks,
    create_pretrain_layout,
)
from unitorch.cli.webuis import SimpleWebUI


class CopyWebUI(SimpleWebUI):
    def __init__(self, config: CoreConfigureParser):
        # create elements
        product_image = create_element("image", "Product Image")
        background_image = create_element("image", "Background Image")
        height = create_element(
            "slider", "Height", min_value=0, max_value=1024, step=1, default=0
        )
        width = create_element(
            "slider", "Width", min_value=0, max_value=1024, step=1, default=0
        )
        threshold = create_element(
            "slider", "White Threshold", min_value=0, max_value=255, step=1, default=200
        )
        generate = create_element("button", "Generate")
        output_image = create_element("image", "Output Image")

        left = create_column(
            create_row(product_image, background_image),
            height,
            width,
            threshold,
            generate,
        )
        right = create_column(output_image)
        iface = create_blocks(create_row(left, right))

        # create events
        iface.__enter__()

        generate.click(
            fn=self.copy,
            inputs=[product_image, background_image, height, width, threshold],
            outputs=[output_image],
        )

        iface.__exit__()

        super().__init__(config, iname="Copy", iface=iface)

    def copy(
        self,
        product,
        background,
        height: Optional[int] = 0,
        width: Optional[int] = 0,
        threshold: Optional[int] = 200,
    ):
        if height > 0 and width > 0:
            product = product.resize((width, height))
            background = background.resize((width, height))
        if product.size != background.size:
            product = product.resize(background.size)

        product = product.convert("RGBA")
        background = background.convert("RGBA")
        product_data = product.getdata()
        new_product_data = []
        for item in product_data:
            if item[0] > threshold and item[1] > threshold and item[2] > threshold:
                new_product_data.append((255, 255, 255, 0))
            else:
                new_product_data.append(item)
        product.putdata(new_product_data)
        background.paste(product, (0, 0), product)
        return background


class PicassoWebUI(SimpleWebUI):
    def __init__(self, config: CoreConfigureParser):
        webuis = [
            CopyWebUI(config),
        ]
        iface = gr.TabbedInterface(
            [webui.iface for webui in webuis],
            tab_names=[webui.iname for webui in webuis],
        )
        super().__init__(config, iname="Picasso", iface=iface)
