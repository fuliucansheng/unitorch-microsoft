# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import cv2
import gc
import io
import requests
import torch
import numpy as np
import gradio as gr
from PIL import Image, ImageDraw, ImageOps
from transformers import AutoModelForImageSegmentation

from torchvision import transforms
from unitorch import mktempfile
from unitorch.utils import read_file
from unitorch.models import GenericOutputs
from unitorch.cli import CoreConfigureParser

from unitorch.cli.webuis import SimpleWebUI
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

try:
    from openai import OpenAI
except ImportError:
    raise ImportError(
        "Please install the openai package by running `pip install openai`"
    )

supported_image_sizes = [
    (1024, 1024),
    (1365, 1024),
    (1024, 1365),
    (1536, 1024),
    (1024, 1536),
    (1820, 1024),
    (1024, 1820),
    (1024, 2048),
    (2048, 1024),
    (1434, 1024),
    (1024, 1434),
    (1024, 1280),
    (1280, 1024),
    (1024, 1707),
    (1707, 1024),
]

supported_image_styles = [
    "realistic_image",
    "digital_illustration",
    "vector_illustration",
    "icon",
    "logo_raster",
]


class CreateImgWebUI(SimpleWebUI):
    def __init__(self, config: CoreConfigureParser):
        self.token = config.getdefault("microsoft/spaces/recraft", "token", None)
        if self.token is None:
            raise ValueError("Please provide a valid OpenAI Recraft API key.")
        self._status = getattr(self, "_status", "Stopped")
        # create elements
        toper_menus = create_toper_menus()
        footer = create_footer()
        header = create_element(
            "markdown",
            label=f"# <div style='margin-top:10px'>🖼️ Text to Image Generation</div>",
            interactive=False,
        )
        description = create_element(
            "markdown",
            label="description",
            interactive=False,
        )

        status = create_element("text", "Status", default="Stopped", interactive=False)
        start = create_element("button", "Start", variant="primary")
        stop = create_element("button", "Stop", variant="stop")
        prompt = create_element("text", "Input Prompt", lines=3)
        supported_widths = list(set([size[0] for size in supported_image_sizes]))
        supported_heights = list(set([size[1] for size in supported_image_sizes]))
        width = create_element("radio", "Width", default=1024, values=supported_widths)
        height = create_element(
            "radio", "Height", default=1024, values=supported_heights
        )
        generate = create_element("button", "Generate")
        output_image = create_element("image", "Output Image")

        left = create_column(prompt, width, height, generate)
        right = create_column(output_image)
        iface = create_blocks(
            toper_menus,
            create_row(
                create_column(header, description, scale=1),
                create_row(status, start, stop),
            ),
            create_row(
                left, right, elem_classes="ut-bg-transparent ut-ms-min-70-height"
            ),
            footer,
        )
        iface._title = "Text to Image"
        iface._description = "This is a demo for text to image with Recraft."

        # create events
        iface.__enter__()

        start.click(
            self.start,
            outputs=[status],
            trigger_mode="once",
        )
        stop.click(
            self.stop,
            outputs=[status],
            trigger_mode="once",
        )

        generate.click(
            fn=self.serve,
            inputs=[prompt, width, height],
            outputs=[output_image],
            trigger_mode="once",
        )

        iface.load(
            fn=lambda: self._status,
            outputs=status,
        )

        iface.__exit__()

        super().__init__(config, iname="Text to Image Generation", iface=iface)

    def start(self):
        self.client = OpenAI(
            base_url="https://external.api.recraft.ai/v1",
            api_key=self.token,
        )
        self._status = "Running"
        return self._status

    def stop(self):
        del self.client
        gc.collect()
        torch.cuda.empty_cache()
        self._status = "Stopped"
        return self._status

    def serve(self, prompt, width, height):
        if width != 1024 and height != 1024:
            gr.Warning(
                "Please note that the width or height must have one value as 1024."
            )
            return None
        response = self.client.images.generate(
            prompt=prompt,
            style="realistic_image",
            size=f"{width}x{height}",
        )
        url = response.data[0].url
        doc = requests.get(url)
        result = Image.open(io.BytesIO(doc.content))
        return result
