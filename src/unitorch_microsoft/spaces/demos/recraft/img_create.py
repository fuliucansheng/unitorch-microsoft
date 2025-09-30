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


from torchvision import transforms
from unitorch import mktempfile
from unitorch.utils import read_file
from unitorch.models import GenericOutputs
from unitorch.cli import CoreConfigureParser

from unitorch.cli.webuis import SimpleWebUI
from unitorch_microsoft import cached_path
from unitorch_microsoft.chatgpt.recraft import (
    supported_image_sizes,
    get_image as get_recraft_image,
    get_inpainting_image as get_recraft_inpainting_image,
    get_change_background_image as get_recraft_change_background_image,
    get_resolution_image as get_recraft_resolution_image,
    get_remove_background_image as get_recraft_remove_background_image,
)
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
from unitorch_microsoft.scripts.tools.report_items import reported_item


class CreateImgWebUI(SimpleWebUI):
    _title = "Recraft Image Generation"
    _description = "This is a demo for text to image generation using Recraft. You can input a prompt, and the model will generate an image based on the prompt."

    def __init__(self, config: CoreConfigureParser):
        self._status = getattr(self, "_status", "Stopped")
        # create elements
        toper_menus = create_toper_menus()
        footer = create_footer()
        header = create_element(
            "markdown",
            label=f"# <div style='margin-top:10px'>🖼️ {self._title}</div>",
            interactive=False,
        )
        description = create_element(
            "markdown",
            label=self._description,
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
        iface._title = self._title
        iface._description = self._description

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
            fn=self.generate,
            inputs=[prompt, width, height],
            outputs=[output_image],
            trigger_mode="once",
        )

        iface.load(
            fn=lambda: self._status,
            outputs=status,
        )

        iface.__exit__()

        super().__init__(config, iname=self._title, iface=iface)

    def start(self):
        self._status = "Running"
        return self._status

    def stop(self):
        self._status = "Stopped"
        return self._status

    def generate(self, prompt, width, height):
        result = get_recraft_image(
            prompt,
            width=width,
            height=height,
        )
        reported_item(
            record={
                "prompt": prompt,
                "style": "realistic_image",
                "width": width,
                "height": height,
                "tags": "#Recraft#T2I",
            },
            images={"result": result},
        )

        return result
