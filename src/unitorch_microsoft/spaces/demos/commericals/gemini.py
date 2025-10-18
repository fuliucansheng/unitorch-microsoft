# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import io
import gc
import re
import base64
from openai import api_key
import requests
import pandas as pd
import gradio as gr
from PIL import Image
from google import genai
from unitorch.cli import CoreConfigureParser
from unitorch.cli import register_webui
from unitorch.cli.webuis import SimpleWebUI
from unitorch_microsoft.spaces import (
    create_element,
    create_row,
    create_column,
    create_flex_layout,
    create_blocks,
    create_toper_menus,
    create_footer,
    create_tab,
    create_tabs,
    create_dashboard_card,
    create_card,
    create_dashboard_cards_group,
    create_cards_group,
)


class NanoBananaWebUI(SimpleWebUI):
    _title = "Nano Banana Image Generation"
    _description = "This is a demo for Nano Banana, a model for generating images from text prompts and input images. You can input multiple images and a prompt, and the model will generate an image based on the prompt and the input images."

    def __init__(self, config: CoreConfigureParser):
        self._status = getattr(self, "_status", "Stopped")

        toper_menus = create_toper_menus()
        footer = create_footer()
        header = create_element(
            "markdown",
            label=f"# <div style='margin-top:10px'>🛠️ {self._title}</div>",
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

        prompt = create_element("text", "Prompt", lines=3)
        num_input_images = 5
        input_images = [
            create_element("image", "Image") for _ in range(num_input_images)
        ]
        image_brush = create_element("image_editor", "Input Image")
        images = create_tabs(
            *[create_tab(ele, name=f"Image {i}") for i, ele in enumerate(input_images)],
            create_tab(image_brush, name="Edit Image"),
        )
        output_image = create_element("image", "Output Image")
        generate = create_element("button", "Generate", variant="primary")

        # layout
        left = create_column(images, prompt, generate)
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

        iface.__enter__()
        start.click(self.start, [], [status])
        stop.click(self.stop, [], [status])
        generate.click(
            self.generate,
            [prompt, image_brush, *input_images],
            [output_image],
        )
        iface.__exit__()
        super().__init__(config, iname=self._title, iface=iface)

    def start(self):
        self._status = "Running"
        self.client = genai.Client()
        return self._status

    def stop(self):
        self._status = "Stopped"
        self.client = None
        return self._status

    def generate(self, prompt, image_brush, *images):
        if self._status != "Running":
            self.start()
        _images = [im for im in images if im is not None]
        if image_brush.get("composite") is not None:
            composite = image_brush["composite"]
            _images = _images + [composite]
        response = self.client.models.generate_content(
            model="gemini-2.5-flash-image-preview",
            contents=_images + [prompt],
        )
        image_parts = [
            part.inline_data.data
            for part in response.candidates[0].content.parts
            if part.inline_data
        ]
        image = Image.open(io.BytesIO(image_parts[0]))
        return image
