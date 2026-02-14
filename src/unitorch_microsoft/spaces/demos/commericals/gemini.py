# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import io
import gc
import re
import base64
import requests
import pandas as pd
import gradio as gr
from PIL import Image
from google import genai
from google.genai import types
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
        model = create_element("radio", "Model", values=["Base", "Pro"], default="Pro")
        ratio = create_element(
            "radio",
            "Aspect Ratio",
            values=[
                "1:1",
                "2:3",
                "3:2",
                "3:4",
                "4:3",
                "4:5",
                "5:4",
                "9:16",
                "16:9",
                "21:9",
            ],
            default="16:9",
        )
        generate = create_element("button", "Generate", variant="primary")

        # layout
        left = create_column(images, prompt, model, ratio, generate)
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
            [prompt, model, ratio, image_brush, *input_images],
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

    def generate(self, prompt, model, ratio, image_brush, *images):
        if self._status != "Running":
            self.start()
        _images = [im for im in images if im is not None]
        if image_brush is not None and image_brush.get("composite") is not None:
            composite = image_brush["composite"]
            _images = _images + [composite]
        response = self.client.models.generate_content(
            model=(
                "gemini-3-pro-image-preview"
                if model == "Pro"
                else "gemini-2.5-flash-image"
            ),
            contents=_images + [prompt],
            config=types.GenerateContentConfig(
                image_config=types.ImageConfig(
                    aspect_ratio=ratio,
                )
            ),
        )
        image_parts = [
            part.inline_data.data
            for part in response.candidates[0].content.parts
            if part.inline_data
        ]
        image = Image.open(io.BytesIO(image_parts[0]))
        return image


class Gemini3WebUI(SimpleWebUI):
    _title = "Gemini3"
    _description = "This is a demo for Gemini 3. You can input images and a prompt (images are optional), and Gemini 3 will generate a result."

    def __init__(self, config: CoreConfigureParser):
        self._status = getattr(self, "_status", "Stopped")
        self.client = None
        # create elements
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
        num_input_images = 5
        input_images = [
            create_element("image", "Image") for _ in range(num_input_images)
        ]
        input_prompt = create_element("text", "Input Prompt", lines=3)
        generate = create_element("button", "Generate")
        output_answer = create_element("text", "Output Answer", lines=5)

        images = create_tabs(
            *[create_tab(ele, name=f"Image {i}") for i, ele in enumerate(input_images)]
        )
        left = create_column(images, input_prompt, generate)
        right = create_column(output_answer)

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
            inputs=[input_prompt, *input_images],
            outputs=[output_answer],
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
        self.client = genai.Client()
        return self._status

    def stop(self):
        self._status = "Stopped" if self._pipe is None else "Running"
        self.client = None
        return self._status

    def generate(self, prompt, *images):
        if self._status != "Running":
            self.start()
        images = [im for im in images if im is not None]
        result = self.client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[prompt, *images],
        )
        result = result.text.replace("\t", " ").replace("\r", " ").replace("\n", " ")
        return result
