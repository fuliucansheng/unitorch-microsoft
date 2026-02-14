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
from openai import OpenAI
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


def pil_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return "data:image/png;base64," + img_str


def base64_to_pil(img_str):
    img_data = base64.b64decode(img_str)
    image = Image.open(io.BytesIO(img_data))
    return image


def get_seedream_response(
    prompt, images, size="1440x2560", model="doubao-seedream-4-5-251128"
):
    token = "5f1c7685-1bd8-4a71-a1e9-0591004685b4"

    client = OpenAI(
        # 此为默认路径，您可根据业务所在地域进行配置
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        # 从环境变量中获取您的 API Key。此为默认方式，您可根据需要进行修改
        api_key=token,
    )
    response = client.images.generate(
        model=model,
        prompt=prompt,
        extra_body={
            "image": [pil_to_base64(im) for im in images],
            "sequential_image_generation": "disabled",
            "watermark": False,
        },
        size=size,
        response_format="b64_json",
    )
    if len(response.data) == 0:
        raise ValueError(f"Error in Seedream API: {response}")
    image = base64_to_pil(response.data[0].b64_json)
    return image


class SeedreamWebUI(SimpleWebUI):
    _title = "Seedream Image Generation"
    _description = "This is a demo for Seedream, a model for generating images from text prompts and input images. You can input multiple images and a prompt, and the model will generate an image based on the prompt and the input images."

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
        model = create_element(
            "radio",
            "Model",
            values=["doubao-seedream-4-5-251128"],
            default="doubao-seedream-4-5-251128",
        )
        size = create_element(
            "radio",
            "Size",
            values=[
                "2048x2048",
                "2304x1728",
                "1728x2304",
                "2560x1440",
                "1440x2560",
                "2496x1664",
                "1664x2496",
                "3024x1296",
            ],
            default="1440x2560",
        )
        height = create_element(
            "slider", "Image Height", min_value=1, max_value=4096, step=1, default=2560
        )
        width = create_element(
            "slider", "Image Width", min_value=1, max_value=4096, step=1, default=1440
        )
        notes = create_element(
            "markdown",
            label="**Note:** Seedream model currently only supports width x height in the range of (3686400, 16777216) pixels.",
            interactive=False,
        )
        generate = create_element("button", "Generate", variant="primary")

        # layout
        left = create_column(
            images,
            prompt,
            create_row(
                create_column(model, size, scale=2), create_column(width, height)
            ),
            notes,
            generate,
        )
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
        size.change(
            lambda x: (int(x.split("x")[0]), int(x.split("x")[1])),
            inputs=[size],
            outputs=[width, height],
        )
        generate.click(
            self.generate,
            [prompt, model, width, height, image_brush, *input_images],
            [output_image],
        )
        iface.__exit__()
        super().__init__(config, iname=self._title, iface=iface)

    def start(self):
        self._status = "Running"
        return self._status

    def stop(self):
        self._status = "Stopped"
        return self._status

    def generate(self, prompt, model, width, height, image_brush, *images):
        if self._status != "Running":
            self.start()
        _images = [im for im in images if im is not None]
        if image_brush is not None and image_brush.get("composite") is not None:
            composite = image_brush["composite"]
            _images = _images + [composite]

        size = f"{width}x{height}"

        image = get_seedream_response(
            prompt,
            _images,
            size=size,
            model=model,
        )
        return image
