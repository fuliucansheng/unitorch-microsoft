# Copyright (c) MICROSOFT.
# Licensed under the MIT License.
import os
import io
import cv2
import gc
import base64
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
from unitorch_microsoft.chatgpt.papyrus import get_gpt_image_response
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
    create_tabs,
    create_tab,
)


class GPTCreateImgWebUI(SimpleWebUI):
    _title = "GPT Image Generation"
    _description = "This is a demo for GPT image generation. You can input images and a prompt, and GPT will generate an image based on the prompt and the input images."

    def __init__(self, config: CoreConfigureParser):
        self._status = getattr(self, "_status", "Stopped")

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
        image_brush = create_element("image_editor", "Input Image")
        mask_brush = create_element("image", "Input Image Mask")
        input_prompt = create_element("text", "Input Prompt", lines=3)
        image_size = create_element(
            "radio",
            "Image Size",
            values=["1024x1024", "1536x1024", "1024x1536"],
            default="1024x1024",
        )
        generate = create_element("button", "Generate")
        output_image = create_element("image", "Output Image")

        images = create_tabs(
            *[create_tab(ele, name=f"Image {i}") for i, ele in enumerate(input_images)],
            create_tab(create_row(image_brush, mask_brush), name="Edit Image"),
        )
        left = create_column(images, input_prompt, image_size, generate)
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
        image_brush.change(
            fn=self.composite_images, inputs=[image_brush], outputs=[mask_brush]
        )

        generate.click(
            fn=self.serve,
            inputs=[input_prompt, image_size, image_brush, mask_brush, *input_images],
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
        self._status = "Stopped" if self._pipe is None else "Running"
        return self._status

    def composite_images(self, images):
        if images is None:
            return None
        layers = images["layers"]
        if len(layers) == 0:
            return None
        image = layers[0]
        for i in range(1, len(layers)):
            image = Image.alpha_composite(image, layers[i])
        image = image.convert("L")
        image = image.point(lambda p: p < 5 and 255)
        image = ImageOps.invert(image)
        return image

    def serve(self, prompt, size, image_brush, mask_brush, *images):
        if image_brush is None or mask_brush is None:
            result = get_gpt_image_response(
                prompt,
                size=size,
                images=images,
                mask=None,
            )
        else:
            image_brush = image_brush["background"]
            rgba = image_brush.convert("RGBA")
            r, g, b, _ = rgba.split()
            alpha = mask_brush.convert("L").point(lambda p: 0 if p > 127 else 255)
            mask = Image.merge("RGBA", (r, g, b, alpha))
            result = get_gpt_image_response(
                prompt,
                size=size,
                images=[image_brush] + list(images),
                mask=mask,
            )

        return result
