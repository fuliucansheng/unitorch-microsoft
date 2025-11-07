# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import cv2
import math
import gc
import json
import requests
import torch
import numpy as np
import gradio as gr
from PIL import Image, ImageDraw

from torchvision import transforms
from unitorch import mktempfile
from unitorch.utils import read_file
from unitorch.models import GenericOutputs
from unitorch.cli import CoreConfigureParser
from unitorch.cli.webuis import SimpleWebUI
from unitorch_microsoft import cached_path
from unitorch_microsoft.externals.papyrus import get_gpt5_response
from unitorch_microsoft.externals.recraft import (
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
    call_fastapi,
)


class ExpandBGWebUI(SimpleWebUI):
    _title = "Expand Background"
    _description = "This is a demo for expanding the background of images using Recraft. You can input an image and a ratio, and the model will generate a new image with the specified background expanded."

    def __init__(self, config: CoreConfigureParser):
        self._status = getattr(self, "_status", "Stopped")

        # create elements
        toper_menus = create_toper_menus()
        footer = create_footer()
        header = create_element(
            "markdown",
            label=f"# <div style='margin-top:10px'>✨ {self._title} </div>",
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
        input_image = create_element("image", "Image")
        input_ratio = create_element(
            "slider", "Ratio", default=1.9, min_value=0.1, max_value=10.0, step=0.01
        )
        generate = create_element("button", "Generate")
        output_image = create_element("image", "Output Image", lines=5)

        left = create_column(input_image, input_ratio, generate)
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
            inputs=[input_image, input_ratio],
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

    def process(self, image, ratio):
        width, height = image.size

        longest_side = 2048
        shortest_side = (
            int(longest_side * ratio) if ratio < 1 else int(longest_side / ratio)
        )
        size = (
            (longest_side, shortest_side)
            if ratio > 1
            else (shortest_side, longest_side)
        )

        scale = min(size[0] / width, size[1] / height)
        if scale > 1:
            size = (int(size[0] // scale), int(size[1] // scale))
        if size[0] < 512:
            size = (512, int(size[1] * 512 / size[0]))
        if size[1] < 512:
            size = (int(size[0] * 512 / size[1]), 512)

        size = (size[0] // 8 * 8, size[1] // 8 * 8)

        scale = min(size[0] / width, size[1] / height)

        new_width = math.ceil(width * scale)
        new_height = math.ceil(height * scale)

        image = image.resize(
            (new_width // 8 * 8, new_height // 8 * 8), resample=Image.LANCZOS
        )

        im_width, im_height = image.size

        mask = Image.new("L", (size[0], size[1]), 255)
        black = Image.new("RGB", (im_width, im_height), (0, 0, 0))
        mask.paste(black, ((size[0] - im_width) // 2, (size[1] - im_height) // 2))
        new_image = Image.new("RGB", (size[0], size[1]), (255, 255, 255))
        new_image.paste(image, ((size[0] - im_width) // 2, (size[1] - im_height) // 2))

        return new_image, mask

    def generate(self, image, ratio):
        prompt = get_gpt5_response(
            "Describe the background of this image, maintaining its colors, textures, and lighting. Ensure seamless blending without adding new objects, text, or artifacts. The caption is in a single short paragraph. Don't mention any object in foreground.",
            images=[image],
        )
        pad_ratio = 0.4

        ratio2 = ratio
        _ratio = image.size[0] / image.size[1]
        if ratio > (1 + pad_ratio) * _ratio:
            ratio = (1 + pad_ratio) * _ratio
        if ratio < _ratio / (1 + pad_ratio):
            ratio = _ratio / (1 + pad_ratio)

        new_image, new_mask = self.process(image, ratio)
        result = get_recraft_inpainting_image(
            prompt,
            new_image,
            new_mask,
        )
        raw_width, raw_height = image.size
        if raw_width / raw_height > ratio:
            result = result.resize(
                (raw_width, int(raw_width / ratio)), resample=Image.LANCZOS
            )
        else:
            result = result.resize(
                (int(raw_height * ratio), raw_height), resample=Image.LANCZOS
            )

        if ratio == ratio2:
            return result

        ratio = ratio2
        new_image, new_mask = self.process(result, ratio)
        result = get_recraft_inpainting_image(
            prompt,
            new_image,
            new_mask,
        )
        raw_width, raw_height = image.size
        if raw_width / raw_height > ratio:
            result = result.resize(
                (raw_width, int(raw_width / ratio)), resample=Image.LANCZOS
            )
        else:
            result = result.resize(
                (int(raw_height * ratio), raw_height), resample=Image.LANCZOS
            )
        return result
