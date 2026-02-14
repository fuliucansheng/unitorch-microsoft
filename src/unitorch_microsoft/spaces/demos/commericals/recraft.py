# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
from unittest import result
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
from unitorch_microsoft.externals.recraft import (
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


class RecraftCreateImgWebUI(SimpleWebUI):
    _title = "Recraft Create Image"
    _description = "This is a demo for creating images using Recraft. You can input a prompt, and the model will generate an image based on the prompt."

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
        return result


class RecraftChangeBGWebUI(SimpleWebUI):
    _title = "Recraft Change Background"
    _description = "This is a demo for changing background of images using Recraft. You can input an image and a prompt, and the model will generate a new image with the specified background."

    def __init__(self, config: CoreConfigureParser):
        self._status = getattr(self, "_status", "Stopped")
        # create elements
        toper_menus = create_toper_menus()
        footer = create_footer()
        header = create_element(
            "markdown",
            label=f"# <div style='margin-top:10px'>✨ {self._title}</div>",
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
        prompt = create_element(
            "text", "Input Prompt", lines=3, default="simple background"
        )
        generate = create_element("button", "Generate")
        output_image = create_element("image", "Output Image")

        left = create_column(input_image, prompt, generate)
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
            inputs=[input_image, prompt],
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

    def generate(self, image, prompt):
        result = get_recraft_change_background_image(
            prompt,
            image=image,
        )
        return result


class RecraftExpandBGWebUI(SimpleWebUI):
    _title = "Recraft Expand Background"
    _description = "This is a demo for expanding the background of images using Recraft. You can input an image and a prompt, and the model will generate a new image with the specified background."

    def __init__(self, config: CoreConfigureParser):
        self._status = getattr(self, "_status", "Stopped")
        # create elements
        toper_menus = create_toper_menus()
        footer = create_footer()
        header = create_element(
            "markdown",
            label=f"# <div style='margin-top:10px'>✨ {self._title}</div>",
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
        prompt = create_element(
            "text", "Input Prompt", lines=3, default="simple background"
        )
        width = create_element(
            "slider", "Width", default=512, min_value=1, max_value=2048, step=1
        )
        height = create_element(
            "slider", "Height", default=512, min_value=1, max_value=2048, step=1
        )
        generate = create_element("button", "Generate")
        output_image = create_element("image", "Output Image")

        left = create_column(input_image, prompt, width, height, generate)
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
            inputs=[input_image, prompt, width, height],
            outputs=[output_image],
            trigger_mode="once",
        )

        input_image.upload(
            lambda x: x.size,
            inputs=[input_image],
            outputs=[width, height],
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

    def generate(self, image, prompt, width, height):
        mask = Image.new("L", (width, height), 255)
        im_width, im_height = image.size
        assert width >= im_width and height >= im_height
        black = Image.new("RGB", (im_width, im_height), (0, 0, 0))
        mask.paste(black, ((width - im_width) // 2, (height - im_height) // 2))
        new_image = Image.new("RGB", (width, height), (255, 255, 255))
        new_image.paste(image, ((width - im_width) // 2, (height - im_height) // 2))
        result = get_recraft_inpainting_image(
            prompt,
            image=new_image,
            mask=mask,
        )
        return result


class RecraftRemoveBGWebUI(SimpleWebUI):
    _title = "Recraft Remove Background"
    _description = "This is a demo for remove the background of images using Recraft. You can input an image, and the model will generate a new image with the background removed."

    def __init__(self, config: CoreConfigureParser):
        self._status = getattr(self, "_status", "Stopped")
        # create elements
        toper_menus = create_toper_menus()
        footer = create_footer()
        header = create_element(
            "markdown",
            label=f"# <div style='margin-top:10px'>✨ {self._title}</div>",
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
        generate = create_element("button", "Generate")
        output_image = create_element("image", "Output Image")

        left = create_column(input_image, generate)
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
            inputs=[input_image],
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

    def generate(self, image):
        result = get_recraft_remove_background_image(image=image)
        return result
