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
from PIL import Image, ImageDraw
from torchvision import transforms
from unitorch import mktempfile
from unitorch.utils import read_file
from unitorch.models import GenericOutputs
from unitorch.cli import CoreConfigureParser
from unitorch.cli.webuis import SimpleWebUI
from unitorch_microsoft import cached_path
from unitorch_microsoft.chatgpt.papyrus import get_gpt4_response
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


class GPT4WebUI(SimpleWebUI):
    _title = "GPT-4"
    _description = "This is a demo for GPT-4. You can input images and a prompt (images are optional), and GPT-4 will generate a result."

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
            fn=self.serve,
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
        return self._status

    def stop(self):
        self._status = "Stopped" if self._pipe is None else "Running"
        return self._status

    def serve(self, prompt, *images):
        result = get_gpt4_response(
            prompt,
            images=images,
        )
        result = result.replace("\t", " ").replace("\r", " ").replace("\n", " ")
        return result
