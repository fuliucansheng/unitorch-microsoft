# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import cv2
import gc
import re
import json
import torch
import requests
import numpy as np
import gradio as gr
from PIL import Image, ImageDraw
from torchvision import transforms
from unitorch import mktempfile
from unitorch.utils import read_file
from unitorch.models import GenericOutputs
from unitorch.cli import CoreConfigureParser
from unitorch.cli.webuis import SimpleWebUI
from unitorch.cli.pipelines.tools.controlnet.canny import canny
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
    call_fastapi,
)


class GenerateObjectWebUI(SimpleWebUI):
    _title = "Generate Object"
    _description = "This is a demo for generating objects using FLUX. You can input an image and a new object description, and the model will generate new images with the specified object."

    def __init__(self, config: CoreConfigureParser):
        self._status = getattr(self, "_status", "Stopped")

        config.set_default_section("microsoft/spaces/picasso/obj_generate")
        self._bria_endpoint = config.getoption("bria_endpoint", None)
        self._joycaption2_endpoint = config.getoption("joycaption2_endpoint", None)
        self._flux_t2i_endpoint = config.getoption("flux_t2i_endpoint", None)
        self._flux_ctrl_endpoint = config.getoption("flux_ctrl_endpoint", None)

        # create elements
        toper_menus = create_toper_menus()
        footer = create_footer()
        header = create_element(
            "markdown",
            label=f"# <div style='margin-top:10px'>🎢 {self._title} </div>",
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
        prompt = create_element("text", "New Object", lines=3)
        generate = create_element("button", "Generate")
        output_image1 = create_element("image", "Output Image 1", lines=5)
        output_image2 = create_element("image", "Output Image 2", lines=5)
        output_image3 = create_element("image", "Output Image 3", lines=5)

        left = create_column(input_image, prompt, generate)
        right = create_column(create_row(output_image1, output_image2, output_image3))

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
            outputs=[output_image1, output_image2, output_image3],
            trigger_mode="once",
        )

        iface.load(
            fn=lambda: self._status,
            outputs=status,
        )

        iface.__exit__()

        super().__init__(config, iname=self._title, iface=iface)

    def start(self):
        checked_endpoints = [
            self._bria_endpoint,
            self._joycaption2_endpoint,
            self._flux_t2i_endpoint,
            self._flux_ctrl_endpoint,
        ]
        for endpoint in checked_endpoints:
            __status = requests.get(endpoint + "/status").json()
            if __status == "running":
                continue
            resp = requests.get(endpoint + "/start", timeout=1200)
            resp.raise_for_status()
        self._status = "Running"
        return self._status

    def stop(self):
        checked_endpoints = [
            self._bria_endpoint,
            self._joycaption2_endpoint,
            self._flux_t2i_endpoint,
            self._flux_ctrl_endpoint,
        ]
        for endpoint in checked_endpoints:
            __status = requests.get(endpoint + "/status").json()
            if __status == "stopped":
                continue
            resp = requests.get(endpoint + "/stop", timeout=1200)
            resp.raise_for_status()
        self._status = "Stopped"
        return self._status

    def generate(self, image, prompt):
        caption = call_fastapi(
            self._joycaption2_endpoint + "/generate",
            images={"image": image},
            params={
                "text": "Write a descriptive caption for this image in a formal tone."
            },
        )

        new_image1 = call_fastapi(
            self._flux_t2i_endpoint + "/generate1",
            params={
                "text": caption,
                "height": 1024,
                "width": 1024,
                "guidance_scale": 3.5,
                "num_timesteps": 50,
                "strength": 1.0,
                "prompt_embeds_scale": 1.0,
                "pooled_prompt_embeds_scale": 1.0,
                "seed": 42,
            },
            resp_type="image",
        )

        new_caption = get_gpt4_response(
            f"Optimize the prompt by refining its structure and updating the described object to the new object. \n Original Prompt: {caption} \n New Object: {prompt}. Output the optimized prompt in <Ans> </Ans> format.",
            images=[],
        )

        match = re.search(r"<Ans>(.*?)</Ans>", new_caption, re.DOTALL)
        new_caption = match.group(1).strip()

        new_image2 = call_fastapi(
            self._flux_t2i_endpoint + "/generate1",
            params={
                "text": new_caption,
                "height": 1024,
                "width": 1024,
                "guidance_scale": 3.5,
                "num_timesteps": 50,
                "strength": 1.0,
                "prompt_embeds_scale": 1.0,
                "pooled_prompt_embeds_scale": 1.0,
                "seed": 42,
            },
            resp_type="image",
        )

        mask = call_fastapi(
            self._bria_endpoint + "/generate",
            images={"image": new_image2},
            params={"threshold": 0.5},
            resp_type="image",
        )

        result = Image.new("RGBA", new_image2.size, (0, 0, 0, 64))
        mask = mask.convert("L").resize(new_image2.size)
        result.paste(new_image2, (0, 0), mask)

        canny_image = canny(result)
        new_image3 = call_fastapi(
            self._flux_ctrl_endpoint + "/generate2",
            images={"redux_image": image, "image": canny_image},
            params={
                "text": "",
                "guidance_scale": 30,
                "strength": 1.0,
                "num_timesteps": 50,
                "prompt_embeds_scale": 0.5,
                "pooled_prompt_embeds_scale": 1.0,
                "seed": 42,
            },
            resp_type="image",
        )
        return new_image1, new_image2, new_image3
