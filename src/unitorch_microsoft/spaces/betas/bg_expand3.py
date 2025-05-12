# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import cv2
import gc
import json
import requests
import torch
import numpy as np
import gradio as gr
from PIL import Image, ImageDraw, ImageOps, ImageFilter, ImageEnhance
from transformers import AutoModelForImageSegmentation
from torchvision import transforms
from unitorch import mktempfile
from unitorch.utils import read_file
from unitorch.models import GenericOutputs
from unitorch.cli import CoreConfigureParser
from unitorch.cli.webuis import SimpleWebUI
from unitorch_microsoft import cached_path
from unitorch_microsoft.chatgpt.azure import get_gpt4o_respone
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


class ExpandBG3WebUI(SimpleWebUI):
    def __init__(self, config: CoreConfigureParser):
        self._status = getattr(self, "_status", "Stopped")

        config.set_default_section("microsoft/spaces/betas/expand_bg")
        self._flux_endpoint = config.getoption("flux_endpoint", None)

        # create elements
        toper_menus = create_toper_menus()
        footer = create_footer()
        header = create_element(
            "markdown",
            label=f"# <div style='margin-top:10px'>✨ Expand Background 3</div>",
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
        input_image = create_element("image", "Image")
        input_ratio = create_element(
            "slider", "Ratio", default=1.9, min_value=0.1, max_value=10.0, step=0.01
        )
        input_prompt = create_element("text", "Prompt")
        generate = create_element("button", "Generate")
        output_image1 = create_element("image", "Output Image 1", lines=5)
        output_image2 = create_element("image", "Output Image 2", lines=5)
        debug_image1 = create_element("image", "Debug Image 1", lines=5)
        debug_image2 = create_element("image", "Debug Image 2", lines=5)
        debug_image3 = create_element("image", "Debug Image 3", lines=5)

        left = create_column(input_image, input_ratio, input_prompt, generate)
        right = create_column(
            create_row(output_image1, output_image2),
            create_row(debug_image1, debug_image2, debug_image3),
        )

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
        iface._title = "Expand Background 3"
        iface._description = "This is a demo for betas expand background 3."

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
            inputs=[input_image, input_ratio, input_prompt],
            outputs=[output_image1, output_image2],
            trigger_mode="once",
        )

        iface.load(
            fn=lambda: self._status,
            outputs=status,
        )

        iface.__exit__()

        super().__init__(config, iname="Expand Background 3", iface=iface)

    def start(self):
        if self._status == "Running":
            return self._status

        requests.post(
            self._flux_endpoint + "/start",
            timeout=1200,
            params={
                "pretrained_name": "stable-flux-dev-fill",
            },
            data=json.dumps(
                {
                    "pretrained_lora_names": "stable-flux-lora-ms-dev-fill-simple",
                    "pretrained_lora_weights": 0.2,
                    "pretrained_lora_alphas": 32,
                }
            ),
            headers={"Content-type": "application/json"},
        )
        self._status = "Running"
        return self._status

    def stop(self):
        if self._status == "Stopped":
            return self._status
        requests.get(self._flux_endpoint + "/stop", timeout=1200).raise_for_status()
        self._status = "Stopped"
        return self._status

    def process(self, image, ratio, max_size=1024):
        width, height = image.size

        longest_side = max_size
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

        new_width = int(width * scale)
        new_height = int(height * scale)

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

    def process2(self, image, mask, ratio=1.9):
        width, height = image.size
        longest_side = 2048
        shortest_side = (
            int(longest_side * ratio) if ratio < 1 else int(longest_side / ratio)
        )
        size = (
            (longest_side, shortest_side)
            if width > height
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
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = image.resize(
            (new_width // 8 * 8, new_height // 8 * 8), resample=Image.LANCZOS
        )
        mask = mask.resize(
            (new_width // 8 * 8, new_height // 8 * 8), resample=Image.LANCZOS
        )
        return image, mask

    def serve(self, image, ratio, prompt):
        caption = prompt

        new_image, new_mask = self.process(image, ratio)
        debug_image1 = new_image.copy()
        debug_image2 = new_mask.copy()
        image_np = np.array(new_image.convert("RGB"))
        mask_np = np.array(new_mask.convert("L")).astype(np.uint8)

        _, binary_mask = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
        inpainted_image = cv2.inpaint(image_np, binary_mask, 10, cv2.INPAINT_TELEA)
        latent_image = Image.fromarray(inpainted_image)
        result = call_fastapi(
            self._flux_endpoint + "/generate",
            images={
                "image": latent_image,
                "mask_image": new_mask,
            },
            params={
                "text": caption,
                "guidance_scale": 30.0,
                "strength": 0.9,
                "num_timesteps": 50,
                "seed": 42,
            },
            resp_type="image",
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

        result1 = result
        new_image, new_mask = self.process(image, ratio, max_size=2048)

        new_image = new_image.resize(result.size, resample=Image.LANCZOS).convert(
            "RGBA"
        )
        new_mask = new_mask.resize(result.size, resample=Image.LANCZOS)
        mask_image = ImageOps.invert(new_mask).convert("L")
        result = result.convert("RGBA")
        new_image = Image.composite(new_image, result, mask_image)
        mask = mask_image.filter(ImageFilter.GaussianBlur(100))
        mask = ImageEnhance.Contrast(mask).enhance(0.8)
        result.paste(new_image, (0, 0), mask)
        result = result.convert("RGB")

        new_image, new_mask = self.process2(result, new_mask)
        result = call_fastapi(
            self._flux_endpoint + "/generate",
            images={
                "image": new_image,
                "mask_image": new_mask,
            },
            params={
                "text": caption,
                "guidance_scale": 30.0,
                "strength": 0.8,
                "num_timesteps": 50,
                "seed": 1123,
            },
            resp_type="image",
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
        return result1, result, debug_image1, debug_image2, None
