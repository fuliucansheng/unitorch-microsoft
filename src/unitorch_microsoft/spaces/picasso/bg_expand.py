# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import cv2
import gc
import torch
import numpy as np
import gradio as gr
from PIL import Image, ImageDraw
from transformers import AutoModelForImageSegmentation
from torchvision import transforms
from unitorch import mktempfile
from unitorch.utils import read_file
from unitorch.models import GenericOutputs
from unitorch.cli import CoreConfigureParser
from unitorch.cli.pipelines.llava import LlavaMistralClipForGenerationPipeline
from unitorch.cli.fastapis.stable_flux.inpainting import (
    StableFluxForImageInpaintingFastAPIPipeline,
)
from unitorch.cli.webuis import SimpleWebUI
from unitorch_microsoft import cached_path
import unitorch_microsoft.models.sam
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


class ExpandBGWebUI(SimpleWebUI):
    def __init__(self, config: CoreConfigureParser):
        self._status = getattr(self, "_status", "Stopped")
        # create elements
        toper_menus = create_toper_menus()
        footer = create_footer()
        header = create_element(
            "markdown",
            label=f"# <div style='margin-top:10px'>🛠️ Expand Background</div>",
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
            "slider", "Ratio", default=1.9, min_value=0.2, max_value=5.0, step=0.1
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
        iface._title = "Expand Background"
        iface._description = "This is a demo for picasso expand background."

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
            inputs=[input_image, input_ratio],
            outputs=[output_image],
            trigger_mode="once",
        )

        iface.load(
            fn=lambda: self._status,
            outputs=status,
        )

        iface.__exit__()

        super().__init__(config, iname="Expand Background", iface=iface)

    def start(self):
        self._pipe1 = LlavaMistralClipForGenerationPipeline.from_core_configure(
            config=self._config,
            pretrained_name="llava-v1.6-mistral-7b-hf",
            quant_config_path="configs/quantization/8bit.json",
        )
        self._pipe2 = StableFluxForImageInpaintingFastAPIPipeline.from_core_configure(
            config=self._config,
            pretrained_name="stable-flux-dev-fill",
        )
        self._status = "Running"
        return self._status

    def stop(self):
        self._pipe1.to("cpu")
        del self._pipe1
        self._pipe2.to("cpu")
        del self._pipe2
        gc.collect()
        torch.cuda.empty_cache()
        self._status = "Stopped"
        return self._status

    def process(self, image, ratio):
        width, height = image.size
        longest_side = 1024
        shortest_side = (
            int(longest_side * ratio) if ratio < 1 else int(longest_side / ratio)
        )
        size = (
            (longest_side, shortest_side)
            if ratio > 1
            else (shortest_side, longest_side)
        )

        scale = min(size[0] / width, size[1] / height)
        new_width = int(width * scale)
        new_height = int(height * scale)

        image = image.resize((new_width // 8 * 8, new_height // 8 * 8))

        im_width, im_height = image.size

        mask = Image.new("L", (size[0], size[1]), 255)
        black = Image.new("RGB", (im_width, im_height), (0, 0, 0))
        mask.paste(black, ((size[0] - im_width) // 2, (size[1] - im_height) // 2))
        new_image = Image.new("RGB", (size[0], size[1]), (255, 255, 255))
        new_image.paste(image, ((size[0] - im_width) // 2, (size[1] - im_height) // 2))

        return new_image, mask

    def serve(self, image, ratio):
        new_prompt = f"[INST] <image>\n Describe the image background. [/INST]"
        caption = self._pipe1(
            new_prompt,
            image,
            lora_checkpoints=[],
            lora_weights=[],
            lora_alphas=[],
            lora_urls=[],
            lora_files=[],
        )
        new_image, new_mask = self.process(image, ratio)
        image_np = np.array(new_image.convert("RGB"))
        mask_np = np.array(new_mask.convert("L")).astype(np.uint8)

        _, binary_mask = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
        inpainted_image = cv2.inpaint(image_np, binary_mask, 10, cv2.INPAINT_TELEA)
        init_image = Image.fromarray(inpainted_image)
        result = self._pipe2(
            caption,
            init_image,
            new_mask,
            "nsfw, paintings, sketches, (worst quality:2), (low quality:2) lowers, normal quality, ((monochrome)), ((grayscale)), logo, word, character, nudity, naked, disfigured, nude, blurry, blurry background",
            width=new_image.width // 8 * 8,
            height=new_image.height // 8 * 8,
            guidance_scale=30,
            strength=0.95,
            num_timesteps=50,
            seed=42,
        )
        return result
