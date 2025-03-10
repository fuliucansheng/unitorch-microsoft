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


class CaptionImgWebUI(SimpleWebUI):
    def __init__(self, config: CoreConfigureParser):
        self._status = getattr(self, "_status", "Stopped")
        # create elements
        toper_menus = create_toper_menus()
        footer = create_footer()
        header = create_element(
            "markdown",
            label=f"# <div style='margin-top:10px'>🛠️ Image Caption</div>",
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
        input_prompt = create_element("text", "Input Prompt", lines=3)
        generate = create_element("button", "Generate")
        output_caption = create_element("text", "Output Caption", lines=5)

        left = create_column(input_image, input_prompt, generate)
        right = create_column(output_caption)

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
        iface._title = "Llava Image Caption"
        iface._description = "This is a demo for image caption."

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
            inputs=[input_prompt, input_image],
            outputs=[output_caption],
            trigger_mode="once",
        )

        iface.load(
            fn=lambda: self._status,
            outputs=status,
        )

        iface.__exit__()

        super().__init__(config, iname="Image Caption", iface=iface)

    def start(self):
        self._pipe = LlavaMistralClipForGenerationPipeline.from_core_configure(
            config=self._config,
            pretrained_name="llava-v1.6-mistral-7b-hf",
            quant_config_path="configs/quantization/8bit.json",
        )
        self._status = "Running"
        return self._status

    def stop(self):
        self._pipe.to("cpu")
        del self._pipe
        gc.collect()
        torch.cuda.empty_cache()
        self._pipe = None
        self._status = "Stopped" if self._pipe is None else "Running"
        return self._status

    def serve(self, prompt, image):
        new_prompt = f"[INST] <image>\n {prompt} [/INST]"
        caption = self._pipe(
            new_prompt,
            image,
            lora_checkpoints=[],
            lora_weights=[],
            lora_alphas=[],
            lora_urls=[],
            lora_files=[],
        )
        return caption
