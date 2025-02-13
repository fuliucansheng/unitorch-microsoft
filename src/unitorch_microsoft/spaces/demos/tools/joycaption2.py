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
from unitorch.cli.pipelines.llava import LlavaLlamaSiglipForGenerationPipeline
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


class JoyCaption2WebUI(SimpleWebUI):
    def __init__(self, config: CoreConfigureParser):
        self._status = getattr(self, "_status", "Stopped")
        # create elements
        toper_menus = create_toper_menus()
        footer = create_footer()
        header = create_element(
            "markdown",
            label=f"# <div style='margin-top:10px'>✨ JoyCaption 2</div>",
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
        examples = gr.Examples(
            examples=[
                "Write a descriptive caption for this image in a formal tone.",
                "Write a descriptive caption for this image in a formal tone within {word_count} words.",
                "Write a {length} descriptive caption for this image in a formal tone.",
                "Write a descriptive caption for this image in a casual tone.",
                "Write a descriptive caption for this image in a casual tone within {word_count} words.",
                "Write a {length} descriptive caption for this image in a casual tone.",
                "Write a stable diffusion prompt for this image.",
                "Write a stable diffusion prompt for this image within {word_count} words.",
                "Write a {length} stable diffusion prompt for this image.",
                "Write a MidJourney prompt for this image.",
                "Write a MidJourney prompt for this image within {word_count} words.",
                "Write a {length} MidJourney prompt for this image.",
                "Write a list of Booru tags for this image.",
                "Write a list of Booru tags for this image within {word_count} words.",
                "Write a {length} list of Booru tags for this image.",
                "Write a list of Booru-like tags for this image.",
                "Write a list of Booru-like tags for this image within {word_count} words.",
                "Write a {length} list of Booru-like tags for this image.",
                "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc.",
                "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it within {word_count} words.",
                "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it {length}.",
                "Write a caption for this image as though it were a product listing.",
                "Write a caption for this image as though it were a product listing. Keep it under {word_count} words.",
                "Write a {length} caption for this image as though it were a product listing.",
                "Write a caption for this image as if it were being used for a social media post.",
                "Write a caption for this image as if it were being used for a social media post. Limit the caption to {word_count} words.",
                "Write a {length} caption for this image as if it were being used for a social media post.",
            ],
            label="Prompt Examples",
            inputs=[input_prompt],
            examples_per_page=10,
        )
        generate = create_element("button", "Generate")
        output_caption = create_element("text", "Output Caption", lines=5)

        left = create_column(input_image, input_prompt, generate, examples.dataset)
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
        iface._title = "JoyCaption 2"
        iface._description = "This is a demo for JoyCaption 2."

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

        examples.create()

        iface.load(
            fn=lambda: self._status,
            outputs=status,
        )

        iface.__exit__()

        super().__init__(config, iname="JoyCaption 2", iface=iface)

    def start(self):
        self._pipe = LlavaLlamaSiglipForGenerationPipeline.from_core_configure(
            config=self._config,
            pretrained_name="llava-v1.6-joycaption-2",
            quant_config_path="configs/quantization/8bit.json",
        )
        self._status = "Running"
        return self._status

    def stop(self):
        self._pipe.to("cpu")
        del self._pipe
        gc.collect()
        torch.cuda.empty_cache()
        self._pipe = None if not hasattr(self, "_pipe") else self._pipe
        self._status = "Stopped" if self._pipe is None else "Running"
        return self._status

    def serve(self, prompt, image):
        new_prompt = f"<|start_header_id|>system<|end_header_id|>\\n\\nCutting Knowledge Date: December 2023\\nToday Date: 26 July 2024\\n\\nYou are a helpful image captioner.<|eot_id|><|start_header_id|>user<|end_header_id|>\\n\\n<|reserved_special_token_70|><|reserved_special_token_69|><|reserved_special_token_71|>{prompt}|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n"
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
