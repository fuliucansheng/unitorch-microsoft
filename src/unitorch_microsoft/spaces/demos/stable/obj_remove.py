# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import cv2
import gc
import torch
import numpy as np
import gradio as gr
from PIL import Image, ImageDraw, ImageOps
from transformers import AutoModelForImageSegmentation
from diffusers import (
    StableDiffusionInpaintPipeline,
    StableDiffusionControlNetInpaintPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)
from torchvision import transforms
from unitorch import mktempfile
from unitorch.utils import read_file
from unitorch.models import GenericOutputs
from unitorch.cli import CoreConfigureParser
from unitorch.cli.pipelines.sam import SamForSegmentationPipeline
from unitorch.cli.fastapis.controlnet import ControlNetForImageInpaintingFastAPIPipeline
from unitorch.cli.pipelines.tools import depth, canny
from unitorch.cli.webuis import SimpleWebUI
from unitorch_microsoft import cached_path
import unitorch_microsoft.models.sam
from unitorch_microsoft.spaces import (
    create_element,
    create_row,
    create_column,
    create_flex_layout,
    create_tab,
    create_tabs,
    create_blocks,
    create_toper_menus,
    create_footer,
    create_dashboard_card,
    create_card,
    create_dashboard_cards_group,
    create_cards_group,
)


class RemoveObjWebUI(SimpleWebUI):
    def __init__(self, config: CoreConfigureParser):
        self._status = getattr(self, "_status", "Stopped")
        # create elements
        toper_menus = create_toper_menus()
        footer = create_footer()
        header = create_element(
            "markdown",
            label=f"# <div style='margin-top:10px'>✨ Remove Object</div>",
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
        image_click = create_element("image", "Input Image")
        mask_click = create_element("image", "Input Image Mask")
        prompt_click = create_element(
            "text", "Input Prompt", lines=3, default="minimalist background"
        )
        mask_threshold = create_element(
            "slider",
            "Mask Threshold",
            default=0,
            min_value=-20,
            max_value=20,
            step=0.1,
            scale=4,
        )
        reset = create_element("button", "Reset", variant="stop")
        segment = create_element("button", "Segment")
        generate_click = create_element("button", "Generate")

        image_brush = create_element("image_editor", "Input Image")
        mask_brush = create_element("image", "Input Image Mask")
        prompt_brush = create_element(
            "text",
            "Input Prompt",
            lines=3,
            default="minimalist background, simple background",
        )
        generate_brush = create_element("button", "Generate")
        output_image = create_element("image", "Output Image")

        click_tab = create_tab(
            create_row(image_click, mask_click),
            create_row(mask_threshold, reset, segment),
            prompt_click,
            generate_click,
            name="Click",
        )
        brush_tab = create_tab(
            create_row(image_brush, mask_brush),
            prompt_brush,
            generate_brush,
            name="Brush",
        )

        left = create_tabs(click_tab, brush_tab)
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
        iface._title = "Remove Object"
        iface._description = "This is a demo for removing object."

        # create events
        iface.__enter__()

        origin_image_click = gr.State(None)
        points = gr.State([])

        image_click.upload(
            lambda image: (image.copy() if image is not None else None, []),
            inputs=[image_click],
            outputs=[origin_image_click, points],
        )
        image_click.select(
            self.add_click_points,
            [origin_image_click, points],
            [image_click, points],
        )
        reset.click(
            lambda x: (x, []),
            inputs=[origin_image_click],
            outputs=[image_click, points],
            trigger_mode="once",
        )

        segment.click(
            self.segment,
            inputs=[
                origin_image_click,
                points,
                mask_threshold,
            ],
            outputs=[mask_click],
            trigger_mode="once",
        )

        image_brush.change(
            fn=self.composite_images, inputs=[image_brush], outputs=[mask_brush]
        )

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

        generate_click.click(
            fn=self.serve_click,
            inputs=[prompt_click, origin_image_click, mask_click],
            outputs=[output_image],
            trigger_mode="once",
        )

        generate_brush.click(
            self.serve_brush,
            inputs=[prompt_brush, image_brush, mask_brush],
            outputs=[output_image],
            trigger_mode="once",
        )

        iface.load(
            fn=lambda: self._status,
            outputs=status,
        )

        iface.__exit__()

        super().__init__(config, iname="Remove Object", iface=iface)

    def start(self):
        self._pipe1 = SamForSegmentationPipeline.from_core_configure(
            config=self._config,
            pretrained_name="sam-vit-large",
        )
        self._pipe2 = ControlNetForImageInpaintingFastAPIPipeline.from_core_configure(
            config=self._config,
            pretrained_name="stable-v1.5-realistic-v5.1-inpainting",
            pretrained_controlnet_names=[],
            pretrained_inpainting_controlnet_name="stable-v1.5-controlnet-inpainting",
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

    def add_click_points(self, image, click_points, evt: gr.SelectData):
        x, y = evt.index[0], evt.index[1]
        click_points = click_points + [(x, y)]
        new_image = image.copy()
        draw = ImageDraw.Draw(new_image)
        point_color = (255, 0, 0)
        radius = 3
        for point in click_points:
            x, y = point
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius), fill=point_color
            )

        return new_image, click_points

    def segment(self, image, points, mask_threshold):
        mask = self._pipe1(
            image,
            points=points,
            mask_threshold=mask_threshold,
            lora_checkpoints=["sam-lora-dis5k"],
            lora_weights=[0.5],
            lora_alphas=[32.0],
            lora_urls=[None],
            lora_files=[None],
        )
        return mask

    def composite_images(self, images):
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

    def serve_click(self, prompt, image, mask):
        return self.serve(prompt, image, mask)

    def serve_brush(self, prompt, image, mask):
        return self.serve(prompt, image["background"], mask)

    def serve(self, prompt, image, mask):
        image = image.convert("RGB")
        white = Image.new("RGB", (image.width, image.height), (255, 255, 255))
        image.paste(white, mask=mask.convert("L"))
        # pos_prompt = f"cinematic photo of {prompt}, realistic, extremely detailed, photorealistic, best quality"
        pos_prompt = (
            f"{prompt}, realistic, extremely detailed, photorealistic, best quality"
        )
        neg_prompt = "nsfw, paintings, sketches, (worst quality:2), (low quality:2) lowers, normal quality, ((monochrome)), ((grayscale)), logo, word, character, nudity, naked, disfigured, nude, blurry, blurry background"
        result = self._pipe2(
            pos_prompt,
            image,
            mask,
            neg_text=neg_prompt,
            width=image.width,
            height=image.height,
            guidance_scale=7.5,
            strength=1.0,
            num_timesteps=25,
            seed=42,
            controlnet_images=[],
            controlnet_guidance_scales=[],
            inpaint_controlnet_image=image,
            inpaint_controlnet_guidance_scale=0.8,
        )
        # new_image = image.resize(result.size, resample=Image.LANCZOS).convert("RGB")
        # result = result.convert("RGB")
        # mask = mask.resize(result.size, resample=Image.LANCZOS)
        # mask = np.array(mask.convert("1"))[:, :, None]
        # result = Image.fromarray(
        #     (np.array(result) * mask + np.array(new_image) * (1 - mask)).astype(
        #         np.uint8
        #     )
        # )
        return result
