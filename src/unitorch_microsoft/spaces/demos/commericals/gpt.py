# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import io
import cv2
import gc
import time
import base64
import torch
import logging
import hashlib
import numpy as np
import gradio as gr
from PIL import Image, ImageDraw, ImageOps
from openai import OpenAI
from unitorch import mktempfile
from unitorch.utils import read_file
from unitorch.models import GenericOutputs
from unitorch.cli import CoreConfigureParser
from unitorch.cli.webuis import SimpleWebUI
from unitorch_microsoft import cached_path
from unitorch_microsoft.externals.github_copilot import (
    get_response as get_gpt5_response,
)
from unitorch_microsoft.externals.papyrus import (
    get_image_response as get_gpt_image_response,
)
from unitorch_microsoft.externals.recraft import (
    get_inpainting_image as get_recraft_inpainting_image,
)
from unitorch_microsoft.spaces import (
    get_temp_folder,
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


class GPT5WebUI(SimpleWebUI):
    _title = "GPT-5"
    _description = "This is a demo for GPT-5. You can input images and a prompt (images are optional), and GPT-5 will generate a result."

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
            fn=self.generate,
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

    def generate(self, prompt, *images):
        result = get_gpt5_response(
            prompt,
            images=images,
        )
        result = result.replace("\t", " ").replace("\r", " ").replace("\n", " ")
        return result


class GPTImageWebUI(SimpleWebUI):
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
            fn=self.generate,
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

    def generate(self, prompt, size, image_brush, mask_brush, *images):
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


class SORA2CreateVideoWebUI(SimpleWebUI):
    _title = "SORA2 Video Generation"
    _description = "This is a demo for SORA2 video generation. You can input image and a prompt, and SORA2 will generate a video based on the prompt and the input image."

    def __init__(self, config: CoreConfigureParser):
        self._status = getattr(self, "_status", "Stopped")
        self.client = None

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
        input_image = create_element("image", "Image")
        input_prompt = create_element("text", "Input Prompt", lines=3)
        video_size = create_element(
            "radio",
            "Video Size",
            values=["720x1280", "1280x720"],
            default="720x1280",
        )
        duration = create_element("radio", "Duration", values=[4, 8, 12], default=4)
        option = create_element(
            "radio",
            "Option",
            values=["Center-Crop", "Out-Painting"],
            default="Center-Crop",
        )
        generate = create_element("button", "Generate")
        output_video = create_element("video", "Output Video")
        left = create_column(
            input_image, input_prompt, video_size, option, duration, generate
        )
        right = create_column(output_video)

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
            inputs=[input_prompt, input_image, video_size, duration, option],
            outputs=[output_video],
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
        resource_name = os.getenv("AZURE_OPENAI_RESOURCE_NAME")
        self.client = OpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            base_url=f"https://{resource_name}.openai.azure.com/openai/v1/",
        )
        return self._status

    def stop(self):
        self._status = "Stopped" if self._pipe is None else "Running"
        self.client = None
        return self._status

    def status(self):
        if getattr(self, "client", None) is None:
            self._status = "Stopped"
        else:
            self._status = "Running"
        return self._status

    def prepare_image(self, image, size, option):
        target_w, target_h = size.split("x")
        target_w = int(target_w)
        target_h = int(target_h)
        orig_w, orig_h = image.size
        scale1 = max(target_w / orig_w, target_h / orig_h)
        scale2 = min(target_w / orig_w, target_h / orig_h)
        if option == "Center-Crop":
            new_w = int(orig_w * scale1)
            new_h = int(orig_h * scale1)
            image = image.resize((new_w, new_h), Image.LANCZOS)
            left = (new_w - target_w) // 2
            top = (new_h - target_h) // 2
            right = left + target_w
            bottom = top + target_h
            new_image = image.crop((left, top, right, bottom))
            return new_image
        elif option == "Out-Painting":
            new_w = int(orig_w * scale2)
            new_h = int(orig_h * scale2)
            image = image.resize((new_w, new_h), Image.LANCZOS)
            black = Image.new("RGB", (target_w, target_h), (0, 0, 0))
            mask = Image.new("L", (target_w, target_h), 255)
            mask.paste(black, ((target_w - new_w) // 2, (target_h - new_h) // 2))
            new_image = Image.new("RGB", (target_w, target_h), (255, 255, 255))
            new_image.paste(image, ((target_w - new_w) // 2, (target_h - new_h) // 2))
            prompt = get_gpt5_response(
                "Describe the background of this image, maintaining its colors, textures, and lighting. Ensure seamless blending without adding new objects, text, or artifacts. The caption is in a single short paragraph. Don't mention any object in foreground.",
                images=[image],
            )
            new_image = get_recraft_inpainting_image(
                prompt,
                image=new_image,
                mask=mask,
            )
            return new_image
        return None

    def generate(self, prompt, image, size, duration, option):
        if self._status != "Running":
            self.start()

        def get_image(im):
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            buf.seek(0)
            return buf

        duration = str(duration)
        if image is not None:
            image = self.prepare_image(image, size, option)
            response = self.client.videos.create(
                prompt=prompt,
                model="sora-2",
                size=size,
                seconds=duration,
                input_reference=("image.png", get_image(image), "image/png"),
            )
        else:
            response = self.client.videos.create(
                prompt=prompt,
                model="sora-2",
                size=size,
                seconds=duration,
            )
        track_id = response.id
        name = hashlib.md5(track_id.encode()).hexdigest() + ".mp4"
        path = f"{get_temp_folder()}/{name}"
        while True:
            response = self.client.videos.retrieve(track_id)
            status = response.status
            logging.info(f"Video {track_id} generation status: {status}.")
            if status == "completed":
                video_id = response.id
                content = self.client.videos.download_content(video_id, variant="video")
                content.write_to_file(path)
                return path
            elif status in ["failed", "cancelled"]:
                gr.Error("Video generation failed!")
                logging.info(
                    f"Video {track_id} generation failed with info: {response}."
                )
                break
            time.sleep(5)
        return None
