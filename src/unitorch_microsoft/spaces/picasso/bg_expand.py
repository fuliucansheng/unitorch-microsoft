# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import io
import cv2
import math
import base64
import gc
import json
import requests
import torch
import numpy as np
import gradio as gr
from PIL import Image, ImageDraw
from openai import OpenAI
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor
from unitorch import mktempfile
from unitorch.utils import read_file, retry
from unitorch.models import GenericOutputs
from unitorch.cli import CoreConfigureParser
from unitorch.cli.webuis import SimpleWebUI
from unitorch_microsoft import cached_path
from unitorch_microsoft.externals.github_copilot import (
    get_response as get_gpt5_response,
)
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
    create_tab,
    create_tabs,
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


def process(image, ratio):
    width, height = image.size

    longest_side = 1024
    shortest_side = (
        int(longest_side * ratio) if ratio < 1 else int(longest_side / ratio)
    )
    size = (longest_side, shortest_side) if ratio > 1 else (shortest_side, longest_side)

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


def generate_by_recraft(image, ratio):
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

    new_image, new_mask = process(image, ratio)
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
    new_image, new_mask = process(result, ratio)
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


def generate_by_flux(image, ratio, flux_endpoint=None):
    caption = get_gpt5_response(
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

    new_image, new_mask = process(image, ratio)
    image_np = np.array(new_image.convert("RGB"))
    mask_np = np.array(new_mask.convert("L")).astype(np.uint8)

    _, binary_mask = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
    inpainted_image = cv2.inpaint(image_np, binary_mask, 10, cv2.INPAINT_TELEA)
    latent_image = Image.fromarray(inpainted_image)
    result = call_fastapi(
        flux_endpoint + "/generate",
        images={
            "image": latent_image,
            "mask_image": new_mask,
        },
        params={
            "text": "no text, no watermark, no logos, no people. " + caption,
            "guidance_scale": 30,
            "strength": 0.95,
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

    if ratio == ratio2:
        return result

    ratio = ratio2
    new_image, new_mask = process(result, ratio)
    image_np = np.array(new_image.convert("RGB"))
    mask_np = np.array(new_mask.convert("L")).astype(np.uint8)

    _, binary_mask = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
    inpainted_image = cv2.inpaint(image_np, binary_mask, 10, cv2.INPAINT_TELEA)
    latent_image = Image.fromarray(inpainted_image)
    result = call_fastapi(
        flux_endpoint + "/generate",
        images={
            "image": latent_image,
            "mask_image": new_mask,
        },
        params={
            "text": "no text, no watermark, no logos, no people. " + caption,
            "guidance_scale": 30,
            "strength": 0.95,
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
    return result

def generate_by_center_crop(image, ratio):
    image_width, image_height = image.size
    image_ratio = image_width / image_height

    if image_ratio > ratio:
        # Image is too wide
        new_height = image_height
        new_width = int(ratio * new_height)
        new_x = (image_width - new_width) // 2
        new_y = 0
    else:
        # Image is too tall
        new_width = image_width
        new_height = int(new_width / ratio)
        new_x = 0
        new_y = (image_height - new_height) // 2

    cropped_image = image.crop(
        (new_x, new_y, new_x + new_width, new_y + new_height)
    )
    return cropped_image

def pil_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return "data:image/png;base64," + img_str


def base64_to_pil(img_str):
    img_data = base64.b64decode(img_str)
    image = Image.open(io.BytesIO(img_data))
    return image


def get_seedream_response(
    prompt, images, size="1440x2560", model="doubao-seedream-4-5-251128"
):
    token = "5f1c7685-1bd8-4a71-a1e9-0591004685b4"

    client = OpenAI(
        # 此为默认路径，您可根据业务所在地域进行配置
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        # 从环境变量中获取您的 API Key。此为默认方式，您可根据需要进行修改
        api_key=token,
    )
    response = client.images.generate(
        model=model,
        prompt=prompt,
        extra_body={
            "image": [pil_to_base64(im) for im in images],
            "sequential_image_generation": "disabled",
            "watermark": False,
        },
        size=size,
        response_format="b64_json",
    )
    if len(response.data) == 0:
        raise ValueError(f"Error in Seedream API: {response}")
    image = base64_to_pil(response.data[0].b64_json)
    return image

def generate_by_seedream(image, ratio):
    prompt = """Extend the image evenly on both the left and right sides for an audience advertising creative.
 
Keep the original image content centered in the final composition. The main subject must remain in the center and unchanged in size, position, and appearance.
 
Keep the original main subject and logos, texts exactly the same in shape, size, position, and appearance. Do not alter, deform, or add any new main objects.
 
Generate a clean, realistic, and brand-safe background that seamlessly continues the original scene. The extended area should have natural lighting, consistent perspective, and smooth color transitions.
 
The background should be simple, non-distracting, and suitable for advertisements, leaving sufficient empty space for text or call-to-action elements.
 
High visual quality, sharp details, no blur, no artifacts, no strange textures.
No text, no watermark, no logo, no extra objects.
"""
    if ratio > 1.0:
        width = int(2048 * ratio) // 8 * 8
        height = 2048
    else:
        width = 2048
        height = int(2048 / ratio) // 8 * 8
    result = get_seedream_response(
        prompt, [image], size=f"{width}x{height}",
    )
    return result


class ExpandBGWebUI(SimpleWebUI):
    _title = "Generative Outpainting"
    _description = "This is a demo for outpaint the image using Prod/Flux/Seedream. You can input an image and a ratio, and the model will generate a new image with the specified background expanded."

    def __init__(self, config: CoreConfigureParser):
        self._status = getattr(self, "_status", "Stopped")

        config.set_default_section("microsoft/spaces/picasso/expand_bg")
        self._flux_endpoint = config.getoption("flux_endpoint", None)
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

        input_image = create_element("image", "Image")
        input_ratio = create_element(
            "slider", "Ratio", default=1.91, min_value=0.1, max_value=10.0, step=0.01
        )
        generate = create_element("button", "Generate")
        output_prod = create_element("image", "Prod Output", lines=5)
        output_flux = create_element("image", "Flux Output", lines=5)
        output_seedream = create_element("image", "Seedream Output", lines=5)

        examples = gr.Examples(
            examples=[
                cached_path("spaces/picasso/examples/op1.jpg"),
                cached_path("spaces/picasso/examples/op2.jpg"),
                cached_path("spaces/picasso/examples/op3.jpg"),
                cached_path("spaces/picasso/examples/op4.jpg"),
                cached_path("spaces/picasso/examples/op5.jpg"),
            ],
            label="Image Examples",
            inputs=[input_image],
            examples_per_page=10,
        )

        left = create_column(input_image, examples.dataset, input_ratio, generate)
        right = create_column(
            output_prod,
            output_flux,
            output_seedream,
        )

        iface = create_blocks(
            toper_menus,
            create_row(
                create_column(header, description, scale=1),
                create_column(),
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

        generate.click(
            fn=self.generate,
            inputs=[input_image, input_ratio],
            outputs=[output_prod, output_flux, output_seedream],
            trigger_mode="once",
        )

        examples.create()

        iface.__exit__()

        super().__init__(config, iname=self._title, iface=iface)

    def start(self):
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
        requests.get(self._flux_endpoint + "/stop", timeout=1200).raise_for_status()
        self._status = "Stopped"
        return self._status
    
    def status(self):
        status = requests.get(
            self._flux_endpoint + "/status", timeout=1200
        ).json()
        self._status = "Running" if status == "running" else "Stopped"
        return self._status

    def generate(self, image, ratio):
        if self.status() != "Running":
            self.start()
        with ThreadPoolExecutor(max_workers=3) as executor:
            f_prod = executor.submit(generate_by_center_crop, image, ratio)
            f_flux = executor.submit(
                generate_by_flux, image, ratio, flux_endpoint=self._flux_endpoint
            )
            f_seedream = executor.submit(generate_by_seedream, image, ratio)

            prod_result = f_prod.result()
            flux_result = f_flux.result()
            seedream_result = f_seedream.result()

        return prod_result, flux_result, seedream_result


