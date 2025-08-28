# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import io
import re
import json
import asyncio
import time
import base64
from turtle import width
from numpy import isin
import requests
import tempfile
from PIL import Image
from typing import Optional, Any, Union, List, Type
from pydantic import BaseModel, Field
from transformers.utils import is_remote_url
from unitorch_microsoft import cached_path
from unitorch_microsoft.chatgpt.papyrus import (
    get_gpt_image_response,
    get_gpt4_response,
    get_gpt4_tools_response,
    get_gpt5_response,
    get_gpt5_tools_response,
)
from unitorch_microsoft.chatgpt.recraft import (
    get_image as get_recraft_image,
    get_inpainting_image as get_recraft_inpainting_image,
    get_change_background_image as get_recraft_change_background_image,
    get_resolution_image as get_recraft_resolution_image,
    get_remove_background_image as get_recraft_remove_background_image,
)
from unitorch_microsoft.agents.utils import call_fastapi
from unitorch_microsoft.agents.components import (
    GenericTool,
    ImageResult,
    GenericResult,
    GenericError,
    ToolChoice,
    ToolCollection,
)
from unitorch_microsoft.agents.components.tools import GPT4FormatTool
from unitorch_microsoft.agents.components.picasso import get_picasso_temp_dir

_PICASSOINTERNAL_DESCRIPTION = """
Use this tool to generate images based on the provided prompt & images. 

You can use the following actions:
1. `create`: Generate an image based on the provided prompt, if the refer_images are provided, it will use them to generate the image. The prompt should be a description of the image you want to generate or how to combine the reference images. The generated image will be of size (width, height) specified in the parameters.
2. `editing`: Edit the input image based on the provided prompt, usually used for modifying the input image with instructions like changing colors, design a new size of (width, height) specified in the parameters.
3. `fitting`: Fit the input image to the expected ratio. The ratio is a float value, where 1 means square, < 1 means portrait, and > 1 means landscape.
4. `padding`: Pad the input image with specified pad_size (left, top, right, bottom) pixels to fit the expected size controlled by prompt. The new image size will be (width + left + right, height + top + bottom).
5. `change_background`: Change the background of the input image based on the provided prompt.
6. `remove_background`: Remove the background of the input image. set crop_object to false if you don't want to remove useless transparent background. It will return the image with transparent background.

Note:
- Parameters `width`, `height` are only used for `create` and `editing` actions, they specify the size of the generated image.
- Parameters `ratio` is only used for `fitting` action, it specifies the ratio of the image to be generated. The ratio should be between 0.1 and 10,
    where 1 means square, < 1 means portrait, and > 1 means landscape
- Parameters `pad_size` is only used for `padding` action, it specifies the padding pixels for left, top, right, bottom.
- Parameters `refer_images` is only used for `create` action, it specifies the reference images to be used for image generation. It should be a list of image paths, with a minimum of 1 and a maximum of 5 images.
- Parameters `crop_object` is only used for `remove_background` action, it specifies whether to crop the foreground object after removing the background.
"""


def processing_outpainting(image, ratio):
    if isinstance(image, str):
        image = Image.open(image)
    width, height = image.size

    longest_side = 2048
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

    new_width = int(width * scale)
    new_height = int(height * scale)

    image = image.resize(
        (new_width // 8 * 8, new_height // 8 * 8), resample=Image.LANCZOS
    )

    im_width, im_height = image.size

    mode = image.mode
    if mode not in ["RGB", "RGBA"]:
        image = image.convert("RGB")
        mode = "RGB"

    mask = Image.new("L", (size[0], size[1]), 255)
    black = Image.new(
        mode, (im_width, im_height), (0, 0, 0) if mode == "RGB" else (0, 0, 0, 0)
    )
    mask.paste(black, ((size[0] - im_width) // 2, (size[1] - im_height) // 2))
    new_image = Image.new(
        mode,
        (size[0], size[1]),
        (255, 255, 255) if mode == "RGB" else (255, 255, 255, 0),
    )
    new_image.paste(image, ((size[0] - im_width) // 2, (size[1] - im_height) // 2))

    return new_image, mask


def resized_image(image, min_pixel=256, max_pixel=2048):
    if isinstance(image, str):
        image = Image.open(image)
    width, height = image.size

    if max(width, height) > max_pixel:
        scale = max_pixel / max(width, height)
        new_size = (int(width * scale), int(height * scale))
        image = image.resize(
            (new_size[0] // 8 * 8, new_size[1] // 8 * 8), resample=Image.LANCZOS
        )
    elif min(width, height) < min_pixel:
        scale = min_pixel / min(width, height)
        new_size = (int(width * scale), int(height * scale))
        image = image.resize(
            (new_size[0] // 8 * 8, new_size[1] // 8 * 8), resample=Image.LANCZOS
        )

    return image


def fit_image_by_smart_roi(image, ratio):
    endpoint = "https://NorthCentralUS.bing.prod.dlis.binginternal.com/route/PicassoAdsCreative.ROIDetection"
    cert = (
        cached_path("agents/components/picasso/10.224.120.184.cer"),
        cached_path("agents/components/picasso/private.key"),
    )

    image = Image.open(image)
    image = resized_image(image, min_pixel=512, max_pixel=2048)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    data = {
        "image": base64.b64encode(buf.getvalue()).decode(),
        "return_saliency_map": False,
        "roi_offset": 0,
        "return_strict_roi": True,
        "keep_original_strict_roi": False,
    }

    result = requests.post(
        endpoint,
        data=json.dumps(data),
        cert=cert,
        headers={"Content-Type": "application/json"},
    ).json()
    if result.get("code", None) != 200:
        return None

    result = result.get("result", {})
    roi = result.get("roi", None)
    if roi is None or not isinstance(roi, str):
        return None
    if len(roi.split(",")) != 4:
        return None
    x, y, w, h = roi.split(",")
    x, y, w, h = int(x), int(y), int(w), int(h)

    image_width, image_height = image.size
    image_ratio = image_width / image_height
    ratio_range = (w / image_height, image_width / h)

    if ratio < ratio_range[0] or ratio > ratio_range[1]:
        return None

    if image_ratio > ratio:
        new_h = image_height
        new_w = ratio * new_h
        w_diff = new_w - w
        left_distance = x
        right_distance = image_width - x - w

        if left_distance >= w_diff / 2 and right_distance >= w_diff / 2:
            new_x = x - w_diff / 2
        elif left_distance < w_diff / 2:
            new_x = 0
        else:
            new_x = image_width - new_w
        new_y = 0
    else:
        new_w = image_width
        new_h = new_w / ratio
        h_diff = new_h - h
        top_distance = y
        bottom_distance = image_height - y - h

        if top_distance >= h_diff / 2 and bottom_distance >= h_diff / 2:
            new_y = y - h_diff / 2
        elif top_distance < h_diff / 2:
            new_y = 0
        else:
            new_y = image_height - new_h

        new_x = 0

    # Clamp and round values
    new_x = int(round(max(0, min(new_x, image_width - new_w))))
    new_y = int(round(max(0, min(new_y, image_height - new_h))))
    new_w = int(round(min(new_w, image_width - new_x)))
    new_h = int(round(min(new_h, image_height - new_y)))

    return image.crop((new_x, new_y, new_x + new_w, new_y + new_h))


class PicassoInternalTool(GenericTool):
    """Add a tool to generate images based on the provided prompt & images."""

    name: str = "picasso_internal_tool"
    description: str = _PICASSOINTERNAL_DESCRIPTION
    parameters: str = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "create",
                    "editing",
                    "fitting",
                    "padding",
                    "change_background",
                    "remove_background",
                ],
                "description": """
                The action to perform on the image. Options are: 
                * 'create' for prompt-to-image generation. refer_images can be used to provide reference images. leave it empty if you don't want to use reference images.
                * 'editing' for editing the input image based on the prompt.
                * 'fitting' for fitting the input image to the expected ratio.
                * 'padding' for padding the input image to with specified pad_size (left, top, right, bottom) pixels controlled by prompt.
                * 'change_background' for change the input image background.
                * 'remove_background' for remove the input image background. crop_object is used to remove useless transparent background.
                """,
            },
            "prompt": {
                "type": "string",
                "description": """
                The prompt for create & change_background actions. 
                * For create, it describes the image to be generated. 
                * For editing, it describes the modifications to be made to the input image.
                * For change_background, it describes the new background of the image. 
                """,
            },
            "image": {
                "type": "string",
                "description": "The input image path to be used for fitting, padding, change_background, remove_background.",
            },
            "width": {
                "type": "integer",
                "description": "The width of the image to be generated for create action.",
            },
            "height": {
                "type": "integer",
                "description": "The height of the image to be generated for create action.",
            },
            "ratio": {
                "type": "number",
                "description": "The ratio of the image to be generated for fitting action.",
                "minimum": 0.1,
                "maximum": 10,
            },
            "pad_size": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "The padding pixels for left, top, right, bottom for padding action.",
                "minItems": 4,
                "maxItems": 4,
            },
            "refer_images": {
                "type": "array",
                "items": {"type": "string"},
                "description": "A list of reference image paths for image generation.",
                "minItems": 1,
                "maxItems": 5,
            },
            "crop_object": {
                "type": "boolean",
                "default": True,
                "description": "Whether to crop the foreground object after remove_background action.",
            },
        },
        "required": ["action"],
        "dependencies": {
            "create": ["prompt", "width", "height", "refer_images"],
            "editing": ["prompt", "image", "width", "height"],
            "fitting": ["image", "ratio"],
            "padding": ["image", "pad_size"],
            "change_background": ["prompt", "image"],
            "remove_background": ["image", "crop_object"],
        },
    }

    async def execute(
        self,
        action: str,
        prompt: Optional[str] = None,
        image: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        ratio: Optional[float] = None,
        pad_size: Optional[list[int]] = None,
        refer_images: Optional[list[str]] = None,
        crop_object: bool = True,
    ) -> str:
        try:
            if action == "create":
                if not prompt:
                    raise GenericError("prompt is required for create action.")
                if not width or not height:
                    raise GenericError(
                        "width and height are required for create action."
                    )
                ratio = width / height
                size = "1024x1024"
                if ratio > 0.8:
                    size = "1024x1536"
                elif ratio < 1.2:
                    size = "1024x1024"
                else:
                    size = "1536x1024"
                result = get_gpt_image_response(
                    prompt=prompt,
                    images=(
                        [Image.open(im) for im in refer_images]
                        if refer_images
                        else None
                    ),
                    size=size,
                )
                result = call_fastapi(
                    "http://br1u43-s2-01:5000/core/fastapi/stable_flux/kontext2image/generate",
                    images={
                        "image": result,
                    },
                    params={
                        "text": "",
                        "guidance_scale": 3.5,
                        "num_timesteps": 30,
                        "seed": 42,
                        "width": width,
                        "height": height,
                    },
                    resp_type="image",
                )
                # result = result.convert("RGB")
            elif action == "fitting":
                if not image:
                    raise GenericError("Image is required for fitting action.")
                if not ratio:
                    raise GenericError("Ratio is required for fitting action.")
                if ratio <= 0.1 or ratio > 10:
                    raise GenericError("Ratio must be between 0.1 and 10.")
                # if Image.open(image).mode == "RGBA":
                #     raise GenericError(
                #         "Image with transparent background is not supported for fitting action."
                #     )

                if Image.open(image).mode != "RGBA":
                    result = fit_image_by_smart_roi(image, ratio)
                else:
                    result = None
                if result is None:
                    image, mask = processing_outpainting(Image.open(image), ratio)
                    caption = get_gpt4_response(
                        "Describe the background of this image, maintaining its colors, textures, and lighting. Ensure seamless blending without adding new objects, text, or artifacts. The caption is in a single short paragraph. Don't mention any object in foreground.",
                        images=[image],
                    )
                    result = get_recraft_inpainting_image(
                        image=image,
                        mask=mask,
                        prompt="no text, no watermark, no logos, no people. " + caption,
                    )
            elif action == "editing":
                if not prompt:
                    raise GenericError("prompt is required for editing action.")
                if not image:
                    raise GenericError("Image is required for editing action.")

                if isinstance(image, str):
                    image = Image.open(image)

                is_rgba = image.mode == "RGBA"
                if is_rgba:
                    image = image.convert("RGB")

                if width is None or height is None:
                    width, height = image.size

                result = call_fastapi(
                    "http://br1u43-s2-01:5000/core/fastapi/stable_flux/kontext2image/generate",
                    images={
                        "image": image,
                    },
                    params={
                        "text": prompt,
                        "guidance_scale": 3.5,
                        "num_timesteps": 30,
                        "seed": 42,
                        "width": width,
                        "height": height,
                    },
                    resp_type="image",
                )
                if is_rgba:
                    result = get_recraft_remove_background_image(image=result)
            elif action == "padding":
                if not image:
                    raise GenericError("Image is required for padding action.")
                if not pad_size:
                    raise GenericError("pad_size is required for padding action.")
                if len(pad_size) != 4:
                    raise GenericError(
                        "pad_size must be a list of four integers: [left, top, right, bottom]."
                    )

                image = Image.open(image)
                # if image.mode == "RGBA":
                #     raise GenericError(
                #         "Image with transparent background is not supported for padding action."
                #     )

                caption = get_gpt4_response(
                    "Describe the background of this image, maintaining its colors, textures, and lighting. Ensure seamless blending without adding new objects, text, or artifacts. The caption is in a single short paragraph. Don't mention any object in foreground.",
                    images=[image],
                )
                # image = image.convert("RGB")
                mode = image.mode
                if mode not in ["RGB", "RGBA"]:
                    image = image.convert("RGB")
                    mode = "RGB"
                new_image = Image.new(
                    mode,
                    (
                        image.width + pad_size[0] + pad_size[2],
                        image.height + pad_size[1] + pad_size[3],
                    ),
                    (255, 255, 255) if mode == "RGB" else (255, 255, 255, 0),
                )
                new_image.paste(image, (pad_size[0], pad_size[1]))
                new_mask = Image.new(
                    "L",
                    (
                        image.width + pad_size[0] + pad_size[2],
                        image.height + pad_size[1] + pad_size[3],
                    ),
                    255,
                )
                new_mask.paste(
                    Image.new("L", image.size, 0), (pad_size[0], pad_size[1])
                )
                result = get_recraft_inpainting_image(
                    image=new_image,
                    mask=new_mask,
                    prompt="no text, no watermark, no logos, no people." + caption,
                )
            elif action == "change_background":
                if not prompt:
                    raise GenericError(
                        "Prompt is required for change_background action."
                    )
                if not image:
                    raise GenericError(
                        "Image is required for change_background action."
                    )
                if isinstance(image, str):
                    image = Image.open(image)
                input_size = image.size
                image = resized_image(image, min_pixel=512, max_pixel=2048)
                result = get_recraft_change_background_image(
                    image=image,
                    prompt="no text, no watermark, no logos. " + prompt,
                )
                result = result.resize(input_size, resample=Image.LANCZOS)
            elif action == "remove_background":
                if not image:
                    raise GenericError(
                        "Image is required for remove_background action."
                    )
                if isinstance(image, str):
                    image = Image.open(image)
                input_size = image.size
                image = resized_image(image, min_pixel=512, max_pixel=2048)
                result = get_recraft_remove_background_image(image=image)
                result = result.resize(input_size, resample=Image.LANCZOS)
                # result is png image object
                if crop_object:
                    result = result.convert("RGBA")
                    alpha = result.getchannel("A")
                    alpha = alpha.point(lambda p: 255 if p > 50 else 0)
                    bbox = alpha.getbbox()
                    if bbox:
                        result = result.crop(bbox)
            else:
                raise GenericError(f"Unknown action: {action}")
        except Exception as e:
            raise GenericError(f"Error occurred while processing the image: {str(e)}")

        if isinstance(result, Image.Image):
            temp_file = tempfile.NamedTemporaryFile(
                suffix=".png", dir=get_picasso_temp_dir(), delete=False
            )
            result.save(temp_file.name)
            result = temp_file.name

        if result is None:
            raise GenericError("Generated image is blocked because of NSFW content.")

        res = Image.open(result)

        return GenericResult(
            output=f"Generated image path: {result} . Width: {res.width}, Height: {res.height}.",
            images={"path": result},
            meta={
                "_width": res.width,
                "_height": res.height,
                "_image": result,
            },
        )
