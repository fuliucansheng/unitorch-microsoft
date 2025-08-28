# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import io
import asyncio
import time
import tempfile
from PIL import Image
from typing import Optional, Any, Union, List, Type
from pydantic import BaseModel, Field
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

_PICASSOIMAGE_DESCRIPTION = """
Use this tool to generate images based on the provided prompt & images. Don't use the tool to add text, as it may not work well.

You can use the following actions:
1. `create`: Generate an image based on the provided prompt, usually used for generating background without any text.
2. `editing`: Edit the input image based on the provided prompt, usually used for modifying the input image with instructions like changing colors, etc.
3. `fitting`: Fit the input image to the expected ratio. The ratio is a float value, where 1 means square, < 1 means portrait, and > 1 means landscape.
4. `padding`: Pad the input image with specified pad_size (left, top, right, bottom) pixels to fit the expected size controlled by prompt. The new image size will be (width + left + right, height + top + bottom).
5. `change_background`: Change the background of the input image based on the provided prompt.
6. `remove_background`: Remove the background of the input image. set crop_object to true if you want to remove useless transparent background. It will return the image with transparent background.
7. `refer_create`: Generate an image based on the provided prompt and reference images. The created image size only supports three values: '1024x1024', '1024x1536', '1536x1024'.

Notes:
- The actions `editing`, `fitting` and `padding` do not support images with transparent background.
"""


def processing_outpainting(image, ratio):
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
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

    mask = Image.new("L", (size[0], size[1]), 255)
    black = Image.new("RGB", (im_width, im_height), (0, 0, 0))
    mask.paste(black, ((size[0] - im_width) // 2, (size[1] - im_height) // 2))
    new_image = Image.new("RGB", (size[0], size[1]), (255, 255, 255))
    new_image.paste(image, ((size[0] - im_width) // 2, (size[1] - im_height) // 2))

    return new_image, mask


class PicassoImageTool(GenericTool):
    """Add a tool to generate images based on the provided prompt & images."""

    name: str = "picasso_image_tool"
    description: str = _PICASSOIMAGE_DESCRIPTION
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
                    "refer_create",
                ],
                "description": """
                The action to perform on the image. Options are: 
                * 'create' for prompt-to-image generation, usually used for generating background.
                * 'editing' for editing the input image based on the prompt
                * 'fitting' for fitting the input image to the expected ratio.
                * 'padding' for padding the input image to with specified pad_size (left, top, right, bottom) pixels controlled by prompt.
                * 'change_background' for change the input image background
                * 'remove_background' for remove the input image background
                * 'refer_create' the prompt-to-image generation with reference images.
                """,
            },
            "prompt": {
                "type": "string",
                "description": """
                The prompt for create & editing & change_background & refer_create actions. 
                * For create, it describes the image to be generated. 
                * For editing, it describes the modifications to be made to the input image.
                * For change_background, it describes the new background of the image. 
                * For refer_create, it describes the instruction based on the reference images.
                """,
            },
            "image": {
                "type": "string",
                "description": "The input image path to be used for fitting, padding, change_background, remove_background.",
            },
            "width": {
                "type": "integer",
                "description": "The width of the image to be generated for create & editing action.",
            },
            "height": {
                "type": "integer",
                "description": "The height of the image to be generated for create & editing action.",
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
                "description": "A list of reference image paths for image-to-image generation.",
                "minItems": 1,
                "maxItems": 5,
            },
            "refer_created_size": {
                "type": "string",
                "enum": ["1024x1024", "1024x1536", "1536x1024"],
                "description": "The size of the image to be generated for refer_create action. Only supports three values: '1024x1024', '1024x1536', '1536x1024'.",
            },
            "crop_object": {
                "type": "boolean",
                "default": False,
                "description": "Whether to crop the foreground object after remove_background action.",
            },
        },
        "required": ["action"],
        "dependencies": {
            "create": ["prompt", "width", "height"],
            "editing": ["prompt", "image", "width", "height"],
            "fitting": ["image", "ratio"],
            "padding": ["image", "pad_size"],
            "change_background": ["prompt", "image"],
            "remove_background": ["image", "crop_object"],
            "refer_create": ["prompt", "refer_images", "refer_created_size"],
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
        refer_created_size: Optional[str] = "1024x1024",
        crop_object: bool = False,
    ) -> str:
        try:
            if action == "create":
                if not prompt:
                    raise GenericError("prompt is required for create action.")
                result = get_recraft_image(
                    prompt=prompt,
                    width=width or 1024,
                    height=height or 1024,
                )
                result = result.convert("RGB")
            elif action == "fitting":
                if not image:
                    raise GenericError("Image is required for fitting action.")
                if not ratio:
                    raise GenericError("Ratio is required for fitting action.")
                if ratio <= 0.1 or ratio > 10:
                    raise GenericError("Ratio must be between 0.1 and 10.")
                if Image.open(image).mode == "RGBA":
                    raise GenericError(
                        "Image with transparent background is not supported for fitting action."
                    )
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
                result = result.convert("RGB")
            elif action == "editing":
                if not prompt:
                    raise GenericError("prompt is required for editing action.")
                if not image:
                    raise GenericError("Image is required for editing action.")

                if isinstance(image, str):
                    image = Image.open(image)

                if image.mode == "RGBA":
                    raise GenericError(
                        "Image with transparent background is not supported for editing action."
                    )

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
                result = result.convert("RGB")
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
                if image.mode == "RGBA":
                    raise GenericError(
                        "Image with transparent background is not supported for padding action."
                    )

                caption = get_gpt4_response(
                    "Describe the background of this image, maintaining its colors, textures, and lighting. Ensure seamless blending without adding new objects, text, or artifacts. The caption is in a single short paragraph. Don't mention any object in foreground.",
                    images=[image],
                )
                image = image.convert("RGB")
                new_image = Image.new(
                    "RGB",
                    (
                        image.width + pad_size[0] + pad_size[2],
                        image.height + pad_size[1] + pad_size[3],
                    ),
                    (255, 255, 255),
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
                result = result.convert("RGB")
            elif action == "change_background":
                if not prompt:
                    raise GenericError(
                        "Prompt is required for change_background action."
                    )
                if not image:
                    raise GenericError(
                        "Image is required for change_background action."
                    )
                result = get_recraft_change_background_image(image=image, prompt=prompt)
                result = result.convert("RGB")
            elif action == "remove_background":
                if not image:
                    raise GenericError(
                        "Image is required for remove_background action."
                    )
                result = get_recraft_remove_background_image(image=image)
                # result is png image object
                if crop_object:
                    result = result.convert("RGBA")
                    alpha = result.getchannel("A")
                    alpha = alpha.point(lambda p: 255 if p > 50 else 0)
                    bbox = alpha.getbbox()
                    if bbox:
                        result = result.crop(bbox)
            elif action == "refer_create":
                if not prompt or not refer_images:
                    raise GenericError(
                        "prompt and refer_images are required for refer_create action."
                    )
                if refer_created_size not in ["1024x1024", "1024x1536", "1536x1024"]:
                    raise GenericError(
                        "refer_created_size must be one of '1024x1024', '1024x1536', '1536x1024'."
                    )
                result = get_gpt_image_response(
                    prompt=prompt,
                    images=[Image.open(im) for im in refer_images],
                    size=refer_created_size,
                )
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
