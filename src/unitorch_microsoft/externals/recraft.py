# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import io
import json
import re
import fire
import hashlib
import random
import logging
import requests
import pandas as pd
from PIL import Image, ImageOps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

try:
    from openai import OpenAI
except ImportError:
    raise ImportError(
        "Please install the openai package by running `pip install openai`"
    )

supported_image_sizes = [
    (1024, 1024),
    (1365, 1024),
    (1024, 1365),
    (1536, 1024),
    (1024, 1536),
    (1820, 1024),
    (1024, 1820),
    (1024, 2048),
    (2048, 1024),
    (1434, 1024),
    (1024, 1434),
    (1024, 1280),
    (1280, 1024),
    (1024, 1707),
    (1707, 1024),
]

supported_image_styles = [
    "realistic_image",
    "digital_illustration",
    "vector_illustration",
    "icon",
    "logo_raster",
]

token = os.getenv("RECRAFT_API_KEY", None)
client = OpenAI(
    base_url="https://external.api.recraft.ai/v1",
    api_key=token,
)


def get_image(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    style: str = "realistic_image",
):
    _width, _height = width, height
    if (width, height) not in supported_image_sizes:
        needed_ratio = width / height
        closest_size = min(
            supported_image_sizes,
            key=lambda size: abs((size[0] / size[1]) - needed_ratio),
        )
        width, height = closest_size
    response = client.images.generate(
        prompt=prompt,
        style=style,
        size=f"{width}x{height}",
    )
    result = response.data[0].url
    return Image.open(requests.get(result, stream=True).raw).resize((_width, _height))


def get_inpainting_image(
    prompt: str,
    image: Union[str, Image.Image],
    mask: Optional[Union[str, Image.Image]] = None,
    style: str = "realistic_image",
):
    if isinstance(image, str):
        image = Image.open(image)
    if isinstance(mask, str):
        mask = Image.open(mask)

    if not isinstance(image, Image.Image):
        raise ValueError("Image must be a PIL Image or a file path to an image.")

    if mask is not None and not isinstance(mask, Image.Image):
        raise ValueError("Mask must be a PIL Image or a file path to a mask.")

    image_buffer = io.BytesIO()
    image.save(image_buffer, format="PNG")
    mask_image_buffer = io.BytesIO()
    mask = mask.convert("L")
    mask = mask.point(lambda x: 255 if x > 127 else 0, "L")
    mask.save(mask_image_buffer, format="PNG")
    image_buffer.seek(0)
    mask_image_buffer.seek(0)

    response = client.post(
        path="/images/inpaint",
        cast_to=object,
        options={"headers": {"Content-Type": "multipart/form-data"}},
        files={
            "image": ("image.png", image_buffer, "image/png"),
            "mask": ("mask_image.png", mask_image_buffer, "image/png"),
        },
        body={
            "style": style,
            "prompt": prompt,
        },
    )
    result = response["data"][0]["url"]
    return Image.open(requests.get(result, stream=True).raw)


def get_remove_background_image(
    image: Union[str, Image.Image],
):
    if isinstance(image, str):
        image = Image.open(image)

    image_buffer = io.BytesIO()
    image.save(image_buffer, format="PNG")
    image_buffer.seek(0)

    response = client.post(
        path="/images/removeBackground",
        cast_to=object,
        options={"headers": {"Content-Type": "multipart/form-data"}},
        files={
            "file": ("image.png", image_buffer, "image/png"),
        },
    )
    result = response["image"]["url"]
    return Image.open(requests.get(result, stream=True).raw)


def get_resolution_image(
    image: Union[str, Image.Image],
    method: Optional[str] = "clarity",  # clarity, generative
):
    if isinstance(image, str):
        image = Image.open(image)

    image_buffer = io.BytesIO()
    image.save(image_buffer, format="PNG")
    image_buffer.seek(0)

    response = client.post(
        path=(
            "/images/clarityUpscale"
            if method == "clarity"
            else "/images/generativeUpscale"
        ),
        cast_to=object,
        options={"headers": {"Content-Type": "multipart/form-data"}},
        files={
            "file": ("image.png", image_buffer, "image/png"),
        },
    )
    result = response["image"]["url"]
    return Image.open(requests.get(result, stream=True).raw)


def get_change_background_image(
    prompt: str,
    image: Union[str, Image.Image],
    style: Optional[str] = "realistic_image",
):
    if isinstance(image, str):
        image = Image.open(image)

    image_buffer = io.BytesIO()
    image.save(image_buffer, format="PNG")
    image_buffer.seek(0)

    response = client.post(
        path="/images/replaceBackground",
        cast_to=object,
        options={"headers": {"Content-Type": "multipart/form-data"}},
        files={
            "image": ("image.png", image_buffer, "image/png"),
        },
        body={
            "style": style,
            "prompt": prompt,
        },
    )
    result = response["data"][0]["url"]
    return Image.open(requests.get(result, stream=True).raw)
