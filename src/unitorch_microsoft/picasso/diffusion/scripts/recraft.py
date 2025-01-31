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
from torch.hub import download_url_to_file
from unitorch.cli import CoreConfigureParser, GenericScript
from unitorch.cli import register_script, cached_path

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


def save_image_from_url(folder, url):
    name = hashlib.md5(url.encode()).hexdigest() + ".jpg"
    path = f"{folder}/{name}"
    try:
        download_url_to_file(url, path, progress=False)
        return path
    except:
        return None


def save_image(folder, image):
    name = hashlib.md5(image.tobytes()).hexdigest() + ".jpg"
    image.save(f"{folder}/{name}")
    return f"{folder}/{name}"


def text2image(
    token: str,
    data_file: str,
    cache_dir: str,
    names: Union[str, List[str]],
    prompt_col: str,
    style_prompt_col: Optional[str] = None,
    style_prompt: Optional[str] = "realistic_image",
    height: Optional[int] = 1024,
    width: Optional[int] = 1024,
):
    assert (width, height) in supported_image_sizes
    client = OpenAI(
        base_url="https://external.api.recraft.ai/v1",
        api_key=token,
    )

    if isinstance(names, str) and names.strip() == "*":
        names = None
    if isinstance(names, str):
        names = re.split(r"[,;]", names)
        names = [n.strip() for n in names]

    data = pd.read_csv(
        data_file,
        names=names,
        sep="\t",
        quoting=3,
        header=None,
    )

    os.makedirs(cache_dir, exist_ok=True)

    assert prompt_col in data.columns, f"Column {prompt_col} not found in data."

    output_file = f"{cache_dir}/output.jsonl"

    if os.path.exists(output_file):
        uniques = []
        with open(output_file, "r") as f:
            for line in f:
                row = json.loads(line)
                uniques.append(row["prompt"] + " - " + row["style"])
        data = data[
            ~data.apply(
                lambda x: x[prompt_col]
                + " - "
                + (style_prompt if style_prompt is not None else x[style_prompt_col])
                in uniques,
                axis=1,
            )
        ]

    writer = open(output_file, "a+")

    for _, row in data.iterrows():
        _prompt = row[prompt_col]
        _style_prompt = (
            style_prompt if style_prompt is not None else row[style_prompt_col]
        )
        try:
            response = client.images.generate(
                prompt=_prompt,
                style=_style_prompt,
                size=f"{width}x{height}",
            )
            result = response.data[0].url
            record = {
                "prompt": _prompt,
                "style": _style_prompt,
                "result": save_image_from_url(cache_dir, result),
            }
            writer.write(json.dumps(record) + "\n")
            writer.flush()
        except:
            pass


# inpainting processing images
def __in_processing_image(image, mask_image, reversed_mask=False):
    if isinstance(image, str):
        image = Image.open(image)
    if isinstance(mask_image, str):
        mask_image = Image.open(mask_image)
    if reversed_mask:
        mask_image = ImageOps.invert(mask_image)

    # process your image/mask_image here

    return image, mask_image


def inpainting(
    token: str,
    data_file: str,
    cache_dir: str,
    names: Union[str, List[str]],
    image_col: str,
    mask_image_col: str,
    prompt_col: Optional[str] = None,
    prompt_text: Optional[str] = None,
    style_prompt_col: Optional[str] = None,
    style_prompt: Optional[str] = "realistic_image",
    reversed_mask: Optional[str] = False,
    processor_name: Optional[str] = "default",
):
    client = OpenAI(
        base_url="https://external.api.recraft.ai/v1",
        api_key=token,
    )

    if isinstance(names, str) and names.strip() == "*":
        names = None
    if isinstance(names, str):
        names = re.split(r"[,;]", names)
        names = [n.strip() for n in names]

    data = pd.read_csv(
        data_file,
        names=names,
        sep="\t",
        quoting=3,
        header=None,
    )

    processors = {
        "default": __in_processing_image,
    }

    assert processor_name in processors.keys()

    process_func = processors[processor_name]

    os.makedirs(cache_dir, exist_ok=True)

    assert prompt_col in data.columns, f"Column {prompt_col} not found in data."

    output_file = f"{cache_dir}/output.jsonl"

    if os.path.exists(output_file):
        uniques = []
        with open(output_file, "r") as f:
            for line in f:
                row = json.loads(line)
                uniques.append(
                    row["prompt"]
                    + " - "
                    + row["style"]
                    + " - "
                    + row["image"]
                    + " - "
                    + row["mask_image"]
                )
        data = data[
            ~data.apply(
                lambda x: (prompt_text if prompt_text is not None else x[prompt_col])
                + " - "
                + (style_prompt if style_prompt is not None else x[style_prompt_col])
                + " - "
                + x[image_col]
                + " - "
                + x[mask_image_col]
                in uniques,
                axis=1,
            )
        ]

    writer = open(output_file, "a+")
    for _, row in data.iterrows():
        _prompt = prompt_text if prompt_text is not None else row[prompt_col]
        _style_prompt = (
            style_prompt if style_prompt is not None else row[style_prompt_col]
        )
        image = row[image_col]
        mask_image = row[mask_image_col]
        p_image, p_mask_image = process_func(
            image, mask_image, reversed_mask=reversed_mask
        )

        image_buffer = io.BytesIO()
        p_image.save(image_buffer, format="JPEG")
        mask_image_buffer = io.BytesIO()
        p_mask_image = p_mask_image.convert("L")
        p_mask_image = p_mask_image.point(lambda x: 255 if x > 127 else 0, "L")
        p_mask_image.save(mask_image_buffer, format="PNG")
        image_buffer.seek(0)
        mask_image_buffer.seek(0)

        try:
            response = client.post(
                path="/images/inpaint",
                cast_to=object,
                options={"headers": {"Content-Type": "multipart/form-data"}},
                files={
                    "image": ("image.jpg", image_buffer, "image/jpeg"),
                    "mask": ("mask_image.jpg", mask_image_buffer, "image/jpeg"),
                },
                body={
                    "style": _style_prompt,
                    "prompt": _prompt,
                },
            )
            result = response["data"][0]["url"]
            record = {
                "prompt": _prompt,
                "style": _style_prompt,
                "image": image,
                "mask_image": mask_image,
                "result": save_image_from_url(cache_dir, result),
            }
            writer.write(json.dumps(record) + "\n")
            writer.flush()
        except:
            pass


# remove background
def remove_background(
    token: str,
    data_file: str,
    cache_dir: str,
    names: Union[str, List[str]],
    image_col: str,
):
    client = OpenAI(
        base_url="https://external.api.recraft.ai/v1",
        api_key=token,
    )

    if isinstance(names, str) and names.strip() == "*":
        names = None
    if isinstance(names, str):
        names = re.split(r"[,;]", names)
        names = [n.strip() for n in names]

    data = pd.read_csv(
        data_file,
        names=names,
        sep="\t",
        quoting=3,
        header=None,
    )

    os.makedirs(cache_dir, exist_ok=True)

    output_file = f"{cache_dir}/output.jsonl"

    if os.path.exists(output_file):
        uniques = []
        with open(output_file, "r") as f:
            for line in f:
                row = json.loads(line)
                uniques.append(row["image"])
        data = data[
            ~data.apply(
                lambda x: x[image_col] in uniques,
                axis=1,
            )
        ]

    writer = open(output_file, "a+")
    for _, row in data.iterrows():
        image = row[image_col]
        raw_image = Image.open(image).convert("RGB")

        image_buffer = io.BytesIO()
        raw_image.save(image_buffer, format="JPEG")
        image_buffer.seek(0)

        try:
            response = client.post(
                path="/images/removeBackground",
                cast_to=object,
                options={"headers": {"Content-Type": "multipart/form-data"}},
                files={
                    "file": ("image.jpg", image_buffer, "image/jpeg"),
                },
            )
            result = response["image"]["url"]
            record = {
                "image": image,
                "result_object": save_image_from_url(cache_dir, result),
            }
            record["result_mask"] = save_image(
                cache_dir,
                Image.open(record["result_object"])
                .convert("L")
                .point(lambda x: 0 if x < 128 else 255, "1"),
            )
            writer.write(json.dumps(record) + "\n")
            writer.flush()
        except:
            pass


# resolution
def resolution(
    token: str,
    data_file: str,
    cache_dir: str,
    names: Union[str, List[str]],
    image_col: str,
    method: Optional[str] = "clarity",  # clarity, generative
):
    assert method in ["clarity", "generative"]

    client = OpenAI(
        base_url="https://external.api.recraft.ai/v1",
        api_key=token,
    )

    if isinstance(names, str) and names.strip() == "*":
        names = None
    if isinstance(names, str):
        names = re.split(r"[,;]", names)
        names = [n.strip() for n in names]

    data = pd.read_csv(
        data_file,
        names=names,
        sep="\t",
        quoting=3,
        header=None,
    )

    os.makedirs(cache_dir, exist_ok=True)

    output_file = f"{cache_dir}/output.jsonl"

    if os.path.exists(output_file):
        uniques = []
        with open(output_file, "r") as f:
            for line in f:
                row = json.loads(line)
                uniques.append(row["image"] + " - " + row["method"])
        data = data[
            ~data.apply(
                lambda x: x[image_col] + " - " + method in uniques,
                axis=1,
            )
        ]

    writer = open(output_file, "a+")
    for _, row in data.iterrows():
        image = row[image_col]
        raw_image = Image.open(image).convert("RGB")

        image_buffer = io.BytesIO()
        raw_image.save(image_buffer, format="JPEG")
        image_buffer.seek(0)

        try:
            response = client.post(
                path="/images/clarityUpscale"
                if method == "clarity"
                else "/images/generativeUpscale",
                cast_to=object,
                options={"headers": {"Content-Type": "multipart/form-data"}},
                files={
                    "file": ("image.jpg", image_buffer, "image/jpeg"),
                },
            )
            result = response["image"]["url"]
            record = {
                "image": image,
                "result": save_image_from_url(cache_dir, result),
            }
            writer.write(json.dumps(record) + "\n")
            writer.flush()
        except:
            pass


# change background
def change_background(
    token: str,
    data_file: str,
    cache_dir: str,
    names: Union[str, List[str]],
    image_col: str,
    prompt_col: Optional[str] = None,
    prompt_text: Optional[str] = None,
    style_prompt_col: Optional[str] = None,
    style_prompt: Optional[str] = "realistic_image",
):
    client = OpenAI(
        base_url="https://external.api.recraft.ai/v1",
        api_key=token,
    )

    if isinstance(names, str) and names.strip() == "*":
        names = None
    if isinstance(names, str):
        names = re.split(r"[,;]", names)
        names = [n.strip() for n in names]

    data = pd.read_csv(
        data_file,
        names=names,
        sep="\t",
        quoting=3,
        header=None,
    )

    os.makedirs(cache_dir, exist_ok=True)

    assert prompt_col in data.columns, f"Column {prompt_col} not found in data."

    output_file = f"{cache_dir}/output.jsonl"

    if os.path.exists(output_file):
        uniques = []
        with open(output_file, "r") as f:
            for line in f:
                row = json.loads(line)
                uniques.append(
                    row["prompt"] + " - " + row["style"] + " - " + row["image"]
                )
        data = data[
            ~data.apply(
                lambda x: (prompt_text if prompt_text is not None else x[prompt_col])
                + " - "
                + (style_prompt if style_prompt is not None else x[style_prompt_col])
                + " - "
                + x[image_col]
                in uniques,
                axis=1,
            )
        ]

    writer = open(output_file, "a+")
    for _, row in data.iterrows():
        _prompt = prompt_text if prompt_text is not None else row[prompt_col]
        _style_prompt = (
            style_prompt if style_prompt is not None else row[style_prompt_col]
        )
        image = row[image_col]
        raw_image = Image.open(image).convert("RGB")

        image_buffer = io.BytesIO()
        raw_image.save(image_buffer, format="JPEG")
        image_buffer.seek(0)

        try:
            response = client.post(
                path="/images/replaceBackground",
                cast_to=object,
                options={"headers": {"Content-Type": "multipart/form-data"}},
                files={
                    "image": ("image.jpg", image_buffer, "image/jpeg"),
                },
                body={
                    "style": _style_prompt,
                    "prompt": _prompt,
                },
            )
            result = response["data"][0]["url"]
            record = {
                "prompt": _prompt,
                "style": _style_prompt,
                "image": image,
                "result": save_image_from_url(cache_dir, result),
            }
            writer.write(json.dumps(record) + "\n")
            writer.flush()
        except:
            pass


# outpainting processing images
def __out_processing_image(image, ratio):
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    assert ratio in [0.5, 1.0, 2.0]
    size_dict = {"1.0": (768, 768), "0.5": (512, 1024), "2.0": (1024, 512)}
    # size_dict = {"1.0": (1024, 1024), "0.5": (768, 1536), "2.0": (1536, 768)}

    width, height = image.size
    size = size_dict[str(ratio)]

    while width > size[0] or height > size[1]:
        image = image.resize((width // 2, height // 2))
        width = width // 2
        height = height // 2

    im_width, im_height = image.size

    mask = Image.new("L", (size[0], size[1]), 255)
    black = Image.new("RGB", (im_width, im_height), (0, 0, 0))
    mask.paste(black, ((size[0] - im_width) // 2, (size[1] - im_height) // 2))
    new_image = Image.new("RGB", (size[0], size[1]), (255, 255, 255))
    new_image.paste(image, ((size[0] - im_width) // 2, (size[1] - im_height) // 2))

    return new_image, mask


def __out_processing1_image(image, ratio):
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    assert ratio in [0.5, 1.0, 2.0]
    size_dict = {"1.0": (768, 768), "0.5": (512, 1024), "2.0": (1024, 512)}
    # size_dict = {"1.0": (1024, 1024), "0.5": (768, 1536), "2.0": (1536, 768)}

    width, height = image.size
    size = size_dict[str(ratio)]
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


def outpainting(
    token: str,
    data_file: str,
    cache_dir: str,
    names: Union[str, List[str]],
    image_col: str,
    prompt_col: Optional[str] = None,
    prompt_text: Optional[str] = None,
    style_prompt_col: Optional[str] = None,
    style_prompt: Optional[str] = "realistic_image",
    processor_name: Optional[str] = "default",
    ratios: Optional[Union[str, List[float]]] = [0.5, 1.0, 2.0],
):
    client = OpenAI(
        base_url="https://external.api.recraft.ai/v1",
        api_key=token,
    )

    if isinstance(names, str) and names.strip() == "*":
        names = None
    if isinstance(names, str):
        names = re.split(r"[,;]", names)
        names = [n.strip() for n in names]

    data = pd.read_csv(
        data_file,
        names=names,
        sep="\t",
        quoting=3,
        header=None,
    )

    processors = {
        "default": __out_processing_image,
        "p1": __out_processing1_image,
    }

    if isinstance(ratios, str):
        ratios = re.split(r"[,;]", ratios)
        ratios = [float(n.strip()) for n in ratios]

    assert processor_name in processors.keys()

    process_func = processors[processor_name]

    os.makedirs(cache_dir, exist_ok=True)

    assert image_col in data.columns, f"Column {image_col} not found in data."

    output_file = f"{cache_dir}/output.jsonl"

    if os.path.exists(output_file):
        uniques = []
        with open(output_file, "r") as f:
            for line in f:
                row = json.loads(line)
                uniques.append(
                    row["prompt"] + " - " + row["style"] + " - " + row["image"]
                )
        data = data[
            ~data.apply(
                lambda x: (prompt_text if prompt_text is not None else x[prompt_col])
                + " - "
                + (style_prompt if style_prompt is not None else x[style_prompt_col])
                + " - "
                + x[image_col]
                in uniques,
                axis=1,
            )
        ]

    writer = open(output_file, "a+")

    for _, row in data.iterrows():
        _prompt = prompt_text if prompt_text is not None else row[prompt_col]
        _style_prompt = (
            style_prompt if style_prompt is not None else row[style_prompt_col]
        )
        raw_image = row[image_col]
        record = {"prompt": _prompt, "image": row[image_col], "style": _style_prompt}
        for ratio in ratios:
            p_image, p_mask_image = process_func(raw_image, ratio)
            image_buffer = io.BytesIO()
            p_image.save(image_buffer, format="JPEG")
            mask_image_buffer = io.BytesIO()
            p_mask_image = p_mask_image.convert("L")
            p_mask_image = p_mask_image.point(lambda x: 255 if x > 127 else 0, "L")
            p_mask_image.save(mask_image_buffer, format="PNG")
            image_buffer.seek(0)
            mask_image_buffer.seek(0)
            try:
                response = client.post(
                    path="/images/inpaint",
                    cast_to=object,
                    options={"headers": {"Content-Type": "multipart/form-data"}},
                    files={
                        "image": ("image.jpg", image_buffer, "image/jpeg"),
                        "mask": ("mask_image.jpg", mask_image_buffer, "image/jpeg"),
                    },
                    body={
                        "style": _style_prompt,
                        "prompt": _prompt,
                    },
                )
                result = response["data"][0]["url"]
                record[f"result_{ratio}"] = save_image_from_url(cache_dir, result)
            except:
                pass
        writer.write(json.dumps(record) + "\n")
        writer.flush()


if __name__ == "__main__":
    fire.Fire()
