# Copyright (c) MICROSOFT.
# Licensed under the MIT License.
import os
import re
import io
import fire
import torch
import json
import logging
import hashlib
import requests
import pandas as pd
from PIL import Image, ImageOps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torch.hub import download_url_to_file
from multiprocessing import Process, Queue

supported_image_ratios = [
    "ASPECT_10_16",
    "ASPECT_16_10",
    "ASPECT_9_16",
    "ASPECT_16_9",
    "ASPECT_3_2",
    "ASPECT_2_3",
    "ASPECT_4_3",
    "ASPECT_3_4",
    "ASPECT_1_1",
    "ASPECT_1_3",
    "ASPECT_3_1",
]

supported_style_types = [
    "AUTO",
    "GENERAL",
    "REALISTIC",
    "DESIGN",
    "RENDER_3D",
    "ANIME",
]

supported_image_sizes = [
    (512, 1536),
    (576, 1408),
    (576, 1472),
    (576, 1536),
    (640, 1024),
    (640, 1344),
    (640, 1408),
    (640, 1472),
    (640, 1536),
    (704, 1152),
    (704, 1216),
    (704, 1280),
    (704, 1344),
    (704, 1408),
    (704, 1472),
    (720, 1280),
    (736, 1312),
    (768, 1024),
    (768, 1088),
    (768, 1152),
    (768, 1216),
    (768, 1232),
    (768, 1280),
    (768, 1344),
    (832, 960),
    (832, 1024),
    (832, 1088),
    (832, 1152),
    (832, 1216),
    (832, 1248),
    (864, 1152),
    (896, 960),
    (896, 1024),
    (896, 1088),
    (896, 1120),
    (896, 1152),
    (960, 832),
    (960, 896),
    (960, 1024),
    (960, 1088),
    (1024, 640),
    (1024, 768),
    (1024, 832),
    (1024, 896),
    (1024, 960),
    (1024, 1024),
    (1088, 768),
    (1088, 832),
    (1088, 896),
    (1088, 960),
    (1120, 896),
    (1152, 704),
    (1152, 768),
    (1152, 832),
    (1152, 864),
    (1152, 896),
    (1216, 704),
    (1216, 768),
    (1216, 832),
    (1232, 768),
    (1248, 832),
    (1280, 704),
    (1280, 720),
    (1280, 768),
    (1280, 800),
    (1312, 736),
    (1344, 640),
    (1344, 704),
    (1344, 768),
    (1408, 576),
    (1408, 640),
    (1408, 704),
    (1472, 576),
    (1472, 640),
    (1472, 704),
    (1536, 512),
    (1536, 576),
    (1536, 640),
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
    style_type_col: Optional[str] = None,
    style_type: Optional[str] = "REALISTIC",
    height: Optional[int] = 1024,
    width: Optional[int] = 1024,
):
    assert (width, height) in supported_image_sizes, "Unsupported image size."

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
                + (style_type if style_type is not None else x[style_type_col])
                in uniques,
                axis=1,
            )
        ]

    writer = open(output_file, "a+")

    for _, row in data.iterrows():
        _prompt = row[prompt_col]
        _style_type = style_type if style_type is not None else row[style_type_col]
        try:
            headers = {
                "Api-Key": token,
                "Content-type": "application/json",
            }
            response = requests.post(
                "https://api.ideogram.ai/generate",
                timeout=60,
                json={
                    "image_request": {
                        "prompt": _prompt,
                        "resolution": f"RESOLUTION_{width}_{height}",
                        "model": "V_2",
                        "magic_prompt_option": "AUTO",
                        "style_type": _style_type,
                    },
                },
                headers=headers,
            ).json()
            result = response["data"][0]["url"]
            record = {
                "prompt": _prompt,
                "style": _style_type,
                "result": save_image_from_url(cache_dir, result),
                "result_is_image_safe": response["data"][0]["is_image_safe"],
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
    style_type_col: Optional[str] = None,
    style_type: Optional[str] = "REALISTIC",
    reversed_mask: Optional[str] = False,
    processor_name: Optional[str] = "default",
):
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
                + (style_type if style_type is not None else x[style_type_col])
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
        _style_type = style_type if style_type is not None else row[style_type_col]
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
            headers = {
                "Api-Key": token,
            }
            response = requests.post(
                "https://api.ideogram.ai/edit",
                files={
                    "image_file": ("image.jpg", image_buffer, "image/jpeg"),
                    "mask": ("mask_image.jpg", mask_image_buffer, "image/jpeg"),
                },
                data={
                    "prompt": _prompt,
                    "model": "V_2",
                    "magic_prompt_option": "AUTO",
                    "style_type": _style_type,
                },
                headers=headers,
            ).json()
            result = response["data"][0]["url"]
            record = {
                "prompt": _prompt,
                "style": _style_type,
                "image": image,
                "mask_image": mask_image,
                "result": save_image_from_url(cache_dir, result),
                "result_is_image_safe": response["data"][0]["is_image_safe"],
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
        image = image.resize((width // 2, height // 2), resample=Image.LANCZOS)
        width = width // 2
        height = height // 2

    im_width, im_height = image.size

    mask = Image.new("L", (size[0], size[1]), 255)
    black = Image.new("RGB", (im_width, im_height), (0, 0, 0))
    mask.paste(black, ((size[0] - im_width) // 2, (size[1] - im_height) // 2))
    new_image = Image.new("RGB", (size[0], size[1]), (255, 255, 255))
    new_image.paste(image, ((size[0] - im_width) // 2, (size[1] - im_height) // 2))

    return new_image, ImageOps.invert(mask)


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

    image = image.resize(
        (new_width // 8 * 8, new_height // 8 * 8), resample=Image.LANCZOS
    )

    im_width, im_height = image.size

    mask = Image.new("L", (size[0], size[1]), 255)
    black = Image.new("RGB", (im_width, im_height), (0, 0, 0))
    mask.paste(black, ((size[0] - im_width) // 2, (size[1] - im_height) // 2))
    new_image = Image.new("RGB", (size[0], size[1]), (255, 255, 255))
    new_image.paste(image, ((size[0] - im_width) // 2, (size[1] - im_height) // 2))

    return new_image, ImageOps.invert(mask)


def outpainting(
    token: str,
    data_file: str,
    cache_dir: str,
    names: Union[str, List[str]],
    image_col: str,
    prompt_col: Optional[str] = None,
    prompt_text: Optional[str] = None,
    style_type_col: Optional[str] = None,
    style_type: Optional[str] = "REALISTIC",
    processor_name: Optional[str] = "default",
    ratios: Optional[Union[str, List[float]]] = [0.5, 1.0, 2.0],
):
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
                + (style_type if style_type is not None else x[style_type_col])
                + " - "
                + x[image_col]
                in uniques,
                axis=1,
            )
        ]

    writer = open(output_file, "a+")

    for _, row in data.iterrows():
        _prompt = prompt_text if prompt_text is not None else row[prompt_col]
        _style_type = style_type if style_type is not None else row[style_type_col]
        raw_image = row[image_col]
        record = {"prompt": _prompt, "image": row[image_col], "style": _style_type}
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
                headers = {
                    "Api-Key": token,
                }
                response = requests.post(
                    "https://api.ideogram.ai/edit",
                    files={
                        "image_file": ("image.jpg", image_buffer, "image/jpeg"),
                        "mask": ("mask_image.jpg", mask_image_buffer, "image/jpeg"),
                    },
                    data={
                        "prompt": _prompt,
                        "model": "V_2",
                        "magic_prompt_option": "AUTO",
                        "style_type": _style_type,
                    },
                    headers=headers,
                ).json()
                result = response["data"][0]["url"]
                record[f"result_{ratio}"] = save_image_from_url(cache_dir, result)
                record[f"result_is_image_safe_{ratio}"] = response["data"][0][
                    "is_image_safe"
                ]
                writer.write(json.dumps(record) + "\n")
                writer.flush()
            except:
                pass
        writer.write(json.dumps(record) + "\n")
        writer.flush()


if __name__ == "__main__":
    fire.Fire()
