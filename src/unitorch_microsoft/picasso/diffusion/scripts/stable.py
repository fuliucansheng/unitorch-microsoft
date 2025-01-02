# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import re
import io
import fire
import torch
import json
import hashlib
import requests
import pandas as pd
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from multiprocessing import Process, Queue
from unitorch.models import GenericOutputs
from unitorch.utils import pop_value, nested_dict_value, read_file, read_json_file
from unitorch.cli import CoreConfigureParser
from unitorch.cli.fastapis.stable import (
    StableForText2ImageFastAPIPipeline,
    StableForImageInpaintingFastAPIPipeline,
)


def save_image(folder, image):
    name = hashlib.md5(image.tobytes()).hexdigest() + ".jpg"
    image.save(f"{folder}/{name}")
    return f"{folder}/{name}"


def text2image(
    data_file: str,
    cache_dir: str,
    names: Union[str, List[str]],
    prompt_col: str,
    pretrained_name: Optional[str] = "stable-v1.5-realistic-v5.1",
    neg_prompt_text: Optional[str] = None,
    height: Optional[int] = 512,
    width: Optional[int] = 512,
    guidance_scale: Optional[float] = 7.5,
    num_timesteps: Optional[int] = 50,
    seed: Optional[int] = 1123,
    pad_token: Optional[str] = "<|endoftext|>",
    weight_path: Optional[str] = None,
    lora_weight_path: Optional[str] = None,
    lora_weight: Optional[float] = 1.0,
    lora_alpha: Optional[float] = 32.0,
    device: Optional[Union[str, int]] = "cpu",
):
    pipe = StableForText2ImageFastAPIPipeline.from_core_configure(
        config=CoreConfigureParser(),
        pretrained_name=pretrained_name,
        pad_token=pad_token,
        pretrained_weight_path=weight_path,
        pretrained_lora_weights_path=lora_weight_path,
        pretrained_lora_weights=lora_weight,
        pretrained_lora_alphas=lora_alpha,
        device=device,
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
        prompts = []
        with open(output_file, "r") as f:
            for line in f:
                prompts.append(json.loads(line)["prompt"])
        data = data[~data[prompt_col].isin(prompts)]

    writer = open(output_file, "a+")

    if neg_prompt_text is not None:
        for prompt in data[prompt_col]:
            result = pipe(
                prompt,
                neg_prompt_text,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_timesteps=num_timesteps,
                seed=seed,
            )
            record = {
                "prompt": prompt,
                "result": save_image(cache_dir, result),
            }
            writer.write(json.dumps(record) + "\n")
            writer.flush()
    else:
        for prompt in data[prompt_col]:
            result = pipe(
                prompt,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_timesteps=num_timesteps,
                seed=seed,
            )
            record = {
                "prompt": prompt,
                "result": save_image(cache_dir, result),
            }
            writer.write(json.dumps(record) + "\n")
            writer.flush()


# inpainting processing images
def __in_processing_image(image, mask_image):
    if isinstance(image, str):
        image = Image.open(image)
    if isinstance(mask_image, str):
        mask_image = Image.open(mask_image)

    # process your image/mask_image here

    return image, mask_image


def inpainting(
    data_file: str,
    cache_dir: str,
    names: Union[str, List[str]],
    image_col: str,
    mask_image_col: str,
    prompt_col: Optional[str] = None,
    prompt_text: Optional[str] = None,
    pretrained_name: Optional[str] = "stable-v1.5-realistic-v5.1-inpainting",
    neg_prompt_text: Optional[str] = None,
    guidance_scale: Optional[float] = 7.5,
    num_timesteps: Optional[int] = 50,
    seed: Optional[int] = 1123,
    pad_token: Optional[str] = "<|endoftext|>",
    weight_path: Optional[str] = None,
    lora_weight_path: Optional[str] = None,
    lora_weight: Optional[float] = 1.0,
    lora_alpha: Optional[float] = 32.0,
    device: Optional[Union[str, int]] = "cpu",
    processor_name: Optional[str] = "default",
):
    pipe = StableForImageInpaintingFastAPIPipeline.from_core_configure(
        config=CoreConfigureParser(),
        pretrained_name=pretrained_name,
        pad_token=pad_token,
        pretrained_weight_path=weight_path,
        pretrained_lora_weights_path=lora_weight_path,
        pretrained_lora_weights=lora_weight,
        pretrained_lora_alphas=lora_alpha,
        device=device,
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
                    row["prompt"] + " - " + row["image"] + " - " + row["mask_image"]
                )
        data = data[
            ~data.apply(
                lambda x: (prompt_text if prompt_text is not None else x[prompt_col])
                + " - "
                + x[image_col]
                + " - "
                + x[mask_image_col]
                in uniques,
                axis=1,
            )
        ]

    writer = open(output_file, "a+")

    if neg_prompt_text is not None:
        for _, row in data.iterrows():
            prompt = prompt_text if prompt_text is not None else row[prompt_col]
            image = row[image_col]
            mask_image = row[mask_image_col]
            p_image, p_mask_image = process_func(image, mask_image)
            result = pipe(
                prompt,
                p_image,
                p_mask_image,
                neg_prompt_text,
                guidance_scale=guidance_scale,
                num_timesteps=num_timesteps,
                seed=seed,
            )
            record = {
                "prompt": prompt,
                "image": image,
                "mask_image": mask_image,
                "result": save_image(cache_dir, result),
            }
            writer.write(json.dumps(record) + "\n")
            writer.flush()
    else:
        for _, row in data.iterrows():
            prompt = prompt_text if prompt_text is not None else row[prompt_col]
            image = row[image_col]
            mask_image = row[mask_image_col]
            p_image, p_mask_image = process_func(image, mask_image)
            result = pipe(
                prompt,
                p_image,
                p_mask_image,
                guidance_scale=guidance_scale,
                num_timesteps=num_timesteps,
                seed=seed,
            )
            record = {
                "prompt": prompt,
                "image": image,
                "mask_image": mask_image,
                "result": save_image(cache_dir, result),
            }
            writer.write(json.dumps(record) + "\n")
            writer.flush()


# outpainting processing images
def __out_processing_image(image, ratio):
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    assert ratio in [0.5, 1, 2]
    size_dict = {"1": (768, 768), "0.5": (512, 1024), "2": (1024, 512)}

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
    assert ratio in [0.5, 1, 2]
    size_dict = {"1": (768, 768), "0.5": (512, 1024), "2": (1024, 512)}

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
    data_file: str,
    cache_dir: str,
    names: Union[str, List[str]],
    image_col: str,
    prompt_col: Optional[str] = None,
    prompt_text: Optional[str] = None,
    pretrained_name: Optional[str] = "stable-v1.5-realistic-v5.1-inpainting",
    neg_prompt_text: Optional[str] = None,
    guidance_scale: Optional[float] = 7.5,
    num_timesteps: Optional[int] = 50,
    seed: Optional[int] = 1123,
    pad_token: Optional[str] = "<|endoftext|>",
    weight_path: Optional[str] = None,
    lora_weight_path: Optional[str] = None,
    lora_weight: Optional[float] = 1.0,
    lora_alpha: Optional[float] = 32.0,
    device: Optional[Union[str, int]] = "cpu",
    processor_name: Optional[str] = "default",
):
    pipe = StableForImageInpaintingFastAPIPipeline.from_core_configure(
        config=CoreConfigureParser(),
        pretrained_name=pretrained_name,
        pad_token=pad_token,
        pretrained_weight_path=weight_path,
        pretrained_lora_weights_path=lora_weight_path,
        pretrained_lora_weights=lora_weight,
        pretrained_lora_alphas=lora_alpha,
        device=device,
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
                uniques.append(row["prompt"] + " - " + row["image"])
        data = data[
            ~data.apply(
                lambda x: (prompt_text if prompt_text is not None else x[prompt_col])
                + " - "
                + x[image_col]
                in uniques,
                axis=1,
            )
        ]

    writer = open(output_file, "a+")

    if neg_prompt_text is not None:
        for _, row in data.iterrows():
            prompt = prompt_text if prompt_text is not None else row[prompt_col]
            raw_image = row[image_col]
            record = {
                "prompt": prompt,
                "image": row[image_col],
            }
            for ratio in [0.5, 1, 2]:
                p_image, p_mask_image = process_func(raw_image, ratio)
                result = pipe(
                    prompt,
                    p_image,
                    p_mask_image,
                    neg_prompt_text,
                    guidance_scale=guidance_scale,
                    num_timesteps=num_timesteps,
                    seed=seed,
                )
                record[f"result_{ratio}"] = save_image(cache_dir, result)
            writer.write(json.dumps(record) + "\n")
            writer.flush()
    else:
        for _, row in data.iterrows():
            prompt = prompt_text if prompt_text is not None else row[prompt_col]
            raw_image = row[image_col]
            record = {
                "prompt": prompt,
                "image": row[image_col],
            }
            for ratio in [0.5, 1, 2]:
                p_image, p_mask_image = process_func(raw_image, ratio)
                result = pipe(
                    prompt,
                    p_image,
                    p_mask_image,
                    guidance_scale=guidance_scale,
                    num_timesteps=num_timesteps,
                    seed=seed,
                )
                record[f"result_{ratio}"] = save_image(cache_dir, result)
            writer.write(json.dumps(record) + "\n")
            writer.flush()


if __name__ == "__main__":
    fire.Fire()
