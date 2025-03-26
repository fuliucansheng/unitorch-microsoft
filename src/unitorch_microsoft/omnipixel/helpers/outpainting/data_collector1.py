# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import re
import cv2
import io
import fire
import torch
import json
import logging
import hashlib
import requests
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from multiprocessing import Process, Queue
from unitorch.models import GenericOutputs
from unitorch.utils import pop_value, nested_dict_value, read_file, read_json_file
from unitorch.cli import CoreConfigureParser
import unitorch_microsoft.models.diffusers

endpoints = [
    # "http://br1t44-s3-17:5050/core/fastapi/stable_flux",
    # "http://br1t44-s3-17:5051/core/fastapi/stable_flux",
    "http://10.224.120.163:5050/core/fastapi/stable_flux",
    "http://10.224.120.163:5051/core/fastapi/stable_flux",
    # "http://br1u43-s2-01:5050/core/fastapi/stable_flux",
    # "http://br1u43-s2-01:5051/core/fastapi/stable_flux",
    "http://10.224.120.219:5050/core/fastapi/stable_flux",
    "http://10.224.120.219:5051/core/fastapi/stable_flux",
    # "http://br1t43-s3-25.guest.corp.microsoft.com:5050/core/fastapi/stable_flux",
    # "http://br1t43-s3-25.guest.corp.microsoft.com:5051/core/fastapi/stable_flux",
    "http://10.224.120.67:5050/core/fastapi/stable_flux",
    "http://10.224.120.67:5051/core/fastapi/stable_flux",
    # "http://br1t45-s1-01:5050/core/fastapi/stable_flux",
    # "http://br1t45-s1-01:5051/core/fastapi/stable_flux",
    "http://10.224.120.184:5050/core/fastapi/stable_flux",
    # "http://10.224.120.184:5051/core/fastapi/stable_flux",
    # "http://br1t43-s3-17.guest.corp.microsoft.com:5050/core/fastapi/stable_flux",
    # "http://br1t43-s3-17.guest.corp.microsoft.com:5051/core/fastapi/stable_flux",
    "http://10.224.120.81:5050/core/fastapi/stable_flux",
    "http://10.224.120.81:5051/core/fastapi/stable_flux",
]


def save_image(folder, image):
    name = hashlib.md5(image.tobytes()).hexdigest() + ".jpg"
    image.save(f"{folder}/{name}")
    return f"{folder}/{name}"


def __out_processing_image(image):
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    width, height = image.size
    size = (1024, 1024)

    while width >= size[0] or height >= size[1]:
        image = image.resize((width // 2, height // 2), resample=Image.LANCZOS)
        width = width // 2
        height = height // 2

    im_width, im_height = image.size

    mask = Image.new("L", (size[0], size[1]), 255)
    black = Image.new("RGB", (im_width, im_height), (0, 0, 0))
    mask.paste(black, ((size[0] - im_width) // 2, (size[1] - im_height) // 2))
    new_image = Image.new("RGB", (size[0], size[1]), (255, 255, 255))
    new_image.paste(image, ((size[0] - im_width) // 2, (size[1] - im_height) // 2))

    return new_image, mask


def main(
    data_file: str,
    cache_dir: str,
    image_col: str,
    names: Union[str, List[str]],
    prompt_col: Optional[str] = None,
    prompt_text: Optional[str] = None,
    pretrained_name: Optional[str] = "stable-flux-dev-fill",
    guidance_scale: Optional[float] = 30.0,
    num_timesteps: Optional[int] = 50,
    seed: Optional[int] = 1123,
    lora_name: Optional[str] = None,
    lora_weight: Optional[float] = 1.0,
    lora_alpha: Optional[float] = 32.0,
    strength: Optional[float] = 0.95,
    processor_name: Optional[str] = "default",
    force_restart: Optional[bool] = False,
    force_stop: Optional[bool] = False,
    do_opencv_inpainting: Optional[bool] = True,
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

    def call_api(endpoint, part, Q):
        status = requests.get(endpoint + "/status").json()
        if status != "running" or force_restart:
            headers = {"Content-type": "application/json"}
            requests.post(
                endpoint + "/start",
                timeout=1200,
                params={
                    "pretrained_name": pretrained_name,
                },
                data=json.dumps(
                    {
                        "pretrained_lora_names": lora_name,
                        "pretrained_lora_weights": lora_weight,
                        "pretrained_lora_alphas": lora_alpha,
                    }
                ),
                headers=headers,
            )
        for _, row in part.iterrows():
            prompt = prompt_text if prompt_text is not None else row[prompt_col]
            raw_image = Image.open(row[image_col])

            record = {
                "prompt": prompt,
                "image": row[image_col],
            }

            image, mask_image = process_func(raw_image)
            if do_opencv_inpainting:
                image_np = np.array(image.convert("RGB"))
                mask_np = np.array(mask_image.convert("L")).astype(np.uint8)

                _, binary_mask = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
                inpainted_image = cv2.inpaint(
                    image_np, binary_mask, 10, cv2.INPAINT_TELEA
                )
                image = Image.fromarray(inpainted_image)
            image_buffer = io.BytesIO()
            image.save(image_buffer, format="JPEG")
            mask_image_buffer = io.BytesIO()
            mask_image.save(mask_image_buffer, format="JPEG")
            image_buffer.seek(0)
            mask_image_buffer.seek(0)

            files = {
                "image": ("image.jpg", image_buffer, "image/jpeg"),
                "mask_image": (
                    "mask_image.jpg",
                    mask_image_buffer,
                    "image/jpeg",
                ),
            }

            response = requests.post(
                endpoint + "/generate",
                params={
                    "text": prompt,
                    "guidance_scale": guidance_scale,
                    "num_timesteps": num_timesteps,
                    "seed": seed,
                    "strength": strength,
                },
                files=files,
            )
            result = Image.open(io.BytesIO(response.content))

            record[f"result"] = save_image(cache_dir, result)
            Q.put(record)
        Q.put("Done")
        if force_stop:
            requests.get(endpoint + "/stop")

    def write_file(fpath, Q, cnt):
        f = open(fpath, "a+")
        done = 0
        while True:
            item = Q.get()
            if item == "Done":
                done += 1
                if done == cnt:
                    break
            else:
                f.write(json.dumps(item) + "\n")
                f.flush()

    api_endpoints = [f"{endpoint}/inpainting" for endpoint in endpoints]
    num_processes = len(api_endpoints)
    data_parts = []
    for i in range(num_processes):
        data_parts.append(data.iloc[i::num_processes])

    processes = []
    queue = Queue()
    for i in range(num_processes):
        p = Process(
            target=call_api,
            args=(
                api_endpoints[i],
                data_parts[i],
                queue,
            ),
        )
        processes.append(p)

    io_process = Process(target=write_file, args=(output_file, queue, num_processes))
    processes.append(io_process)

    for p in processes:
        p.start()

    # wait for all processes to finish
    for p in processes:
        p.join()


if __name__ == "__main__":
    fire.Fire(main)
