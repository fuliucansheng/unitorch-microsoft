# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import re
import io
import fire
import torch
import json
import time
import queue
import base64
import threading
import logging
import hashlib
import requests
import pandas as pd
from PIL import Image, ImageOps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torch.hub import download_url_to_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(threadName)s - %(levelname)s - %(message)s",
)


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
    height: Optional[int] = 1024,
    width: Optional[int] = 1024,
    prompt_upsampling: Optional[bool] = False,
    max_queue_size: Optional[int] = 10,
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

    os.makedirs(cache_dir, exist_ok=True)

    assert prompt_col in data.columns, f"Column {prompt_col} not found in data."

    output_file = f"{cache_dir}/output.jsonl"

    if os.path.exists(output_file):
        uniques = []
        with open(output_file, "r") as f:
            for line in f:
                row = json.loads(line)
                uniques.append(row["prompt"])
        data = data[
            ~data.apply(
                lambda x: x[prompt_col] in uniques,
                axis=1,
            )
        ]

    writer = open(output_file, "a+")
    Q = queue.Queue(maxsize=max_queue_size)

    def producer():
        for _, row in data.iterrows():
            while Q.full():
                time.sleep(2)

            _prompt = row[prompt_col]
            try:
                headers = {
                    "X-Key": token,
                    "Content-type": "application/json",
                }
                response = requests.post(
                    "https://api.us1.bfl.ai/v1/flux-pro-1.1",
                    timeout=60,
                    json={
                        "prompt": _prompt,
                        "width": width,
                        "height": height,
                        "prompt_upsampling": prompt_upsampling,
                        "output_format": "jpeg",
                    },
                    headers=headers,
                ).json()

                Q.put(response["id"])
            except:
                pass
        Q.put("Done")

    def consumer():
        is_produder_done = False
        while True:
            if is_produder_done and Q.empty():
                break
            trackid = Q.get()
            if trackid == "Done":
                is_produder_done = True
                continue

            try:
                response = requests.get(
                    "https://api.us1.bfl.ai/v1/get_result",
                    headers={
                        "X-Key": token,
                    },
                    params={
                        "id": trackid,
                    },
                ).json()

                if response["status"] == "Ready":
                    result = response["result"]["sample"]
                    _prompt = response["result"]["prompt"]
                    record = {
                        "prompt": _prompt,
                        "result": save_image_from_url(cache_dir, result),
                    }
                    writer.write(json.dumps(record) + "\n")
                    writer.flush()
                else:
                    Q.put(trackid)
                    time.sleep(2)
            except:
                pass

    producer_thread = threading.Thread(target=producer)
    consumer_thread = threading.Thread(target=consumer)

    producer_thread.start()
    consumer_thread.start()

    producer_thread.join()
    consumer_thread.join()


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
    reversed_mask: Optional[str] = False,
    processor_name: Optional[str] = "default",
    prompt_upsampling: Optional[bool] = False,
    max_queue_size: Optional[int] = 100,
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
    Q = queue.Queue(maxsize=max_queue_size)

    def producer():
        for _, row in data.iterrows():
            while Q.qsize() >= max_queue_size - 1:
                time.sleep(2)

            _prompt = prompt_text if prompt_text is not None else row[prompt_col]
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
                    "X-Key": token,
                    "Content-type": "application/json",
                }
                response = requests.post(
                    "https://api.us1.bfl.ai/v1/flux-pro-1.0-fill",
                    timeout=60,
                    json={
                        "prompt": _prompt,
                        "image": base64.b64encode(image_buffer.getvalue()).decode(),
                        "mask": base64.b64encode(mask_image_buffer.getvalue()).decode(),
                        "prompt_upsampling": prompt_upsampling,
                        "output_format": "jpeg",
                    },
                    headers=headers,
                ).json()

                Q.put((response["id"], _prompt, image, mask_image))
            except:
                pass
        Q.put(("Done", None, None, None))

    def consumer():
        is_produder_done = False
        while True:
            if is_produder_done and Q.empty():
                break
            trackid, prompt, image, mask_image = Q.get()

            if trackid == "Done":
                is_produder_done = True
                continue

            try:
                response = requests.get(
                    "https://api.us1.bfl.ai/v1/get_result",
                    headers={
                        "X-Key": token,
                    },
                    params={
                        "id": trackid,
                    },
                ).json()

                if response["status"] == "Ready":
                    result = response["result"]["sample"]
                    record = {
                        "prompt": prompt,
                        "image": image,
                        "mask_image": mask_image,
                        "result": save_image_from_url(cache_dir, result),
                    }
                    writer.write(json.dumps(record) + "\n")
                    writer.flush()
                elif response["status"] == "Pending":
                    Q.put((trackid, prompt, image, mask_image))
                    time.sleep(1)
                else:
                    logging.warning(
                        f"TrackId: {trackid} - Prompt: {prompt} - Image: {image} - Mask: {mask_image} - Status: {response['status']}"
                    )
            except:
                pass

    producer_thread = threading.Thread(target=producer)
    consumer_thread = threading.Thread(target=consumer)

    producer_thread.start()
    consumer_thread.start()

    producer_thread.join()
    consumer_thread.join()


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
    processor_name: Optional[str] = "default",
    prompt_upsampling: Optional[bool] = False,
    max_queue_size: Optional[int] = 10,
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
    Q = queue.Queue(maxsize=max_queue_size)

    def producer():
        for _, row in data.iterrows():
            while Q.full():
                time.sleep(2)

            _prompt = prompt_text if prompt_text is not None else row[prompt_col]
            raw_image = row[image_col]
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
                        "X-Key": token,
                        "Content-type": "application/json",
                    }
                    response = requests.post(
                        "https://api.us1.bfl.ai/v1/flux-pro-1.0-fill",
                        timeout=60,
                        json={
                            "prompt": _prompt,
                            "image": base64.b64encode(image_buffer.getvalue()).decode(),
                            "mask": base64.b64encode(
                                mask_image_buffer.getvalue()
                            ).decode(),
                            "prompt_upsampling": prompt_upsampling,
                            "output_format": "jpeg",
                        },
                        headers=headers,
                    ).json()

                    Q.put((response["id"], raw_image, ratio))
                except:
                    pass
        Q.put(("Done", None, None))

    def consumer():
        is_produder_done = False
        while True:
            if is_produder_done and Q.empty():
                break
            trackid, image, ratio = Q.get()
            if trackid == "Done":
                is_produder_done = True
                continue

            try:
                response = requests.get(
                    "https://api.us1.bfl.ai/v1/get_result",
                    headers={
                        "X-Key": token,
                    },
                    params={
                        "id": trackid,
                    },
                ).json()

                if response["status"] == "Ready":
                    result = response["result"]["sample"]
                    _prompt = response["result"]["prompt"]
                    record = {
                        "prompt": _prompt,
                        "image": image,
                        "ratio": ratio,
                        "result": save_image_from_url(cache_dir, result),
                    }
                    writer.write(json.dumps(record) + "\n")
                    writer.flush()
                else:
                    Q.put((trackid, image, ratio))
                    time.sleep(2)
            except:
                pass

    producer_thread = threading.Thread(target=producer)
    consumer_thread = threading.Thread(target=consumer)

    producer_thread.start()
    consumer_thread.start()

    producer_thread.join()
    consumer_thread.join()


if __name__ == "__main__":
    fire.Fire()
