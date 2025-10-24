# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import io
import cv2
import re
import gc
import fire
import torch
import hashlib
import requests
import logging
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torchvision import transforms
from unitorch import mktempfile
from unitorch.utils import read_file
from unitorch.models import GenericOutputs
from unitorch_microsoft import cached_path

FASTAPI_ENDPOINT = None


def save_image(folder, image):
    name = hashlib.md5(image.tobytes()).hexdigest() + ".jpg"
    image.save(f"{folder}/{name}")
    return f"{folder}/{name}"


def call_fastapi(url, params={}, images=None, req_type="POST", resp_type="json"):
    assert resp_type in ["json", "image"], f"Unsupported response type: {resp_type}"

    def process_image(image):
        image = image.convert("RGB")
        byte_arr = io.BytesIO()
        image.save(byte_arr, format="JPEG")
        byte_arr.seek(0)
        return byte_arr

    if images is None:
        files = {}
    else:
        files = {
            k: (f"{k}.jpg", process_image(v), "image/jpeg") for k, v in images.items()
        }
    if req_type == "POST" or images is not None:
        resp = (
            requests.post(url, params=params, files=files)
            if images is not None
            else requests.post(url, params=params)
        )
    else:
        resp = requests.get(url, params=params)
    if resp_type == "json":
        result = resp.json()
    elif resp_type == "image":
        result = Image.open(io.BytesIO(resp.content)).convert("RGB")
    return result


def get_quality_score(image):
    result = call_fastapi(
        FASTAPI_ENDPOINT + "/microsoft/spaces/fastapi/siglip2/generate1",
        images={
            "image": image,
        },
    )
    return result["Bad Cropped"]


def get_click_score(image):
    result = call_fastapi(
        FASTAPI_ENDPOINT
        + "/microsoft/spaces/fastapi/bletchley/msan/image_click/generate1",
        images={
            "image": image,
        },
    )
    return result


def get_aesthetics_score(image):
    result = call_fastapi(
        FASTAPI_ENDPOINT + "/microsoft/spaces/fastapi/bletchley/v3/generate2",
        images={
            "image": image,
        },
    )
    return result["Bad Aesthetics"]


def generate(image, ratio, step):
    w, h = image.size
    if w / h > ratio:
        crop_w, crop_h = int(h * ratio), h
    else:
        crop_w, crop_h = w, int(w / ratio)
    step_w, step_h = int(crop_w * step), int(crop_h * step)
    all_crops = []
    for i in range(0, w - crop_w + 1, step_w):
        for j in range(0, h - crop_h + 1, step_h):
            box = (i, j, i + crop_w, j + crop_h)
            cropped_image = image.crop(box)
            quality_score = get_quality_score(cropped_image)
            click_score = get_click_score(cropped_image)
            aesthetics_score = get_aesthetics_score(cropped_image)
            all_crops.append(
                {
                    "image": cropped_image,
                    "box": box,
                    "quality_score": quality_score,
                    "click_score": click_score,
                    "aesthetics_score": aesthetics_score,
                }
            )
    crops = pd.DataFrame(all_crops)
    best_crop1 = crops.loc[crops["quality_score"].idxmin()]
    best_crop2 = crops.loc[crops["click_score"].idxmax()]
    best_crop3 = crops.loc[crops["aesthetics_score"].idxmin()]
    keep_crops = crops[crops["quality_score"] <= 0.45]
    if len(keep_crops) == 0:
        logging.warning("All crops have bad quality, select the best quality crop.")
        best_crop4 = crops.loc[crops["quality_score"].idxmin()]
        best_crop5 = crops.loc[crops["quality_score"].idxmin()]
    else:
        best_crop4 = keep_crops.loc[keep_crops["click_score"].idxmax()]
        best_crop5 = keep_crops.loc[keep_crops["aesthetics_score"].idxmin()]
    result1_image = image.crop(best_crop1["box"])
    result2_image = image.crop(best_crop2["box"])
    result3_image = image.crop(best_crop3["box"])
    result4_image = image.crop(best_crop4["box"])
    result5_image = image.crop(best_crop5["box"])
    return result1_image, result2_image, result3_image, result4_image, result5_image


def main(
    data_file: str,
    cache_dir: str,
    names: Union[str, List[str]],
    endpoint: str,
    image_col: str = None,
    ratio: float = None,
    step: float = 0.1,
    disp_image_col: str = None,
):
    global FASTAPI_ENDPOINT
    FASTAPI_ENDPOINT = endpoint

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

    assert image_col in data.columns, f"image_col {image_col} not in data columns."

    if disp_image_col is not None:
        assert (
            disp_image_col in data.columns
        ), f"disp_image_col {disp_image_col} not in data columns."

    assert (ratio is not None) or (
        disp_image_col is not None
    ), "Either ratio or disp_image_col should be provided."

    ratio2 = ratio
    results = []
    for idx, row in data.iterrows():
        image_path = row[image_col]
        image = Image.open(image_path).convert("RGB")
        if disp_image_col is not None:
            disp_image = Image.open(row[disp_image_col]).convert("RGB")
            ratio = disp_image.width / disp_image.height
        else:
            ratio = ratio2
        result_images = generate(image, ratio, step)
        saved_paths = []
        for result_image in result_images:
            saved_path = save_image(cache_dir, result_image)
            saved_paths.append(saved_path)
        results.append(saved_paths)
    result_df = pd.DataFrame(
        results,
        columns=[
            "best_quality_crop",
            "best_click_crop",
            "best_aesthetics_crop",
            "good_click_crop",
            "good_aesthetics_crop",
        ],
    )
    result_df = pd.concat([data, result_df], axis=1)
    result_df.to_csv(f"{cache_dir}/fit_probe_results.csv", index=False)


if __name__ == "__main__":
    fire.Fire(main)
