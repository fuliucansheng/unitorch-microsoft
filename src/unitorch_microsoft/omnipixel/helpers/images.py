# Copyright (c) MICROSOFT.
# Licensed under the MIT License.
import os
import re
import fire
import json
import logging
import hashlib
import subprocess
import tempfile
import pandas as pd
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union


def save_image(folder, image):
    name = hashlib.md5(image.tobytes()).hexdigest() + ".jpg"
    image.save(f"{folder}/{name}")
    return f"{folder}/{name}"


def crop_image(image, ratio, step, folder):
    w, h = image.size
    if w / h > ratio:
        crop_w, crop_h = int(h * ratio), h
    else:
        crop_w, crop_h = w, int(w / ratio)
    step_w, step_h = int(crop_w * step), int(crop_h * step)
    for i in range(0, w - crop_w + 1, step_w):
        for j in range(0, h - crop_h + 1, step_h):
            box = (i, j, i + crop_w, j + crop_h)
            cropped_image = image.crop(box)
            cropped_image.save(os.path.join(folder, f"crop_{i}_{j}.jpg"))

    return


def crop(
    data_file: str,
    cache_dir: str,
    names: Union[str, List[str]],
    image_col: str,
    ratio: float = 1.0,
    step: float = 0.1,
):
    os.makedirs(cache_dir, exist_ok=True)

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

    assert image_col in data.columns, f"Column {image_col} not found in data file"

    images = data[image_col].unique().tolist()
    results = []
    for image in images:
        im = Image.open(image).convert("RGB")
        w, h = im.size
        if w / h > ratio:
            crop_w, crop_h = int(h * ratio), h
        else:
            crop_w, crop_h = w, int(w / ratio)
        step_w, step_h = int(crop_w * step), int(crop_h * step)
        for i in range(0, w - crop_w + 1, step_w):
            for j in range(0, h - crop_h + 1, step_h):
                box = (i, j, i + crop_w, j + crop_h)
                cropped_image = im.crop(box)
                name = save_image(cache_dir, cropped_image)
                results.append((image, box, name))
    results = pd.DataFrame(results, columns=["image", "box", "cropped_image"])
    results.to_csv(
        os.path.join(cache_dir, "output.tsv"),
        index=False,
        header=None,
        sep="\t",
        quoting=3,
    )


if __name__ == "__main__":
    fire.Fire(
        {
            "crop": crop,
        }
    )
