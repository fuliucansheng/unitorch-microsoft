# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import time
import os
import re
import io
import fire
import torch
import json
import hashlib
import requests
import pandas as pd
from PIL import Image, ImageOps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from PIL import Image
from google import genai

gemini_clients = [
    genai.Client(api_key="Your API Key Here"),
]

os.makedirs("gemini", exist_ok=True)


def get_gemini_client():
    idx = int(time.time()) % len(gemini_clients)
    return gemini_clients[idx]


def save_image(folder, image):
    name = hashlib.md5(image.tobytes()).hexdigest() + ".jpg"
    image.save(f"{folder}/{name}")
    return f"{folder}/{name}"


def outpainting(
    data_file: str,
    cache_dir: str,
    names: Union[str, List[str]],
    image_col: str,
    prompt_col: Optional[str] = None,
    prompt_text: Optional[str] = None,
    template_image: Optional[str] = None,
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

    output_file = f"{cache_dir}/output.jsonl"

    if os.path.exists(output_file):
        uniques = []
        with open(output_file, "r") as f:
            for line in f:
                row = json.loads(line)
                uniques.append(
                    row[image_col]
                    + " - "
                    + (row[prompt_col] if prompt_col is not None else "")
                )
        data = data[
            ~data.apply(
                lambda x: x[image_col]
                + " - "
                + (row[prompt_col] if prompt_col is not None else "")
                in uniques,
                axis=1,
            )
        ]

    writer = open(output_file, "a+")

    for _, row in data.iterrows():
        _prompt = row[prompt_col] if prompt_col is not None else prompt_text
        _image = Image.open(row[image_col]).convert("RGB")
        try:
            resp = get_gemini_client().models.generate_content(
                model="gemini-2.5-flash-image-preview",
                contents=[
                    _image,
                    (
                        template_image
                        if template_image is not None
                        else Image.new("RGB", (1910, 1000), (0, 0, 0))
                    ),
                    _prompt,
                ],
            )
            image_parts = [
                part.inline_data.data
                for part in resp.candidates[0].content.parts
                if part.inline_data
            ]
            result = Image.open(io.BytesIO(image_parts[0]))
            record = {
                "prompt": _prompt,
                "image": row[image_col],
                "result": save_image(cache_dir, result),
            }
            writer.write(json.dumps(record) + "\n")
            writer.flush()
        except KeyboardInterrupt:
            break
        except Exception as e:
            pass

    writer.close()


if __name__ == "__main__":
    fire.Fire(
        {
            "outpainting": outpainting,
        }
    )
