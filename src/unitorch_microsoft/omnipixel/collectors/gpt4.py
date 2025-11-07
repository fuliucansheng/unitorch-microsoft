# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import re
import fire
import json
import hashlib
import pandas as pd
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch_microsoft.externals.papyrus import (
    get_gpt_image_response,
    get_gpt5_response,
)


def save_image(folder, image):
    name = hashlib.md5(image.tobytes()).hexdigest() + ".jpg"
    image.save(f"{folder}/{name}")
    return f"{folder}/{name}"


def text_generate(
    data_file: str,
    cache_dir: str,
    names: Optional[Union[str, List[str]]] = None,
    image_cols: Optional[Union[str, List[str]]] = None,
    prompt_col: Optional[str] = None,
    prompt_text: Optional[str] = None,
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

    if isinstance(image_cols, str):
        image_cols = re.split(r"[,;]", image_cols)
        image_cols = [n.strip() for n in image_cols]

    os.makedirs(cache_dir, exist_ok=True)

    output_file = f"{cache_dir}/output.jsonl"

    if os.path.exists(output_file):
        uniques = []
        with open(output_file, "r") as f:
            for line in f:
                row = json.loads(line)
                uniques.append(row["prompt"] + " - " + row.get("images", ""))
        data = data[
            ~data.apply(
                lambda x: (prompt_text if prompt_text is not None else x[prompt_col])
                + " - "
                + ",".join(
                    [x[col] for col in image_cols] if image_cols is not None else ""
                )
                in uniques,
                axis=1,
            )
        ]

    writer = open(output_file, "a")
    for _, row in data.iterrows():
        prompt = prompt_text if prompt_text is not None else row[prompt_col]
        images = [row[col] for col in image_cols] if image_cols is not None else None
        result = get_gpt5_response(
            prompt=prompt,
            images=(
                [Image.open(image) for image in images] if images is not None else None
            ),
        )
        if result != "":
            writer.write(
                json.dumps(
                    {
                        "prompt": prompt,
                        "images": ",".join(images) if images is not None else "",
                        "result": result,
                    }
                )
                + "\n"
            )
            writer.flush()


def image_generate(
    data_file: str,
    cache_dir: str,
    names: Optional[Union[str, List[str]]] = None,
    image_cols: Optional[Union[str, List[str]]] = None,
    prompt_col: Optional[str] = None,
    prompt_text: Optional[str] = None,
    size: Optional[str] = "1024x1024",
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

    if isinstance(image_cols, str):
        image_cols = re.split(r"[,;]", image_cols)
        image_cols = [n.strip() for n in image_cols]

    os.makedirs(cache_dir, exist_ok=True)

    output_file = f"{cache_dir}/output.jsonl"

    if os.path.exists(output_file):
        uniques = []
        with open(output_file, "r") as f:
            for line in f:
                row = json.loads(line)
                uniques.append(row["prompt"] + " - " + row.get("images", ""))
        data = data[
            ~data.apply(
                lambda x: (prompt_text if prompt_text is not None else x[prompt_col])
                + " - "
                + ",".join(
                    [x[col] for col in image_cols] if image_cols is not None else ""
                )
                in uniques,
                axis=1,
            )
        ]

    writer = open(output_file, "a")
    for _, row in data.iterrows():
        prompt = prompt_text if prompt_text is not None else row[prompt_col]
        images = [row[col] for col in image_cols] if image_cols is not None else None
        try:
            result = get_gpt_image_response(
                prompt=prompt,
                images=(
                    [Image.open(image) for image in images]
                    if images is not None
                    else None
                ),
                size=size,
            )
            if isinstance(result, Image.Image):
                writer.write(
                    json.dumps(
                        {
                            "prompt": prompt,
                            "images": ",".join(images) if images is not None else "",
                            "size": size,
                            "result": save_image(cache_dir, result),
                        }
                    )
                    + "\n"
                )
                writer.flush()
        except Exception as e:
            print(
                f"Error generating image for prompt '{prompt} and images {images}': {e}"
            )
            continue


if __name__ == "__main__":
    fire.Fire(
        {
            "text_generate": text_generate,
            "image_generate": image_generate,
        }
    )
