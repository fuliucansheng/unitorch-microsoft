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


def main(
    folder: str = "./results",
    prompt_col: str = "prompt",
    image_cols: str = "image",
    result_cols: str = "result",
    port: int = 12310,
    temp_folder: str = None,
):
    res = pd.read_csv(f"{folder}/output.jsonl", names=["jsonl"], sep="\t", quoting=3)
    res["obj"] = res.jsonl.map(json.loads)

    res["prompt"] = res.obj.map(lambda x: x[prompt_col] if prompt_col in x else None)
    image_cols = re.split(r"[,;]", image_cols)
    result_cols = re.split(r"[,;]", result_cols)
    for image_col in image_cols:
        res[f"Im_{image_col}"] = res.obj.map(
            lambda x: x[image_col] if image_col in x else None
        )
    for result_col in result_cols:
        res[f"Re_{result_col}"] = res.obj.map(
            lambda x: x[result_col] if result_col in x else None
        )

    text_cols = [prompt_col]
    image_cols = [f"Im_{col}" for col in image_cols] + [
        f"Re_{col}" for col in result_cols
    ]

    if temp_folder is None:
        temp_folder = tempfile.gettempdir()
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
    data_file = tempfile.mktemp(suffix=".data.txt", dir=temp_folder)
    res_file = tempfile.mktemp(suffix=".result.txt", dir=temp_folder)

    res[text_cols + image_cols].to_csv(
        data_file, sep="\t", index=False, header=False, quoting=3
    )

    logging.info(f"Data file: {data_file}")
    logging.info(f"Result file: {res_file}")

    choices = ["good", "fair", "bad"]
    num_images_per_row = 5
    for i in [5, 4, 3]:
        if len(image_cols) % i == 0:
            num_images_per_row = i
            break

    process = subprocess.Popen(
        [
            "unitorch-webui",
            "configs/labeling/classification.ini",
            "--data_file",
            data_file,
            "--result_file",
            res_file,
            "--names",
            ";".join(text_cols + image_cols),
            "--text_cols",
            ";".join(text_cols),
            "--image_cols",
            ";".join(image_cols),
            "--choices",
            ";".join(choices),
            "--checkbox",
            "False",
            "--microsoft/webui/labeling/classification@num_images_per_row",
            str(num_images_per_row),
            "--force_to_relabel",
            "True",
            "--port",
            str(port),
        ]
    )

    process.wait()


if __name__ == "__main__":
    fire.Fire(main)
