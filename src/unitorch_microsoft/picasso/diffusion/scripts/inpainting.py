# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import re
import io
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

from unitorch.cli import (
    hf_endpoint_url,
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
)
from unitorch.cli import CoreConfigureParser, GenericScript
from unitorch.cli import register_script
from unitorch.cli.models.image_utils import ImageProcessor


@register_script("microsoft/picasso/script/diffusion/inpainting")
class ImageInpaintingScript(GenericScript):
    def __init__(self, config: CoreConfigureParser):
        self.config = config

    def launch(self, **kwargs):
        config = self.config

        config.set_default_section("microsoft/picasso/script/diffusion/inpainting")

        api_endpoints = config.getoption("api_endpoints", None)

        assert api_endpoints is not None, "api_endpoints is required"

        if isinstance(api_endpoints, str):
            api_endpoints = [api_endpoints]

        data_file = config.getoption("data_file", None)
        names = config.getoption("names", None)
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

        prompt_col = config.getoption("prompt_col", None)
        prompt_text = config.getoption("prompt_text", None)
        if prompt_text is None:
            assert prompt_col in data.columns, f"Column {prompt_col} not found in data."

        image_folder = config.getoption("image_folder", None)
        assert image_folder is not None, "image_folder is required"
        os.makedirs(image_folder, exist_ok=True)

        image_col = config.getoption("image_col", None)
        mask_image_col = config.getoption("mask_image_col", None)

        guidance_scale = config.getoption("guidance_scale", 3.5)
        num_timesteps = config.getoption("num_timesteps", 50)
        seed = config.getoption("seed", 42)
        jsonl_file = config.getoption("jsonl_file", None)

        if os.path.exists(jsonl_file):
            prompts = []
            with open(jsonl_file, "r") as f:
                for line in f:
                    prompts.append(json.loads(line)["prompt"])
            if prompt_col is not None:
                data = data[~data[prompt_col].isin(prompts)]

        def processing_image(image, mask_image):
            if isinstance(image, str):
                image = Image.open(image)
            if isinstance(mask_image, str):
                mask_image = Image.open(mask_image)

            # process your image/mask_image here

            return image, mask_image

        def save_image(image: Image.Image):
            md5 = hashlib.md5()
            md5.update(image.tobytes())
            name = md5.hexdigest() + ".jpg"
            output_path = f"{image_folder}/{name}"
            image.save(output_path)
            return name

        def call_api(endpoint, part, Q):
            status = requests.get(endpoint + "/status").json()
            if status != "running":
                requests.get(endpoint + "/start", timeout=1200)
            for _, row in part.iterrows():
                prompt = prompt_text if prompt_text is not None else row[prompt_col]
                image = row[image_col] if image_col in row else None
                mask_image = row[mask_image_col] if mask_image_col in row else None

                image, mask_image = processing_image(image, mask_image)
                image_buffer = io.BytesIO()
                image.save(image_buffer, format="JPEG")
                mask_image_buffer = io.BytesIO()
                mask_image.save(mask_image_buffer, format="JPEG")
                image_buffer.seek(0)
                mask_image_buffer.seek(0)

                files = {
                    "image": ("image.jpg", image_buffer, "image/jpeg"),
                    "mask_image": ("mask_image.jpg", mask_image_buffer, "image/jpeg"),
                }

                response = requests.post(
                    endpoint + "/generate",
                    params={
                        "text": prompt,
                        "guidance_scale": guidance_scale,
                        "num_timesteps": num_timesteps,
                        "seed": seed,
                    },
                    files=files,
                )

                result = Image.open(io.BytesIO(response.content))

                response = {
                    "prompt": prompt,
                    "image": save_image(result),
                }

                Q.put(response)
            Q.put("Done")
            requests.get(endpoint + "/stop")

        def write_file(fpath, Q, cnt):
            f = open(fpath, "w")
            done = 0
            while True:
                item = Q.get()
                if item == "Done":
                    done += 1
                    if done == cnt:
                        break
                else:
                    f.write(json.dumps(item) + "\n")

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

        io_process = Process(target=write_file, args=(jsonl_file, queue, num_processes))
        processes.append(io_process)

        for p in processes:
            p.start()

        # wait for all processes to finish
        for p in processes:
            p.join()
