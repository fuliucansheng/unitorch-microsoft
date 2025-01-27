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
from multiprocessing import Process, Queue


def save_to_zip(image):
    md5 = hashlib.md5()
    md5.update(image.tobytes())
    name = md5.hexdigest() + ".png"
    saved_buffer = io.BytesIO()
    image.save(saved_buffer, format="PNG")
    saved_buffer = saved_buffer.getvalue()
    files = {"file": saved_buffer}
    requests.post(f"http://0.0.0.0:11231/?name={name}", files=files)
    return name


def save_image(url):
    headers = {
        "User-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.1.2 Safari/605.1.15"
    }
    response = requests.get(url, timeout=300, headers=headers)
    image_bytes = io.BytesIO(response.content)
    image = Image.open(image_bytes)
    tile_size = 1024
    results = {}
    for i in range(2):  # 行
        for j in range(2):  # 列
            left = j * tile_size
            upper = i * tile_size
            right = left + tile_size
            lower = upper + tile_size
            sub_image = image.crop((left, upper, right, lower))
            results[f"image{i}{j}"] = save_to_zip(sub_image)
    return results


def crawl(part, Q):
    for _, row in part.iterrows():
        image_url = row["image_url"]
        try:
            results = save_image(image_url)
            record = {"prompt": row["prompt"], **results}
            Q.put(record)
        except Exception as e:
            logging.error(f"Failed to crawl {image_url}: {e}")
    Q.put("Done")


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


data = pd.read_parquet("./midjourney-v6/data/train-00000-of-00001.parquet")
output_file = "output.jsonl"
num_processes = 10
data_parts = []
for i in range(num_processes):
    data_parts.append(data.iloc[i::num_processes])

processes = []
queue = Queue()
for i in range(num_processes):
    p = Process(
        target=crawl,
        args=(
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
