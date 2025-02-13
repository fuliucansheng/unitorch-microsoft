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
from torch.hub import download_url_to_file
from PIL import Image, ImageOps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from multiprocessing import Process, Queue


def save_to_zip(image):
    md5 = hashlib.md5()
    md5.update(image.tobytes())
    name = md5.hexdigest() + ".jpg"
    saved_buffer = io.BytesIO()
    image.save(saved_buffer, format="JPEG")
    saved_buffer = saved_buffer.getvalue()
    files = {"file": saved_buffer}
    requests.post(f"http://0.0.0.0:11231/?name={name}", files=files)
    return name


def crawl(parts, Q, download_folder):
    for data_url in parts:
        parquet_file = os.path.join(download_folder, f"{os.getpid()}.parquet")
        try:
            download_url_to_file(data_url, parquet_file)
            part = pd.read_parquet(parquet_file)
            for _, row in part.iterrows():
                image, label = row["image"], row["class"]
                image = io.BytesIO(image)
                image = Image.open(image)
                try:
                    name = save_to_zip(image)
                    record = {"image": name, "label": label}
                    Q.put(record)
                except Exception as e:
                    logging.error(f"Failed to process image in {data_url}: {e}")
        except Exception as e:
            logging.error(f"Failed to download {data_url}: {e}")
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


def main(
    start: int = 0,
    stop: int = 7760,
    download_folder: str = "./",
    output_file: str = "./output.jsonl",
    num_processes: int = 10,
    max_queue_size: Optional[int] = 1000,
):
    base_folder = os.path.dirname(output_file)
    if not os.path.exists(base_folder):
        os.makedirs(base_folder, exist_ok=True)

    if not os.path.exists(download_folder):
        os.makedirs(download_folder, exist_ok=True)

    stop = min(7760, stop)
    data = [
        f"https://huggingface.co/datasets/gmongaras/Imagenet21K/resolve/main/data/train-{str(i).rjust(5, '0')}-of-07760.parquet"
        for i in range(start, stop + 1)
    ]

    num_processes = min(num_processes, len(data))
    data_parts = []
    for i in range(num_processes):
        data_parts.append(data[i::num_processes])

    processes = []
    queue = Queue(maxsize=max_queue_size)
    for i in range(num_processes):
        p = Process(
            target=crawl,
            args=(
                data_parts[i],
                queue,
                download_folder,
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
