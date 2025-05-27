# Copyright (c) MICROSOFT.
# Licensed under the MIT License.
import os
import re
import io
import fire
import torch
import json
import queue
import logging
import hashlib
import requests
import threading
import pandas as pd
from torch.hub import download_url_to_file
from PIL import Image, ImageOps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch_microsoft import cached_path


def main(
    start: int = 0,
    stop: int = 72,
    download_folder: str = "./",
    output_file: str = "./output.jsonl",
    num_processes: int = 10,
    max_queue_size: int = 1000,
):
    base_folder = os.path.dirname(output_file)
    if not os.path.exists(base_folder):
        os.makedirs(base_folder, exist_ok=True)

    if not os.path.exists(download_folder):
        os.makedirs(download_folder, exist_ok=True)

    stop = min(72, stop)
    data = [
        f"https://huggingface.co/datasets/huggan/wikiart/resolve/main/data/train-{str(i).rjust(5, '0')}-of-00072.parquet"
        for i in range(start, stop + 1)
    ]

    infos = json.load(
        open(
            cached_path(
                "https://huggingface.co/datasets/huggan/wikiart/resolve/main/dataset_infos.json"
            )
        )
    )["huggan--wikiart"]
    artists = infos["features"]["artist"]["names"]
    genres = infos["features"]["genre"]["names"]
    styles = infos["features"]["style"]["names"]

    infos = {
        "artists": {i: artist for i, artist in enumerate(artists)},
        "genres": {i: genre for i, genre in enumerate(genres)},
        "styles": {i: style for i, style in enumerate(styles)},
    }

    saved_images = set()
    Q = queue.Queue(maxsize=max_queue_size)
    num_processes = min(num_processes, len(data))

    def save_to_zip(image):
        md5 = hashlib.md5()
        md5.update(image.tobytes())
        name = md5.hexdigest() + ".jpg"
        if name in saved_images:
            return name
        saved_buffer = io.BytesIO()
        image.save(saved_buffer, format="JPEG")
        saved_buffer = saved_buffer.getvalue()
        files = {"file": saved_buffer}
        requests.post(f"http://0.0.0.0:11231/?name={name}", files=files)
        saved_images.add(name)
        return name

    def crawl(parts):
        for data_url in parts:
            parquet_file = os.path.join(download_folder, f"{os.getpid()}.parquet")
            try:
                download_url_to_file(data_url, parquet_file)
                part = pd.read_parquet(parquet_file)
                for _, row in part.iterrows():
                    image, artist, genre, style = (
                        row["image"]["bytes"],
                        int(row["artist"]),
                        int(row["genre"]),
                        int(row["style"]),
                    )
                    image = io.BytesIO(image)
                    image = Image.open(image)
                    try:
                        name = save_to_zip(image)
                        record = {
                            "image": name,
                            "artist_id": artist,
                            "genre_id": genre,
                            "style_id": style,
                            "artist": infos["artists"].get(artist, "Unknown"),
                            "genre": infos["genres"].get(genre, "Unknown"),
                            "style": infos["styles"].get(style, "Unknown"),
                        }
                        Q.put(record)
                    except Exception as e:
                        logging.error(f"Failed to process image in {data_url}: {e}")
            except Exception as e:
                logging.error(f"Failed to download {data_url}: {e}")
        Q.put("Done")

    def write_file(cnt):
        f = open(output_file, "a+")
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

    data_parts = []
    for i in range(num_processes):
        data_parts.append(data[i::num_processes])

    processes = []
    for i in range(num_processes):
        p = threading.Thread(
            target=crawl,
            args=(data_parts[i],),
        )
        processes.append(p)

    io_process = threading.Thread(target=write_file, args=(num_processes,))
    processes.append(io_process)

    for p in processes:
        p.start()

    # wait for all processes to finish
    for p in processes:
        p.join()


if __name__ == "__main__":
    fire.Fire(main)
