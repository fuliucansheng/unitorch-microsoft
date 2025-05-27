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
from urllib.parse import urlparse
import random
import numpy as np
import jwt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(threadName)s - %(levelname)s - %(message)s",
)


def save_image_from_url(folder, url):
    urls = url.split("[SEP]")
    result = ""
    for url in urls:
        name = hashlib.md5(url.encode()).hexdigest() + ".jpg"
        path = f"{folder}/{name}"
        try:
            download_url_to_file(url, path, progress=False)
            if result != "":
                result += "[SEP]"
            result += path
        except:
            pass
    return result


def save_video_from_url(folder, url):
    urls = url.split("[SEP]")
    result = ""
    for url in urls:
        name = hashlib.md5(url.encode()).hexdigest() + ".mp4"
        path = f"{folder}/{name}"
        try:
            download_url_to_file(url, path, progress=False)
            if result != "":
                result += "[SEP]"
            result += path
        except:
            pass
    return result


def get_videofile(token, file_id):
    url = f"https://devapi.pika.art/videos/{file_id}"
    headers = {"Accept": "application/json", "X-API-KEY": token}
    response = requests.get(
        url,
        headers=headers,
    ).json()
    print(response)
    return response


def send_request_retry(token, api, params, image, retry_cnt=5):
    headers = {"Accept": "application/json", "X-API-KEY": token}
    response = requests.post(
        api, timeout=60, data=params, headers=headers, files=image
    ).json()
    time.sleep(2)
    retry = 0
    print(retry, response)
    while "video_id" not in response and retry <= 5:
        retry += 1
        time.sleep(2)
        response = requests.post(
            api,
            timeout=60,
            json=params,
            headers=headers,
        ).json()
        print(retry, response)
    return response


def prepare_image_byte(image):
    from urllib.request import urlopen

    print("prepare image", image)
    if image == None:
        return None
    if "http" in image:
        # try:
        if True:
            with urlopen(image, timeout=10) as html:
                img_bytes = html.read()
            return img_bytes
        # except:
        #    return None

    if os.path.exists(image):
        # try:
        if True:
            data = open(image, "rb")
            return data
        # except Exception as e:
        #    return None
    return None


def get_random_cameras():
    cameras = [
        "Static",
        "Move Left",
        "Move Right",
        "Move Up",
        "Move Down",
        "Push In",
        "Pull Out",
        "Zoom In",
        "Zoom Out",
        "Pan Left",
        "Pan Right",
        "Crane Up",
        "Crane Down",
    ]
    motion = random.choice(cameras)
    return " Camera Motion: " + motion


def image2video(
    data_file: str,
    cache_dir: str,
    names: Union[str, List[str]],
    prompt_col: str,
    start_frame_col: Optional[str] = None,
    end_frame_col: Optional[str] = None,
    neg_prompt_col: Optional[str] = None,
    resolution: Optional[str] = "1080p",
    duration: Optional[str] = "5",
    model: Optional[str] = "pika2.2",  # I2V-01 I2V-01-live I2V-01-Director
    max_queue_size: Optional[int] = 2000,
    index_col: Optional[str] = None,
):
    if isinstance(names, str) and names.strip() == "*":
        names = None
    if isinstance(names, str):
        names = re.split(r"[,;]", names)
        names = [n.strip() for n in names]
    data = pd.read_csv(data_file, names=names, sep="\t", quoting=3, header=None)
    os.makedirs(cache_dir, exist_ok=True)
    api_key = os.getenv("PIKA_API_KEY")
    assert api_key != None, f"Column api_key not found"

    assert prompt_col in data.columns, f"Column {prompt_col} not found in data."
    assert (
        start_frame_col in data.columns or end_frame_col in data.columns
    ), f"At least one image needed."

    process_file = f"{cache_dir}/process.jsonl"
    proc_writer = open(process_file, "w")
    output_file = f"{cache_dir}/output.jsonl"
    if os.path.exists(output_file):
        print(f"Before df {len(data)} ")
        uniques = []
        with open(output_file, "r") as f:
            for line in f:
                row = json.loads(line)
                uniques.append(
                    str(row["prompt"])
                    + " - "
                    + str(row["neg_prompt"])
                    + " - "
                    + str(row["start_frame"])
                    + " - "
                    + str(row["end_frame"])
                )
        print(f"unique size {len(uniques)}")
        data = data[
            ~data.apply(
                lambda x: (
                    x[prompt_col]
                    if prompt_col is not None and not pd.isna(x[prompt_col])
                    else ""
                )
                + " - "
                + (
                    x[neg_prompt_col]
                    if neg_prompt_col is not None and not pd.isna(x[neg_prompt_col])
                    else ""
                )
                + " - "
                + (
                    x[start_frame_col]
                    if start_frame_col is not None and not pd.isna(x[start_frame_col])
                    else ""
                )
                + " - "
                + (
                    x[end_frame_col]
                    if end_frame_col is not None and not pd.isna(x[end_frame_col])
                    else ""
                )
                in uniques,
                axis=1,
            )
        ]
        print(f"Need to handle {len(data)} files")
    writer = open(output_file, "a+")
    Q = queue.Queue(maxsize=max_queue_size)

    def producer():
        cnt = 0
        for _, row in data.iterrows():
            cnt += 1
            print(f"process {cnt} file")
            while Q.full():
                time.sleep(2)
            _prompt = row[prompt_col] if not pd.isna(row[prompt_col]) else ""
            _neg_prompt = ""
            if neg_prompt_col != None:
                _neg_prompt = (
                    row[neg_prompt_col] if not pd.isna(row[neg_prompt_col]) else ""
                )
            _index_id = ""
            if index_col != None:
                _index_id = row[index_col] if not pd.isna(row[index_col]) else ""
            _start_frame = ""
            _end_frame = ""
            if start_frame_col != None:
                _start_frame = (
                    row[start_frame_col] if not pd.isna(row[start_frame_col]) else ""
                )
            if end_frame_col != None:
                _end_frame = (
                    row[end_frame_col] if not pd.isna(row[end_frame_col]) else ""
                )
            params = {"promptText": _prompt, "negativePrompt": _neg_prompt}
            image_byte = prepare_image_byte(_start_frame)
            if image_byte == None:
                continue
            image = {"image": ("image.jpg", image_byte, "image/jpg")}
            try:
                response = send_request_retry(
                    api_key, "https://devapi.pika.art/generate/2.2/i2v", params, image
                )
                print(response)
                Q.put(
                    (
                        response["video_id"],
                        _prompt,
                        _neg_prompt,
                        _index_id,
                        _start_frame,
                        _end_frame,
                    )
                )

                proc_record = {
                    "taskid": response["video_id"],
                    "prompt": _prompt,
                    "neg_prompt": _neg_prompt,
                    "index_id": _index_id,
                    "start_frame": _start_frame,
                    "end_frame": _end_frame,
                }
                proc_writer.write(json.dumps(proc_record) + "\n")
                proc_writer.flush()
            except:
                pass
        Q.put(("Done", "Done", "Done", "Done", "Done", "Done"))
        print("finish")

    def consumer():
        is_produder_done = False
        while True:
            if is_produder_done and Q.empty():
                break
            trackid, _prompt, _neg_prompt, _index_id, _start_frame, _end_frame = Q.get()
            if trackid == "Done":
                is_produder_done = True
                continue
            try:
                response = get_videofile(api_key, trackid)
                if response["status"] == "finished":
                    video_url = response["url"]
                    record = {
                        "prompt": _prompt,
                        "neg_prompt": _neg_prompt,
                        "index_id": _index_id,
                        "start_frame": _start_frame,
                        "end_frame": _end_frame,
                        "url": video_url,
                        "result": save_video_from_url(cache_dir, video_url),
                    }
                    writer.write(json.dumps(record) + "\n")
                    writer.flush()
                elif response["status"] == "Fail":
                    logging.warning(
                        f"TrackId: {trackid} - Prompt: {_prompt} - Status: {response['status']} - Reason: {response['base_resp']['status_msg']}"
                    )
                else:
                    Q.put(
                        (
                            trackid,
                            _prompt,
                            _neg_prompt,
                            _index_id,
                            _start_frame,
                            _end_frame,
                        )
                    )
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
