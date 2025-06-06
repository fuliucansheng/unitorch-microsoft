# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

# https://docs.lumalabs.ai/docs/python-image-generation

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(threadName)s - %(levelname)s - %(message)s",
)


def save_image_from_url(folder, url):
    name = hashlib.md5(url.encode()).hexdigest() + ".jpg"
    path = f"{folder}/{name}"
    try:
        download_url_to_file(url, path, progress=False)
        return path
    except:
        return None


def save_image(folder, image):
    name = hashlib.md5(image.tobytes()).hexdigest() + ".jpg"
    image.save(f"{folder}/{name}")
    return f"{folder}/{name}"


def save_video_from_url(folder, url):
    name = hashlib.md5(url.encode()).hexdigest() + ".mp4"
    path = f"{folder}/{name}"
    try:
        download_url_to_file(url, path, progress=False)
        return path
    except Exception as e:
        print("save video error {!r}".format(e))
        return None


def send_i2v_request(api, params, retry_cnt=5):
    token = os.getenv("LUMA_API_TOKEN")
    assert token != None, f"Column api_key not found"
    headers = {
        "accept": "application/json",
        "authorization": "Bearer " + token,
        "content-type": "application/json",
    }
    response = requests.post(
        api,
        timeout=600,
        json=params,
        headers=headers,
    ).json()
    time.sleep(2)
    print(response)
    retry = 0
    while "id" not in response and retry <= retry_cnt:
        retry += 1
        time.sleep(60)
        response = requests.post(
            api,
            timeout=600,
            json=params,
            headers=headers,
        ).json()
        print(response)
    return response


def get_image_url_with_azure(image, account_name, connect_key, subfolder):
    from azure.storage.blob import BlobServiceClient
    from azure.storage.blob import BlobClient, ContentSettings

    if "http" in image:
        return image
    if os.path.exists(image):
        # upload to azure storage
        try:
            remote_name = subfolder + "/" + os.path.basename(image)
            connect_str = (
                "DefaultEndpointsProtocol=https;AccountName="
                + account_name
                + ";AccountKey="
                + connect_key
                + ";EndpointSuffix=core.windows.net"
            )
            container_name = "i2v"
            blob_service_client = BlobServiceClient.from_connection_string(connect_str)
            container_client = blob_service_client.get_container_client(container_name)
            image_blob = container_client.get_blob_client(remote_name)
            with open(image, "rb") as data:
                image_blob.upload_blob(data, overwrite=True)
            url = f"https://{account_name}.blob.core.windows.net/i2v/{remote_name}"
            return url
        except Exception as e:
            print("upload img2azure failed {!r}".format(e))
            return None
    else:
        return None


def text2image(
    data_file: str,
    cache_dir: str,
    names: Union[str, List[str]],
    prompt_col: str,
    aspect_ratio: Optional[str] = "16:9",
    model: Optional[str] = "photon-1",
    max_queue_size: Optional[int] = 2000,
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
    assert prompt_col in data.columns, f"Column {prompt_col} not found in data."
    output_file = f"{cache_dir}/output.jsonl"
    if os.path.exists(output_file):
        uniques = []
        with open(output_file, "r") as f:
            for line in f:
                row = json.loads(line)
                uniques.append(row["prompt"])
        data = data[
            ~data.apply(
                lambda x: x[prompt_col] in uniques,
                axis=1,
            )
        ]
    writer = open(output_file, "a+")
    Q = queue.Queue(maxsize=max_queue_size)

    def producer():
        token = os.getenv("LUMA_API_TOKEN")
        assert token != None, f"Column api_key not found"
        for _, row in data.iterrows():
            while Q.full():
                time.sleep(2)
            _prompt = row[prompt_col]
            try:
                api = "https://api.lumalabs.ai/dream-machine/v1/generations/image"
                param = {
                    "prompt": _prompt,
                    "aspect_ratio": aspect_ratio,
                    "model": model,
                }
                response = send_i2v_request(api, param)
                Q.put(response["id"])
            except:
                pass
        Q.put("Done")

    def consumer():
        token = os.getenv("LUMA_API_TOKEN")
        assert token != None, f"Column api_key not found"
        is_produder_done = False
        while True:
            if is_produder_done and Q.empty():
                break
            trackid = Q.get()
            if trackid == "Done":
                is_produder_done = True
                continue
            try:
                response = requests.get(
                    "https://api.lumalabs.ai/dream-machine/v1/generations/" + trackid,
                    headers={
                        "authorization": "Bearer " + token,
                    },
                ).json()
                if response["state"] == "completed":
                    result = response["assets"]["image"]
                    _prompt = response["request"]["prompt"]
                    record = {
                        "prompt": _prompt,
                        "url": result,
                        "result": save_image_from_url(cache_dir, result),
                    }
                    writer.write(json.dumps(record) + "\n")
                    writer.flush()
                elif response["state"] == "failed":
                    logging.warning(
                        f"TrackId: {trackid} - Prompt: {response['request']['prompt']} - Status: {response['state']} - Reason: {response['failure_reason']}"
                    )
                else:
                    Q.put(trackid)
                    time.sleep(2)
            except:
                pass

    producer_thread = threading.Thread(target=producer)
    consumer_thread = threading.Thread(target=consumer)
    producer_thread.start()
    consumer_thread.start()
    producer_thread.join()
    consumer_thread.join()


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


def text2video(
    data_file: str,
    cache_dir: str,
    names: Union[str, List[str]],
    prompt_col: str,
    resolution: Optional[str] = "720p",
    duration: Optional[str] = "5s",
    model: Optional[str] = "ray-2",
    aspect_ratio: Optional[str] = "16:9",
    max_queue_size: Optional[int] = 2000,
    random_camera_motion: Optional[bool] = False,
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

    assert prompt_col in data.columns, f"Column {prompt_col} not found in data."

    output_file = f"{cache_dir}/output.jsonl"

    if os.path.exists(output_file):
        uniques = []
        with open(output_file, "r") as f:
            for line in f:
                row = json.loads(line)
                uniques.append(row["prompt"])
        data = data[
            ~data.apply(
                lambda x: x[prompt_col] in uniques,
                axis=1,
            )
        ]

    writer = open(output_file, "a+")
    Q = queue.Queue(maxsize=max_queue_size)

    def producer():
        token = os.getenv("LUMA_API_TOKEN")
        assert token != None, f"Column api_key not found"
        camera = ""
        for _, row in data.iterrows():
            while Q.full():
                time.sleep(2)
            if random_camera_motion:
                camera = get_random_cameras()
            _prompt = row[prompt_col] + camera
            try:
                api = "https://api.lumalabs.ai/dream-machine/v1/generations"
                param = {
                    "prompt": _prompt,
                    "aspect_ratio": aspect_ratio,
                    "model": model,
                    "resolution": resolution,
                    "duration": duration,
                }
                response = send_i2v_request(api, param)
                Q.put(response["id"])
            except:
                pass
        Q.put("Done")

    def consumer():
        token = os.getenv("LUMA_API_TOKEN")
        assert token != None, f"Column api_key not found"
        is_produder_done = False
        while True:
            if is_produder_done and Q.empty():
                break
            trackid = Q.get()
            if trackid == "Done":
                is_produder_done = True
                continue

            try:
                response = requests.get(
                    "https://api.lumalabs.ai/dream-machine/v1/generations/" + trackid,
                    headers={
                        "accept": "application/json",
                        "authorization": "Bearer " + token,
                    },
                ).json()

                if response["state"] == "completed":
                    result = response["assets"]["video"]
                    _prompt = response["request"]["prompt"]
                    record = {
                        "prompt": _prompt,
                        "url": result,
                        "result": save_video_from_url(cache_dir, result),
                    }
                    writer.write(json.dumps(record) + "\n")
                    writer.flush()
                elif response["state"] == "failed":
                    logging.warning(
                        f"TrackId: {trackid} - Prompt: {response['request']['prompt']} - Status: {response['state']} - Reason: {response['failure_reason']}"
                    )
                else:
                    Q.put(trackid)
                    time.sleep(2)
            except:
                pass

    producer_thread = threading.Thread(target=producer)
    consumer_thread = threading.Thread(target=consumer)

    producer_thread.start()
    consumer_thread.start()

    producer_thread.join()
    consumer_thread.join()


def image2video(
    data_file: str,
    cache_dir: str,
    names: Union[str, List[str]],
    prompt_col: str,
    neg_prompt_col: Optional[str] = None,
    start_frame_col: Optional[str] = None,
    end_frame_col: Optional[str] = None,
    resolution: Optional[str] = "720p",
    duration: Optional[str] = "5s",
    model: Optional[str] = "ray-2",
    aspect_ratio: Optional[str] = "16:9",
    max_queue_size: Optional[int] = 2000,
    azure_account_name: Optional[str] = "unitorchazureblob",
    azure_save_folder: Optional[str] = "test",
    random_camera_motion: Optional[bool] = False,
):
    if isinstance(names, str) and names.strip() == "*":
        names = None
    if isinstance(names, str):
        names = re.split(r"[,;]", names)
        names = [n.strip() for n in names]

    data = pd.read_csv(data_file, names=names, sep="\t", quoting=3, header=None)

    os.makedirs(cache_dir, exist_ok=True)

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
                    row["prompt"]
                    + " - "
                    + row["neg_prompt"]
                    + " - "
                    + row["start_frame"]
                    + " - "
                    + row["end_frame"]
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
        token = os.getenv("LUMA_API_TOKEN")
        assert token != None, f"Column api_key not found"
        azure_account_key = os.getenv("AZURE_ACCOUNT_KEY")
        assert azure_account_key != None, f"Need Azure account key"
        camera = ""
        cnt = 0
        for _, row in data.iterrows():
            cnt += 1
            print(f"process {cnt} file")
            while Q.full():
                time.sleep(2)
            if random_camera_motion:
                camera = get_random_cameras()
            _prompt = row[prompt_col] if not pd.isna(row[prompt_col]) else "" + camera
            _neg_prompt = ""
            if neg_prompt_col != None:
                _neg_prompt = (
                    row[neg_prompt_col] if not pd.isna(row[neg_prompt_col]) else ""
                )
            keyframes = {}
            start_frame = ""
            if start_frame_col != None and start_frame_col in data.columns:
                url = get_image_url_with_azure(
                    row[start_frame_col],
                    azure_account_name,
                    azure_account_key,
                    azure_save_folder,
                )
                if url != None:
                    keyframes["frame0"] = {"type": "image", "url": url}
                    start_frame = url
            print(start_frame, _prompt)
            end_frame = ""
            if end_frame_col != None and end_frame_col in data.columns:
                url = get_image_url_with_azure(
                    row[end_frame_col],
                    azure_account_name,
                    azure_account_key,
                    azure_save_folder,
                )
                if url != None:
                    keyframes["frame1"] = {"type": "image", "url": url}
                    end_frame = url
            if len(keyframes) == 0:
                continue
            try:
                api = "https://api.lumalabs.ai/dream-machine/v1/generations"
                param = {
                    "prompt": _prompt,
                    "aspect_ratio": aspect_ratio,
                    "model": model,
                    "resolution": resolution,
                    "duration": duration,
                    "keyframes": keyframes,
                }
                response = send_i2v_request(api, param)
                print(response)
                Q.put((response["id"], _neg_prompt))
                proc_record = {
                    "taskid": response["id"],
                    "prompt": _prompt,
                    "neg_prompt": _neg_prompt,
                    "index_id": "",
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                }
                proc_writer.write(json.dumps(proc_record) + "\n")
                proc_writer.flush()
            except Exception as e:
                print("Producer error {!r}".format(e))
                pass
        Q.put(("Done", "Done"))

    def consumer():
        token = os.getenv("LUMA_API_TOKEN")
        assert token != None, f"Column api_key not found"
        is_produder_done = False
        while True:
            if is_produder_done and Q.empty():
                break
            trackid, _neg_prompt = Q.get()
            if trackid == "Done":
                is_produder_done = True
                continue

            try:
                response = requests.get(
                    "https://api.lumalabs.ai/dream-machine/v1/generations/" + trackid,
                    headers={
                        "accept": "application/json",
                        "authorization": "Bearer " + token,
                    },
                ).json()

                if response["state"] == "completed":
                    result = response["assets"]["video"]
                    _prompt = response["request"]["prompt"]
                    start_frame = ""
                    end_frame = ""
                    if (
                        "frame0" in response["request"]["keyframes"]
                        and response["request"]["keyframes"]["frame0"] != None
                        and "url" in response["request"]["keyframes"]["frame0"]
                    ):
                        start_frame = response["request"]["keyframes"]["frame0"]["url"]
                    if (
                        "frame1" in response["request"]["keyframes"]
                        and response["request"]["keyframes"]["frame1"] != None
                        and "url" in response["request"]["keyframes"]["frame1"]
                    ):
                        end_frame = response["request"]["keyframes"]["frame1"]["url"]
                    record = {
                        "prompt": _prompt,
                        "neg_prompt": "",
                        "index_id": "",
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "url": result,
                        "result": save_video_from_url(cache_dir, result),
                    }
                    writer.write(json.dumps(record) + "\n")
                    writer.flush()
                elif response["state"] == "failed":
                    logging.warning(
                        f"TrackId: {trackid} - Prompt: {response['request']['prompt']} - Status: {response['state']} - Reason: {response['failure_reason']}"
                    )
                else:
                    Q.put((trackid, _neg_prompt))
                    time.sleep(2)
            except Exception as e:
                print("consumer {!r}".format(e))
                pass

    producer_thread = threading.Thread(target=producer)
    consumer_thread = threading.Thread(target=consumer)

    producer_thread.start()
    consumer_thread.start()

    producer_thread.join()
    consumer_thread.join()


if __name__ == "__main__":
    fire.Fire()
