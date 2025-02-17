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

def get_image_url_with_azure(image, account_name, connect_key, subfolder):
    from azure.identity import DefaultAzureCredential
    from azure.storage.blob import BlobServiceClient
    from azure.storage.blob import BlobClient, ContentSettings
    from azureml.core.run import Run
    if "http" in image:
        return image
    if os.path.exists(image):
        #upload to azure storage
        try:
            remote_name = subfolder+'/'+os.path.basename(image)
            connect_str = 'DefaultEndpointsProtocol=https;AccountName='+account_name+';AccountKey='+connect_key+';EndpointSuffix=core.windows.net'
            container_name = 'i2v'
            blob_service_client = BlobServiceClient.from_connection_string(connect_str)
            container_client = blob_service_client.get_container_client(container_name)
            image_blob = container_client.get_blob_client(remote_name)
            with open(image, "rb") as data:
                image_blob.upload_blob(data, overwrite=True)
            url = f"https://{account_name}.blob.core.windows.net/i2v/{remote_name}"
            return  url   
        except Exception as e:
            print("upload img2azure failed {!r}".format(e))
            return None 
    else:
        return None    
   
def text2image(
    token: str,
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
        for _, row in data.iterrows():
            while Q.full():
                time.sleep(2)
            _prompt = row[prompt_col]
            try:
                headers = {
                    "authorization": "Bearer "+token,
                    "content-type": "application/json",
                }
                response = requests.post(
                    "https://api.lumalabs.ai/dream-machine/v1/generations/image",
                    timeout=60,
                    json={
                        "prompt": _prompt,
                        "aspect_ratio": aspect_ratio,
                        "model": model
                    },
                    headers=headers,
                ).json()
                Q.put(response["id"])
            except:
                pass
        Q.put("Done")
    def consumer():
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
                    "https://api.lumalabs.ai/dream-machine/v1/generations/"+trackid,
                    headers={
                        "authorization": "Bearer "+token,
                    }
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


def text2video(
    token: str,
    data_file: str,
    cache_dir: str,
    names: Union[str, List[str]],
    prompt_col: str,
    resolution: Optional[str] = "720p",
    duration: Optional[str] = "5s",
    model: Optional[str] = "ray-2",
    aspect_ratio: Optional[str] = "16:9",
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
        for _, row in data.iterrows():
            while Q.full():
                time.sleep(2)

            _prompt = row[prompt_col]
            try:
                headers = {
                    "accept":"application/json",
                    "authorization": "Bearer "+token,
                    "content-type": "application/json",
                }
                response = requests.post(
                    "https://api.lumalabs.ai/dream-machine/v1/generations",
                    timeout=600,
                    json={
                        "prompt": _prompt,
                        "aspect_ratio": aspect_ratio,
                        "model": model,
                        "resolution":resolution,
                        "duration":duration
                    },
                    headers=headers,
                ).json()

                Q.put(response["id"])
            except:
                pass
        Q.put("Done")

    def consumer():
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
                    "https://api.lumalabs.ai/dream-machine/v1/generations/"+trackid,
                    headers={
                        "accept":"application/json",
                        "authorization": "Bearer "+token,
                    }
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
    token: str,
    data_file: str,
    cache_dir: str,
    names: Union[str, List[str]],
    prompt_col: str,
    azure_account_key: str,
    start_frame_col:Optional[str] = None,
    end_frame_col: Optional[str] = None,
    resolution: Optional[str] = "720p",
    duration: Optional[str] = "5s",
    model: Optional[str] = "ray-2",
    aspect_ratio: Optional[str] = "16:9",
    max_queue_size: Optional[int] = 2000,
    azure_account_name: Optional[str] = "unitorchazureblob",
    azure_save_folder: Optional[str] = "test"
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
    assert start_frame_col in data.columns or end_frame_col in data.columns, f"At least one image needed."


    output_file = f"{cache_dir}/output.jsonl"



    writer = open(output_file, "a+")
    Q = queue.Queue(maxsize=max_queue_size)

    def producer():
        cnt = 0
        for _, row in data.iterrows():
            while Q.full():
                time.sleep(2)

            _prompt = row[prompt_col]
            keyframes = {}
            if start_frame_col != None and start_frame_col in data.columns:
                url = get_image_url_with_azure(row[start_frame_col], azure_account_name, azure_account_key, azure_save_folder)
                if url != None:
                    keyframes["frame0"] = {"type":"image", "url":url}
            if end_frame_col != None and end_frame_col in data.columns:
                url = get_image_url_with_azure(row[end_frame_col], azure_account_name, azure_account_key, azure_save_folder)
                if url != None:
                    keyframes["frame1"] = {"type":"image", "url":url}
            if len(keyframes) == 0:
                continue
            try:
                headers = {
                    "accept":"application/json",
                    "authorization": "Bearer "+token,
                    "content-type": "application/json",
                }
                response = requests.post(
                    "https://api.lumalabs.ai/dream-machine/v1/generations",
                    timeout=600,
                    json={
                        "prompt": _prompt,
                        "aspect_ratio": aspect_ratio,
                        "model": model,
                        "resolution":resolution,
                        "duration":duration,
                        "keyframes":keyframes
                    },
                    headers=headers,
                ).json()
                cnt += 1
                Q.put(response["id"])
            except Exception as e:
                print("Producer error {!r}".format(e))
                pass
        Q.put("Done")

    def consumer():
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
                    "https://api.lumalabs.ai/dream-machine/v1/generations/"+trackid,
                    headers={
                        "accept":"application/json",
                        "authorization": "Bearer "+token,
                    }
                ).json()

                if response["state"] == "completed":
                    result = response["assets"]["video"]
                    _prompt = response["request"]["prompt"]
                    start_frame = ""
                    end_frame = ""
                    if "frame0" in response["request"]["keyframes"] and response["request"]["keyframes"]["frame0"] != None and "url" in response["request"]["keyframes"]["frame0"]:
                        start_frame = response["request"]["keyframes"]["frame0"]["url"]
                    if "frame1" in response["request"]["keyframes"] and response["request"]["keyframes"]["frame1"] != None and "url" in response["request"]["keyframes"]["frame1"]:
                        end_frame = response["request"]["keyframes"]["frame1"]["url"]
                    record = {
                        "prompt": _prompt,
                        "url": result,
                        "start_frame": start_frame,
                        "end_frame": end_frame,
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

