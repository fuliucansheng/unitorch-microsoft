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

def encode_jwt_token(ak, sk):
    headers = {
        "alg": "HS256",
        "typ": "JWT"
    }
    payload = {
        "iss": ak,
        "exp": int(time.time()) + 1800, # The valid time, in this example, represents the current time+1800s(30min)
        "nbf": int(time.time()) - 5 # The time when it starts to take effect, in this example, represents the current time minus 5s
    }
    token = jwt.encode(payload, sk, headers=headers)
    return token

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
    group_id = os.getenv('MINIMAX_GroupId')

    url = f'https://api.minimaxi.chat/v1/files/retrieve?GroupId={group_id}&file_id={file_id}'
    headers = {
        'authority': 'api.minimaxi.chat',
        'content-type': 'application/json',
        'Authorization': f'Bearer {token}'
    }
    response = requests.get(
        url,
        headers=headers,
        ).json()
    return response['file']['download_url']

def send_request_retry(token, api, params, retry_cnt=5):
    headers = {
        "accept":"application/json",
        "authorization": "Bearer "+ token,
        "content-type": "application/json"
        }
    response = requests.post(
        api,
        timeout=60,
        json=params, 
        headers=headers,
        ).json()
    time.sleep(2)
    retry = 0
    print(retry,response)
    while 'task_id' not in response and retry <= 5:
        retry += 1
        time.sleep(2)
        response = requests.post(
            api,
            timeout=60,
            json=params,
            headers=headers,
            ).json()
        print(retry,response)
    return response

def get_api_response(api, taskid):
    api_key = os.getenv('MINIMAX_API_KEY')
    assert api_key != None, f"Column api_key not found"
    headers = {
        "accept":"application/json",
        "authorization": "Bearer "+ api_key,
        "content-type": "application/json"
        }
    response = requests.get(
        api+'/'+taskid,
        timeout=60,
        headers=headers,
        ).json()
    print(response)
    return

def prepare_image(image):
    print("prepare image", image)
    if image == None:
        return ""
    if "http" in image:
        return image
    if os.path.exists(image):
        try:
            image = Image.open(image).convert("RGB")
            image_buffer = io.BytesIO()
            image.save(image_buffer, format="JPEG")
            image_buffer.seek(0)
            data = base64.b64encode(image_buffer.getvalue()).decode('utf-8')
            return f"data:image/jpeg;base64,{data}"
        except Exception as e:
            print("prepare image failed {!r}".format(e))
            return "" 
    else:
        return ""       

def get_random_cameras():
    cameras = ["Static","Move Left","Move Right","Move Up","Move Down","Push In","Pull Out","Zoom In","Zoom Out","Pan Left","Pan Right","Crane Up","Crane Down"]
    motion = random.choice(cameras)
    return " Camera Motion: "+motion


def image2video(
    data_file: str,
    cache_dir: str,
    names: Union[str, List[str]],
    prompt_col: str,
    start_frame_col:Optional[str] = None,
    end_frame_col: Optional[str] = None,
    auth_ak:Optional[str] = None,
    auth_sk:Optional[str] = None,
    neg_prompt_col:Optional[str] = None,
    aspect_ratio: Optional[str] = "16:9",
    duration: Optional[str] = '5',
    model: Optional[str] = "I2V-01", #I2V-01 I2V-01-live I2V-01-Director
    max_queue_size: Optional[int] = 2000,
    index_col: Optional[str] = None
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
        header=None
    )
    os.makedirs(cache_dir, exist_ok=True)
    api_key = os.getenv('MINIMAX_API_KEY')
    assert api_key != None, f"Column api_key not found"

    assert prompt_col in data.columns, f"Column {prompt_col} not found in data."
    assert start_frame_col in data.columns or end_frame_col in data.columns, f"At least one image needed."

    output_file = f"{cache_dir}/output.jsonl"
    if os.path.exists(output_file):
        uniques = []
        with open(output_file, "r") as f:
            for line in f:
                row = json.loads(line)
                uniques.append(
                    row["prompt"] + " - " + row["neg_prompt"] + " - " +row["start_frame"] + " - " + row["end_frame"]
                )
        data = data[
            ~data.apply(
                lambda x: (x[prompt_col] if prompt_col is not None and not pd.isna(x[prompt_col]) else "")
                + " - "
                + (x[neg_prompt_col] if neg_prompt_col is not None and not pd.isna(x[neg_prompt_col]) else "")
                + " - "
                + (x[start_frame_col] if start_frame_col is not None and not pd.isna(x[start_frame_col]) else "")
                + " - "
                + (x[end_frame_col] if end_frame_col is not None and not pd.isna(x[end_frame_col]) else "")
                in uniques,
                axis=1,
            )
        ]
    writer = open(output_file, "a+")
    Q = queue.Queue(maxsize=max_queue_size)
    def producer():
        for _, row in data.iterrows():
            while Q.full():
                time.sleep(2)
            _prompt = row[prompt_col] if not pd.isna(row[prompt_col]) else ""
            _neg_prompt = ""
            if neg_prompt_col != None:
                _neg_prompt = row[neg_prompt_col] if not pd.isna(row[neg_prompt_col]) else ""
            _index_id = ""
            if index_col != None:
                _index_id = row[index_col] if not pd.isna(row[index_col]) else ""
            _start_frame = ""
            _end_frame = ""
            if start_frame_col != None:
                _start_frame = row[start_frame_col] if not pd.isna(row[start_frame_col]) else ""
            if end_frame_col != None:
                _end_frame = row[end_frame_col] if not pd.isna(row[end_frame_col]) else ""
            params = { 
                "prompt": _prompt,
                "model": model,
                "first_frame_image": prepare_image(_start_frame)
            }
            try:
                response = send_request_retry(api_key, "https://api.minimaxi.chat/v1/video_generation", params)
                print(response)
                Q.put((response["task_id"],_prompt, _neg_prompt, _index_id, _start_frame, _end_frame))
            except:
                pass
        Q.put(("Done","Done","Done","Done","Done","Done"))
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
                response = requests.get(
                    f"http://api.minimaxi.chat/v1/query/video_generation?task_id={trackid}",
                    headers={
                        "authorization": "Bearer "+api_key,
                    }
                ).json()
                if response["status"] == "Success":
                    fileid = response["file_id"]
                    #get real video data
                    video_url = get_videofile(api_key, fileid)
                    record = {
                        "prompt": _prompt,
                        "neg_prompt": _neg_prompt,
                        "index_id": _index_id,
                        "start_frame": _start_frame,
                        "end_frame":_end_frame,
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
                    Q.put((trackid, _prompt, _neg_prompt, _index_id, _start_frame, _end_frame))
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



