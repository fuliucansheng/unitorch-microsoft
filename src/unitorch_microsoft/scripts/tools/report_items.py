# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import re
import logging
import fire
import base64
import torch
import json
import subprocess
import pandas as pd
from transformers.utils import is_remote_url
from torch.hub import download_url_to_file
from typing import Optional, Union, List
from unitorch_microsoft.fastapis.collector import reported_item


def report_jsonl_file(
    data_file,
    image_cols: Optional[str] = None,
    video_cols: Optional[str] = None,
    audio_cols: Optional[str] = None,
    log_freq: Optional[int] = 1000,
    tags: Optional[str] = None,
):
    if isinstance(image_cols, str):
        image_cols = re.split(r"[,;]", image_cols)
        image_cols = [n.strip() for n in image_cols]

    if isinstance(video_cols, str):
        video_cols = re.split(r"[,;]", video_cols)
        video_cols = [n.strip() for n in video_cols]

    if isinstance(audio_cols, str):
        audio_cols = re.split(r"[,;]", audio_cols)
        audio_cols = [n.strip() for n in audio_cols]

    data = pd.read_csv(data_file, names=["jsonl"], header=None, sep="\t", quoting=3)
    for i, row in data.iterrows():
        item = json.loads(row["jsonl"])
        images = {k: item[k] for k in item if k in image_cols} if image_cols else {}
        videos = {k: item[k] for k in item if k in video_cols} if video_cols else {}
        audios = {k: item[k] for k in item if k in audio_cols} if audio_cols else {}

        ignore_cols = list(
            set(image_cols or []) | set(video_cols or []) | set(audio_cols or [])
        )
        record = {k: v for k, v in item.items() if k not in ignore_cols}
        if tags is not None:
            record["tags"] = tags
        reported_item(
            record=record,
            images=images,
            videos=videos,
            audios=audios,
        )

        if (i + 1) % log_freq == 0:
            logging.info(f"Processed {i + 1} items")


def report_folder(
    data_folder,
    prompt_extension: Optional[str] = ".txt",
    image_extension: Optional[str] = ".jpg",
    video_extension: Optional[str] = ".mp4",
    audio_extension: Optional[str] = ".wav",
    log_freq: Optional[int] = 1000,
    tags: Optional[str] = None,
):
    if isinstance(prompt_extension, str):
        prompt_extension = [prompt_extension]
    if isinstance(image_extension, str):
        image_extension = [image_extension]
    if isinstance(video_extension, str):
        video_extension = [video_extension]
    if isinstance(audio_extension, str):
        audio_extension = [audio_extension]

    data_files = os.listdir(data_folder)
    for i, file_name in enumerate(data_files):
        file_path = os.path.join(data_folder, file_name)
        if not os.path.isfile(file_path):
            continue

        item = {}
        if any(file_name.endswith(ext) for ext in prompt_extension):
            with open(file_path, "r") as f:
                item["prompt"] = f.read().strip()
        else:
            continue

        file_name = file_name.rsplit(".", 1)[0]
        image, video, audio = None, None, None
        for ext in image_extension:
            image_file_path = f"{file_name}{ext}"
            if os.path.isfile(image_file_path):
                image = image_file_path
                break

        for ext in video_extension:
            video_file_path = f"{file_name}{ext}"
            if os.path.isfile(video_file_path):
                video = video_file_path
                break

        for ext in audio_extension:
            audio_file_path = f"{file_name}{ext}"
            if os.path.isfile(audio_file_path):
                audio = audio_file_path
                break

        if tags is not None:
            item["tags"] = tags

        reported_item(
            record=item,
            images={"image": image} if image else None,
            videos={"video": video} if video else None,
            audios={"audio": audio} if audio else None,
        )

        if (i + 1) % log_freq == 0:
            logging.info(f"Processed {i + 1} items")


def report_file(
    data_file,
    names: Optional[str] = None,
    image_cols: Optional[str] = None,
    video_cols: Optional[str] = None,
    audio_cols: Optional[str] = None,
    zip_folder: Optional[str] = None,
    zip_cols: Optional[str] = None,
    base64_cols: Optional[str] = None,
    log_freq: Optional[int] = 1000,
    tags: Optional[str] = None,
):
    if isinstance(names, str) and names.strip() == "*":
        names = None
    if isinstance(names, str):
        names = re.split(r"[,;]", names)
        names = [n.strip() for n in names]

    if isinstance(image_cols, str):
        image_cols = re.split(r"[,;]", image_cols)
        image_cols = [n.strip() for n in image_cols]

    if isinstance(video_cols, str):
        video_cols = re.split(r"[,;]", video_cols)
        video_cols = [n.strip() for n in video_cols]

    if isinstance(audio_cols, str):
        audio_cols = re.split(r"[,;]", audio_cols)
        audio_cols = [n.strip() for n in audio_cols]

    data = pd.read_csv(
        data_file,
        names=names,
        sep="\t",
        quoting=3,
        header=None if names is None else "infer",
    )

    if zip_cols is not None:
        assert (
            zip_folder is not None
        ), "zip_folder must be specified if zip_cols is provided"
        http_process = subprocess.Popen(
            [
                "unitorch-service",
                "start",
                "services/zip_files/config.ini",
                "--daemon_mode",
                "False",
                "--zip_folder",
                str(zip_folder),
                "--port",
                "41230",
            ],
        )
        http_url = "http://0.0.0.0:41230/?file={0}"
    else:
        http_process = None
        http_url = None

    for i, row in data.iterrows():
        item = json.loads(row["jsonl"])
        item = {
            k: (v if k not in (zip_cols or []) else http_url.format(v))
            for k, v in item.items()
        }
        item = {
            k: (v if k not in (base64_cols or []) else base64.b64decode(v))
            for k, v in item.items()
        }
        images = {k: item[k] for k in item if k in image_cols} if image_cols else {}
        videos = {k: item[k] for k in item if k in video_cols} if video_cols else {}
        audios = {k: item[k] for k in item if k in audio_cols} if audio_cols else {}

        ignore_cols = list(
            set(image_cols or []) | set(video_cols or []) | set(audio_cols or [])
        )
        record = {k: v for k, v in item.items() if k not in ignore_cols}
        if tags is not None:
            record["tags"] = tags
        reported_item(
            record=record,
            images=images,
            videos=videos,
            audios=audios,
        )

        if (i + 1) % log_freq == 0:
            logging.info(f"Processed {i + 1} items")


if __name__ == "__main__":
    fire.Fire(
        {
            "report_file": report_file,
            "report_folder": report_folder,
            "report_jsonl_file": report_jsonl_file,
        }
    )
