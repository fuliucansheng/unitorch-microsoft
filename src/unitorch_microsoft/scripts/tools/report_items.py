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
            "report_jsonl_file": report_jsonl_file,
        }
    )
