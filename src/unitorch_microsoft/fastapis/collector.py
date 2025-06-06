# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import io
import re
import gc
import time
import json
import logging
import psutil
import torch
import hashlib
import asyncio
import socket
import requests
import pandas as pd
from PIL import Image
from fastapi import APIRouter, UploadFile, File, Form, Body
from fastapi.responses import StreamingResponse
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch import is_xformers_available
from unitorch.utils import is_remote_url
from unitorch.utils import pop_value, nested_dict_value
from unitorch.cli import (
    cached_path,
    register_fastapi,
    add_default_section_for_init,
    add_default_section_for_function,
)
from unitorch.cli import CoreConfigureParser, GenericFastAPI

UNITORCH_MS_FASTAPI_ENDPOINT = os.environ.get("UNITORCH_MS_FASTAPI_ENDPOINT", None)


def save_image(image: Union[Image.Image, bytes], media_folder: str) -> str:
    if isinstance(image, bytes):
        image = Image.open(io.BytesIO(image))
    name = hashlib.md5(image.tobytes()).hexdigest() + ".jpg"
    part = f"{name[:2]}/{name[2:4]}/{name[4:6]}"
    folder = os.path.join(media_folder, part)
    os.makedirs(folder, exist_ok=True)
    image.save(f"{folder}/{name}")
    return name


def save_video(video: bytes, media_folder: str) -> str:
    name = hashlib.md5(video).hexdigest() + ".mp4"
    part = f"{name[:2]}/{name[2:4]}/{name[4:6]}"
    folder = os.path.join(media_folder, part)
    os.makedirs(folder, exist_ok=True)
    with open(f"{folder}/{name}", "wb") as f:
        f.write(video)
    return name


def save_audio(audio: bytes, media_folder: str) -> str:
    name = hashlib.md5(audio).hexdigest() + ".wav"
    part = f"{name[:2]}/{name[2:4]}/{name[4:6]}"
    folder = os.path.join(media_folder, part)
    os.makedirs(folder, exist_ok=True)
    with open(f"{folder}/{name}", "wb") as f:
        f.write(audio)
    return name


def collect(
    record: Dict[str, Any],
    images: Dict[str, Any] = None,
    videos: Dict[str, Any] = None,
    audios: Dict[str, Any] = None,
    db_folder: str = "ms_collector_db",
    media_folder: str = "ms_collector_db/media",
):
    sample = {**record}
    for name, value in images.items():
        sample[f"🖼️{name}"] = save_image(value, media_folder)

    for name, value in videos.items():
        sample[f"🎞️{name}"] = save_video(value, media_folder)

    for name, value in audios.items():
        sample[f"🎵{name}"] = save_audio(value, media_folder)

    date = time.strftime("%Y-%m-%d", time.localtime())
    dump_file = os.path.join(db_folder, f"{date}.jsonl")
    with open(dump_file, "a") as f:
        f.write(json.dumps(sample) + "\n")
        f.flush()


@register_fastapi("microsoft/fastapi/collector")
class CollectorFastAPI(GenericFastAPI):
    def __init__(self, config: CoreConfigureParser):
        self.config = config
        config.set_default_section(f"microsoft/fastapi/collector")
        router = config.getoption("router", "/microsoft/fastapi/collector")
        self._db_folder = config.getoption("db_folder", "ms_collector_db")
        self._media_folder = os.path.join(self._db_folder, "media")
        os.makedirs(self._db_folder, exist_ok=True)
        os.makedirs(self._media_folder, exist_ok=True)

        self._router = APIRouter(prefix=router)
        self._router.add_api_route(
            "/report",
            self.report,
            methods=["POST"],
            response_model=None,
            status_code=200,
        )
        self._lock = asyncio.Lock()

    @property
    def router(self):
        return self._router

    def start(self):
        return "start success"

    def stop(self):
        return "stop success"

    async def report(
        self,
        record: str = Form(...),
        images: List[UploadFile] = File(default=[]),
        videos: List[UploadFile] = File(default=[]),
        audios: List[UploadFile] = File(default=[]),
    ):
        try:
            record = json.loads(record)
        except json.JSONDecodeError:
            return {"status": "error", "message": "Invalid JSON format in 'record'"}

        image_files = {file.filename: await file.read() for file in images}
        video_files = {file.filename: await file.read() for file in videos}
        audio_files = {file.filename: await file.read() for file in audios}

        try:
            collect(
                record=record,
                images=image_files,
                videos=video_files,
                audios=audio_files,
                db_folder=self._db_folder,
                media_folder=self._media_folder,
            )
            return {"status": "success", "message": "Data collected successfully"}
        except Exception as e:
            logging.error(f"Error collecting data: {e}")
            return {"status": "error", "message": str(e)}


# Utility functions for reporting items
def cache_url_file_to_bytes(url_file_or_bytes) -> bytes:
    headers = {
        "User-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.1.2 Safari/605.1.15"
    }
    if is_remote_url(url_file_or_bytes):
        response = requests.get(url_file_or_bytes, timeout=30, headers=headers)
        response.raise_for_status()
        return response.content
    elif os.path.exists(url_file_or_bytes):
        with open(url_file_or_bytes, "rb") as f:
            return f.read()
    return url_file_or_bytes


def reported_item(
    record: Dict[str, Any],
    images: Dict[str, Union[Image.Image, str, bytes]] = None,
    videos: Dict[str, Union[str, bytes]] = None,
    audios: Dict[str, Union[str, bytes]] = None,
):
    endpoint = UNITORCH_MS_FASTAPI_ENDPOINT + "/microsoft/fastapi/collector/report"
    if not endpoint:
        return

    try:
        if images is not None:
            images = {
                name: (
                    value.tobytes()
                    if isinstance(value, Image.Image)
                    else cache_url_file_to_bytes(value)
                )
                for name, value in images.items()
            }
        if videos is not None:
            videos = {
                name: cache_url_file_to_bytes(value) for name, value in videos.items()
            }
        if audios is not None:
            audios = {
                name: cache_url_file_to_bytes(value) for name, value in audios.items()
            }

        files = []
        if images:
            for name, value in (images or {}).items():
                files.append(("images", (name, io.BytesIO(value), "image/jpeg")))

        if videos:
            for name, value in videos.items():
                files.append(("videos", (name, io.BytesIO(value), "video/mp4")))

        if audios:
            for name, value in audios.items():
                files.append(("audios", (name, io.BytesIO(value), "audio/wav")))

        requests.post(
            endpoint,
            data={
                "record": json.dumps(record),
            },
            files=files,
        )
        return {"status": "success", "message": "Record reported successfully"}
    except Exception as e:
        logging.error(f"Failed to report record: {e}")
        return {"status": "error", "message": str(e)}
