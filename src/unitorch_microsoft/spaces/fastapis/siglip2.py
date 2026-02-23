# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import io
import re
import gc
import json
import logging
import torch
import hashlib
import asyncio
import pandas as pd
from PIL import Image
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import StreamingResponse
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from unitorch.utils import is_remote_url
from unitorch.utils import pop_value, nested_dict_value
from unitorch.cli import (
    cached_path,
    register_fastapi,
    add_default_section_for_init,
    add_default_section_for_function,
)
from unitorch.cli import CoreConfigureParser, GenericFastAPI
from unitorch_microsoft.models.siglip.pipeline import Siglip2ForMatchingV2Pipeline


@register_fastapi("microsoft/spaces/fastapi/siglip2")
class Siglip2FastAPI(GenericFastAPI):
    def __init__(self, config: CoreConfigureParser):
        self._config = config
        config.set_default_section(f"microsoft/spaces/fastapi/siglip2")
        router = config.getoption("router", "/microsoft/spaces/fastapi/siglip2")
        self._pipe1 = None
        self._pipe2 = None
        self._router = APIRouter(prefix=router)
        self._router.add_api_route(
            "/generate1", self.generate1, methods=["POST"]
        )  # bad crop
        self._router.add_api_route(
            "/generate2", self.generate2, methods=["POST"]
        )  # bad whitepad
        self._router.add_api_route("/status", self.status, methods=["GET"])
        self._router.add_api_route("/start", self.start, methods=["GET"])
        self._router.add_api_route("/stop", self.stop, methods=["GET"])
        self._lock = asyncio.Lock()

    @property
    def router(self):
        return self._router

    def start(self):
        if self._pipe1 is not None and self._pipe2 is not None:
            return "running"
        self._pipe1 = Siglip2ForMatchingV2Pipeline.from_core_configure(
            self._config,
            pretrained_name="siglip2-so400m-patch14-384",
            pretrained_lora_weight_path="https://unitorchazureblob.blob.core.windows.net/shares/models/adsplus/lora/siglip/pytorch_model.v2.lora4.badcrop.2506.bin",
            label_dict={
                "Bad Cropped": "bad cropped, cut off, mutilated",
            },
            act_fn="sigmoid",
        )
        self._pipe2 = Siglip2ForMatchingV2Pipeline.from_core_configure(
            self._config,
            pretrained_name="siglip2-so400m-patch14-384",
            pretrained_lora_weight_path="https://unitorchazureblob.blob.core.windows.net/shares/models/adsplus/lora/siglip/pytorch_model.v2.lora4.whitepad.2510.v2.bin",
            label_dict={
                "Bad White Padding": "bad white padding",
            },
            act_fn="sigmoid",
        )
        return "running"

    def stop(self):
        self._pipe1 = None
        self._pipe2 = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return "stopped"

    def status(self):
        if self._pipe1 is not None and self._pipe2 is not None:
            return "running"
        return "stopped"

    async def generate1(
        self,
        image: UploadFile,
    ):
        image_bytes = await image.read()
        image = Image.open(io.BytesIO(image_bytes))
        async with self._lock:
            if self.status() != "running":
                self.start()
            results = self._pipe1(image)

        return results

    async def generate2(
        self,
        image: UploadFile,
    ):
        image_bytes = await image.read()
        image = Image.open(io.BytesIO(image_bytes))
        async with self._lock:
            if self.status() != "running":
                self.start()
            results = self._pipe2(image)

        return results
