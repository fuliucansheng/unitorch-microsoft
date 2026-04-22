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
from PIL import Image, ImageDraw
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
from unitorch.cli.pipelines.detr import DetrForDetectionPipeline
import unitorch_microsoft.models.detr


@register_fastapi("microsoft/apps/spaces/picasso/detr")
class DetrFastAPI(GenericFastAPI):
    def __init__(self, config: CoreConfigureParser):
        self._config = config
        config.set_default_section(f"microsoft/apps/spaces/picasso/detr")
        router = config.getoption("router", "/microsoft/apps/spaces/picasso/detr")
        self._pipe1 = None
        self._router = APIRouter(prefix=router)
        self._router.add_api_route(
            "/generate1", self.generate1, methods=["POST"]
        )
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
        self._pipe1 = DetrForDetectionPipeline.from_core_configure(
            config=self._config,
            pretrained_name="detr-resnet-50-picasso-roi-v2",
        )
        return "running"

    def stop(self):
        self._pipe1 = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return "stopped"

    def status(self):
        if self._pipe1 is not None:
            return "running"
        return "stopped"

    async def generate1(
        self,
        image: UploadFile,
        threshold: float = 0.5,
    ):
        image_bytes = await image.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        async with self._lock:
            if self.status() != "running":
                self.start()
            result = self._pipe1(
                image,
                threshold=threshold,
            )
        
        buffer = io.BytesIO()
        result.save(buffer, format="PNG")

        return StreamingResponse(
            io.BytesIO(buffer.getvalue()),
            media_type="image/png",
        )
