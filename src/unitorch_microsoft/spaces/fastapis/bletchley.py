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
from unitorch_microsoft.models.bletchley.pipeline_v1 import (
    BletchleyForMatchingV2Pipeline as BletchleyV1ForMatchingV2Pipeline,
    BletchleyForMatchingPipeline as BletchleyV1ForMatchingPipeline,
)
from unitorch_microsoft.models.bletchley.pipeline_v3 import (
    BletchleyForMatchingPipeline as BletchleyV3ForMatchingPipeline,
    BletchleyForMatchingV2Pipeline as BletchleyV3ForMatchingV2Pipeline,
    BletchleyForImageClassificationPipeline as BletchleyV3ForImageClassificationPipeline,
)
from unitorch_microsoft.omnipixel.bletchley import BletchleyForImageClickModelPipeline


@register_fastapi("microsoft/spaces/fastapi/bletchley/v1")
class BletchleyV1FastAPI(GenericFastAPI):
    def __init__(self, config: CoreConfigureParser):
        self._config = config
        config.set_default_section(f"microsoft/spaces/fastapi/bletchley/v1")
        self._pipe1 = None
        self._pipe2 = None
        self._pipe3 = None
        router = config.getoption("router", "/microsoft/spaces/fastapi/bletchley/v1")
        self._router = APIRouter(prefix=router)
        self._router.add_api_route(
            "/generate1", self.generate1, methods=["POST"]
        )  # bg/im type
        self._router.add_api_route(
            "/generate2", self.generate2, methods=["POST"]
        )  # blurry
        self._router.add_api_route(
            "/generate3", self.generate3, methods=["POST"]
        )  # bg type
        self._router.add_api_route("/status", self.status, methods=["GET"])
        self._router.add_api_route("/start", self.start, methods=["GET"])
        self._router.add_api_route("/stop", self.stop, methods=["GET"])
        self._lock = asyncio.Lock()

    @property
    def router(self):
        return self._router

    def start(self):
        if (
            self._pipe1 is not None
            and self._pipe2 is not None
            and self._pipe3 is not None
        ):
            return "running"
        self._pipe1 = BletchleyV1ForMatchingV2Pipeline.from_core_configure(
            self._config,
            config_type="0.8B",
            pretrained_weight_path="https://huggingface.co/datasets/fuliucansheng/unitorchblobfuse/resolve/main/models/bletchley/v1/pytorch_model.0.8B.bin",
            pretrained_lora_weight_path="https://huggingface.co/datasets/fuliucansheng/unitorchblobfuse/resolve/main/models/adsplus/lora/bletchley/pytorch_model.v1.lora4.bg_type.2501.bin",
            label_dict={
                "Complex": "complex background, objects in the background or even no background",
                "Simple": "clean background, no objects in the background",
                "White": "white background, no objects in the background",
                "Poster": "poster image, composed of multiple objects, logo, text, etc.",
                "Real": "a real image, not a poster or a logo",
                "Logo": "logo image, composed of logo only",
            },
            act_fn="sigmoid",
        )
        self._pipe2 = BletchleyV1ForMatchingV2Pipeline.from_core_configure(
            self._config,
            config_type="2.5B",
            pretrained_weight_path="https://huggingface.co/datasets/fuliucansheng/unitorchblobfuse/resolve/main/models/bletchley/v1/pytorch_model.2.5B.bin",
            pretrained_lora_weight_path="https://huggingface.co/datasets/fuliucansheng/unitorchblobfuse/resolve/main/models/adsplus/lora/bletchley/pytorch_model.v1.lora4.blurry.2409.bin",
            label_dict={
                "Blurry": "blurry",
            },
            act_fn="sigmoid",
        )
        self._pipe3 = BletchleyV1ForMatchingV2Pipeline.from_core_configure(
            self._config,
            config_type="0.8B",
            pretrained_weight_path="https://huggingface.co/datasets/fuliucansheng/unitorchblobfuse/resolve/main/models/bletchley/v1/pytorch_model.0.8B.bin",
            pretrained_lora_weight_path="https://huggingface.co/datasets/fuliucansheng/unitorchblobfuse/resolve/main/models/picasso/pytorch_model.bletchley.background.classification.202504.bin",
            label_dict={
                "Complex": "A product, poster, or lifestyle image with a complex background that contains many visual elements and textures",
                "Simple": "objects on a plain or softly blurred background with soft colors or smooth gradients",
                "White": "objects on a clean white background with no textures, watermarks, logos, or color boundaries",
            },
            act_fn="sigmoid",
        )
        return "running"

    def stop(self):
        self._pipe1 = None
        self._pipe2 = None
        self._pipe3 = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return "stopped"

    def status(self):
        if (
            self._pipe1 is not None
            and self._pipe2 is not None
            and self._pipe3 is not None
        ):
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

    async def generate3(
        self,
        image: UploadFile,
    ):
        image_bytes = await image.read()
        image = Image.open(io.BytesIO(image_bytes))
        async with self._lock:
            if self.status() != "running":
                self.start()
            results = self._pipe3(image)

        return results


@register_fastapi("microsoft/spaces/fastapi/bletchley/v3")
class BletchleyV3FastAPI(GenericFastAPI):
    def __init__(self, config: CoreConfigureParser):
        self._config = config
        config.set_default_section(f"microsoft/spaces/fastapi/bletchley/v3")
        self._pipe1 = None
        self._pipe2 = None
        router = config.getoption("router", "/microsoft/spaces/fastapi/bletchley/v3")
        self._router = APIRouter(prefix=router)
        self._router.add_api_route(
            "/generate1", self.generate1, methods=["POST"]
        )  # watermark
        self._router.add_api_route(
            "/generate2", self.generate2, methods=["POST"]
        )  # aesthetics
        self._router.add_api_route("/status", self.status, methods=["GET"])
        self._router.add_api_route("/start", self.start, methods=["GET"])
        self._router.add_api_route("/stop", self.stop, methods=["GET"])
        self._lock = asyncio.Lock()

    @property
    def router(self):
        return self._router

    def start(self):
        if self._pipe1 is not None:
            return "running"
        self._pipe1 = BletchleyV3ForMatchingV2Pipeline.from_core_configure(
            self._config,
            config_type="2.5B",
            pretrained_weight_path="https://huggingface.co/datasets/fuliucansheng/unitorchblobfuse/resolve/main/models/bletchley/v3/pytorch_model.large.bin",
            pretrained_lora_weight_path="https://huggingface.co/datasets/fuliucansheng/unitorchblobfuse/resolve/main/models/adsplus/lora/bletchley/pytorch_model.v3.2.5B.lora4.watermark.2410.bin",
            label_dict={
                "Watermark": "watermarked, no watermark signature, brand logo",
            },
            act_fn="sigmoid",
        )
        self._pipe2 = BletchleyV3ForMatchingV2Pipeline.from_core_configure(
            self._config,
            config_type="2.5B",
            pretrained_weight_path="https://huggingface.co/datasets/fuliucansheng/unitorchblobfuse/resolve/main/models/bletchley/v3/pytorch_model.large.bin",
            pretrained_lora_weight_path="https://huggingface.co/datasets/fuliucansheng/unitorchblobfuse/resolve/main/models/adsplus/lora/bletchley/pytorch_model.v3.lora4.aesthetics.tad66k.2412.bin",
            label_dict={
                "Bad Aesthetics": "draft, disgusting, blurry, disfigured",
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


@register_fastapi("microsoft/spaces/fastapi/bletchley/msan/image_click")
class BletchleyImageClickModelFastAPI(GenericFastAPI):
    def __init__(self, config: CoreConfigureParser):
        self._config = config
        config.set_default_section(
            f"microsoft/spaces/fastapi/bletchley/msan/image_click"
        )
        self._pipe1 = None
        router = config.getoption(
            "router", "/microsoft/spaces/fastapi/bletchley/msan/image_click"
        )
        self._router = APIRouter(prefix=router)
        self._router.add_api_route(
            "/generate1", self.generate1, methods=["POST"]
        )  # watermark
        self._router.add_api_route("/status", self.status, methods=["GET"])
        self._router.add_api_route("/start", self.start, methods=["GET"])
        self._router.add_api_route("/stop", self.stop, methods=["GET"])
        self._lock = asyncio.Lock()

    @property
    def router(self):
        return self._router

    def start(self):
        if self._pipe1 is not None:
            return "running"
        self._pipe1 = BletchleyForImageClickModelPipeline.from_core_configure(
            self._config
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
    ):
        image_bytes = await image.read()
        image = Image.open(io.BytesIO(image_bytes))
        async with self._lock:
            if self.status() != "running":
                self.start()
            results = self._pipe1(image)

        return results
