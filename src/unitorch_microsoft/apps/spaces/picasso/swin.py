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
from unitorch.models.swin import SwinProcessor, SwinForImageClassification
from unitorch.cli import (
    cached_path,
    register_fastapi,
    add_default_section_for_init,
    add_default_section_for_function,
)
from unitorch.cli import CoreConfigureParser, GenericFastAPI


@register_fastapi("microsoft/apps/spaces/picasso/swin/googlecate")
class GoogleCateFastAPI(GenericFastAPI):
    def __init__(self, config: CoreConfigureParser):
        self._config = config
        config.set_default_section(f"microsoft/apps/spaces/picasso/swin/googlecate")
        self._model = None
        self._cates = pd.read_csv(
            cached_path(
                "https://huggingface.co/datasets/fuliucansheng/unitorchblobfuse/raw/main/models/googlecategory_dic.txt"
            ),
            names=["cate", "id"],
            sep="\t",
            quoting=3,
        )
        self._device = config.getoption("device", "cpu")
        self._cates = {row["id"]: row["cate"] for _, row in self._cates.iterrows()}
        router = config.getoption("router", "/microsoft/apps/spaces/picasso/swin/googlecate")
        self._router = APIRouter(prefix=router)
        self._router.add_api_route("/generate", self.generate, methods=["POST"])
        self._router.add_api_route("/status", self.status, methods=["GET"])
        self._router.add_api_route("/start", self.start, methods=["GET"])
        self._router.add_api_route("/stop", self.stop, methods=["GET"])
        self._lock = asyncio.Lock()

    @property
    def router(self):
        return self._router

    def start(self):
        if self._model is not None:
            return "running"
        self._processor = SwinProcessor(
            cached_path(
                "https://huggingface.co/microsoft/swin-base-patch4-window7-224/resolve/main/preprocessor_config.json"
            )
        )
        self._model = SwinForImageClassification(
            config_path=cached_path(
                "https://huggingface.co/microsoft/swin-base-patch4-window7-224/resolve/main/config.json"
            ),
            num_classes=4559,
        )
        self._model.from_pretrained(
            cached_path(
                "https://huggingface.co/datasets/fuliucansheng/unitorchblobfuse/resolve/main/models/pytorch_model.swin.base.patch4.window7.224.0.79.bin"
            ),
        )
        return "running"

    def stop(self):
        self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return "stopped"

    def status(self):
        if self._model is not None:
            return "running"
        return "stopped"

    async def generate(
        self,
        image: UploadFile,
        topk: int = 5,
    ):
        image_bytes = await image.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        async with self._lock:
            if self.status() != "running":
                self.start()
            inputs = self._processor.classification(image)
            pixel_values = inputs.pixel_values.unsqueeze(0)  # batch size 1
            results = (
                self._model(pixel_values)
                .softmax(dim=-1)
                .detach()
                .cpu()
                .numpy()
                .tolist()[0]
            )
            topk_indices = sorted(
                range(len(results)), key=lambda i: results[i], reverse=True
            )[:topk]
            results = [
                {
                    "category": self._cates.get(idx, "unknown"),
                    "score": results[idx],
                }
                for idx in topk_indices
            ]

        return results
