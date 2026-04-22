# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import io
import base64
import asyncio
import httpx
from PIL import Image
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import StreamingResponse
from typing import List, Optional

import litellm

from unitorch.cli import (
    register_fastapi,
)
from unitorch.cli import CoreConfigureParser, GenericFastAPI


@register_fastapi("microsoft/apps/spaces/gpt/image-15")
class GPTImageFastAPI(GenericFastAPI):
    def __init__(self, config: CoreConfigureParser):
        self._config = config
        config.set_default_section("microsoft/apps/spaces/gpt/image")
        self._base_url = config.getoption("base_url", "http://127.0.0.1:4000")
        self._generate_model = config.getoption("generate_model", "papyrus-gpt-image-15-eval")
        self._edit_model = config.getoption("edit_model", "papyrus-gpt-image-15-eval")
        router = config.getoption("router", "/microsoft/apps/spaces/gpt/image-15")
        self._router = APIRouter(prefix=router)
        self._router.add_api_route("/generate", self.generate, methods=["POST"])
        self._router.add_api_route("/edit", self.edit, methods=["POST"])
        self._router.add_api_route("/status", self.status, methods=["GET"])
        self._router.add_api_route("/start", self.start, methods=["GET"])
        self._router.add_api_route("/stop", self.stop, methods=["GET"])
        self._lock = asyncio.Lock()
        self._running = False

    @property
    def router(self):
        return self._router

    def start(self):
        self._running = True
        return "running"

    def stop(self):
        self._running = False
        return "stopped"

    def status(self):
        return "running" if self._running else "stopped"

    async def generate(
        self,
        prompt: str,
        size: Optional[str] = "1024x1024",
        background: Optional[str] = "transparent",
    ):
        data = {
            "model": self._generate_model,
            "prompt": prompt,
            "size": size,
            "background": background,
            "quality": "medium",
        }

        async with self._lock:
            if self.status() != "running":
                self.start()
            async with httpx.AsyncClient() as client:
                request = client.build_request(
                    "POST",
                    f"{self._base_url}/v1/images/generations",
                    headers={"Authorization": "Bearer litellm"},
                    json=data,
                    timeout=120,
                )
                resp = await client.send(request)
                resp.raise_for_status()
                response = resp.json()

        image_data = base64.b64decode(response["data"][0]["b64_json"])
        result = Image.open(io.BytesIO(image_data)).convert("RGB")
        buffer = io.BytesIO()
        result.save(buffer, format="PNG")
        return StreamingResponse(
            io.BytesIO(buffer.getvalue()),
            media_type="image/png",
        )

    async def edit(
        self,
        prompt: str,
        images: List[UploadFile] = File(...),
        mask: Optional[UploadFile] = File(default=None),
        size: Optional[str] = "1024x1024",
        background: Optional[str] = "transparent",
    ):
        image_bytes = [(f.filename or f"image{i}.png", await f.read()) for i, f in enumerate(images)]
        mask_bytes = (mask.filename or "mask.png", await mask.read()) if mask is not None else None

        files = [("image[]", (name, data, "image/png")) for name, data in image_bytes]
        if mask_bytes is not None:
            files.append(("mask", (mask_bytes[0], mask_bytes[1], "image/png")))

        data = {
            "model": self._edit_model,
            "prompt": prompt,
            "size": size,
            "background": background,
            "quality": "medium",
            "input_fidelity": "high",
        }

        async with self._lock:
            if self.status() != "running":
                self.start()
            async with httpx.AsyncClient() as client:
                request = client.build_request(
                    "POST",
                    f"{self._base_url}/v1/images/edits",
                    headers={"Authorization": "Bearer litellm"},
                    files=files,
                    data=data,
                    timeout=120,
                )
                resp = await client.send(request)
                resp.raise_for_status()
                response = resp.json()

        image_data = base64.b64decode(response["data"][0]["b64_json"])
        result = Image.open(io.BytesIO(image_data)).convert("RGB")
        buffer = io.BytesIO()
        result.save(buffer, format="PNG")
        return StreamingResponse(
            io.BytesIO(buffer.getvalue()),
            media_type="image/png",
        )

