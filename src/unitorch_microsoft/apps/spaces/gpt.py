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
from unitorch.cli import (
    register_fastapi,
)
from unitorch.cli import Config, GenericFastAPI


@register_fastapi("microsoft/apps/spaces/gpt/image")
class GPTImageFastAPI(GenericFastAPI):
    def __init__(self, config: Config):
        self._config = config
        config.set_default_section("microsoft/apps/spaces/gpt/image")
        self._base_url = config.getoption("base_url", "http://127.0.0.1:4000")
        self._api_key = config.getoption("api_key", "litellm")
        self._generate_model = config.getoption("generate_model", "papyrus-gpt-image-2-eval")
        self._edit_model = config.getoption("edit_model", "papyrus-gpt-image-2-eval")
        router = config.getoption("router", "/microsoft/apps/spaces/gpt/image")
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
                    headers={"Authorization": f"Bearer {self._api_key}"},
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
                    headers={"Authorization": f"Bearer {self._api_key}"},
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


@register_fastapi("microsoft/apps/spaces/gpt/chat")
class GPTChatFastAPI(GenericFastAPI):
    def __init__(self, config: Config):
        self._config = config
        config.set_default_section("microsoft/apps/spaces/gpt/chat")
        self._model = config.getoption("model", "github_copilot/gpt-5.5")
        self._api_base = config.getoption("api_base", "http://127.0.0.1:4000")
        self._api_key = config.getoption("api_key", "litellm")
        self._max_tokens = config.getoption("max_tokens", 2048)
        self._temperature = config.getoption("temperature", 0.7)
        router = config.getoption("router", "/microsoft/apps/spaces/gpt/chat")
        self._router = APIRouter(prefix=router)
        self._router.add_api_route("/generate", self.generate, methods=["POST"])
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
        model: Optional[str] = None,
        system: Optional[str] = None,
        images: Optional[List[UploadFile]] = File(default=None),
        max_tokens: Optional[int] = None,
    ):
        content = [{"type": "text", "text": prompt}]
        if images:
            for image in images:
                image_data = await image.read()
                b64 = base64.b64encode(image_data).decode("utf-8")
                media_type = image.content_type or "image/png"
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{b64}"},
                })

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})

        async with self._lock:
            if self.status() != "running":
                self.start()
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{self._api_base}/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json={
                        "model": model or self._model,
                        "messages": messages,
                        "max_tokens": max_tokens or self._max_tokens,
                    },
                    timeout=120,
                )
                resp.raise_for_status()
                response = resp.json()

        return {
            "content": response["choices"][0]["message"]["content"],
            "model": response["model"],
            "usage": {
                "prompt_tokens": response["usage"]["prompt_tokens"],
                "completion_tokens": response["usage"]["completion_tokens"],
                "total_tokens": response["usage"]["total_tokens"],
            },
        }
