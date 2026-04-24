# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import uuid
import asyncio
import aiofiles
from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel
from unitorch.cli import register_fastapi, CoreConfigureParser, GenericFastAPI


class UploadResponse(BaseModel):
    path: str
    filename: str
    size: int


@register_fastapi("microsoft/apps/studios/utils")
class StudioUtilsFastAPI(GenericFastAPI):
    def __init__(self, config: CoreConfigureParser):
        self._config = config
        config.set_default_section("microsoft/apps/studios/utils")
        router = config.getoption("router", "/microsoft/apps/studio/utils")
        self._upload_dir = config.getoption("upload_dir", "/tmp/studios")

        os.makedirs(self._upload_dir, exist_ok=True)

        self._router = APIRouter(prefix=router)
        self._router.add_api_route("/upload", self.upload_file, methods=["POST"])
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

    async def upload_file(self, file: UploadFile = File(...)) -> UploadResponse:
        ext = os.path.splitext(file.filename or "")[1]
        save_name = f"{uuid.uuid4().hex}{ext}"
        save_path = os.path.join(self._upload_dir, save_name)

        size = 0
        async with aiofiles.open(save_path, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)  # 1 MB chunks
                if not chunk:
                    break
                await f.write(chunk)
                size += len(chunk)

        return UploadResponse(
            path=save_path,
            filename=file.filename or save_name,
            size=size,
        )
