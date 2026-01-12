# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import json
import re
import fire
import random
import logging
import requests
import uvicorn
import pandas as pd
import gradio as gr
import httpx
import importlib_resources
from functools import partial
from urllib.parse import urljoin
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from unitorch.utils import read_file
from unitorch.cli import CoreConfigureParser
from unitorch.cli.webuis import SimpleWebUI
from unitorch.cli import (
    import_library,
    cached_path,
    init_registered_module,
)
from unitorch_microsoft import cached_path
from unitorch_microsoft.spaces.home import home_routers
from unitorch_microsoft.spaces.picasso import picasso_routers
from unitorch_microsoft.spaces.models import models_routers
from unitorch_microsoft.spaces.demos import demos_routers
from unitorch_microsoft.spaces.docs import docs_routers
from unitorch_microsoft.spaces.betas import betas_routers

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

assets_folder = os.path.join(
    importlib_resources.files("unitorch_microsoft"), "spaces/assets"
)
app.mount("/_publics", StaticFiles(directory=assets_folder), name="publics")

icon_path = os.path.join(assets_folder, "icon.ico")
pkg_theme_css = read_file(
    os.path.join(importlib_resources.files("unitorch"), "cli/assets/style.css")
)
app_theme_css = read_file(os.path.join(assets_folder, "style.css"))
theme_css = pkg_theme_css + "\n" + app_theme_css

page_routers = {
    **home_routers,
    **picasso_routers,
    **models_routers,
    **demos_routers,
    **docs_routers,
    **betas_routers,
}

page_routers = [(k, v) for k, v in page_routers.items()]
page_routers.sort(key=lambda x: len(x[0]), reverse=False)
page_routers_replaces = []

for index, (router, page) in enumerate(page_routers):
    page_routers_replaces.append((router, f"/_page{index}"))
    app = gr.mount_gradio_app(
        app,
        page,
        path=f"/_page{index}",
        favicon_path=icon_path,
        allowed_paths=["/"],
        css=theme_css,
    )

page_routers_replaces.reverse()

from fastapi import Request, Response
from urllib.parse import urljoin, urlparse, urlunparse
import httpx


@app.middleware("http")
async def ReverseMiddleware(request: Request, call_next):
    if not request.url.path.startswith("/_"):
        new_path = None
        for router, replace in page_routers_replaces:
            if router != "/" and request.url.path.startswith(router):
                new_path = request.url.path.replace(router, replace, 1)
                break
            elif router == "/":
                new_path = replace + request.url.path
                break

        if new_path:
            parsed_url = request.url
            new_url = urljoin(str(parsed_url), new_path)
            parsed_new = urlparse(new_url)
            new_url = urlunparse(parsed_new._replace(query=parsed_url.query))

            headers = dict(request.headers)
            headers.pop("host", None)
            headers.pop("content-length", None)

            async with httpx.AsyncClient(timeout=3600.0) as client:
                if request.method.upper() in ["GET", "DELETE"]:
                    response = await client.request(
                        method=request.method,
                        url=new_url,
                        headers=headers,
                        follow_redirects=True,
                    )
                else:
                    response = await client.request(
                        method=request.method,
                        url=new_url,
                        headers=headers,
                        content=await request.body(),
                        follow_redirects=True,
                    )

            # 🔥 关键修复点在这里
            excluded_headers = {
                "content-length",
                "transfer-encoding",
                "connection",
                "content-encoding",
            }

            clean_headers = {
                k: v
                for k, v in response.headers.items()
                if k.lower() not in excluded_headers
            }

            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=clean_headers,
            )

    return await call_next(request)


def cli_main(host: str = "0.0.0.0", port: int = 7654):
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    fire.Fire(cli_main)
