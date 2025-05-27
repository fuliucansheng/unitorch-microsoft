# Copyright (c) MICROSOFT.
# Licensed under the MIT License.
import os
import json
import re
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
from unitorch.cli.webuis.utils import create_blocks
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
for router, page in page_routers.items():
    page.theme_css = theme_css
    page.css = theme_css

page_routers = [(k, v) for k, v in page_routers.items()]
page_routers.sort(key=lambda x: len(x[0]), reverse=False)
page_routers_replaces = []

for index, (router, page) in enumerate(page_routers):
    page_routers_replaces.append((router, f"/_page{index}"))
    app = gr.mount_gradio_app(app, page, path=f"/_page{index}", favicon_path=icon_path)

page_routers_replaces.reverse()


@app.middleware("http")
async def ReverseMiddleware(request: Request, call_next):
    if not request.url.path.startswith("/_"):
        for router, replace in page_routers_replaces:
            if router != "/" and request.url.path.startswith(router):
                new_path = request.url.path.replace(router, replace)
                break
            elif router == "/":
                new_path = replace + request.url.path
                break
        new_url = urljoin(str(request.url), new_path)
        async with httpx.AsyncClient() as client:
            response = await client.request(
                request.method,
                new_url,
                headers=request.headers.raw,
                data=await request.body(),
                follow_redirects=True,
            )

        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=response.headers,
        )

    return await call_next(request)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7654)
