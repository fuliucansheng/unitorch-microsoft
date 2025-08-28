# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import glob
import os
import io
import logging
import tempfile
import subprocess
from urllib import response
import requests
from PIL import Image
from typing import Optional, Any
from pydantic import BaseModel, Field
from playwright.async_api import async_playwright
from unitorch_microsoft import cached_path
from unitorch_microsoft.chatgpt.recraft import (
    get_remove_background_image as get_recraft_remove_background_image,
)
from unitorch_microsoft.agents.components import (
    GenericTool,
    GenericResult,
    GenericError,
)
from unitorch_microsoft.agents.components.picasso import get_picasso_temp_dir

tailwind_file = cached_path("agents/components/picasso/tailwind-browser.js")

_browser = None
_playwright = None
_http_file_process = None

async def init_browser():
    global _browser, _playwright, _http_file_process
    if _browser is None:
        _playwright = await async_playwright().start()
        _browser = await _playwright.chromium.launch(headless=True)
        print("Browser initialized successfully.")
    else:
        print("Browser already initialized.")
    if _http_file_process is None:
        _http_file_process = subprocess.Popen(
            ["python3", "-m", "http.server", "49876"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd="/",
        )
    return _browser

async def html_to_image(html: str, viewport=(1920, 1080), use_tailwind_template=True) -> Image.Image:
    global _browser
    if _browser is None:
        _browser = await init_browser()
    page = await _browser.new_page(
        viewport={"width": viewport[0], "height": viewport[1]}
    )

    if use_tailwind_template:
        full_html = f"""
        <html>
            <head>
                <meta charset="UTF-8" />
                <meta name="viewport" content="width=device-width, initial-scale=1.0" />
                <script src="http://127.0.0.1:49876/{tailwind_file}"></script>
                <style>
                    body {{
                        margin: 0;
                        padding: 0;
                        font-family:
                            -apple-system, BlinkMacSystemFont, /* macOS/iOS */
                            'Noto Sans',                       /* 谷歌多语种 */
                            'Noto Sans CJK SC',                /* 简体中文 */
                            'WenQuanYi Zen Hei',               /* 开源微软雅黑替代 */
                            'Helvetica Neue', Helvetica, Arial, sans-serif;
                    }}
                </style>
            </head>
            <body>
                <div id="rendered-content">
                    {html}
                </div>
            </body>
        </html>
        """
    else:
        full_html = f"""
        <html>
        <head>
            <meta charset="UTF-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        </head>
        <body>
            <div id="rendered-content">
                {html}
            </div>
        </body>
        """

    await page.set_content(full_html, wait_until="networkidle")

    element = await page.query_selector("#rendered-content")
    screenshot_bytes = await element.screenshot(type="png")

    await page.close()

    return Image.open(io.BytesIO(screenshot_bytes))

_HTML_DESCRIPTION = """
Use the html_to_image tool to convert structured HTML content into high-quality image output. This is ideal for generating posters, social media cards, banners, or any composite layout involving styled text, images, and layout elements.

* Input: A valid HTML string starting with a <div> tag. Tailwind css files is included by default.
* Output: A rendered image (PNG format) from your HTML content.
* Styling: Supports Tailwind CSS for utility-first styling. Tailwind css files is included by default.
* Assets: Embed image assets using URLs like http://127.0.0.1:49876/<absolute-path-to-file>.
* Viewport: Default rendering size is 1920x1080 pixels.

Follow these Tailwind guidelines to ensure your poster designs are visually consistent, pixel-perfect, and responsive to layout constraints:

1. Consistent Element Sizing
Use fixed sizing: w-[value], h-[value], max-w-[value] to avoid layout shifts.
Add min-w-0 on flex children to prevent overflow clipping.
Apply the same width/height rules across similar components (e.g. buttons, avatars, thumbnails).
2. Precise Alignment & Spacing
Use flex utilities:
Horizontal: flex flex-row items-center justify-between
Vertical: flex flex-col items-center justify-center
Apply spacing between siblings using space-x-* or space-y-* for consistent gaps.
3. Typography Hierarchy
Follow text sizing rules: text-sm, text-base, text-lg, text-xl, etc.
Set line height thoughtfully: leading-tight, leading-normal, or leading-relaxed for readability.
Use font-medium, font-bold etc. to match your visual branding.
4. Uniform Padding & Margin
Use Tailwind’s scale: p-4, px-6, mt-8, etc.
Avoid custom or arbitrary spacing (style="margin: 7px"), unless explicitly required.
Combine spacing utilities (e.g., px-6 py-4) for balanced padding.
5. Box Model & Layout Structure
Always use box-border to include padding and border in width/height calculations.
For grid layouts, use grid with gap-* to define spacing, e.g. grid-cols-3 gap-6.
"""

class PicassoHtmlTool(GenericTool):
    """Add a tool to generate images based on the provided prompt & images."""

    name: str = "picasso_html_tool"
    description: str = _HTML_DESCRIPTION
    parameters: str = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "html_to_image",
                ],
                "description": """
                The action to perform on the image. Options are: 
                * 'html_to_image' render the raw_html to image.
                """,
            },
            "raw_html": {
                "type": "string",
                "description": "The raw html text to be rendered",
            },
            "viewport": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "The viewport size for html_to_image action. Default is [1920, 1080].",
                "default": [1920, 1080],
                "minItems": 2,
                "maxItems": 2,
            },
        },
        "required": ["raw_html"],
    }

    async def execute(
        self,
        action: str,
        raw_html: str,
        viewport: Optional[tuple[int, int]] = (1920, 1080),
    ) -> str:
        if action == "html_to_image":
            if not raw_html:
                raise ValueError("raw_html is required for html_to_image action.")
            result = await html_to_image(raw_html, viewport=viewport)

        if isinstance(result, Image.Image):
            temp_file = tempfile.NamedTemporaryFile(
                suffix=".png", dir=get_picasso_temp_dir(), delete=False
            )
            result.save(temp_file.name)
            result = temp_file.name

        if result is None:
            return {
                "result": {
                    "image": None,
                    "width": None,
                    "height": None,
                },
                "message": "Failed to generate image.",
            }

        res = Image.open(result)

        return GenericResult(
            output=f"Generated image path: {result} . Width: {res.width}, Height: {res.height}.",
            images={"path": result},
            meta={
                "_width": res.width,
                "_height": res.height,
                "_image": result,
            },
        )


async def layout_to_image(
    title: str,
    description: str,
    call_to_action: str,
    call_to_action_size: tuple[int, int],
    product_image: str,
    logo_image: str,
    background_prompt: Optional[str] = None,
    viewport: Optional[tuple[int, int]] = (1920, 1080),
):
    api_url = "http://br1t45-s1-01:8787/generate"
    asset_url = "http://10.172.118.59:49876/{0}"
    response = requests.post(
                api_url,
                json={
                    "grid_width": viewport[0],
                    "grid_height": viewport[1],
                    "headline": title,
                    "subcopy": description,
                    "cta": call_to_action,
                    "cta_width": call_to_action_size[0],
                    "cta_height": call_to_action_size[1],
                    "background_prompt": background_prompt,
                    "product_image_url": asset_url.format(product_image),
                    "logo_image_url": asset_url.format(logo_image),
                },
                headers={"Content-Type": "application/json"},
                timeout=60,
    ).json()
    html_results = response.get('html_files', [])
    for html_results in html_results:
        html_str = html_results.get('content', '')
        if not html_str:
            continue
        logging.info(f"HTML content from interal layout tool: {html_str}")
        image = await html_to_image(html_str, viewport=viewport, use_tailwind_template=False)
        temp_file = tempfile.NamedTemporaryFile(
                suffix=".png", dir=get_picasso_temp_dir(), delete=False
        )
        image.save(temp_file.name)
        return temp_file.name
    return None
    


_LAYOUT_DESCRIPTION = """
Use the layout_to_image tool to create visually appealing layouts for posters, banners, or social media cards with the provided inputs.
"""

class PicassoLayoutTool(GenericTool):
    """Add a tool to generate images based on the provided layout parameters."""

    name: str = "picasso_layout_tool"
    description: str = _LAYOUT_DESCRIPTION
    parameters: str = {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "The title of the generated layout.",
            },
            "description": {
                "type": "string",
                "description": "The description of the generated layout.",
            },
            "product_image": {
                "type": ["string"],
                "description": (
                    'The product image path.'
                ),
            },
            "logo_image": {
                "type": ["string"],
                "description": (
                    'The logo image path'
                ),
            },
            "call_to_action": {
                "type": "string",
                "description": "The call to action text.",
            },
            "call_to_action_size": {
                "type": "array",
                "items": {"type": "integer"},
                "description": (
                    "The size of the call to action button as [width, height]. (default is [200, 50])"
                ),
                "default": [200, 50],
                "minItems": 2,
                "maxItems": 2,
            },
            "background_prompt": {
                "type": ["string", "null"],
                "description": (
                    'The background prompt for the background of the generatted layout image. (default is None)'
                ),
            },
            "viewport": {
                "type": ["array", "null"],
                "items": {"type": "integer"},
                "description": (
                    "The viewport size for layout_to_image action. Default is [1920, 1080]."
                ),
                "default": [1920, 1080],
                "minItems": 2,
                "maxItems": 2,
            },
        },
        "required": [
            "title",
            "description",
            "product_image",
            "logo_image",
            "call_to_action",
        ],
    }

    async def execute(
        self,
        title: str,
        description: str,
        product_image: str,
        logo_image: str,
        call_to_action: str,
        call_to_action_size: Optional[tuple[int, int]] = (200, 50),
        background_prompt: Optional[str] = None,
        viewport: Optional[tuple[int, int]] = (1920, 1080),
    ) -> GenericResult:
        if not title or not description or not product_image or not logo_image:
            raise ValueError(
                "title, description, product_image and logo_image are required."
            )
        
        product = Image.open(product_image)
        logo = Image.open(logo_image)

        if product.mode != "RGBA":
            product = get_recraft_remove_background_image(product)
            temp_file = tempfile.NamedTemporaryFile(
                suffix=".png", dir=get_picasso_temp_dir(), delete=False
            )
            product.save(temp_file.name)
            product_image = temp_file.name
        if logo.mode != "RGBA":
            logo = get_recraft_remove_background_image(logo)
            temp_file = tempfile.NamedTemporaryFile(
                suffix=".png", dir=get_picasso_temp_dir(), delete=False
            )
            logo.save(temp_file.name)
            logo_image = temp_file.name

        try:
            result = await layout_to_image(
                title=title,
                description=description,
                call_to_action=call_to_action,
                call_to_action_size=call_to_action_size,
                product_image=product_image,
                logo_image=logo_image,
                background_prompt=background_prompt,
                viewport=viewport,
            )
            if result is None:
                return GenericError(
                    "Failed to generate layout image.",
                )
            res = Image.open(result)
        except Exception as e:
            return GenericError(
                f"Failed to generate layout image: {str(e)}",
            )
        
        return GenericResult(
            output=f"Generated layout image path: {result} . Width: {res.width}, Height: {res.height}.",
            images={"path": result},
            meta={
                "_width": res.width,
                "_height": res.height,
                "_image": result,
            },
        )