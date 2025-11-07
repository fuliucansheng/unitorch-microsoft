# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import glob
import os
import io
import re
import json
import logging
import tempfile
import subprocess
from datetime import datetime
from urllib import response
import requests
from PIL import Image
from typing import Optional, Any
from pydantic import BaseModel, Field
from playwright.async_api import async_playwright
from unitorch_microsoft import cached_path
from unitorch_microsoft.externals.papyrus import (
    get_gpt5_response,
)
from unitorch_microsoft.externals.recraft import (
    get_remove_background_image as get_recraft_remove_background_image,
)
from unitorch_microsoft.agents.components import (
    GenericTool,
    GenericResult,
    GenericError,
    ToolCollection,
    Memory,
    Message,
    ToolCall,
    ToolChoice,
)
from unitorch_microsoft.agents.components.tools import GPT4FormatTool, TerminateTool
from unitorch_microsoft.agents.utils.chatgpt import GPTModel
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


async def html_to_image(
    html: str, viewport=(1920, 1080), use_tailwind_template=True
) -> Image.Image:
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
* Output: A rendered image (PNG format) from your HTML content. Feedback about the result will be provided.
* Styling: Supports Tailwind CSS for utility-first styling. Tailwind css files is included by default.
* Assets: Embed image assets using URLs like http://127.0.0.1:49876/<absolute-path-to-file>.
* Viewport: Default rendering size is 1920x1080 pixels.
* Refinement: You can set `auto_refine_steps` to automatically refine the design. Suggest to set it at least to 20 for high quality result.

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

_HTML_REFINE_SYSTEM_PROMPT = """
You are a professional graphic designer. You need to create a poster image based on the user's prompt with html_to_image tool. 

* You may need multiple iterations to refine the html code, by checking the previous generated image to ensure it meets the requirements finally.
* Only use one tool call in each step.

You can use the following tools to complete tasks:
1. `gpt4_format`: Use this tool to put the refined html code into the required format which will be used to generate the image.
2. `terminate`: The designed image is perfect, and you can terminate the process.

Design the image with Tailwind CSS, including text, images, and styles. Follow these guidelines to ensure your poster designs are visually consistent, pixel-perfect, and responsive to layout constraints:
* Overall Style:
    * Sophisticated, futuristic, high-tech aesthetic.
    * Flat design with a card-based layout; harmonious card colors, spacing, and alignment with the overall theme.
        * Choose a color scheme suited to the content (e.g., Morandi palette, advanced gray tones, Memphis, Mondrian).
        * No gradients. Maintain high text-background contrast for readability.
        * Avoid plain white backgrounds unless explicitly requested.
        * Avoid ordinary or tacky designs.
        * Treat the page as one unified container — no nested cards or isolated container blocks.
* Layout Guidelines:
    * Avoid overlap, clutter, emptiness, overflow, or truncation.
    * Color palette must be harmonious and consistent with the theme; avoid harsh tones.
    * Place content in the visual focal area. If minimal content, enlarge fonts slightly and center the card.
    * For multiple cards, ensure a clear arrangement, alignment, and proportionate sizing with minimal unused white space. No overlapping or cluttering of cards.
    * All elements must be fully visible within the page boundaries with proper size. Adjust font sizes, card/image/video dimensions, spacing, and positioning to maintain balance.
    * Horizontal (landscape) images should generally use a left–right structure — main subject and key visual elements balanced or contrasted between the left and right sides.
    * Vertical (portrait) images should generally use a top–bottom structure — main subject and key visual elements arranged from top to bottom in a visually balanced flow.
    * The overall layout should be centered horizontally and vertically within the canvas.
    * Don't put the logo in the corner of the image.
    * Don't show any grid lines or borders even it's opcaity is low.

When checking the previous designed image, consider:
1. Is the overall composition visually appealing and professionally executed?
2. Does the design clearly communicate its intended purpose or message?
3. Does the aesthetic reflect a modern and refined visual style?
4. Are there flaws such as text, product or any key elements overlap (including the elements from background from visual view), inconsistent spacing, imbalanced sizing or misaligned elements?
5. Are horizontal and vertical alignments precise and intentional?
6. What specific, actionable improvements can enhance the visual outcome?
7. Is the product image preserved and integrated effectively into the design if a product is provided?
8. Does the image meet the user's requirements and expectations?
9. Does the image have wired artifacts, lines, or visual noise?
10. Does the image has any issues from the visual design perspective to be improved?

If flaws are identified, refine it or create a new one right now. Please focus on the high-priority issues that can significantly enhance the design quality.
If the overall looks prefect, use the `terminate` tool to end the process.
"""


class RefinedHtml(BaseModel):
    raw_html: str = Field(
        description="The raw html text to be rendered",
    )


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
            "auto_refine_steps": {
                "type": "integer",
                "description": "The maximum number of steps to automatically refine the html design. Default is 20. If you don't want to refine, set it to 0. Suggest to set it at least to 20 for high quality result.",
                "default": 20,
                "minimum": 0,
                "maximum": 30,
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
    gpt: Any = GPTModel()
    available_tools: ToolCollection = ToolCollection(
        GPT4FormatTool(RefinedHtml),
        TerminateTool(),
    )

    async def execute(
        self,
        action: str,
        raw_html: str,
        auto_refine_steps: Optional[int] = 20,
        viewport: Optional[tuple[int, int]] = (1920, 1080),
    ) -> str:
        if action == "html_to_image":
            if not raw_html:
                raise ValueError("raw_html is required for html_to_image action.")
            result = await html_to_image(raw_html, viewport=viewport)
            temp_file = tempfile.NamedTemporaryFile(
                suffix=".png", dir=get_picasso_temp_dir(), delete=False
            )
            result.save(temp_file.name)
            result = temp_file.name

            refined_html = raw_html
            designed_html, designed_image = raw_html, result
            system_message = Message.system_message(content=_HTML_REFINE_SYSTEM_PROMPT)
            current_date = datetime.now().strftime("%Y-%m-%d")
            user_message = Message.user_message(
                content=f"""
refine the previous design or create a new one if it's not perfect. 

* Don't change the assets content in the html code including title, images, videos. 
    * Background image (if any) could be changed to fit the designed canvas size. Make sure it's visible in the design.
    * Make sure these assets are visible and properly placed in the design.
* Don't change the viewport size.
* You can optimize the existing assets in terms of size, color, and layout, and add more assets to make the design look more professional and appealing.
* Today is {current_date}, please don't use any outdated elements in the design.
* Don't add any new links or qrcodes in the design because they may not be accessible.
* Ensure that assets do not visually overlap with important background elements which may affect the overall visual experience.
""",
            )
            histories = [result]
            logging.info(f"Designed image: {result}")
            for _ in range(auto_refine_steps):
                assistant_message = Message.assistant_message(
                    content=f"""
The previous design is as follows. The HTML code is:
{designed_html}
The designed image is: {designed_image}
You need to check if the designed image looks perfect. If not, refine or rewrite the HTML code to improve it.
""",
                    images=[
                        {
                            "path": designed_image,
                            "width": None,
                            "height": None,
                            "priority": "high",
                        }
                    ],
                )

                for _ in range(3):
                    resp = self.gpt.ask_tools(
                        messages=Memory(
                            messages=[
                                system_message,
                                user_message,
                                assistant_message,
                            ]
                        ).to_dict_list(),
                        tools=self.available_tools.to_params(),
                        tool_choice=ToolChoice.REQUIRED,
                    )
                    tool_calls = [ToolCall(**tc) for tc in resp.tool_calls]
                    if len(tool_calls) == 1:
                        break
                    print("Poster Tool calls:", tool_calls)

                tool_call = tool_calls[0]
                if tool_call.function.name == "terminate":
                    logging.info("Finish the html refine process.")
                    break
                args = json.loads(tool_calls[0].function.arguments)
                raw_html = args.get("raw_html", None)
                designed_image = await html_to_image(raw_html, viewport=viewport)
                temp_file = tempfile.NamedTemporaryFile(
                    suffix=".png", dir=get_picasso_temp_dir(), delete=False
                )
                designed_image.save(temp_file.name)
                designed_html, designed_image = raw_html, temp_file.name
                #                 ans = get_gpt5_response(
                #                     """
                # You are an unbiased visual evaluator. Compare Image1 (the first image) and Image2 (the second image) purely from a user’s visual experience perspective.

                # * Consider clarity, composition, visual appeal, and overall user-friendliness.
                # * Ignore minor cosmetic details that do not significantly affect the user’s perception.
                # * Select only the image that would be more visually appealing and engaging to a general audience.

                # Output Format:
                # Respond only with:
                # <ans>image1</ans> or <ans>image2</ans>""",
                #                     images=[result, designed_image],
                #                 )
                #                 check_result = re.search(r"<ans>(.*?)</ans>", ans, re.IGNORECASE)
                #                 check_result = (
                #                     check_result.group(1).strip().lower() if check_result else None
                #                 )
                #                 if check_result == "image2":
                result = designed_image
                refined_html = designed_html
                histories.append(designed_image)
                logging.info(f"Designed image: {designed_image}")

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
            output=f"Generated image path: {result} . Width: {res.width}, Height: {res.height}. Feedback: The result looks perfect.",
            images={"path": result},
            meta={
                "refined_html": refined_html,
                "_width": res.width,
                "_height": res.height,
                "_image": result,
                "_histories": histories,
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
    html_results = response.get("html_files", [])
    for html_results in html_results:
        html_str = html_results.get("content", "")
        if not html_str:
            continue
        logging.info(f"HTML content from interal layout tool: {html_str}")
        image = await html_to_image(
            html_str, viewport=viewport, use_tailwind_template=False
        )
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
                "description": ("The product image path."),
            },
            "logo_image": {
                "type": ["string"],
                "description": ("The logo image path"),
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
                    "The background prompt for the background of the generatted layout image. (default is None)"
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
