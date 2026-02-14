# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import io
import re
import cv2
import math
import gc
import json
import requests
import torch
import numpy as np
import gradio as gr
from PIL import Image, ImageDraw, ImageFilter
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor
from google import genai
from google.genai import types
from unitorch import mktempfile
from unitorch.utils import read_file
from unitorch.utils.decorators import retry
from unitorch.models import GenericOutputs
from unitorch.cli import CoreConfigureParser
from unitorch.cli.webuis import SimpleWebUI
from unitorch_microsoft import cached_path
from unitorch_microsoft.externals.github_copilot import (
    get_response as get_gpt5_response,
)
from unitorch_microsoft.externals.recraft import (
    get_image as get_recraft_image,
    get_inpainting_image as get_recraft_inpainting_image,
    get_change_background_image as get_recraft_change_background_image,
    get_resolution_image as get_recraft_resolution_image,
    get_remove_background_image as get_recraft_remove_background_image,
)
from unitorch_microsoft.spaces import (
    create_element,
    create_row,
    create_column,
    create_flex_layout,
    create_blocks,
    create_toper_menus,
    create_footer,
    create_dashboard_card,
    create_card,
    create_dashboard_cards_group,
    create_cards_group,
    call_fastapi,
)


@retry(times=3, base_delay=1, max_delay=5, exceptions=(RuntimeError,))
def call_nano_banana_generate(
    prompt: str, images: List[Image.Image] = [], model="base"
) -> Image.Image:
    gemini_client = genai.Client()
    response = gemini_client.models.generate_content(
        model=(
            "gemini-2.5-flash-image"
            if model == "base"
            else "gemini-3-pro-image-preview"
        ),
        contents=images + [prompt],
        config=types.GenerateContentConfig(
            image_config=types.ImageConfig(
                aspect_ratio="16:9",
            )
        ),
    )

    if not response.candidates:
        raise RuntimeError("No candidates returned")

    candidate = response.candidates[0]

    if not candidate.content or not candidate.content.parts:
        raise ValueError(
            f"No content parts returned. Finish reason: {candidate.finish_reason}"
        )

    image_parts = [
        part.inline_data.data
        for part in candidate.content.parts
        if part.inline_data and part.inline_data.data
    ]

    if not image_parts:
        raise RuntimeError("No image data found in content parts")
    
    image = Image.open(io.BytesIO(image_parts[0]))
    return image

def call_gemini3_generate(prompt: str, images: List[Image.Image] = []):
    gemini_client = genai.Client()
    images = [im for im in images if im is not None]
    result = gemini_client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=[prompt, *images],
    )
    result = result.text.replace("\t", " ").replace("\r", " ").replace("\n", " ")
    return result

def generate_by_whitepad(image, ratio):
    w, h = image.size
    src_ratio = w / h

    # 1. 计算目标画布尺寸（不裁剪原图）
    if src_ratio > ratio:
        # 原图偏宽 → 补高度
        target_w = w
        target_h = int(w / ratio)
    else:
        # 原图偏高 → 补宽度
        target_h = h
        target_w = int(h * ratio)

    # 2. 生成模糊背景
    bg = Image.new("RGB", (target_w, target_h), (255, 255, 255))
    
    # 3. 把原图居中贴回
    offset_x = (target_w - w) // 2
    offset_y = (target_h - h) // 2
    bg.paste(image, (offset_x, offset_y))

    return bg


def place_product_center_pil(
    background: Image.Image,
    product: Image.Image,
    occupy_ratio: float = 0.8,
) -> Image.Image:
    """
    Args:
        background: PIL Image (background)
        product: PIL Image (product, supports alpha)
        occupy_ratio: max ratio for width / height

    Returns:
        PIL Image
    """

    bg = background.convert("RGBA")
    prod = product.convert("RGBA")

    bg_w, bg_h = bg.size
    p_w, p_h = prod.size

    # ✅ 核心：contain 逻辑
    scale = min(
        (bg_w * occupy_ratio) / p_w,
        (bg_h * occupy_ratio) / p_h,
    )

    new_w = int(p_w * scale)
    new_h = int(p_h * scale)

    prod_resized = prod.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # 居中坐标
    paste_x = (bg_w - new_w) // 2
    paste_y = (bg_h - new_h) // 2

    # 透明贴图
    bg.paste(prod_resized, (paste_x, paste_y), prod_resized)

    return bg

def generate_by_center_crop(image, ratio):
    image_width, image_height = image.size
    image_ratio = image_width / image_height

    if image_ratio > ratio:
        # Image is too wide
        new_height = image_height
        new_width = int(ratio * new_height)
        new_x = (image_width - new_width) // 2
        new_y = 0
    else:
        # Image is too tall
        new_width = image_width
        new_height = int(new_width / ratio)
        new_x = 0
        new_y = (image_height - new_height) // 2

    cropped_image = image.crop(
        (new_x, new_y, new_x + new_width, new_y + new_height)
    )
    return cropped_image

def generate_by_solution1(image, ratio):
    prompt1 = """
# Background Image Design Skill – Book Display Ad

## Purpose
Create a subtle, high-performing background image for a **display ad banner** where an existing book cover will be placed on top. The background must **support the book**, preserve its original appearance, and include **intentional empty space** for book product placement.

This skill guides an AI image engine to analyze the book cover’s **visual design language** first, then generate a compatible background that enhances clarity, contrast, and advertising performance.

---

## Inputs
- **Book cover image** (final artwork, unchanged)
- **Target usage**: Display ad banners (multiple sizes and aspect ratios)

---

## Core Principles (Non‑Negotiable)

### 1. Stay Subtle
- The background must *never compete* with the book cover.
- Avoid strong focal points, high-detail illustrations, or narrative scenes.
- The book cover should remain the most visually dominant element.

### 2. Ensure Readability
- Maintain strong contrast between:
  - Book cover vs. background
- Avoid noisy textures, sharp patterns, or high-frequency details.
- Favor soft gradients, light grain, or smooth tonal transitions.
- Lightly integrate book cover design elements into the background to create depth and visual character, avoiding excessive details to preserve a clean aesthetic.

### 3. Design Empty Zones
- Intentionally reserve **clean, quiet areas** for:
  - Supporting copy
- Empty zones should be:
  - Visually calm
  - Low contrast

### 4. Mutate Colors (Do Not Copy)
- Derive the background palette from the book cover **indirectly**:
  - Desaturate
  - Lighten or darken
  - Shift hues slightly toward neutral tones
- Let the book cover carry the color richness and detail.
- Avoid exact color sampling that blends the book into the background.

### 5. Match Mood, Not Content
- Reflect the **tone** of the book, not literal imagery.

Examples:
- Business / Strategy → calm, confident, minimal, structured
- Technical / Non‑fiction → clean, precise, modern
- Fiction → atmospheric, restrained, emotionally suggestive

Do **not** illustrate scenes, characters, or literal metaphors from the book.

### 6. Be Crop‑Safe
- Design for flexibility across formats (e.g. 160×600, 300×250, 728×90).
- Keep all visual interest:
  - Away from edges
  - Center‑weighted or softly distributed
- Assume aggressive cropping may occur.

### 7. Think Performance First
- Prioritize clarity over decoration.
- Fast‑to‑scan visuals outperform complex aesthetics in ads.
- If a choice exists between “beautiful” and “clear,” choose **clear**.

---

## Process Steps

### Step 1: Analyze the Book Cover
Identify and document the book cover’s **visual design language** in detail before generating any background.

Analyze the following dimensions:

**Color & Contrast**
- Primary and secondary color palette
- Saturation level (muted vs. vivid)
- Contrast intensity (high-impact vs. soft)

**Typography Signals**
- Font weight (light, regular, bold)
- Font style (serif, sans-serif, condensed, geometric)
- Overall typographic tone (authoritative, modern, elegant, technical)

**Graphic Shapes & Geometry**
- Dominant shapes (rectangular blocks, circles, lines, grids, diagonals)
- Edge treatment (sharp, rounded, organic)
- Symmetry vs. asymmetry
- Sense of structure (rigid, modular, free-form)

**Patterns & Motifs**
- Repeating elements or visual rhythms
- Use of grids, stripes, dots, or abstract motifs
- Density of repetition (sparse vs. frequent)

**Overall Tone**
- Emotional and professional signal (premium, minimal, bold, calm, technical)

### Step 2: Define Layout Zones
- Book placement zone (dominant visual anchor)
- Safe margins for cropping

### Step 3: Generate Background Visual
- Use muted gradients, soft lighting, or abstract forms
- Avoid sharp edges, symbols, or literal imagery

### Step 4: Validate Against Principles
Confirm:
- Book remains visually dominant
- Background works across multiple aspect ratios

---

## Output Requirements
- No placeholder text or CTAs or product included
- No obvious product elements
- Background image only (book added later in a front view)
- Neutral, adaptable composition
- Suitable for display advertising
- Scales cleanly across common banner sizes

---

## Success Criteria
A successful background:
- Makes the book stand out instantly
- Feels intentional but invisible
- Supports messaging without dictating it
- Performs well in real ad environments, not just mockups

---

## Summary Rule
**The background exists to disappear. The book exists to convert.**
    """
    prompt2 = "Refine the book’s margins and add subtle, realistic shadowing to enhance depth and visual hierarchy. Ensure the book integrates naturally with the background while maintaining visual balance. Do not modify the book object itself and background. No text, no watermark, and no logos."
    prompt3 = "check if there is a book in the image, answer yes or no in <ans></ans> tag. yes for book exists, no for book not exists."

    for _ in range(3):
        step1 = call_nano_banana_generate(prompt1, images=[image], model="pro")
        ans = get_gpt5_response(prompt3, images=[step1]).replace(" ", "")
        if "<ans>no</ans>" in ans.lower():
            break
    step2 = place_product_center_pil(step1, image, occupy_ratio=0.8)
    step3 = call_nano_banana_generate(prompt2, [step2], model="pro")
    step4 = generate_by_center_crop(step3, ratio)
    return step4

def generate_by_solution2(image, ratio):
    prompt = """Can you create the background image for this book if I want to place this book on the generated background as a display ad banner? Follow the guidelines below and avoid copyright policy violation if needed. 

Firstly, understand the visual design language (color, shape, pattern, texture and font) of this book cover.
Secondly, generate the background and consider the design language of the book and follow the visual guidances below
Lastly, place the book in the center of the background. The book size should no smaller than 20% of the background size

Originality:  Keep the original look of book and retain the completeness of the asset as much as possible. Do not change the book cover design and shooting angle. 
Stay subtle: The background should support the book, never compete with it. 
Ensure readability: Keep strong contrast for book; avoid busy textures and complex patterns.
Design empty zones: Leave clean, quiet areas for the book.  
Mutate colors: Use muted or soft tones; let the book cover carry detail and color.
Match mood, not content: Express the book’s tone (business, fiction, technical), not literal scenes.
Be crop-safe: Avoid important details near edges; work across sizes and ratios.
Think performance: Clear, fast-to-scan visuals beat decorative complexity.
"""
    prompt2 = "check if the book in two images are the same, answer yes or no in <ans></ans> tag. yes for same book, no for different book."
    for _ in range(3):
        result = call_nano_banana_generate(prompt, [image], model="pro")
        ans = get_gpt5_response(prompt2, images=[image, result]).replace(" ", "")
        if "<ans>yes</ans>" in ans.lower():
            break
    result = generate_by_center_crop(result, ratio)
    return result

def get_image_score(image):
    url = "http://10.224.120.190:8000/predict"
    resp = call_fastapi(
        url,
        images={"file": image},
    )
    return resp["result_value"]

class BookBGWebUI(SimpleWebUI):
    _title = "Background Synthesis"
    _description = "This is a demo for background synthesis using Gemini & GPT. You can input an image of a product, and the model will generate a new image with an improved background suitable for e-commerce."

    def __init__(self, config: CoreConfigureParser):
        self._status = getattr(self, "_status", "Stopped")

        # create elements
        toper_menus = create_toper_menus()
        footer = create_footer()
        header = create_element(
            "markdown",
            label=f"# <div style='margin-top:10px'>✨ {self._title} </div>",
            interactive=False,
        )
        description = create_element(
            "markdown",
            label=self._description,
            interactive=False,
        )

        input_image = create_element("image", "Image")
        input_ratio = create_element(
            "slider", "Ratio", default=1.91, min_value=0.1, max_value=10.0, step=0.01
        )
        generate = create_element("button", "Generate")
        output_prod = create_element("image", "Prod Output")
        output_result1 = create_element("image", "Our Result 1")
        output_score1 = create_element("text", "Aesthetics Score 1")
        output_result2 = create_element("image", "Our Result 2")
        output_score2 = create_element("text", "Aesthetics Score 2")
        examples = gr.Examples(
            examples=[
                cached_path("spaces/picasso/examples/book1.png"),
                cached_path("spaces/picasso/examples/book2.png"),
                cached_path("spaces/picasso/examples/book3.png"),
                cached_path("spaces/picasso/examples/book4.png"),
                cached_path("spaces/picasso/examples/book5.png"),
                cached_path("spaces/picasso/examples/book6.png"),
            ],
            label="Image Examples",
            inputs=[input_image],
            examples_per_page=10,
        )

        left = create_column(input_image, examples.dataset, input_ratio, generate)
        right = create_column(output_prod, output_result1, output_score1, output_result2, output_score2)

        iface = create_blocks(
            toper_menus,
            create_row(
                create_column(header, description, scale=1),
                create_column(),
            ),
            create_row(
                left, right, elem_classes="ut-bg-transparent ut-ms-min-70-height"
            ),
            footer,
        )
        iface._title = self._title
        iface._description = self._description

        # create events
        iface.__enter__()

        generate.click(
            fn=self.generate,
            inputs=[input_image, input_ratio],
            outputs=[output_prod, output_result1, output_score1, output_result2, output_score2],
            trigger_mode="once",
        )

        examples.create()

        iface.__exit__()

        super().__init__(config, iname=self._title, iface=iface)

    def start(self):
        self._status = "Running"
        return self._status

    def stop(self):
        self._status = "Stopped"
        return self._status

    def generate(self, image, ratio):
        with ThreadPoolExecutor(max_workers=3) as executor:
            f_prod = executor.submit(generate_by_whitepad, image, ratio)
            f_sol1 = executor.submit(generate_by_solution1, image, ratio)
            f_sol2 = executor.submit(generate_by_solution2, image, ratio)

            prod_result = f_prod.result()
            sol1_result = f_sol1.result()
            sol2_result = f_sol2.result()

        sol1_score = get_image_score(sol1_result)
        sol2_score = get_image_score(sol2_result)
        return prod_result, sol1_result, sol1_score, sol2_result, sol2_score
