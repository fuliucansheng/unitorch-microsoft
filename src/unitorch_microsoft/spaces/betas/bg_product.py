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
from PIL import Image, ImageDraw
from typing import Optional, List
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
    image_parts = [
        part.inline_data.data
        for part in response.candidates[0].content.parts
        if part.inline_data
    ]
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


def padding_to_the_ratio(
    image: Image.Image, ratio: float, method: str = "opencv"
) -> Image.Image:
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    width, height = image.size

    longest_side = 1536
    shortest_side = (
        int(longest_side * ratio) if ratio < 1 else int(longest_side / ratio)
    )
    size = (longest_side, shortest_side) if ratio > 1 else (shortest_side, longest_side)

    scale = min(size[0] / width, size[1] / height)
    if scale > 1:
        size = (int(size[0] // scale), int(size[1] // scale))
    if size[0] < 512:
        size = (512, int(size[1] * 512 / size[0]))
    if size[1] < 512:
        size = (int(size[0] * 512 / size[1]), 512)

    size = (size[0] // 8 * 8, size[1] // 8 * 8)

    scale = min(size[0] / width, size[1] / height)

    new_width = int(width * scale)
    new_height = int(height * scale)

    image = image.resize(
        (new_width // 8 * 8, new_height // 8 * 8), resample=Image.LANCZOS
    )

    im_width, im_height = image.size

    mask = Image.new("L", (size[0], size[1]), 255)
    black = Image.new("RGB", (im_width, im_height), (0, 0, 0))
    mask.paste(black, ((size[0] - im_width) // 2, (size[1] - im_height) // 2))
    new_image = Image.new("RGB", (size[0], size[1]), (255, 255, 255))
    new_image.paste(image, ((size[0] - im_width) // 2, (size[1] - im_height) // 2))
    image_np = np.array(new_image.convert("RGB"))
    mask_np = np.array(mask.convert("L")).astype(np.uint8)
    _, binary_mask = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)

    inpainted_image = cv2.inpaint(image_np, binary_mask, 10, cv2.INPAINT_TELEA)

    if method == "opencv":

        padded_image = Image.fromarray(inpainted_image)
    elif method == "recraft":
        padded_image = get_recraft_inpainting_image(
            "simple background", new_image, mask
        )
    else:
        caption = get_gpt5_response(
            "Describe the background of this image, maintaining its colors, textures, and lighting. Ensure seamless blending without adding new objects, text, or artifacts. The caption is in a single short paragraph. Don't mention any object in foreground.",
            images=[image],
        )
        latent_image = Image.fromarray(inpainted_image)
        flux_endpoint = "http://br1t45-s1-01:5050/core/fastapi/stable_flux/inpainting"
        padded_image = call_fastapi(
            flux_endpoint + "/generate",
            images={
                "image": latent_image,
                "mask_image": mask,
            },
            params={
                "text": "no text, no watermark, no logos, no people. " + caption,
                "guidance_scale": 30,
                "strength": 0.95,
                "num_timesteps": 50,
                "seed": 42,
            },
            resp_type="image",
        )
    return padded_image


class ProductBGWebUI(SimpleWebUI):
    _title = "Product Enhancement"
    _description = "This is a demo for enhancing general product images using Gemini and Recraft. You can input an image of a product, and the model will generate a new image with an improved background suitable for e-commerce."

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

        status = create_element("text", "Status", default="Stopped", interactive=False)
        start = create_element("button", "Start", variant="primary")
        stop = create_element("button", "Stop", variant="stop")
        input_image = create_element("image", "Image")
        input_ratio = create_element(
            "slider", "Ratio", default=1.91, min_value=0.1, max_value=10.0, step=0.01
        )
        input_text = create_element("text", "User Prompt", lines=6)
        generate = create_element("button", "Generate")
        output_image1 = create_element("image", "Step 1: Initial Generation")
        output_text2 = create_element("text", "Gemini3 Prompt for Background", lines=6)
        output_image3 = create_element("image", "Step 3: Padded Image")
        output_image4 = create_element("image", "Output Image")

        left = create_column(input_image, input_ratio, input_text, generate)
        right = create_column(
            output_text2,
            create_row(output_image1, output_image3, output_image4),
        )

        iface = create_blocks(
            toper_menus,
            create_row(
                create_column(header, description, scale=1),
                create_row(status, start, stop),
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

        start.click(
            self.start,
            outputs=[status],
            trigger_mode="once",
        )
        stop.click(
            self.stop,
            outputs=[status],
            trigger_mode="once",
        )

        generate.click(
            fn=self.generate,
            inputs=[input_image, input_ratio, input_text],
            outputs=[output_text2, output_image1, output_image3, output_image4],
            trigger_mode="once",
        )

        iface.load(
            fn=lambda: self._status,
            outputs=status,
        )

        iface.__exit__()

        super().__init__(config, iname=self._title, iface=iface)

    def start(self):
        self._status = "Running"
        return self._status

    def stop(self):
        self._status = "Stopped"
        return self._status

    def generate(self, image, ratio, user_prompt):
        prompt0 = """
# Background Image Design Skill – Product Display Ad

## Purpose
Create a subtle, high-performing background image for a **display ad banner** where an existing product will be placed on top. The background must **support the product**, preserve its original appearance, and include **intentional empty space** for product placement.

This skill guides an AI image engine to analyze the product’s **visual design language** first, then generate a compatible background that enhances clarity, contrast, and advertising performance.

---

## Inputs
- **product image** (final artwork, unchanged)
- **Target usage**: Display ad banners (multiple sizes and aspect ratios)

---

## Core Principles (Non‑Negotiable)

### 1. Stay Subtle
- The background must *never compete* with the product.
- Avoid strong focal points, high-detail illustrations, or narrative scenes.
- The product should remain the most visually dominant element.

### 2. Ensure Readability
- Maintain strong contrast between:
  - Product vs. background
- Avoid noisy textures, sharp patterns, or high-frequency details.
- Favor soft gradients, light grain, or smooth tonal transitions.
- Lightly integrate product color into the background to create depth and visual character, avoiding excessive details to preserve a clean aesthetic.

### 3. Design Empty Zones
- Intentionally reserve **clean, quiet areas** for:
  - Supporting copy
- Empty zones should be:
  - Visually calm
  - Low contrast

### 4. Mutate Colors (Do Not Copy)
- Derive the background palette from the product **indirectly**:
  - Desaturate
  - Lighten or darken
  - Shift hues slightly toward neutral tones
- Let the product carry the color richness and detail.
- Avoid exact color sampling that blends the product into the background.

### 5. Match Mood, Not Content
- Reflect the **tone** of the product, not literal imagery.

Examples:
- Business / Strategy → calm, confident, minimal, structured
- Technical / Non‑fiction → clean, precise, modern
- Fiction → atmospheric, restrained, emotionally suggestive

Do **not** illustrate scenes, characters, or literal metaphors from the product.

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

### Step 1: Analyze the Product
Identify and document the product’s **visual design language** in detail before generating any background.

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
- Product placement zone (dominant visual anchor)
- Safe margins for cropping

### Step 3: Generate Background Visual
- Use muted gradients, soft lighting, or abstract forms
- Avoid sharp edges, symbols, or literal imagery

### Step 4: Validate Against Principles
Confirm:
- Product remains visually dominant
- Background works across multiple aspect ratios

---

## Output Requirements
- No placeholder text or CTAs or product included
- No obvious product elements
- Background image only (product added later)
- Neutral, adaptable composition
- Suitable for display advertising
- Scales cleanly across common banner sizes

---

## Success Criteria
A successful background:
- Makes the product stand out instantly
- Feels intentional but invisible
- Supports messaging without dictating it
- Performs well in real ad environments, not just mockups

---

## Summary Rule
**The background exists to disappear. The product exists to convert.**
        """
        prompt1 = "generate a product from the provided the first image in front view, put the product in the center with a proper size, use the second image as the background, adjust the lighting and colors to ensure visual harmony, no text, no watermark, no logos."
        prompt1 = "put the product in the first image into the background of the second image, adjust the lighting and colors to ensure visual harmony, no text, no watermark, no logos."
        if user_prompt is not None and user_prompt.strip() != "":
            prompt0 += f"\n## More Context: {user_prompt}\n"
            prompt1 += f"\n## More Context: {user_prompt}\n"

        step2 = call_nano_banana_generate(prompt0, images=[image], model="pro")
        step3 = call_nano_banana_generate(prompt1, [image, step2], model="base")
        return None, step2, step3, step3
