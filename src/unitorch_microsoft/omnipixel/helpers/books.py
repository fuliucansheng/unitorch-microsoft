# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import io
import re
import fire
import json
import logging
import hashlib
import subprocess
import tempfile
import pandas as pd
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from google import genai
from google.genai import types


def save_image(folder, image):
    name = hashlib.md5(image.tobytes()).hexdigest() + ".jpg"
    image.save(f"{folder}/{name}")
    return f"{folder}/{name}"


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


def process1(image):
    prompt0 = """
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
    prompt1 = "generate a book product from the provided the first cover image in front view, put the product in the center with a proper size, use the second image as the background, adjust the lighting and colors to ensure visual harmony, no text, no watermark, no logos."
    prompt2 = "Refine the book’s margins and add subtle, realistic shadowing to enhance depth and visual hierarchy. Ensure the book integrates naturally with the background while maintaining visual balance. Do not modify the book object itself and background. No text, no watermark, and no logos."

    step2 = None
    for _ in range(3):
        try:
            step2 = call_nano_banana_generate(prompt0, images=[image], model="pro")
            break
        except Exception as e:
            logging.warning(f"Retrying step 2 due to error: {e}")
    if step2 is None:
        raise RuntimeError("Failed to generate step 2 image after multiple attempts.")
    step3 = place_product_center_pil(step2, image, occupy_ratio=0.8)

    step4 = None
    for _ in range(3):
        try:
            step4 = call_nano_banana_generate(prompt2, [step3], model="pro")
            break
        except Exception as e:
            logging.warning(f"Retrying step 4 due to error: {e}")

    if step4 is None:
        raise RuntimeError("Failed to generate step 4 image after multiple attempts.")
    return step4


def batch1(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        try:
            image = Image.open(input_path).convert("RGB")
            processed_image = process1(image)
            processed_image.save(output_path)
            print(f"Processed and saved: {output_path}")
        except Exception as e:
            print(f"Failed to process {input_path}: {e}")

def process2(image):
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
Think performance: Clear, fast-to-scan visuals beat decorative complexity."""
    result = call_nano_banana_generate(prompt, images=[image], model="pro")
    return result

def batch2(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        try:
            image = Image.open(input_path).convert("RGB")
            processed_image = process2(image)
            processed_image.save(output_path)
            print(f"Processed and saved: {output_path}")
        except Exception as e:
            print(f"Failed to process {input_path}: {e}")   

if __name__ == "__main__":
    fire.Fire({
        "batch1": batch1,
        "batch2": batch2,
    })
