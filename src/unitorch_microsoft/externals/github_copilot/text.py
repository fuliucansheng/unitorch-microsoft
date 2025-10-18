# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import io
import base64
import requests
from PIL import Image
from typing import Optional, List

from unitorch_microsoft.externals.github_copilot import (
    get_access_token,
    get_copilot_token,
)


def get_response(
    prompt,
    system_prompt: Optional[
        str
    ] = "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture.",
    images: Optional[Image.Image] = None,
    model: Optional[str] = "gpt-4.1",
    max_tokens: Optional[int] = 32768,
):
    content = [{"type": "text", "text": prompt}]
    images = images if images is not None else []
    if isinstance(images, str):
        images = Image.open(images)
    if isinstance(images, Image.Image):
        images = [images]
    images = [
        im if isinstance(im, Image.Image) else Image.open(im)
        for im in images
        if isinstance(im, Image.Image) or isinstance(im, str)
    ]
    reported_images = {}
    for i, image in enumerate(images):
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
                },
            }
        )
        reported_images[f"image_{i}"] = image
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {"role": "user", "content": content},
    ]

    headers = {
        "Authorization": f"Bearer {get_copilot_token()}",
        "User-Agent": "GitHub-Copilot-Client/1.0",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Editor-Version": "vscode/1.85.0",
        "Editor-Plugin-Version": "copilot/1.155.0",
    }
    try:
        response = requests.post(
            "https://api.githubcopilot.com/chat/completions",
            headers=headers,
            json={
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "top_p": 1.0,
            },
        ).json()
        result = response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(e)
        return ""
    return result


def get_tools_response(
    prompt,
    tools: List[dict],
    tool_choice: Optional[str] = "auto",
    system_prompt: Optional[
        str
    ] = "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture.",
    images: Optional[Image.Image] = None,
    model: Optional[str] = "gpt-4.1",
    max_tokens: Optional[int] = 32768,
):
    content = [{"type": "text", "text": prompt}]
    images = images if images is not None else []
    if isinstance(images, str):
        images = Image.open(images)
    if isinstance(images, Image.Image):
        images = [images]
    images = [
        im if isinstance(im, Image.Image) else Image.open(im)
        for im in images
        if isinstance(im, Image.Image) or isinstance(im, str)
    ]
    reported_images = {}
    for i, image in enumerate(images):
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
                },
            }
        )
        reported_images[f"image_{i}"] = image
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {"role": "user", "content": content},
    ]

    headers = {
        "Authorization": f"Bearer {get_copilot_token()}",
        "User-Agent": "GitHub-Copilot-Client/1.0",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Editor-Version": "vscode/1.85.0",
        "Editor-Plugin-Version": "copilot/1.155.0",
    }
    try:
        response = requests.post(
            "https://api.githubcopilot.com/chat/completions",
            headers=headers,
            json={
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "top_p": 1.0,
                "tools": tools,
                "tool_choice": tool_choice,
            },
        ).json()
        result = response["choices"][0]["message"]
        tool_calls = result.get("tool_calls", [])
        content = result.get("content", None)
        content = content if content is not None else ""
        content = content.strip()
    except Exception as e:
        print(e)
        return {"content": "", "tool_calls": []}
    return {
        "content": content,
        "tool_calls": tool_calls,
    }


def get_chat_response(
    histories,
    message: str,
    system_prompt: Optional[
        str
    ] = "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture.",
    images: Optional[Image.Image] = None,
    model: Optional[str] = "gpt-4.1",
    max_tokens: Optional[int] = 32768,
):
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ]
    for msg, ans in histories:
        messages += [
            {
                "role": "user",
                "content": msg,
            },
            {
                "role": "assistant",
                "content": ans,
            },
        ]

    content = [{"type": "text", "text": message}]
    images = images if images is not None else []
    if isinstance(images, str):
        images = Image.open(images)
    if isinstance(images, Image.Image):
        images = [images]
    images = [
        im if isinstance(im, Image.Image) else Image.open(im)
        for im in images
        if isinstance(im, Image.Image) or isinstance(im, str)
    ]
    reported_images = {}
    for i, image in enumerate(images):
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
                },
            }
        )
        reported_images[f"image_{i}"] = image

    messages += [
        {
            "role": "user",
            "content": content,
        }
    ]

    headers = {
        "Authorization": f"Bearer {get_copilot_token()}",
        "User-Agent": "GitHub-Copilot-Client/1.0",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Editor-Version": "vscode/1.85.0",
        "Editor-Plugin-Version": "copilot/1.155.0",
    }
    try:
        response = requests.post(
            "https://api.githubcopilot.com/chat/completions",
            headers=headers,
            json={
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "top_p": 1.0,
            },
        ).json()
        result = response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(e)
        return ""

    return result
