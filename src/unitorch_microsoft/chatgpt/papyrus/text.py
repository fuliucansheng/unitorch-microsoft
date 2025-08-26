# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import io
import base64
import requests
from PIL import Image
from typing import Optional, List

from unitorch_microsoft.chatgpt.papyrus import (
    get_access_token,
    papyrus_endpoint3,
    reported_item,
)


def get_response(
    prompt,
    system_prompt: Optional[
        str
    ] = "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture.",
    images: Optional[Image.Image] = None,
    model: Optional[str] = "gpt-41-2025-04-14-Eval",
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
        "Authorization": "Bearer " + get_access_token(),
        "Content-Type": "application/json",
        "papyrus-model-name": model,
        "papyrus-quota-id": "",
        "papyrus-timeout-ms": "120000",
    }
    try:
        response = requests.post(
            papyrus_endpoint3,
            headers=headers,
            json={
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
    reported_item(
        record={
            "prompt": prompt,
            "system_prompt": system_prompt,
            "tags": "#GPT-4",
            "result": result,
        },
        images=reported_images if len(reported_images) > 0 else None,
    )
    return result


def get_tools_response(
    prompt,
    tools: List[dict],
    tool_choice: Optional[str] = "auto",
    system_prompt: Optional[
        str
    ] = "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture.",
    images: Optional[Image.Image] = None,
    model: Optional[str] = "gpt-41-2025-04-14-Eval",
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
        "Authorization": "Bearer " + get_access_token(),
        "Content-Type": "application/json",
        "papyrus-model-name": model,
        "papyrus-quota-id": "",
        "papyrus-timeout-ms": "120000",
    }
    try:
        response = requests.post(
            papyrus_endpoint3,
            headers=headers,
            json={
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
    model: Optional[str] = "gpt-41-2025-04-14-Eval",
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
        "Authorization": "Bearer " + get_access_token(),
        "Content-Type": "application/json",
        "papyrus-model-name": model,
        "papyrus-quota-id": "",
        "papyrus-timeout-ms": "120000",
    }
    try:
        response = requests.post(
            papyrus_endpoint3,
            headers=headers,
            json={
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


def get_gpt5_response(
    prompt,
    system_prompt: Optional[
        str
    ] = "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-5 architecture.",
    images: Optional[Image.Image] = None,
    model: Optional[str] = "gpt-5-2025-08-07-Eval",
    max_tokens: Optional[int] = 16384,
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
        "Authorization": "Bearer " + get_access_token(),
        "Content-Type": "application/json",
        "papyrus-model-name": model,
        "papyrus-quota-id": "",
        "papyrus-timeout-ms": "120000",
    }
    try:
        response = requests.post(
            papyrus_endpoint3,
            headers=headers,
            json={
                "messages": messages,
                "max_completion_tokens": max_tokens,
                "top_p": 1.0,
            },
        ).json()
        result = response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(e)
        return ""
    reported_item(
        record={
            "prompt": prompt,
            "system_prompt": system_prompt,
            "tags": "#GPT-5",
            "result": result,
        },
        images=reported_images if len(reported_images) > 0 else None,
    )
    return result


def get_gpt5_tools_response(
    prompt,
    tools: List[dict],
    tool_choice: Optional[str] = "auto",
    system_prompt: Optional[
        str
    ] = "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture.",
    images: Optional[Image.Image] = None,
    model: Optional[str] = "gpt-5-2025-08-07-Eval",
    max_tokens: Optional[int] = 16384,
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
        "Authorization": "Bearer " + get_access_token(),
        "Content-Type": "application/json",
        "papyrus-model-name": model,
        "papyrus-quota-id": "",
        "papyrus-timeout-ms": "120000",
    }
    try:
        response = requests.post(
            papyrus_endpoint3,
            headers=headers,
            json={
                "messages": messages,
                "max_completion_tokens": max_tokens,
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


def get_gpt5_chat_response(
    histories,
    message: str,
    system_prompt: Optional[
        str
    ] = "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture.",
    images: Optional[Image.Image] = None,
    model: Optional[str] = "gpt-5-2025-08-07-Eval",
    max_tokens: Optional[int] = 16384,
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
        "Authorization": "Bearer " + get_access_token(),
        "Content-Type": "application/json",
        "papyrus-model-name": model,
        "papyrus-quota-id": "",
        "papyrus-timeout-ms": "120000",
    }
    try:
        response = requests.post(
            papyrus_endpoint3,
            headers=headers,
            json={
                "messages": messages,
                "max_completion_tokens": max_tokens,
                "top_p": 1.0,
            },
        ).json()
        result = response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(e)
        return ""

    return result
