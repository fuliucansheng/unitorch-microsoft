# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import io
import time
import base64
import requests
from PIL import Image
from typing import Optional
from azure.identity import AzureCliCredential

"""
pip3 install azure-identity
az login
"""

# https://eng.ms/docs/microsoft-ai/webxt/bing-fundamentals/dlis/dlis/papyrus/serviceusage/serviceusage
# https://eng.ms/docs/microsoft-ai/webxt/bing-fundamentals/dlis/dlis/papyrus/modelmigration/models
# You can only call /images/generations api through papyrus large endpoint.
papyrus_endpoint1 = "https://westus2large.papyrus.binginternal.com/images/generations"
papyrus_endpoint2 = "https://westus2large.papyrus.binginternal.com/images/edits"
papyrus_endpoint3 = "https://westus2large.papyrus.binginternal.com/chat/completions"
verify_scope = "api://5fe538a8-15d5-4a84-961e-be66cd036687/.default"

credential = AzureCliCredential()


def timed_cache(ttl_seconds=300):  # 默认缓存时间为5分钟
    def decorator(func):
        cache = {}

        def wrapper(*args, **kwargs):
            key = (args, frozenset(kwargs.items()))
            now = time.time()

            if key in cache:
                result, timestamp = cache[key]
                if now - timestamp < ttl_seconds:
                    return result

            result = func(*args, **kwargs)
            cache[key] = (result, now)
            return result

        return wrapper

    return decorator


@timed_cache(ttl_seconds=300)
def get_access_token():
    access_token = credential.get_token(verify_scope).token
    return access_token


def get_gpt_image_response(
    prompt, images=None, mask=None, size="1024x1024", quality="medium"
):
    headers = {
        "Authorization": "Bearer " + get_access_token(),
        "papyrus-model-name": "gpt-image-1-2025-04-15-eval",
        "papyrus-quota-id": "",
        "papyrus-timeout-ms": "120000",
    }

    def get_image(im):
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        buf.seek(0)
        return buf

    images = [im for im in images if isinstance(im, Image.Image)]

    try:
        if images is None or len(images) == 0:
            headers["Content-Type"] = "application/json"
            data = {
                "prompt": prompt,
                "size": size,
                "quality": quality,
                "output_compression": 100,
                "output_format": "png",
                "n": 1,
            }

            response = requests.post(papyrus_endpoint1, headers=headers, json=data)
        else:
            data = {
                "prompt": prompt,
                "size": size,
                "quality": quality,
                "n": 1,
            }
            files = [
                ("image[]", (f"image_{i}.png", get_image(image), "image/png"))
                for i, image in enumerate(images)
            ]
            if mask is not None:
                files.append(("mask", ("mask.png", get_image(mask), "image/png")))
            response = requests.post(
                papyrus_endpoint2, headers=headers, data=data, files=files
            )
        response = response.json()
        result = response["data"][0]["b64_json"]
        return Image.open(io.BytesIO(base64.b64decode(result)))
    except Exception as e:
        print(f"Error during request: {e}")
        return None


def get_gpt4_response(
    prompt,
    system_prompt: Optional[
        str
    ] = "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture.",
    images: Optional[Image.Image] = None,
    model: Optional[str] = "gpt-41-2025-04-14-Eval",
    max_tokens: Optional[int] = 4096,
):
    content = [{"type": "text", "text": prompt}]
    images = images if images is not None else []
    for image in images:
        if image is not None:
            if isinstance(image, str):
                image = Image.open(image)
            buf = io.BytesIO()
            image = image.convert("RGB")  # Ensure the image is in RGB format
            image.save(buf, format="JPEG")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"
                    },
                }
            )
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
    return result
