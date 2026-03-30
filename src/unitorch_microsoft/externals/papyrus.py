# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import io
import time
import base64
import requests
from PIL import Image
from typing import Optional, List
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


def timed_cache(ttl_seconds=300):
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


def get_image_response(
    prompt,
    images=None,
    mask=None,
    size="1024x1024",
    quality="medium",
    input_fidelity="high",
):
    headers = {
        "Authorization": "Bearer " + get_access_token(),
        "papyrus-model-name": "gpt-image-15-eval",
        "papyrus-quota-id": "",
        "papyrus-timeout-ms": "120000",
    }

    def get_image(im):
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        buf.seek(0)
        return buf

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

    try:
        if len(images) == 0:
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
                "input_fidelity": input_fidelity,
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

        if "data" in response and len(response["data"]) > 0:
            result = response["data"][0]["b64_json"]
            result = Image.open(io.BytesIO(base64.b64decode(result)))

            return result
        else:
            print(f"Error in response: {response}")
            return None
    except Exception as e:
        print(f"Error during request: {e}")
        return None
