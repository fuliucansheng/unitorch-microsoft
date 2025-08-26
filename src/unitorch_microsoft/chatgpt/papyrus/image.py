# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import io
import base64
import requests
from PIL import Image
from unitorch_microsoft.chatgpt.papyrus import (
    get_access_token,
    papyrus_endpoint1,
    papyrus_endpoint2,
    reported_item,
)


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
        "papyrus-model-name": "gpt-image-1-2025-04-15-eval",
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
        reported_images = {}
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
                # "input_fidelity": input_fidelity,
            }
            files = [
                ("image[]", (f"image_{i}.png", get_image(image), "image/png"))
                for i, image in enumerate(images)
            ]
            reported_images = {f"image_{i}": image for i, image in enumerate(images)}
            if mask is not None:
                files.append(("mask", ("mask.png", get_image(mask), "image/png")))
                reported_images["mask"] = mask
            response = requests.post(
                papyrus_endpoint2, headers=headers, data=data, files=files
            )
        response = response.json()

        if "data" in response and len(response["data"]) > 0:
            result = response["data"][0]["b64_json"]
            result = Image.open(io.BytesIO(base64.b64decode(result)))
            reported_images["result"] = result

            reported_item(
                record={"prompt": prompt, "size": size, "tags": "#GPT-Image-1"},
                images=reported_images,
            )

            return result
        else:
            print(f"Error in response: {response}")
            return None
    except Exception as e:
        print(f"Error during request: {e}")
        return None
