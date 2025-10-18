# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import io
import base64
import cv2
import time
import jwt
import logging
import requests
import hashlib
import numpy as np
import gradio as gr
from torch.hub import download_url_to_file
from PIL import Image, ImageDraw
from unitorch import mktempfile
from unitorch.utils import read_file
from unitorch.models import GenericOutputs
from unitorch.cli import CoreConfigureParser
from unitorch.cli.webuis import SimpleWebUI
from unitorch_microsoft import cached_path
from unitorch_microsoft.spaces import (
    get_temp_folder,
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
)


def get_api_key():
    ak = os.environ.get("KELING_API_AK", None)
    sk = os.environ.get("KELING_API_SK", None)
    headers = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "iss": ak,
        "exp": int(time.time())
        + 1800,  # The valid time, in this example, represents the current time+1800s(30min)
        "nbf": int(time.time())
        - 5,  # The time when it starts to take effect, in this example, represents the current time minus 5s
    }
    token = jwt.encode(payload, sk, headers=headers)
    return token


class KelingImage2VideoWebUI(SimpleWebUI):
    _title = "Keling Image2Video"
    _description = "This is a demo for Keling Image2Video. You can upload an image and the output will be a video generated from the image."

    def __init__(self, config: CoreConfigureParser):
        # create elements
        toper_menus = create_toper_menus()
        footer = create_footer()
        header = create_element(
            "markdown",
            label=f"# <div style='margin-top:10px'>🎬 {self._title}</div>",
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
        first_image = create_element("image", "First Image")
        last_image = create_element("image", "Last Image")
        prompt = create_element("text", "Prompt", lines=3)
        neg_prompt = create_element("text", "Negative Prompt", lines=3)
        duration = create_element("radio", "Duration", values=[5, 10], default=5)
        generate = create_element("button", "Generate")
        output_video = create_element("video", "Output Video")

        left = create_column(
            create_row(first_image, last_image), prompt, neg_prompt, duration, generate
        )
        right = create_column(output_video)
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
            inputs=[prompt, neg_prompt, first_image, last_image, duration],
            outputs=[output_video],
            trigger_mode="once",
        )

        iface.__exit__()

        super().__init__(config, iname=self._title, iface=iface)

    def start(self):
        self._status = "Running"
        return self._status

    def stop(self):
        self._status = "Stopped"
        return self._status

    def generate(self, prompt, neg_prompt, first_image, last_image, duration):
        token = get_api_key()
        headers = {
            "accept": "application/json",
            "authorization": "Bearer " + token,
            "content-type": "application/json",
        }

        def prepare_image(image):
            image_buffer = io.BytesIO()
            image.save(image_buffer, format="JPEG")
            image_buffer.seek(0)
            return base64.b64encode(image_buffer.getvalue()).decode()

        response = requests.post(
            "https://api-singapore.klingai.com/v1/videos/image2video",
            timeout=60,
            json={
                "prompt": prompt,
                "negative_prompt": neg_prompt,
                "model_name": "kling-v2-1",
                "image": prepare_image(first_image),
                "image_tail": (
                    prepare_image(last_image) if last_image is not None else None
                ),
                "mode": "pro",
                "duration": duration,
            },
            headers=headers,
        ).json()
        track_id = response["data"]["task_id"]
        while True:
            response = requests.get(
                "https://api-singapore.klingai.com/v1/videos/image2video/" + track_id,
                headers={"authorization": "Bearer " + token},
            ).json()
            status = response["data"]["task_status"]
            logging.info(f"Video {track_id} generation status: {status}")
            if status == "succeed":
                results = response["data"]["task_result"]["videos"]
                if len(results) > 0:
                    url = results[0]["url"]
                    name = hashlib.md5(url.encode()).hexdigest() + ".mp4"
                    path = f"{get_temp_folder()}/{name}"
                    download_url_to_file(url, path, progress=False)
                    return path
            elif status == "failed":
                gr.Error("Video generation failed!")
                break
            time.sleep(5)
        return None
