# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import logging
import os
from unittest import result
import cv2
import gc
import torch
import numpy as np
import gradio as gr
import pandas as pd
from PIL import Image, ImageDraw

from torchvision import transforms
from unitorch import mktempfile
from unitorch.utils import read_file
from unitorch.models import GenericOutputs
from unitorch.cli import CoreConfigureParser
from unitorch.cli.webuis import SimpleWebUI
from unitorch_microsoft import cached_path
from unitorch_microsoft.spaces import (
    create_element,
    create_row,
    create_column,
    create_tab,
    create_tabs,
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


class FitProbeWebUI(SimpleWebUI):
    _title = "Fit Probe"
    _description = "This is a demo for Fit Probe. You can upload an image and the output will be the best fit image."

    def __init__(self, config: CoreConfigureParser):
        config.set_default_section("microsoft/spaces/picasso/fit_probe")
        self._endpoint = config.getoption("endpoint", None)
        self._status = getattr(self, "_status", "Stopped")
        # create elements
        toper_menus = create_toper_menus()
        footer = create_footer()
        header = create_element(
            "markdown",
            label=f"# <div style='margin-top:10px'>🎢 {self._title} </div>",
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
        ratio = create_element(
            "slider",
            "Ratio",
            default=1.0,
            min_value=0.5,
            max_value=2.0,
            step=0.01,
        )
        step = create_element(
            "slider",
            "Step",
            default=0.05,
            min_value=0.01,
            max_value=0.99,
            step=0.01,
        )
        # option = create_element(
        #     "radio", "Option", values=["Quality Top", "Click Top", "Quality Filter + Click Top"], default="Quality Top"
        # )
        generate = create_element("button", "Generate")
        result1 = create_element("image", "Quality Top Result Image")
        result2 = create_element("image", "Click Top Result Image")
        result3 = create_element("image", "Aesthetics Top Result Image")
        result4 = create_element("image", "Quality Filter + Click Top Result Image")
        result5 = create_element(
            "image", "Quality Filter + Aesthetics Top Result Image"
        )

        left = create_column(input_image, ratio, step, generate, scale=1)
        right = create_column(
            create_row(result1, result2, result3), create_row(result4, result5), scale=1
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
            inputs=[input_image, ratio, step],
            outputs=[result1, result2, result3],
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

    def get_quality_score(self, image):
        result = call_fastapi(
            self._endpoint + "/microsoft/spaces/fastapi/siglip2/generate1",
            images={
                "image": image,
            },
        )
        return result["Bad Cropped"]

    def get_click_score(self, image):
        result = call_fastapi(
            self._endpoint
            + "/microsoft/spaces/fastapi/bletchley/msan/image_click/generate1",
            images={
                "image": image,
            },
        )
        return result

    def get_aesthetics_score(self, image):
        result = call_fastapi(
            self._endpoint + "/microsoft/spaces/fastapi/bletchley/v3/generate2",
            images={
                "image": image,
            },
        )
        return result["Bad Aesthetics"]

    def generate(self, image, ratio, step):
        w, h = image.size
        if w / h > ratio:
            crop_w, crop_h = int(h * ratio), h
        else:
            crop_w, crop_h = w, int(w / ratio)
        step_w, step_h = int(crop_w * step), int(crop_h * step)
        all_crops = []
        for i in range(0, w - crop_w + 1, step_w):
            for j in range(0, h - crop_h + 1, step_h):
                box = (i, j, i + crop_w, j + crop_h)
                cropped_image = image.crop(box)
                quality_score = self.get_quality_score(cropped_image)
                click_score = self.get_click_score(cropped_image)
                aesthetics_score = self.get_aesthetics_score(cropped_image)
                all_crops.append(
                    {
                        "image": cropped_image,
                        "box": box,
                        "quality_score": quality_score,
                        "click_score": click_score,
                        "aesthetics_score": aesthetics_score,
                    }
                )
        crops = pd.DataFrame(all_crops)
        best_crop1 = crops.loc[crops["quality_score"].idxmin()]
        best_crop2 = crops.loc[crops["click_score"].idxmax()]
        best_crop3 = crops.loc[crops["aesthetics_score"].idxmin()]
        keep_crops = crops[crops["quality_score"] <= 0.45]
        if len(keep_crops) == 0:
            logging.warning("All crops have bad quality, select the best quality crop.")
            gr.Warning("All crops have bad quality, select the best quality crop.")
            best_crop4 = crops.loc[crops["quality_score"].idxmin()]
            best_crop5 = crops.loc[crops["quality_score"].idxmin()]
        else:
            best_crop4 = keep_crops.loc[keep_crops["click_score"].idxmax()]
            best_crop5 = keep_crops.loc[keep_crops["aesthetics_score"].idxmin()]
        result1_image = image.crop(best_crop1["box"])
        result2_image = image.crop(best_crop2["box"])
        result3_image = image.crop(best_crop3["box"])
        result4_image = image.crop(best_crop4["box"])
        result5_image = image.crop(best_crop5["box"])
        return result1_image, result2_image, result3_image, result4_image, result5_image
