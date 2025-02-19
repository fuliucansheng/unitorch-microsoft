# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import cv2
import gc
import torch
import numpy as np
import gradio as gr
from PIL import Image, ImageDraw
from transformers import AutoModelForImageSegmentation
from torchvision import transforms
from unitorch import mktempfile
from unitorch.utils import read_file
from unitorch.models import GenericOutputs
from unitorch.cli import CoreConfigureParser
from unitorch.cli.pipelines.bria import BRIAForSegmentationPipeline
from unitorch.cli.webuis import SimpleWebUI
from unitorch_microsoft import cached_path
import unitorch_microsoft.models.sam
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
)
from unitorch_microsoft.models.bletchley.pipeline_v1 import (
    BletchleyForMatchingV2Pipeline as BletchleyV1ForMatchingV2Pipeline,
    BletchleyForMatchingPipeline as BletchleyV1ForMatchingPipeline,
)
from unitorch_microsoft.models.bletchley.pipeline_v3 import (
    BletchleyForMatchingPipeline as BletchleyV3ForMatchingPipeline,
    BletchleyForImageClassificationPipeline as BletchleyV3ForImageClassificationPipeline,
)


class ImgInsightsWebUI(SimpleWebUI):
    def __init__(self, config: CoreConfigureParser):
        self._status = getattr(self, "_status", "Stopped")
        # create elements
        toper_menus = create_toper_menus()
        footer = create_footer()
        header = create_element(
            "markdown",
            label=f"# <div style='margin-top:10px'>🛠️ Image Insights</div>",
            interactive=False,
        )
        description = create_element(
            "markdown",
            label="description",
            interactive=False,
        )

        status = create_element("text", "Status", default="Stopped", interactive=False)
        start = create_element("button", "Start", variant="primary")
        stop = create_element("button", "Stop", variant="stop")
        input_image = create_element("image", "Image")
        generate = create_element("button", "Generate")
        result1 = gr.Label(label="Background Type")
        result2 = gr.Label(label="Image Type")
        result3 = gr.Label(label="General Category")
        result4 = gr.Label(label="Blurry")

        left = create_column(input_image, generate)
        right = create_column(create_row(result1, result2, result4), result3)

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
        iface._title = "Image Insights"
        iface._description = "This is a demo for image insights."

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
            fn=self.serve,
            inputs=[input_image],
            outputs=[result1, result2, result3, result4],
            trigger_mode="once",
        )

        iface.load(
            fn=lambda: self._status,
            outputs=status,
        )

        iface.__exit__()

        super().__init__(config, iname="Image Insights", iface=iface)

    def start(self):
        self._pipe1 = BletchleyV1ForMatchingV2Pipeline.from_core_configure(
            self._config,
            config_type="0.8B",
            pretrained_weight_path="https://unitorchazureblob.blob.core.windows.net/shares/models/bletchley/v1/pytorch_model.0.8B.bin",
            pretrained_lora_weight_path="https://unitorchazureblob.blob.core.windows.net/shares/models/adsplus/lora/bletchley/pytorch_model.v1.lora4.bg_type.2501.bin",
            label_dict={
                "complex": "complex background, objects in the background or even no background",
                "simple": "clean background, no objects in the background",
                "white": "white background, no objects in the background",
                "poster": "poster image, composed of multiple objects, logo, text, etc.",
                "real": "a real image, not a poster or a logo",
                "logo": "logo image, composed of logo only",
            },
        )
        self._pipe2 = BletchleyV3ForImageClassificationPipeline.from_core_configure(
            self._config,
            config_type="2.5B",
            pretrained_weight_path="https://unitorchazureblob.blob.core.windows.net/shares/models/adsplus/image/pytorch_model.bletchley.v3.cate.2502.bin",
            id2label={
                0: "Animals",
                1: "Plants",
                2: "People,Scenery",
                3: "Sports",
                4: "Transportation",
                5: "Food",
                6: "Everyday Items",
                7: "Clothing & Beauty Products",
                8: "Home & Living",
                9: "Products",
                10: "Daily Life Scenes",
                11: "Machinery",
                12: "Medical",
                13: "City",
                14: "Architecture",
                15: "Visual Design",
                16: "Cartoons & Anime",
                17: "Games",
                18: "Others",
            },
        )
        self._pipe3 = BletchleyV1ForMatchingV2Pipeline.from_core_configure(
            self._config,
            config_type="2.5B",
            pretrained_weight_path="https://unitorchazureblob.blob.core.windows.net/shares/models/bletchley/v1/pytorch_model.2.5B.bin",
            pretrained_lora_weight_path="https://unitorchazureblob.blob.core.windows.net/shares/models/adsplus/lora/bletchley/pytorch_model.v1.lora4.blurry.2409.bin",
            label_dict={
                "blurry": "blurry",
            },
        )
        self._status = "Running"
        return self._status

    def stop(self):
        self._pipe1.to("cpu")
        del self._pipe1
        self._pipe2.to("cpu")
        del self._pipe2
        gc.collect()
        torch.cuda.empty_cache()
        self._pipe1 = None if not hasattr(self, "_pipe1") else self._pipe1
        self._pipe2 = None if not hasattr(self, "_pipe2") else self._pipe2
        self._status = (
            "Stopped" if self._pipe1 is None or self._pipe2 is None else "Running"
        )
        return self._status

    def serve(self, image):
        results = self._pipe1(image)
        result1 = {k: results[k] for k in ["white", "simple", "complex"]}
        result2 = {k: results[k] for k in ["poster", "logo", "real"]}
        result3 = self._pipe2(image)
        result4 = self._pipe3(image)

        return result1, result2, result3, result4
