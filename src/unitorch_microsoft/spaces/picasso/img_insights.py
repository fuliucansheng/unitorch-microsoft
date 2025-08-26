# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import cv2
import gc
import torch
import numpy as np
import gradio as gr
from PIL import Image, ImageDraw

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
)
from unitorch_microsoft.models.bletchley.pipeline_v1 import (
    BletchleyForMatchingV2Pipeline as BletchleyV1ForMatchingV2Pipeline,
    BletchleyForMatchingPipeline as BletchleyV1ForMatchingPipeline,
)
from unitorch_microsoft.models.bletchley.pipeline_v3 import (
    BletchleyForMatchingPipeline as BletchleyV3ForMatchingPipeline,
    BletchleyForMatchingV2Pipeline as BletchleyV3ForMatchingV2Pipeline,
    BletchleyForImageClassificationPipeline as BletchleyV3ForImageClassificationPipeline,
)
from unitorch_microsoft.models.siglip.pipeline import Siglip2ForMatchingV2Pipeline


class ImgInsightsWebUI(SimpleWebUI):
    _title = "Image Insights"
    _description = "This is a demo for image insights using Bletchley. You can input an image and the model will generate insights about the image, such as background type, image type, general category, blurry status, ICE category, and watermark status."

    def __init__(self, config: CoreConfigureParser):
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
        generate = create_element("button", "Generate")
        result1 = gr.Label(label="Background Type")
        result2 = gr.Label(label="Image Type")
        result3 = gr.Label(label="General Category")
        result4 = gr.Label(label="Blurry")
        result5 = gr.Label(label="ICE Category")
        result6 = gr.Label(label="Watermark")
        result7 = gr.Label(label="Bad Cropped")

        left = create_column(input_image, generate, scale=1)
        right = create_column(
            create_tabs(
                create_tab(
                    create_row(
                        result4,
                        result6,
                    ),
                    create_row(
                        result7,
                    ),
                    name="Quality",
                ),
                create_tab(result3, name="Open Category"),
                create_tab(
                    result5,
                    name="ICE Category",
                ),
                create_tab(
                    create_row(
                        result1,
                        result2,
                    ),
                    name="Others",
                ),
            )
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
            inputs=[input_image],
            outputs=[result1, result2, result3, result4, result5, result6],
            trigger_mode="once",
        )

        iface.load(
            fn=lambda: self._status,
            outputs=status,
        )

        iface.__exit__()

        super().__init__(config, iname=self._title, iface=iface)

    def start(self):
        if self._status == "Running":
            return self._status
        self._pipe1 = BletchleyV1ForMatchingV2Pipeline.from_core_configure(
            self._config,
            config_type="0.8B",
            pretrained_weight_path="https://huggingface.co/datasets/fuliucansheng/unitorchblobfuse/resolve/main/models/bletchley/v1/pytorch_model.0.8B.bin",
            pretrained_lora_weight_path="https://huggingface.co/datasets/fuliucansheng/unitorchblobfuse/resolve/main/models/adsplus/lora/bletchley/pytorch_model.v1.lora4.bg_type.2501.bin",
            label_dict={
                "complex": "complex background, objects in the background or even no background",
                "simple": "clean background, no objects in the background",
                "white": "white background, no objects in the background",
                "poster": "poster image, composed of multiple objects, logo, text, etc.",
                "real": "a real image, not a poster or a logo",
                "logo": "logo image, composed of logo only",
            },
            act_fn="sigmoid",
        )
        self._pipe2 = BletchleyV3ForImageClassificationPipeline.from_core_configure(
            self._config,
            config_type="2.5B",
            pretrained_weight_path="https://huggingface.co/datasets/fuliucansheng/unitorchblobfuse/resolve/main/models/adsplus/image/pytorch_model.bletchley.v3.cate.2502.bin",
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
            pretrained_weight_path="https://huggingface.co/datasets/fuliucansheng/unitorchblobfuse/resolve/main/models/bletchley/v1/pytorch_model.2.5B.bin",
            pretrained_lora_weight_path="https://huggingface.co/datasets/fuliucansheng/unitorchblobfuse/resolve/main/models/adsplus/lora/bletchley/pytorch_model.v1.lora4.blurry.2409.bin",
            label_dict={
                "blurry": "blurry",
            },
            act_fn="sigmoid",
        )
        self._pipe4 = BletchleyV3ForImageClassificationPipeline.from_core_configure(
            self._config,
            config_type="2.5B",
            pretrained_weight_path="https://huggingface.co/datasets/fuliucansheng/unitorchblobfuse/resolve/main/models/adsplus/image/pytorch_model.bletchley.v3.cate.2.2502.bin",
            id2label={
                0: "Sports & Fitness",
                1: "Home & Garden",
                2: "Apparel",
                3: "Others",
                4: "Food & Groceries",
                5: "Vehicles",
                6: "Beauty & Personal Care",
                7: "Hobbies & Leisure",
                8: "Jobs & Education",
                9: "Travel & Tourism",
                10: "Business & Industrial",
                11: "Arts & Entertainment",
                12: "Occasions & Gifts",
                13: "Health",
                14: "Dining & Nightlife",
                15: "Computers & Consumer Electronics",
                16: "Family & Community",
                17: "Real Estate",
                18: "Law & Government",
                19: "Internet & Telecom",
                20: "Finance",
                21: "Retailers & General Merchandise",
                22: "News Media & Publications",
            },
        )
        self._pipe5 = BletchleyV3ForMatchingV2Pipeline.from_core_configure(
            self._config,
            config_type="2.5B",
            pretrained_weight_path="https://huggingface.co/datasets/fuliucansheng/unitorchblobfuse/resolve/main/models/bletchley/v3/pytorch_model.large.bin",
            pretrained_lora_weight_path="https://huggingface.co/datasets/fuliucansheng/unitorchblobfuse/resolve/main/models/adsplus/lora/bletchley/pytorch_model.v3.2.5B.lora4.watermark.2410.bin",
            label_dict={
                "watermark": "watermarked, no watermark signature, brand logo",
            },
            act_fn="sigmoid",
        )
        self._pipe6 = Siglip2ForMatchingV2Pipeline.from_core_configure(
            self._config,
            pretrained_name="siglip2-so400m-patch14-384",
            pretrained_lora_weight_path="https://huggingface.co/datasets/fuliucansheng/unitorchblobfuse/resolve/main/models/adsplus/lora/siglip/pytorch_model.v2.lora4.badcrop.2506.bin",
            label_dict={
                "bad": "bad cropped, cut off, mutilated",
            },
            act_fn="sigmoid",
        )
        self._status = "Running"
        return self._status

    def stop(self):
        if self._status == "Stopped":
            return self._status
        self._pipe1.to("cpu")
        del self._pipe1
        self._pipe2.to("cpu")
        del self._pipe2
        self._pipe3.to("cpu")
        del self._pipe3
        self._pipe4.to("cpu")
        del self._pipe4
        self._pipe5.to("cpu")
        del self._pipe5
        self._pipe6.to("cpu")
        del self._pipe6
        gc.collect()
        torch.cuda.empty_cache()
        self._status = "Stopped"
        return self._status

    def generate(self, image):
        results = self._pipe1(image)
        result1 = {k: results[k] for k in ["white", "simple", "complex"]}
        result2 = {k: results[k] for k in ["poster", "logo", "real"]}
        result3 = self._pipe2(image)
        result4 = self._pipe3(image)
        result5 = self._pipe4(image)
        result6 = self._pipe5(image)
        result7 = self._pipe6(image)

        return result1, result2, result3, result4, result5, result6, result7
