# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import io
import re
import gc
import torch
import socket
import requests
import tempfile
import hashlib
import subprocess
import pandas as pd
import gradio as gr
from PIL import Image, ImageOps
from collections import Counter, defaultdict
from torch.hub import download_url_to_file
from unitorch import get_temp_home
from unitorch.cli import CoreConfigureParser
from unitorch.cli import register_webui
from unitorch.cli.webuis import (
    create_element,
    create_accordion,
    create_row,
    create_column,
    create_flex_layout,
    create_group,
    create_tab,
    create_tabs,
    create_blocks,
)
from unitorch.cli.webuis import SimpleWebUI
from unitorch_microsoft.omnipixel.bletchley import (
    BletchleyForImageClickModelPipeline,
)


@register_webui("microsoft/omnipixel/webui/image_click")
class OmniPixelImageClickWebUI(SimpleWebUI):
    def __init__(
        self,
        config: CoreConfigureParser,
    ):
        self._config = config
        config.set_default_section("microsoft/omnipixel/webui/image_click")

        image = create_element("image_editor", "Input Image")
        processed_image = create_element("image", "Processed Image")
        score = create_element("text", "Score")

        submit = create_element(
            "button",
            label="Submit",
        )
        left = create_column(create_row(image, processed_image), submit)
        right = create_column(score)
        iface = create_blocks(create_row(left, right))

        iface.__enter__()
        image.change(
            fn=self.composite_images, inputs=[image], outputs=[processed_image]
        )
        submit.click(
            self.generate,
            inputs=[processed_image],
            outputs=[score],
            trigger_mode="once",
        )

        iface.__exit__()

        self.start()

        super().__init__(config, iname="Image Click Model WebUI", iface=iface)

    def start(self, **kwargs):
        config = self._config
        config.set_default_section("microsoft/omnipixel/webui/image_click")
        config_type = config.getoption("config_type", "0.8B")
        max_seq_length = config.getoption("max_seq_length", 36)
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        device = config.getoption("device", "cpu")
        self._pipe = BletchleyForImageClickModelPipeline.from_core_configure(
            self._config,
            config_type=config_type,
            max_seq_length=max_seq_length,
            pretrained_weight_path=pretrained_weight_path,
            device=device,
        )
        self._status = "Running"
        return self._status

    def stop(self, **kwargs):
        self._pipe.to("cpu")
        del self._pipe
        gc.collect()
        torch.cuda.empty_cache()
        self._pipe = None
        self._status = "Stopped" if self._pipe is None else "Running"
        return self._status

    def composite_images(self, images):
        if images is None:
            return None
        return images["composite"]

    def generate(self, image):
        result = self._pipe(image)
        return result
