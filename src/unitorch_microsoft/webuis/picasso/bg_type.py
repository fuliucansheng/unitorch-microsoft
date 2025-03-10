# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import io
import torch
import gc
import gradio as gr
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.cli import CoreConfigureParser, GenericWebUI
from unitorch.cli import register_webui
from unitorch.cli.webuis import (
    matched_pretrained_names,
    create_element,
    create_accordion,
    create_row,
    create_column,
    create_group,
    create_tab,
    create_tabs,
    create_blocks,
    create_pretrain_layout,
    create_lora_layout,
)
from unitorch.cli.webuis import SimpleWebUI
from unitorch_microsoft.models.bletchley.pipeline_v1 import (
    BletchleyForMatchingV2Pipeline,
)


@register_webui("microsoft/webui/picasso/bg_type")
class BackgroundTypeWebUI(SimpleWebUI):
    supported_pretrained_names = ["2.5B"]

    def __init__(self, config: CoreConfigureParser):
        self._pipe = None
        self._status = "Stopped" if self._pipe is None else "Running"
        if len(self.supported_pretrained_names) == 0:
            raise ValueError("No supported pretrained models found.")
        self._name = self.supported_pretrained_names[0]

        # create elements
        pretrain_layout_group = create_pretrain_layout(
            self.supported_pretrained_names, self._name
        )
        name, status, start, stop, pretrain_layout = (
            pretrain_layout_group.name,
            pretrain_layout_group.status,
            pretrain_layout_group.start,
            pretrain_layout_group.stop,
            pretrain_layout_group.layout,
        )

        image = create_element("image", "Input Image", scale=2)
        generate = create_element("button", "Generate")
        result = gr.Label(label="Type")

        # create blocks
        left = create_column(image, generate)
        right = create_row(result)
        iface = create_blocks(pretrain_layout, create_row(left, right))

        # create events
        iface.__enter__()

        start.click(self.start, inputs=[name], outputs=[status], trigger_mode="once")
        stop.click(self.stop, outputs=[status], trigger_mode="once")
        generate.click(
            self.serve,
            inputs=[image],
            outputs=[result],
            trigger_mode="once",
        )

        iface.load(
            fn=lambda: [gr.update(value=self._name), gr.update(value=self._status)],
            outputs=[name, status],
        )

        iface.__exit__()

        super().__init__(config, iname="BGType", iface=iface)

    def start(self, config_type, **kwargs):
        if self._status == "Running":
            return self._status
        self._pipe = BletchleyForMatchingV2Pipeline.from_core_configure(
            self._config,
            config_type="2.5B",
            pretrained_lora_weight_path="https://unitorchazureblob.blob.core.windows.net/shares/models/adsplus/lora/bletchley/pytorch_model.v1.lora4.bg_type.2410.bin",
            label_dict={
                "complex": "complex detailed environment, detailed surroundings",
                "simple": "minimal clean background",
                "white": "white background",
            },
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

    def serve(
        self,
        image,
    ):
        assert self._pipe is not None
        result = self._pipe(image)
        return result
