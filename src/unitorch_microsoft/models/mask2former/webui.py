# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import io
import torch
import gc
import gradio as gr
from typing import List, Tuple
from PIL import Image, ImageDraw
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
)
from unitorch.cli.webuis import SimpleWebUI
from unitorch_microsoft.models.mask2former.pipeline import Mask2FormerPipeline
from unitorch_microsoft.models.mask2former import pretrained_mask2former_infos


@register_webui("microsoft/webui/mask2former")
class Mask2FormerWebUI(SimpleWebUI):
    pretrained_names = list(pretrained_mask2former_infos.keys())
    supported_pretrained_names = matched_pretrained_names(
        pretrained_names,
        ["^picasso-mask2former"],
    )

    def __init__(self, config: CoreConfigureParser):
        self._config = config
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
        input_image = create_element("image", "Input Image")
        mask_threshold = create_element(
            "slider",
            "Mask Threshold",
            default=0.5,
            min_value=0,
            max_value=1,
            step=0.01,
        )
        output_image_type = create_element(
            "radio", "Output Image Type", default="Mask", values=["Mask", "Object"]
        )
        segment = create_element("button", "Segment")
        output_image = create_element("image", "Output Image")

        # create layout
        left = create_column(
            input_image,
            create_row(mask_threshold, output_image_type),
            create_row(segment),
        )
        right = create_column(output_image)
        iface = create_blocks(pretrain_layout, create_row(left, right))

        # create events
        iface.__enter__()

        start.click(self.start, inputs=[name], outputs=[status], trigger_mode="once")
        stop.click(self.stop, outputs=[status], trigger_mode="once")
        segment.click(
            self.serve,
            inputs=[input_image, mask_threshold, output_image_type],
            outputs=[output_image],
            trigger_mode="once",
        )

        iface.load(
            fn=lambda: [gr.update(value=self._name), gr.update(value=self._status)],
            outputs=[name, status],
        )

        iface.__exit__()

        super().__init__(config, iname="Mask2Former", iface=iface)

    def start(self, pretrained_name, **kwargs):
        if self._status == "Running":
            self.stop()
        self._name = pretrained_name
        self._pipe = Mask2FormerPipeline.from_core_configure(
            self._config, pretrained_name=pretrained_name
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
        image: Image.Image,
        mask_threshold: float = 0.5,
        output_image_type: str = "Mask",
    ):
        assert self._pipe is not None
        mask = self._pipe(
            image,
            threshold=mask_threshold,
        )
        if output_image_type == "Object":
            result = Image.new("RGBA", image.size, (0, 0, 0, 0))
            mask = mask.convert("L").resize(image.size)
            result.paste(image, (0, 0), mask)
        else:
            result = mask
        return result
