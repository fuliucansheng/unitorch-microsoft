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
    create_freeu_layout,
)
from unitorch.cli.webuis import SimpleWebUI
from unitorch_microsoft.pipelines.selection.image import (
    BletchleyImageTaggerSelectionPipeline,
)


@register_webui("microsoft/webui/selection/image/tagger")
class ImageTaggerSelectionWebUI(SimpleWebUI):
    supported_pretrained_names = ["0.8B"]

    def __init__(self, config: CoreConfigureParser):
        self._pipe = None if not hasattr(self, "_pipe") else self._pipe
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
        topk = create_element(
            "slider", "Top K", default=10, min_value=1, max_value=30, step=1
        )
        search = create_element("button", "Search")
        result = gr.Label(label="Tags")

        # create blocks
        left = create_column(image, topk, search)
        right = create_row(result)
        iface = create_blocks(pretrain_layout, create_row(left, right))

        # create events
        iface.__enter__()

        start.click(self.start, inputs=[name], outputs=[status], trigger_mode="once")
        stop.click(self.stop, outputs=[status], trigger_mode="once")
        search.click(
            self.serve,
            inputs=[image, topk],
            outputs=[result],
            trigger_mode="once",
        )

        iface.load(
            fn=lambda: [gr.update(value=self._name), gr.update(value=self._status)],
            outputs=[name, status],
        )

        iface.__exit__()

        super().__init__(config, iname="Tagger", iface=iface)

    def start(self, config_type, **kwargs):
        if self._status == "Running":
            return self._status
        self._pipe = BletchleyImageTaggerSelectionPipeline.from_core_configure(
            self._config,
            config_type="0.8B",
            pretrained_lora_weight_path="https://unitorchazureblob.blob.core.windows.net/shares/models/adsplus/lora/bletchley/pytorch_model.v3.retrieval.0.8B.64d.lora.p3.bin",
            output_embed_dim=64,
        )
        self._status = "Running"
        return self._status

    def stop(self, **kwargs):
        self._pipe.to("cpu")
        del self._pipe
        gc.collect()
        torch.cuda.empty_cache()
        self._pipe = None if not hasattr(self, "_pipe") else self._pipe
        self._status = "Stopped" if self._pipe is None else "Running"
        return self._status

    def serve(
        self,
        image,
        topk: Optional[int] = 10,
    ):
        assert self._pipe is not None
        result = self._pipe(image, topk=topk)
        return result
