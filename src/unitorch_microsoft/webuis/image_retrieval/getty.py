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
from unitorch_microsoft.pipelines.image_retrieval import BletchleyV1FaissPipeline


class GettyImageRetrievalWebUI(SimpleWebUI):
    supported_pretrained_names = ["2.5B"]

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

        text = create_element("text", "Input Text", lines=3, scale=2)
        image = create_element("image", "Input Image", scale=2)
        topk = create_element(
            "slider", "Top K", default=10, min_value=1, max_value=250, step=1
        )
        search = create_element("button", "Search")
        result = create_element("dataframe", "Search Result")

        # create blocks
        top1 = create_row(text, image, create_column(topk, search))
        top2 = create_row(result)
        iface = create_blocks(pretrain_layout, top1, top2)

        # create events
        iface.__enter__()

        start.click(self.start, inputs=[name], outputs=[status])
        stop.click(self.stop, outputs=[status])
        search.click(self.serve, inputs=[text, image, topk], outputs=[result])

        iface.load(
            fn=lambda: [gr.update(value=self._name), gr.update(value=self._status)],
            outputs=[name, status],
        )

        iface.__exit__()

        super().__init__(config, iname="Getty", iface=iface)

    def start(self, config_type, **kwargs):
        if self._status == "Running":
            self.stop()
        self._pipe = BletchleyV1FaissPipeline.from_core_configure(
            self._config,
            config_type="2.5B",
            pretrained_weight_path="https://unitorchazureblob.blob.core.windows.net/shares/models/adsplus/image/pytorch_model.bletchley.v1.retrieval.61.28.bin",
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
        text: Optional[str] = None,
        image: Optional[Image.Image] = None,
        topk: Optional[int] = 10,
    ):
        if text.strip() == "":
            text = None
        assert self._pipe is not None
        result = self._pipe(text, image, topk=topk)
        return result
