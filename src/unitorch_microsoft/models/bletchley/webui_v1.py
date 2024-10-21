# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import io
import torch
import gc
import gradio as gr
from typing import List, Tuple
from PIL import Image, ImageDraw
from unitorch.utils import nested_dict_value
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
from unitorch_microsoft.models.bletchley import (
    pretrained_bletchley_v1_infos,
    pretrained_bletchley_v1_extensions_infos,
)
from unitorch_microsoft.models.bletchley.pipeline_v1 import BletchleyForMatchingPipeline


@register_webui("core/webui/matching/bletchley")
class BletchleyMatchingWebUI(SimpleWebUI):
    pretrained_names = list(pretrained_bletchley_v1_infos.keys())
    supported_pretrained_names = matched_pretrained_names(
        pretrained_names,
        ".*",
    )
    pretrained_extension_names = list(pretrained_bletchley_v1_extensions_infos.keys())
    supported_lora_names = matched_pretrained_names(
        pretrained_extension_names,
        ["^lora-"],
    )

    def __init__(self, config: CoreConfigureParser):
        self._config = config
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

        self.num_loras = 5
        lora_layout_group = create_lora_layout(
            self.supported_lora_names, num_loras=self.num_loras
        )
        loras = lora_layout_group.loras
        lora_layout = lora_layout_group.layout
        lora_params = []
        for lora in loras:
            lora_params += [
                lora.checkpoint,
                lora.weight,
                lora.alpha,
                lora.url,
                lora.file,
            ]

        text = create_element("text", "Input Text", lines=3)
        image = create_element("image", "Input Image")
        max_seq_length = create_element(
            "slider", "Max Sequence", default=36, min_value=1, max_value=120, step=1
        )
        generate = create_element("button", "Generate")
        score = create_element("text", "Output Score", lines=1)

        # create blocks
        left = create_column(lora_layout, text, generate)
        right = create_column(image, score)
        iface = create_blocks(pretrain_layout, create_row(left, right))

        # create events
        iface.__enter__()

        start.click(self.start, inputs=[name], outputs=[status])
        stop.click(self.stop, outputs=[status])

        for lora in loras:
            lora.checkpoint.change(
                fn=lambda x: nested_dict_value(
                    pretrained_bletchley_v1_extensions_infos, x, "text"
                ),
                inputs=[lora.checkpoint],
                outputs=[lora.text],
            )

        generate.click(
            self.serve,
            inputs=[
                text,
                image,
                max_seq_length,
                *lora_params,
            ],
            outputs=[score],
        )

        iface.load(
            fn=lambda: [gr.update(value=self._name), gr.update(value=self._status)],
            outputs=[name, status],
        )

        iface.__exit__()

        super().__init__(config, iname="Matching", iface=iface)

    def start(self, pretrained_name, **kwargs):
        if self._name == pretrained_name and self._status == "Running":
            return self._status
        if self._status == "Running":
            self.stop()
        self._name = pretrained_name
        self._pipe = BletchleyForMatchingPipeline.from_core_configure(
            self._config,
            pretrained_name=pretrained_name,
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
        text: str,
        image: Image.Image,
        max_seq_length: int,
        *params,
    ):
        assert self._pipe is not None
        lora_params = params
        lora_checkpoints = lora_params[0::5]
        lora_weights = lora_params[1::5]
        lora_alphas = lora_params[2::5]
        lora_urls = lora_params[3::5]
        lora_files = lora_params[4::5]
        score = self._pipe(
            text,
            image,
            max_seq_length=max_seq_length,
            lora_checkpoints=lora_checkpoints,
            lora_weights=lora_weights,
            lora_alphas=lora_alphas,
            lora_urls=lora_urls,
            lora_files=lora_files,
        )
        return score


@register_webui("core/webui/bletchley/v1")
class BletchleyWebUI(SimpleWebUI):
    def __init__(self, config: CoreConfigureParser):
        webuis = [
            BletchleyMatchingWebUI(config),
        ]
        iface = gr.TabbedInterface(
            [webui.iface for webui in webuis],
            tab_names=[webui.iname for webui in webuis],
        )
        super().__init__(config, iname="Bletchley V1", iface=iface)
