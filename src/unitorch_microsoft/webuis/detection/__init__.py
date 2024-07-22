# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import gradio as gr
from unitorch.cli import CoreConfigureParser
from unitorch.cli import register_webui
from unitorch.cli.webuis import SimpleWebUI
from unitorch.cli.webuis.detr import DetrWebUI


@register_webui("microsoft/webui/detection")
class DetectionWebUI(SimpleWebUI):
    def __init__(self, config: CoreConfigureParser):
        webuis = [
            DetrWebUI(config),
        ]
        iface = gr.TabbedInterface(
            [webui.iface for webui in webuis],
            tab_names=[webui.iname for webui in webuis],
        )
        super().__init__(config, iname="Detection", iface=iface)
