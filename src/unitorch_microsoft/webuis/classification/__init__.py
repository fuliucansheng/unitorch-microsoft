# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import gradio as gr
from unitorch.cli import CoreConfigureParser
from unitorch.cli import register_webui
from unitorch.cli.webuis import SimpleWebUI
from unitorch.cli.webuis.clip import ClipWebUI
from unitorch_microsoft.models.bletchley.webui_v1 import (
    BletchleyWebUI as BletchleyV1WebUI,
)
from unitorch_microsoft.models.bletchley.webui_v3 import (
    BletchleyWebUI as BletchleyV3WebUI,
)


@register_webui("microsoft/webui/classification")
class ClassificationWebUI(SimpleWebUI):
    def __init__(self, config: CoreConfigureParser):
        webuis = [
            ClipWebUI(config),
            BletchleyV1WebUI(config),
            BletchleyV3WebUI(config),
        ]
        iface = gr.TabbedInterface(
            [webui.iface for webui in webuis],
            tab_names=[webui.iname for webui in webuis],
        )
        super().__init__(config, iname="Classification", iface=iface)
