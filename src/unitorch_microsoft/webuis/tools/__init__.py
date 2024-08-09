# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import gradio as gr
from unitorch.cli import CoreConfigureParser
from unitorch.cli import register_webui
from unitorch.cli.webuis import SimpleWebUI
from unitorch.cli.webuis.tools.image import ImageWebUI
from unitorch_microsoft.webuis.tools.picasso import PicassoWebUI


@register_webui("microsoft/webui/tools")
class ToolsWebUI(SimpleWebUI):
    def __init__(self, config: CoreConfigureParser):
        webuis = [
            ImageWebUI(config),
            PicassoWebUI(config),
        ]
        iface = gr.TabbedInterface(
            [webui.iface for webui in webuis],
            tab_names=[webui.iname for webui in webuis],
        )
        super().__init__(config, iname="Tools", iface=iface)
