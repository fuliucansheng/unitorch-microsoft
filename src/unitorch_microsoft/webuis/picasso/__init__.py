# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import gradio as gr
from unitorch.cli import CoreConfigureParser
from unitorch.cli import register_webui
from unitorch.cli.webuis import SimpleWebUI
from unitorch_microsoft.models.mask2former.webui import Mask2FormerWebUI
from unitorch_microsoft.webuis.picasso.video import VideoWebUI
from unitorch_microsoft.webuis.picasso.tools import ToolsWebUI
from unitorch_microsoft.webuis.picasso.bg_type import BackgroundTypeWebUI


@register_webui("microsoft/webui/picasso")
class PicassoWebUI(SimpleWebUI):
    def __init__(self, config: CoreConfigureParser):
        webuis = [
            BackgroundTypeWebUI(config),
            Mask2FormerWebUI(config),
            VideoWebUI(config),
            ToolsWebUI(config),
        ]
        iface = gr.TabbedInterface(
            [webui.iface for webui in webuis],
            tab_names=[webui.iname for webui in webuis],
        )
        super().__init__(config, iname="Picasso", iface=iface)
