# Copyright (c) MICROSOFT.
# Licensed under the MIT License.
import gradio as gr
from unitorch.cli import CoreConfigureParser
from unitorch.cli import register_webui
from unitorch.cli.webuis import SimpleWebUI
from unitorch_microsoft.webuis.selection.image import ImageSelectionWebUI


@register_webui("microsoft/webui/selection")
class SelectionWebUI(SimpleWebUI):
    def __init__(self, config: CoreConfigureParser):
        webuis = [
            ImageSelectionWebUI(config),
        ]
        iface = gr.TabbedInterface(
            [webui.iface for webui in webuis],
            tab_names=[webui.iname for webui in webuis],
        )
        super().__init__(config, iname="Selection", iface=iface)
