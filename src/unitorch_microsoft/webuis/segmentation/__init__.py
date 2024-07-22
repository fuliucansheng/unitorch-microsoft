# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import gradio as gr
from unitorch.cli import CoreConfigureParser
from unitorch.cli import register_webui
from unitorch.cli.webuis import SimpleWebUI
from unitorch.cli.webuis.bria import BRIAWebUI
from unitorch.cli.webuis.sam import SamWebUI
from unitorch_microsoft.models.mask2former.webui import Mask2FormerWebUI


@register_webui("microsoft/webui/segmentation")
class SegmentationWebUI(SimpleWebUI):
    def __init__(self, config: CoreConfigureParser):
        webuis = [
            Mask2FormerWebUI(config),
            BRIAWebUI(config),
            SamWebUI(config),
        ]
        iface = gr.TabbedInterface(
            [webui.iface for webui in webuis],
            tab_names=[webui.iname for webui in webuis],
        )
        super().__init__(config, iname="Segmentation", iface=iface)
