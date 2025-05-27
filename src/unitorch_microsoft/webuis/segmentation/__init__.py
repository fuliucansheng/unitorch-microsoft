# Copyright (c) MICROSOFT.
# Licensed under the MIT License.
import gradio as gr
from unitorch.cli import CoreConfigureParser
from unitorch.cli import register_webui
from unitorch.cli.webuis import SimpleWebUI
from unitorch.cli.webuis.bria import BRIAWebUI
from unitorch.cli.webuis.sam import SamWebUI
from unitorch.cli.webuis.mask2former import Mask2FormerWebUI
from unitorch.cli.webuis.segformer import SegformerWebUI

import unitorch_microsoft.models.sam


@register_webui("microsoft/webui/segmentation")
class SegmentationWebUI(SimpleWebUI):
    def __init__(self, config: CoreConfigureParser):
        webuis = [
            SamWebUI(config),
            BRIAWebUI(config),
            Mask2FormerWebUI(config),
            SegformerWebUI(config),
        ]
        iface = gr.TabbedInterface(
            [webui.iface for webui in webuis],
            tab_names=[webui.iname for webui in webuis],
        )
        super().__init__(config, iname="Segmentation", iface=iface)
