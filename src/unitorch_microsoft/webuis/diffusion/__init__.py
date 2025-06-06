# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import gradio as gr
from unitorch.cli import CoreConfigureParser
from unitorch.cli import register_webui
from unitorch.cli.webuis import SimpleWebUI
from unitorch.cli.webuis.stable import StableWebUI
from unitorch.cli.webuis.stable_xl import StableXLWebUI

from unitorch.cli.webuis.stable_3 import Stable3WebUI
from unitorch.cli.webuis.stable_flux import StableFluxWebUI


@register_webui("microsoft/webui/diffusion")
class DiffusionWebUI(SimpleWebUI):
    def __init__(self, config: CoreConfigureParser):
        webuis = [
            StableWebUI(config),
            StableXLWebUI(config),
            Stable3WebUI(config),
            StableFluxWebUI(config),
        ]
        iface = gr.TabbedInterface(
            [webui.iface for webui in webuis],
            tab_names=[webui.iname for webui in webuis],
        )
        super().__init__(config, iname="Diffusion", iface=iface)
