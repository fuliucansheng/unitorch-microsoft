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
from unitorch.cli.webuis import SimpleWebUI
from unitorch_microsoft.webuis.selection.image.getty import GettyImageSelectionWebUI
from unitorch_microsoft.webuis.selection.image.tagger import ImageTaggerSelectionWebUI


@register_webui("microsoft/webui/selection/image")
class ImageSelectionWebUI(SimpleWebUI):
    def __init__(self, config: CoreConfigureParser):
        webuis = [
            GettyImageSelectionWebUI(config),
            ImageTaggerSelectionWebUI(config),
        ]
        iface = gr.TabbedInterface(
            [webui.iface for webui in webuis],
            tab_names=[webui.iname for webui in webuis],
        )
        super().__init__(config, iname="Image", iface=iface)
