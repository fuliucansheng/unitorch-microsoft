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
from unitorch_microsoft.webuis.image_retrieval.getty import GettyImageRetrievalWebUI


@register_webui("microsoft/webui/image_retrieval")
class ImageRetrievalWebUI(SimpleWebUI):
    def __init__(self, config: CoreConfigureParser):
        webuis = [
            GettyImageRetrievalWebUI(config),
        ]
        iface = gr.TabbedInterface(
            [webui.iface for webui in webuis],
            tab_names=[webui.iname for webui in webuis],
        )
        super().__init__(config, iname="ImageRetrieval", iface=iface)
