# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import gradio as gr
from unitorch.cli import CoreConfigureParser
from unitorch.cli import register_webui
from unitorch.cli.webuis import SimpleWebUI
from unitorch.cli.webuis.bloom import BloomWebUI
from unitorch.cli.webuis.llama import LlamaWebUI
from unitorch.cli.webuis.mistral import MistralWebUI
from unitorch.cli.webuis.llava import LLAVAWebUI
from unitorch.cli.webuis.stable import StableWebUI
from unitorch.cli.webuis.stable_xl import StableXLWebUI
from unitorch.cli.webuis.stable_3 import Stable3WebUI
from unitorch.cli.webuis.bria import BRIAWebUI
from unitorch.cli.webuis.detr import DetrWebUI
from unitorch.cli.webuis.sam import SamWebUI

from unitorch_microsoft.webuis.image_retrieval.getty import GettyImageRetrievalWebUI
from unitorch_microsoft.models.mask2former.webui import Mask2FormerWebUI


@register_webui("microsoft/webui/llm")
class LLMWebUI(SimpleWebUI):
    def __init__(self, config: CoreConfigureParser):
        webuis = [
            BloomWebUI(config),
            LlamaWebUI(config),
            MistralWebUI(config),
            LLAVAWebUI(config),
        ]
        iface = gr.TabbedInterface(
            [webui.iface for webui in webuis],
            tab_names=[webui.iname for webui in webuis],
        )
        super().__init__(config, iname="LLM", iface=iface)


@register_webui("microsoft/webui/diffusion")
class DiffusionWebUI(SimpleWebUI):
    def __init__(self, config: CoreConfigureParser):
        webuis = [
            StableWebUI(config),
            StableXLWebUI(config),
            Stable3WebUI(config),
        ]
        iface = gr.TabbedInterface(
            [webui.iface for webui in webuis],
            tab_names=[webui.iname for webui in webuis],
        )
        super().__init__(config, iname="Diffusion", iface=iface)


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


@register_webui("microsoft/webui/selection")
class SelectionWebUI(SimpleWebUI):
    def __init__(self, config: CoreConfigureParser):
        webuis = [
            GettyImageRetrievalWebUI(config),
        ]
        iface = gr.TabbedInterface(
            [webui.iface for webui in webuis],
            tab_names=[webui.iname for webui in webuis],
        )
        super().__init__(config, iname="Selection", iface=iface)
