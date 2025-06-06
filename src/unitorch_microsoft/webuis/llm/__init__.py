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

import unitorch_microsoft.models.bloom


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
