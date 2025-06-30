# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import json
import re
import random
import logging
import requests
import pandas as pd
import gradio as gr
from typing import Optional
from unitorch.cli import register_webui, CoreConfigureParser
from unitorch.cli.webuis import SimpleWebUI
from unitorch.cli.webuis.utils import create_blocks
from unitorch_microsoft.chatgpt.papyrus import get_gpt4_chat_response


@register_webui("microsoft/webui/chatgpt")
class ChatGPTWebUI(SimpleWebUI):
    def __init__(self, config: CoreConfigureParser):
        config.set_default_section("microsoft/webui/chatgpt")

        chatbot = gr.Chatbot(
            bubble_full_width=False,
            height=800,
            editable="all",
            show_copy_button=True,
        )
        message = gr.Textbox(
            interactive=True,
            placeholder="Enter message...",
            show_label=False,
        )

        iface = create_blocks(chatbot, message)

        iface.__enter__()

        message.submit(self.response, [message, chatbot], [message, chatbot])

        iface.__exit__()

        super().__init__(config, "ChatGPT", iface)

    def response(self, message, histories):
        answer = get_gpt4_chat_response(
            histories,
            message=message,
        )
        histories.append((message, answer))
        return "", histories
