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

try:
    from openai import AzureOpenAI
except ImportError:
    raise ImportError(
        "Please install the openai package by running `pip install openai`"
    )


def get_respone(
    histories,
    message: str,
    system_prompt: Optional[
        str
    ] = "You are an AI assistant that helps people find information.",
    api_endpoint: Optional[str] = None,
    api_deploy_name: Optional[str] = None,
    api_key: Optional[str] = None,
):
    client = AzureOpenAI(
        azure_endpoint=api_endpoint, api_key=api_key, api_version="2023-07-01-preview"
    )

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ]
    for msg, ans in histories:
        messages += [
            {
                "role": "user",
                "content": msg,
            },
            {
                "role": "assistant",
                "content": ans,
            },
        ]

    messages += [
        {
            "role": "user",
            "content": message,
        }
    ]

    try:
        response = client.chat.completions.create(
            model=api_deploy_name,
            messages=messages,
        )
        result = response.choices[0].message.content.strip()
    except Exception as e:
        print(e)
        result = ""

    return result


@register_webui("microsoft/webui/chatgpt")
class ChatGPTWebUI(SimpleWebUI):
    def __init__(self, config: CoreConfigureParser):
        config.set_default_section("microsoft/webui/chatgpt")
        self.api_endpoint = config.getoption("api_endpoint", None)
        self.api_key = config.getoption("api_key", None)
        self.api_deploy_name = config.getoption("api_deploy_name", None)

        chatbot = gr.Chatbot(label="ChatGPT", height=800)
        message = gr.Textbox(label="Message")

        iface = create_blocks(chatbot, message)

        iface.__enter__()

        message.submit(self.response, [message, chatbot], [message, chatbot])

        iface.__exit__()

        super().__init__(config, "ChatGPT", iface)

    def response(self, message, histories):
        answer = get_respone(
            histories,
            message=message,
            api_endpoint=self.api_endpoint,
            api_key=self.api_key,
            api_deploy_name=self.api_deploy_name,
        )
        histories.append((message, answer))
        return "", histories
