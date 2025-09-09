# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import requests
from PIL import Image
from typing import Optional, List
from unitorch.models import GenericOutputs
from unitorch_microsoft.chatgpt.github_copilot import (
    get_copilot_token,
)


class GPTModel:
    def __init__(self, use_gpt5: bool = True):
        self.use_gpt5 = use_gpt5

    def ask(
        self,
        messages: list,
        model: Optional[str] = "gpt-4.1",
        temperature: Optional[float] = 0.0,
        top_p: Optional[float] = 1.0,
        max_tokens: Optional[int] = 32768,
    ) -> GenericOutputs:
        if self.use_gpt5:
            model = "gpt-5"
            max_tokens = 16384

        headers = {
            "Authorization": f"Bearer {get_copilot_token()}",
            "User-Agent": "GitHub-Copilot-Client/1.0",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Editor-Version": "vscode/1.85.0",
            "Editor-Plugin-Version": "copilot/1.155.0",
        }
        try:
            response = requests.post(
                "https://api.githubcopilot.com/chat/completions",
                headers=headers,
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                },
            ).json()
            result = response["choices"][0]["message"]
            content = result.get("content", None)
            content = content if content is not None else ""
            content = content.strip()
            return GenericOutputs(content=content)
        except Exception as e:
            return GenericOutputs(
                content="",
                error=str(e),
            )

    def ask_tools(
        self,
        messages: list,
        tools: List[dict],
        tool_choice: Optional[str] = "auto",
        model: Optional[str] = "gpt-4.1",
        temperature: Optional[float] = 0.0,
        top_p: Optional[float] = 1.0,
        max_tokens: Optional[int] = 32768,
    ):
        if self.use_gpt5:
            model = "gpt-5"
            max_tokens = 16384
        headers = {
            "Authorization": f"Bearer {get_copilot_token()}",
            "User-Agent": "GitHub-Copilot-Client/1.0",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Editor-Version": "vscode/1.85.0",
            "Editor-Plugin-Version": "copilot/1.155.0",
        }
        try:
            response = requests.post(
                "https://api.githubcopilot.com/chat/completions",
                headers=headers,
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "tools": tools,
                    "tool_choice": tool_choice,
                },
            ).json()
            result = response["choices"][0]["message"]
            tool_calls = result.get("tool_calls", [])
            content = result.get("content", None)
            content = content if content is not None else ""
            content = content.strip()
            return GenericOutputs(
                content=content,
                tool_calls=tool_calls,
            )
        except Exception as e:
            return GenericOutputs(
                content="",
                tool_calls=[],
                error=str(e),
            )
