# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import requests
from PIL import Image
from typing import Optional, List
from unitorch.models import GenericOutputs
from unitorch_microsoft.chatgpt.papyrus import (
    papyrus_endpoint3,
    get_access_token,
)


class GPTModel:
    def __init__(self, use_gpt5: bool = True):
        self.use_gpt5 = use_gpt5

    def ask(
        self,
        messages: list,
        model: Optional[str] = "gpt-41-2025-04-14-Eval",
        temperature: Optional[float] = 0.0,
        top_p: Optional[float] = 1.0,
        max_tokens: Optional[int] = 32768,
    ) -> GenericOutputs:
        if self.use_gpt5:
            model = "gpt-5-2025-08-07-Eval"
            max_tokens = 16384
        headers = {
            "Authorization": "Bearer " + get_access_token(),
            "Content-Type": "application/json",
            "papyrus-model-name": model,
            "papyrus-quota-id": "",
            "papyrus-timeout-ms": "120000",
        }
        if not model.startswith("gpt-5"):
            data = {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
        else:
            data = {
                "messages": messages,
                "max_completion_tokens": max_tokens,
                "top_p": top_p,
            }
        try:
            response = requests.post(
                papyrus_endpoint3,
                headers=headers,
                json=data,
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
        model: Optional[str] = "gpt-41-2025-04-14-Eval",
        temperature: Optional[float] = 0.0,
        top_p: Optional[float] = 1.0,
        max_tokens: Optional[int] = 32768,
    ):
        if self.use_gpt5:
            model = "gpt-5-2025-08-07-Eval"
            max_tokens = 16384
        headers = {
            "Authorization": "Bearer " + get_access_token(),
            "Content-Type": "application/json",
            "papyrus-model-name": model,
            "papyrus-quota-id": "",
            "papyrus-timeout-ms": "120000",
        }
        if not model.startswith("gpt-5"):
            data = {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "tools": tools,
                "tool_choice": tool_choice,
            }
        else:
            data = {
                "messages": messages,
                "max_completion_tokens": max_tokens,
                "top_p": top_p,
                "tools": tools,
                "tool_choice": tool_choice,
            }
        try:
            response = requests.post(
                papyrus_endpoint3,
                headers=headers,
                json=data,
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
