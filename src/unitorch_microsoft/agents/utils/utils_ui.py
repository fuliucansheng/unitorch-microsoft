# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import json
import urllib.parse
from unitorch.models import GenericOutputs
from unitorch_microsoft.agents.components import (
    GenericResult,
    ToolCall,
)


def format_tool_for_logs(tool_call: ToolCall) -> str:
    res = f"✅ 🎯 **Tool:** `{tool_call.function.name}` used.\n\n"
    args = json.loads(tool_call.function.arguments)
    res += f"```json \n{json.dumps(args, indent=2, ensure_ascii=False)}\n```"
    return res


def format_tool_for_preview(tool_call: ToolCall, result: GenericResult) -> str:
    res = f"## 🛠 Tool: `{tool_call.function.name}` Execution\n\n"
    if tool_call.function.name == "browser_use":
        res += f"- **Action:** {result.meta.get('_action', 'N/A')}\n- **Goal:** {result.meta.get('_goal', 'N/A')}\n- **URL:** [🔗 Link]({result.meta.get('_url', 'N/A')}\n\n**📸 Screenshot:**\n\n![Browser Screenshot](/gradio_api/file={urllib.parse.quote(result.meta.get('_screenshot', 'N/A'))})"
    elif tool_call.function.name == "web_search":
        search_results = []
        for i, _result in enumerate(result.results):
            search_results.append(
                f"{i+1}. [{_result.title}]({_result.url}) - {_result.description}"
            )
        search_results = "\n".join(search_results)
        res += f"- **Query:** {result.query}\n **📑 Top Results:**\n\n{search_results}"
    elif tool_call.function.name == "editor":
        _path = result.meta.get("_path", "N/A")
        res += f"- **Command:** `{result.meta.get('_command', 'N/A')}`\n - **Path:** [📄 {os.path.basename(_path)}](/gradio_api/file={urllib.parse.quote(_path)})\n\n**📄 File Content:**\n\n```txt\n{result.meta.get('_content', '')}\n```"
    elif tool_call.function.name == "check_image_tool":
        res += f"- **Prompt:** {result.meta.get('_prompt', 'N/A')}\n\n - **Image:** {result.meta.get('_width', 'N/A')}x{result.meta.get('_height', 'N/A')}:\n\n ![📷 Image](/gradio_api/file={urllib.parse.quote(result.meta.get('_image', 'N/A'))})\n\n - **Priority:** {result.meta.get('_priority', 'N/A')}\n\n - **Feedback:**\n\n{result.output if result.output else 'No feedback provided'}"
    elif (
        tool_call.function.name == "picasso_image_tool"
        or tool_call.function.name == "picasso_html_tool"
        or tool_call.function.name == "picasso_internal_tool"
        or tool_call.function.name == "picasso_layout_tool"
    ):
        res += f"- **Image {result.meta.get('_width', 'N/A')}x{result.meta.get('_height', 'N/A')}:**\n\n![📷 Image](/gradio_api/file={urllib.parse.quote(result.meta.get('_image', 'N/A'))})\n"
    else:
        res += f"**📤 Results:**\n\n```txt\n{result.output if result.output else 'No output'}\n```"
    return res
