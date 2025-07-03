# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import re
import sys
import fire
import json
import subprocess
import tempfile
import logging
import pandas as pd
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch_microsoft.chatgpt.papyrus import (
    get_gpt4_response,
    get_gpt4_tools_response,
    get_gpt4_chat_response,
    get_gpt_image_response,
)
from unitorch_microsoft.agents.components.tools import (
    PythonTool,
    BashTool,
    TerminateTool,
)

system_prompt = """
You are an agent that can execute tool calls to finish the user's instruction.

For complex tasks, you can break down the problem and use different tools step by step to solve it. After using each tool, clearly explain the execution results, check the current step and suggest the next steps.

You can use the following tools to assist you in completing the task:
1. `bash`: Execute bash commands. Don't use > redirection in the command, as it will not work as expected.
2. `python`: Execute Python code.
3. `terminate`: Terminate the script execution.
"""
action_prompt = """
Checking the tool results if it's not empty, then based on the current situation and exectued tool results, what's your next action to finish user's instruction?
Choose the most efficient path forward:
1. Is the plan sufficient, or does it need refinement?
2. Can you execute the next step immediately?
3. If you need any help, please use `ask_human` to get the answer from human.
4. Is the task complete? If so, use `terminate` right away.

Be concise in your reasoning, generate a detailed plan, check the current step and then select the appropriate tool or combination of tools or action as next step.

Summary of Current Situation: 
{0}

Executed Tool Results: {1}
User Instruction: {2}
User Inputs: {3}
"""

available_tools = [
    BashTool(),
    PythonTool(),
    TerminateTool(),
]


def function_call(
    name,
    args: Dict[str, Any] = None,
):
    output = None
    for tool in available_tools:
        if tool.name == name:
            if args is None:
                args = {}
            if not isinstance(args, dict):
                raise ValueError("Arguments must be a dictionary.")
            output = tool.execute(**args)

    return {"name": name, "output": output}


def cli_main(
    instruction,
    **kwargs,
):
    summary = ""
    tool_results = []
    step = 0
    while True:
        print("--" * 10, step, "--" * 10)
        print(summary)
        action_output = get_gpt4_tools_response(
            prompt=action_prompt.format(
                summary,
                json.dumps(
                    [res["output"] for res in tool_results],
                    indent=2,
                    ensure_ascii=False,
                ),
                instruction,
                json.dumps(kwargs, indent=2, ensure_ascii=False),
            ),
            tools=[t.to_params() for t in available_tools],
            # tool_choice="required",
            system_prompt=system_prompt,
        )
        tool_calls = action_output.get("tool_calls", [])
        content = action_output.get("content", "")
        print("Thinking:", content)
        print("Tool Calls:", json.dumps(tool_calls, indent=2, ensure_ascii=False))
        tool_results.clear()
        is_terminated = False
        if tool_calls and len(tool_calls) > 0:
            for tool_call in tool_calls:
                if tool_call.get("type", None) != "function":
                    continue
                tool_call = tool_call["function"]
                tool_results.append(
                    function_call(
                        tool_call["name"],
                        json.loads(tool_call["arguments"]),
                    )
                )

                if tool_call["name"] == "terminate":
                    logging.info(
                        f"Terminating the script execution. status: {json.loads(tool_call['arguments'])['status']}"
                    )
                    is_terminated = True
                    break
        if is_terminated:
            break

        summary = get_gpt4_response(
            prompt=f"""
            You are a summarizer. Your task is to generate a new summary based on the current situation, thought, and executed tool results. Please keep all the important tools/steps/information which might be useful for the next steps.
            
            Current Situation: 
            {summary}

            Below is the information you need to add to the summary:
            Thought: {content}
            Executed Tool Results: {json.dumps([res['output'] for res in tool_results], indent=2, ensure_ascii=False)}

            Please provide a concise summary of current situation, thought, executed tool results, and generate a refined detailed plan, current step the next steps.
            """,
        )
        print("--" * 10, step, "--" * 10)
        step += 1


if __name__ == "__main__":
    fire.Fire(cli_main)
