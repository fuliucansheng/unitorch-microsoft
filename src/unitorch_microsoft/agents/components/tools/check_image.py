# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import json
from PIL import Image
from typing import Any, DefaultDict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field
from unitorch_microsoft.agents.components import (
    GenericTool,
    GenericResult,
    GenericError,
    ToolCollection,
    Memory,
    Message,
    ToolCall,
    ToolChoice,
)
from unitorch_microsoft.agents.components.tools import GPT4FormatTool
from unitorch_microsoft.agents.utils.chatgpt import GPTModel

# from unitorch_microsoft.agents.utils.github_copilot import GPTModel


class CheckedResult(BaseModel):
    """Checked Result Model."""

    image: Optional[str] = Field(
        default=None,
        description="The image path of the checked image for evaluation. leave it empty if no image to review.",
    )
    priority: Optional[str] = Field(
        default="low",
        description="The priority of the checked image. Can be 'high', 'medium', or 'low'. The high priority image will be put in the memory (very cost) every time and will be viewed for all future steps. medium priority image will be viewed for the next step, and low priority image will not be viewed for any any step.",
    )
    feedback: Optional[str] = Field(
        default=None,
        description="The comment or feedback on the checked image. leave it empty if no image.",
    )


class CheckImageTool(GenericTool):
    """Add a tool to check the image based on the provided prompt & images."""

    name: str = "check_image_tool"
    description: str = """
Use this tool to check the image with prompt or mark the image to the priority.

Key capabilities include:
* `check_image`: Analyze the image based on the provided prompt.
* `mark_image`: Mark the image to the specified priority. whether the image bytes will be viewed in the future steps depends on the priority.

Parameters:
- `command`: The command to execute. Available commands: `check_image`, `mark_image`.
- `prompt`: The prompt for checking the images. It describes the instruction to be checked.
- `image`: The image path for checking.
- `priority`: The priority of the image. Can be 'high', 'medium', or 'low'. The parameter for `mark_image` command.
    * `high`: the image will be put in the memory (very cost) and will be viewed for all future steps.
    * `medium`: the image will be viewed in the next step.
    * `low`: the image will not be viewed for any step. This is the default value.
    Don't set priority to "high" even if the image is very critical for the task. Mark the image to "medium" when you want to view it.
"""

    parameters: dict = {
        "type": "object",
        "properties": {
            "command": {
                "description": "The command to execute. Available commands: check_image, mark_image.",
                "enum": [
                    "check_image",
                    "mark_image",
                ],
                "type": "string",
            },
            "image": {
                "type": "string",
                "description": "The image path for checking or marking.",
            },
            "prompt": {
                "type": "string",
                "description": "The prompt for checking the images. It describes the instruction to be checked.",
                "default": "",
            },
            "priority": {
                "type": "string",
                "description": "The priority of the checked result. Can be 'high', 'medium', or 'low'. only work for `mark_image` command.",
                "default": "low",
            },
        },
        "required": ["command", "image"],
        "dependencies": {
            "check_image": {
                "required": ["image", "prompt"],
            },
            "mark_image": {
                "required": ["image", "priority"],
            },
        },
    }
    gpt: Any = GPTModel()
    available_tools: ToolCollection = ToolCollection(
        GPT4FormatTool(CheckedResult),
    )

    async def execute(
        self,
        command: str,
        image: str,
        prompt: Optional[str] = None,
        priority: Literal["high", "medium", "low"] = "low",
    ):
        """Execute the tool to check the image."""
        if image is None or not os.path.exists(image):
            raise ValueError("image is required and must exist.")
        if command == "check_image":
            if prompt is None:
                raise ValueError("prompt is required for check_image command.")
            resp = self.gpt.ask_tools(
                messages=Memory(
                    messages=[
                        Message.system_message(
                            content="You are a image analysis AI to finish user's task. Choose only one tool call for action."
                        ),
                        Message.user_message(
                            content=prompt,
                            images=[
                                {
                                    "path": image,
                                    "width": None,
                                    "height": None,
                                    "priority": "high",
                                }
                            ],
                        ),
                    ]
                ).to_dict_list(),
                tools=self.available_tools.to_params(),
                tool_choice=ToolChoice.REQUIRED,
            )
            tool_calls = [ToolCall(**tc) for tc in resp.tool_calls]
            if len(tool_calls) != 1:
                print("Check Image Tool calls:", tool_calls)
                raise ValueError(
                    "The tool call should only return one tool call, but got multiple."
                )
            args = json.loads(tool_calls[0].function.arguments)
            result = CheckedResult(**args)
            im = Image.open(result.image)
            return GenericResult(
                output=f"Checked image {result.image} with prompt: {prompt}. Width: {im.width}, Height: {im.height}. Checked result: {result.feedback if result.feedback else 'No feedback provided'}.",
                images={
                    "path": result.image,
                    "width": im.width,
                    "height": im.height,
                    "priority": "low",
                },
                meta={
                    "_width": im.width,
                    "_height": im.height,
                    "_prompt": prompt,
                    "_image": result.image,
                    "_priority": "low",
                },
            )
        elif command == "mark_image":
            if priority not in ["high", "medium", "low"]:
                raise ValueError(
                    f"priority must be one of 'high', 'medium', or 'low', but got {priority}."
                )
            im = Image.open(image)
            return GenericResult(
                output=f"Marked image {image} with priority: {priority}. Width: {im.width}, Height: {im.height}.",
                images={
                    "path": image,
                    "width": im.width,
                    "height": im.height,
                    "priority": priority,
                },
                meta={
                    "_width": im.width,
                    "_height": im.height,
                    "_image": image,
                    "_priority": priority,
                },
            )
