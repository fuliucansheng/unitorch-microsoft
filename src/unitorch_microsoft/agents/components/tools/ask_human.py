# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

from typing import Any, DefaultDict, List, Literal, Optional, Tuple
from unitorch_microsoft.agents.components import (
    GenericTool,
    GenericResult,
    GenericError,
)

_ASKHUMAN_DESCRIPTION = """
Ask user a question and wait for response. Use for requesting clarification, asking for confirmation, or gathering additional information.
"""


class AskHumanTool(GenericTool):
    """Add a tool to ask human for help."""

    name: str = "ask_human"
    description: str = _ASKHUMAN_DESCRIPTION
    parameters: str = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question you want to ask human.",
            },
            "attachments": {
                "anyOf": [
                    {"type": "string"},
                    {"items": {"type": "string"}, "type": "array"},
                ],
                "description": "(Optional) List of attachments to show to user, can be file paths or URLs",
            },
        },
        "required": ["question"],
    }
    mode: str = "cli"  # "cli" or "ui"

    def __init__(self, mode: str = "cli"):
        """Initialize with a specific mode."""
        super().__init__()
        self.mode = mode

    async def execute(self, question: str, attachments=None):
        if attachments is None:
            attachments = []
        if isinstance(attachments, str):
            attachments = [attachments]
        elif not isinstance(attachments, list):
            raise GenericError("Attachments must be a string or a list of strings.")

        if self.mode == "cli":
            if attachments:
                answer = input(
                    f"""Question: {question}\n\nAttachments: {attachments}\n\nYou: """
                ).strip()
            else:
                answer = input(f"""Question: {question}\n\nYou: """).strip()
        else:
            answer = ""

        result = GenericResult(
            output=answer,
        )
        result.meta["question"] = question
        result.meta["attachments"] = attachments
        result.meta["answer"] = answer

        return result
