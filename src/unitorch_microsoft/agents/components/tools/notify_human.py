# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import logging
from unitorch_microsoft.agents.components import (
    GenericTool,
    GenericResult,
    GenericError,
)

_NOTIFYHUMAN_DESCRIPTION = """
Send a message to user without requiring a response. Use for acknowledging receipt of messages, providing progress updates, reporting task completion, or explaining changes in approach.
"""


class NotifyHumanTool(GenericTool):
    """Add a tool to notify human without requiring a response."""

    name: str = "notify_human"
    description: str = _NOTIFYHUMAN_DESCRIPTION
    parameters: str = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Message text to display to user",
            },
            "attachments": {
                "anyOf": [
                    {"type": "string"},
                    {"items": {"type": "string"}, "type": "array"},
                ],
                "description": "(Optional) List of attachments to show to user, can be file paths or URLs",
            },
        },
        "required": ["text"],
    }

    async def execute(self, text: str, attachments=None) -> str:
        """Execute the tool to notify human."""
        if attachments is None:
            attachments = []
        if isinstance(attachments, str):
            attachments = [attachments]
        elif not isinstance(attachments, list):
            raise GenericError("Attachments must be a string or a list of strings.")

        logging.info(f"Notify human: {text}, attachments: {attachments}")

        result = GenericResult(
            output="Notification sent.",
        )
        result.meta["text"] = text
        result.meta["attachments"] = attachments
        return result
