# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

from unitorch_microsoft.agents.components.tools import GenericTool, GenericResult

_TERMINATE_DESCRIPTION = """
Terminate the interaction when the request is met OR if the assistant cannot proceed further with the task. When you have finished all the tasks, call this tool to end the work.
"""


class TerminateTool(GenericTool):
    """Add a tool to terminate the agent."""

    name: str = "terminate"
    description: str = _TERMINATE_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "description": "The finish status of the interaction.",
                "enum": ["success", "failure"],
            }
        },
        "required": ["status"],
    }

    def execute(self, status: str):
        return {
            "status": status,
            "message": f"The interaction has been completed with status: {status}",
        }
