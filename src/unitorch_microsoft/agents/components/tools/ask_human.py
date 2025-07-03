# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

from unitorch_microsoft.agents.components.tools import GenericTool, GenericResult

_ASKHUMAN_DESCRIPTION = """
Use this tool to ask human for help.
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
            }
        },
        "required": ["question"],
    }

    def execute(self, question: str) -> str:
        answer = input(f"""Question: {question}\n\nYou: """).strip()
        return {
            "question": question,
            "answer": answer,
        }
