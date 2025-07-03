# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import sys
import subprocess
from unitorch_microsoft.agents.components.tools import GenericTool, GenericResult

_BASH_DESCRIPTION = """
Use this tool to execute bash commands. 
Note: 
1. Only the standard output and error are captured, not the return value of the command. 
2. Use echo statements to see results. 
3. Don't use > redirection in the command, as it will not work as expected.
"""


class BashTool(GenericTool):
    """Add a tool to execute bash commands."""

    name: str = "bash"
    description: str = _BASH_DESCRIPTION
    parameters: str = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The bash command you want to execute. ",
            }
        },
        "required": ["command"],
    }

    def execute(self, command: str) -> str:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return {
            "command": command,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "returncode": result.returncode,
        }
