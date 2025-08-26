# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import io
import sys
import subprocess
from typing import Any, Dict, Optional
from numpy import std
from unitorch_microsoft.agents.components import (
    GenericTool,
    GenericResult,
    GenericError,
)

_PYTHON_DESCRIPTION = """
Use this tool to execute Python code. Note that only print outputs are visible, and function return values are not captured. Use print statements to see results.
"""


class PythonTool(GenericTool):
    """Add a tool to execute Python code."""

    name: str = "python"
    description: str = _PYTHON_DESCRIPTION
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The Python code you want to execute.",
            }
        },
        "required": ["code"],
    }

    async def execute(self, code: str) -> Optional[Dict[str, Any]]:
        process = subprocess.Popen(
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = process.communicate()
        return GenericResult(
            output=(
                f"Execution return code: {process.returncode}. " + stdout.strip()
                if stdout
                else ""
            ),
            error=stderr.strip() if stderr else "",
        )
