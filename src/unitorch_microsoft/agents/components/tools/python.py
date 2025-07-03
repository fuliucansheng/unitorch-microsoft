# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import io
import sys
import subprocess
from typing import Any, Dict, Optional
from unitorch_microsoft.agents.components.tools import GenericTool, GenericResult

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

    def execute(self, code: str) -> Optional[Dict[str, Any]]:
        results = {"code": code}
        process = subprocess.Popen(
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = process.communicate()
        results["returncode"] = process.returncode
        results["stdout"] = stdout.strip()
        results["stderr"] = stderr.strip()
        return results
