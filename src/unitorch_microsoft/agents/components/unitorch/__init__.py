# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import sys
import glob
import frontmatter
import subprocess
import importlib_resources
from unitorch_microsoft.chatgpt.papyrus import (
    get_gpt4_response,
    get_gpt4_tools_response,
    get_gpt4_chat_response,
    get_gpt_image_response,
)
from unitorch_microsoft.agents.components.tools import GenericTool, GenericResult
from unitorch_microsoft.agents.components.tools import (
    PythonTool,
    BashTool,
    AskHumanTool,
    TerminateTool,
)

pkg_folder = importlib_resources.files("unitorch_microsoft")
model_docs = glob.glob(
    f"{pkg_folder}/agents/components/unitorch/models/*.md", recursive=True
)
tool_docs = glob.glob(
    f"{pkg_folder}/agents/components/unitorch/tools/*.md", recursive=True
)

docs = {}
for doc in model_docs + tool_docs:
    name = doc.split("/")[-1].replace(".md", "")
    post = frontmatter.load(doc)
    if "<!-- MACHINE_GENERATED -->" in post.content:
        human, machine = post.content.split("<!-- MACHINE_GENERATED -->", 1)
        docs[name] = {
            "title": post.get("title", ""),
            "description": post.get("description", ""),
            "content": machine.strip(),
        }

available_docs = list(docs.keys())

_UNITORCH_DOCS_DESCRIPTION = """
A collection of docs for Unitorch Tools, including different models for training/inference and useful tools for data analysis in the Unitorch framework.
You can use this tool to reference the documentation of Unitorch models and tools.
Available tools:
"""

for name, doc in docs.items():
    _UNITORCH_DOCS_DESCRIPTION += f"\n- **{name}**: {doc['description'] if doc['description'] else 'No description available.'}\n"

_UNITORCH_DOCS_DESCRIPTION += (
    "\nYou can use the `document` parameter to specify which document you want to reference. The available documents are: "
    + ", ".join(available_docs)
    + "."
)


class UnitorchDocsTool(GenericTool):
    """A collection of docs for Unitorch."""

    name: str = "unitorch_docs"
    description: str = _UNITORCH_DOCS_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "document": {
                "type": "string",
                "enum": list(available_docs),
                "description": "The name of the document to reference. Available tools: "
                + ", ".join(available_docs),
            }
        },
        "required": ["document"],
    }

    def execute(self, document: str):
        return docs.get(document, None)


_UNITORCH_DATAPROCESS_DESCRIPTION = """
Use this tool to process data for Unitorch models.
You can provide a detailed instruction for data processing, including the input/output data format, processing steps, and any specific requirements.
The instruction should be clear and concise to ensure accurate data processing without human inputs.
"""


class UnitorchDataProcessTool(GenericTool):
    name: str = "unitorch_data_process"
    description: str = _UNITORCH_DATAPROCESS_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "instruction": {
                "type": "string",
                "description": "A detailed instruction for data processing including the input/output data format, processing steps, and any specific requirements. Please provide a clear and concise instruction to ensure accurate data processing without human inputs.",
            }
        },
        "required": ["instruction"],
    }

    def execute(self, instruction: str):
        result = subprocess.run(
            [sys.executable, "-m", "unitorch_microsoft.agents.tools.auto", instruction],
            capture_output=True,
            text=True,
        )
        return {
            "instruction": instruction,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "returncode": result.returncode,
        }
