# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

from abc import ABC, abstractmethod
from PIL import Image
from typing import Any, Optional
from pydantic import BaseModel, Field


class GenericTool(ABC, BaseModel):
    name: str
    description: str
    parameters: Optional[dict] = None

    @abstractmethod
    def execute(self, **kwargs):
        pass

    def to_params(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class GenericResult(BaseModel):
    output: Any = Field(default=None)
    error: Optional[str] = Field(default=None)
    base64_image: Optional[str] = Field(default=None)
    system: Optional[str] = Field(default=None)


from unitorch_microsoft.agents.components.tools.python import PythonTool
from unitorch_microsoft.agents.components.tools.bash import BashTool
from unitorch_microsoft.agents.components.tools.ask_human import AskHumanTool
from unitorch_microsoft.agents.components.tools.terminate import TerminateTool
