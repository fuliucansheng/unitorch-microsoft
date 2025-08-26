# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import json
import base64
import requests
from abc import ABC, abstractmethod
from enum import Enum
from PIL import Image
from typing import Any, List, Literal, Optional, Union, Dict
from pydantic import BaseModel, Field, model_validator
from transformers.utils import is_remote_url


class GenericTool(ABC, BaseModel):
    name: str
    description: str
    parameters: Optional[dict] = None

    class Config:
        arbitrary_types_allowed = True

    def to_params(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    @abstractmethod
    def execute(self, **kwargs):
        """Execute the tool with the provided parameters."""
        raise NotImplementedError("Subclasses must implement this method.")


class ImageResult(BaseModel):
    path: str = Field(..., description="Path to the image file.")
    priority: Optional[Literal["high", "medium", "low"]] = Field(
        default="low",
        description="The priority level of the image in the result. Allowed values: 'high', 'medium', 'low'.",
    )
    width: Optional[int] = Field(
        default=None,
        description="Width of the image in pixels. Optional if not applicable.",
    )
    height: Optional[int] = Field(
        default=None,
        description="Height of the image in pixels. Optional if not applicable.",
    )
    mode: Optional[str] = Field(
        default=None,
        description="Color mode of the image (e.g., 'RGB', 'RGBA'). Optional if not applicable.",
    )

    @model_validator(mode="after")
    def validate_image_path(self) -> "ImageResult":
        """Ensure the image path is valid and populate dimensions."""
        try:
            with Image.open(self.path) as img:
                object.__setattr__(self, "width", img.width)
                object.__setattr__(self, "height", img.height)
                object.__setattr__(self, "mode", img.mode)
        except Exception as e:
            raise ValueError(f"Failed to open image at '{self.path}': {e}")
        return self


class GenericResult(BaseModel):
    output: Any = Field(default=None)
    error: Optional[str] = Field(default=None)
    images: Optional[Union[ImageResult, List[ImageResult]]] = Field(default=None)
    system: Optional[str] = Field(default=None)
    meta: Optional[dict] = Field(default=dict())

    def __add__(self, other: "GenericResult"):
        def comb(a, b):
            if a and b:
                return a + b
            return a or b

        if self.images is None:
            self.images = []
        elif isinstance(self.images, ImageResult):
            self.images = [self.images]

        if other.images is None:
            other.images = []
        elif isinstance(other.images, ImageResult):
            other.images = [other.images]

        return GenericResult(
            output=comb(self.output, other.output),
            error=comb(self.error, other.error),
            images=comb(self.images, other.images),
            system=comb(self.system, other.system),
        )

    def __str__(self):
        result = ""
        if self.output:
            result += f"{self.output}\n"
        if self.error:
            result += f"Error: {self.error}\n"
        if self.images:
            images = self.images if isinstance(self.images, list) else [self.images]
            images = [image for image in images if image.priority != "low"]
            for i, image in enumerate(images):
                result += f"* {i+1}th image: {image.path}, width: {image.width}, height: {image.height}, mode: {image.mode}\n"

        if self.meta:
            meta = self.meta.copy()
            meta = {
                k: v for k, v in meta.items() if v is not None and not k.startswith("_")
            }
            result += f"Meta: {meta}\n"
        return result.strip() if result else "No output or error or images."


class GenericError(Exception):
    """Generic error for tools."""

    def __init__(self, message: str):
        self.message = message


class Role(str, Enum):
    """Message role options"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


ROLE_VALUES = tuple(role.value for role in Role)
ROLE_TYPE = Literal[ROLE_VALUES]  # type: ignore


class ToolChoice(str, Enum):
    """Tool choice options"""

    NONE = "none"
    AUTO = "auto"
    REQUIRED = "required"


TOOL_CHOICE_VALUES = tuple(choice.value for choice in ToolChoice)
TOOL_CHOICE_TYPE = Literal[TOOL_CHOICE_VALUES]  # type: ignore


class AgentState(str, Enum):
    """Agent execution states"""

    IDLE = "IDLE"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    ERROR = "ERROR"


class Function(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    """Represents a tool/function call in a message"""

    id: str
    type: str = "function"
    function: Function


class Message(BaseModel):
    """Represents a chat message in the conversation"""

    role: ROLE_TYPE = Field(...)  # type: ignore
    content: Optional[str] = Field(default=None)
    tool_calls: Optional[List[ToolCall]] = Field(default=None)
    name: Optional[str] = Field(default=None)
    tool_call_id: Optional[str] = Field(default=None)
    images: Optional[Union[ImageResult, List[ImageResult]]] = Field(default=None)

    def __add__(self, other) -> List["Message"]:
        """支持 Message + list 或 Message + Message 的操作"""
        if isinstance(other, list):
            return [self] + other
        elif isinstance(other, Message):
            return [self, other]
        else:
            raise TypeError(
                f"unsupported operand type(s) for +: '{type(self).__name__}' and '{type(other).__name__}'"
            )

    def __radd__(self, other) -> List["Message"]:
        """支持 list + Message 的操作"""
        if isinstance(other, list):
            return other + [self]
        else:
            raise TypeError(
                f"unsupported operand type(s) for +: '{type(other).__name__}' and '{type(self).__name__}'"
            )

    def _read_image(self, image):
        if is_remote_url(image):
            return requests.get(image, timeout=300).content
        if os.path.exists(image):
            return open(image, "rb").read()
        return base64.b64decode(image)

    def to_dict(self) -> dict:
        """Convert message to dictionary format"""
        message = {"role": self.role}
        if self.tool_calls is not None:
            message["tool_calls"] = [tool_call.dict() for tool_call in self.tool_calls]
        if self.name is not None:
            message["name"] = self.name
        if self.tool_call_id is not None:
            message["tool_call_id"] = self.tool_call_id
        if self.images is not None:
            images = self.images if isinstance(self.images, list) else [self.images]
            images = [image for image in images if image.priority != "low"]

            content = self.content or ""
            for i, image in enumerate(images):
                content += f"\n\n* {i+1}th image: {image.path}, width: {image.width}, height: {image.height}, mode: {image.mode}"

            message["content"] = []
            message["content"].append({"type": "text", "text": content})

            for image in images:
                message["content"].append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64.b64encode(self._read_image(image.path)).decode()}"
                        },
                    }
                )

            if isinstance(self.images, ImageResult):
                # If only one image, set its priority to low if it was medium
                if self.images.priority == "medium":
                    self.images.priority = "low"
            else:
                for image in self.images:
                    if image.priority == "medium":
                        image.priority = "low"
        elif self.content is not None:
            message["content"] = self.content
        return message

    @classmethod
    def user_message(cls, content: str, images: Optional[str] = None):
        """Create a user message"""
        return cls(role=Role.USER, content=content, images=images)

    @classmethod
    def system_message(cls, content: str):
        """Create a system message"""
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def assistant_message(
        cls, content: Optional[str] = None, images: Optional[str] = None
    ):
        """Create an assistant message"""
        return cls(role=Role.ASSISTANT, content=content, images=images)

    @classmethod
    def tool_message(
        cls, content: str, name, tool_call_id: str, images: Optional[str] = None
    ):
        """Create a tool message"""
        return cls(
            role=Role.TOOL,
            content=content,
            name=name,
            tool_call_id=tool_call_id,
            images=images,
        )

    @classmethod
    def from_tool_calls(
        cls,
        tool_calls: List[Any],
        content: Union[str, List[str]] = "",
        images: Optional[str] = None,
        **kwargs,
    ):
        """Create ToolCallsMessage from raw tool calls.

        Args:
            tool_calls: Raw tool calls from LLM
            content: Optional message content
            base64_image: Optional base64 encoded image
        """
        formatted_calls = [
            {"id": call.id, "function": call.function.model_dump(), "type": "function"}
            for call in tool_calls
        ]
        return cls(
            role=Role.ASSISTANT,
            content=content,
            tool_calls=formatted_calls,
            images=images,
            **kwargs,
        )


class Memory(BaseModel):
    messages: List[Message] = Field(default_factory=list)
    max_messages: int = Field(default=100)

    def add_message(self, message: Message) -> None:
        """Add a message to memory"""
        self.messages.append(message)
        # Optional: Implement message limit
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    def add_messages(self, messages: List[Message]) -> None:
        """Add multiple messages to memory"""
        self.messages.extend(messages)
        # Optional: Implement message limit
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    def clear(self) -> None:
        """Clear all messages"""
        self.messages.clear()

    def get_recent_messages(self, n: int) -> List[Message]:
        """Get n most recent messages"""
        return self.messages[-n:]

    def to_dict_list(self) -> List[dict]:
        """Convert messages to list of dicts"""
        return [msg.to_dict() for msg in self.messages]


class ToolCollection:
    def __init__(self, *tools: GenericTool):
        """Initialize the tool collection with optional tools."""
        self.tools = list(tools)
        self.tool_maps = {tool.name: tool for tool in tools}

    def add_tool(self, tool: GenericTool):
        """Add a tool to the collection."""
        self.tools.append(tool)
        self.tool_maps[tool.name] = tool

    def get_tool(self, name: str) -> Optional[GenericTool]:
        """Get a tool by name."""
        return self.tool_maps.get(name)

    def to_params(self):
        """Convert all tools to parameters format."""
        return [tool.to_params() for tool in self.tools]

    async def execute_tool(self, toolcall: ToolCall):
        """Execute a tool by name with optional arguments."""
        name = toolcall.function.name
        tool = self.tool_maps.get(name)
        if not tool:
            return GenericResult(error=f"Tool '{name}' not found in the collection.")
        try:
            args = json.loads(toolcall.function.arguments or "{}")
            result = await tool.execute(**args)
            return (
                result
                if isinstance(result, GenericResult)
                else GenericResult(output=result)
            )
        except GenericError as e:
            return GenericResult(error=e.message)
        except Exception as e:
            return GenericResult(error=str(e))


class GenericAgent(BaseModel):
    name: str = Field(..., description="Unique name of the agent")
    description: Optional[str] = Field(None, description="Optional agent description")

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"  # Allow extra fields for flexibility in subclasses
