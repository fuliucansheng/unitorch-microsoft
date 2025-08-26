# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import json
import sys
import fire
import glob
import time
import uuid
import logging
import requests
import subprocess
import tempfile
import asyncio
from enum import Enum
from pathlib import Path
from collections import UserDict
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, model_validator
from collections import defaultdict
from typing import Any, DefaultDict, List, Literal, Optional, Tuple, Union, Dict


logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S %p",
    level=logging.INFO,
)

use_papyrus = os.environ.get("USE_PAPYRUS", "True") == "True"


class GenericOutputs(UserDict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getattr__(self, key):
        # 允许通过属性访问，例如 output.last_hidden_state
        try:
            return self.data[key]
        except KeyError as e:
            raise AttributeError(
                f"'GenericOutputs' object has no attribute '{key}'"
            ) from e

    def __setattr__(self, key, value):
        if key == "data":
            super().__setattr__(key, value)
        else:
            self.data[key] = value

    def __delattr__(self, key):
        if key in self.data:
            del self.data[key]
        else:
            raise AttributeError(f"'GenericOutputs' object has no attribute '{key}'")


if use_papyrus:
    from azure.identity import AzureCliCredential

    """
    pip3 install pydantic azure-identity
    az login
    """

    papyrus_endpoint3 = "https://westus2large.papyrus.binginternal.com/chat/completions"
    verify_scope = "api://5fe538a8-15d5-4a84-961e-be66cd036687/.default"

    credential = AzureCliCredential()

    def timed_cache(ttl_seconds=300):
        def decorator(func):
            cache = {}

            def wrapper(*args, **kwargs):
                key = (args, frozenset(kwargs.items()))
                now = time.time()

                if key in cache:
                    result, timestamp = cache[key]
                    if now - timestamp < ttl_seconds:
                        return result

                result = func(*args, **kwargs)
                cache[key] = (result, now)
                return result

            return wrapper

        return decorator

    @timed_cache(ttl_seconds=300)
    def get_access_token():
        access_token = credential.get_token(verify_scope).token
        return access_token

    class GPTModel:
        def ask(
            self,
            messages: list,
            model: Optional[str] = "gpt-41-2025-04-14-Eval",
            temperature: Optional[float] = 0.0,
            top_p: Optional[float] = 1.0,
            max_tokens: Optional[int] = 32768,
        ):
            headers = {
                "Authorization": "Bearer " + get_access_token(),
                "Content-Type": "application/json",
                "papyrus-model-name": model,
                "papyrus-quota-id": "",
                "papyrus-timeout-ms": "120000",
            }
            try:
                response = requests.post(
                    papyrus_endpoint3,
                    headers=headers,
                    json={
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                    },
                ).json()
                result = response["choices"][0]["message"]
                content = result.get("content", None)
                content = content if content is not None else ""
                content = content.strip()
                return GenericOutputs(content=content)
            except Exception as e:
                return GenericOutputs(
                    content="",
                    error=str(e),
                )

        def ask_tools(
            self,
            messages: list,
            tools: List[dict],
            tool_choice: Optional[str] = "auto",
            model: Optional[str] = "gpt-41-2025-04-14-Eval",
            temperature: Optional[float] = 0.0,
            top_p: Optional[float] = 1.0,
            max_tokens: Optional[int] = 32768,
        ):
            headers = {
                "Authorization": "Bearer " + get_access_token(),
                "Content-Type": "application/json",
                "papyrus-model-name": model,
                "papyrus-quota-id": "",
                "papyrus-timeout-ms": "120000",
            }
            try:
                response = requests.post(
                    papyrus_endpoint3,
                    headers=headers,
                    json={
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "tools": tools,
                        "tool_choice": tool_choice,
                    },
                ).json()
                result = response["choices"][0]["message"]
                tool_calls = result.get("tool_calls", [])
                content = result.get("content", None)
                content = content if content is not None else ""
                content = content.strip()
                return GenericOutputs(
                    content=content,
                    tool_calls=tool_calls,
                )
            except Exception as e:
                return GenericOutputs(
                    content="",
                    tool_calls=[],
                    error=str(e),
                )

else:

    CLIENT_ID = "Iv1.b507a08c87ecfe98"
    HEADERS = {
        "accept": "application/json",
        "content-type": "application/json",
        "user-agent": "GithubCopilot/1.155.0",
    }
    TOKEN_FILE = str(Path.home() / ".github_copilot_access_token")

    def cached_file(file_path: str):
        """Decorator to cache the result of a function to a file."""

        def decorator(func):
            def wrapper(*args, **kwargs):
                if os.path.exists(file_path):
                    with open(file_path, "r") as f:
                        return f.read()
                result = func(*args, **kwargs)
                if result is None:
                    return None
                with open(file_path, "w") as f:
                    f.write(result)
                return result

            return wrapper

        return decorator

    @cached_file(TOKEN_FILE)
    def get_access_token():
        """获取 GitHub access token"""
        print("🚀 GitHub Copilot Token 获取工具")
        print("=" * 40)

        # 1. 获取设备码
        print("📡 获取设备授权码...")
        response = requests.post(
            "https://github.com/login/device/code",
            json={"client_id": CLIENT_ID, "scope": "read:user"},
            headers=HEADERS,
        )
        auth_data = response.json()

        device_code = auth_data["device_code"]
        user_code = auth_data["user_code"]
        verification_uri = auth_data["verification_uri"]

        print(f"✅ 用户代码: {user_code}")
        print(f"🔗 验证网址: {verification_uri}")

        # 2. 打开浏览器
        input("授权完成后按回车继续...")

        # 3. 轮询获取令牌
        print("🔄 正在获取访问令牌...")
        for i in range(8):
            time.sleep(2)
            try:
                response = requests.post(
                    "https://github.com/login/oauth/access_token",
                    json={
                        "client_id": CLIENT_ID,
                        "device_code": device_code,
                        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    },
                    headers=HEADERS,
                )

                if response.status_code == 200:
                    result = response.json()
                    if "access_token" in result:
                        access_token = result["access_token"]
                        print(f"🎉 获取成功！")
                        print(f"🔑 ACCESS_TOKEN: {access_token}")
                        return access_token
            except:
                pass

            print(f"   尝试 {i+1}/8...")

        print("❌ 获取失败，请重试")
        return None

    def get_copilot_token():
        """获取 Copilot 专用 token"""
        url = "https://api.github.com/copilot_internal/v2/token"

        access_token = get_access_token()
        if not access_token:
            raise Exception("Failed to get GitHub access token")
        headers = {
            "Authorization": f"token {access_token}",
            "User-Agent": "GitHub-Copilot-Client/1.0",
        }

        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            return response.json()["token"]
        else:
            raise Exception(
                f"Failed to get Copilot token: {response.status_code} - {response.text}"
            )

    class GPTModel:
        def ask(
            self,
            messages: list,
            model: Optional[str] = "gpt-4.1",
            temperature: Optional[float] = 0.0,
            top_p: Optional[float] = 1.0,
            max_tokens: Optional[int] = 32768,
        ) -> GenericOutputs:
            headers = {
                "Authorization": f"Bearer {get_copilot_token()}",
                "User-Agent": "GitHub-Copilot-Client/1.0",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Editor-Version": "vscode/1.85.0",
                "Editor-Plugin-Version": "copilot/1.155.0",
            }
            try:
                response = requests.post(
                    "https://api.githubcopilot.com/chat/completions",
                    headers=headers,
                    json={
                        "model": model,
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                    },
                ).json()
                result = response["choices"][0]["message"]
                content = result.get("content", None)
                content = content if content is not None else ""
                content = content.strip()
                return GenericOutputs(content=content)
            except Exception as e:
                return GenericOutputs(
                    content="",
                    error=str(e),
                )

        def ask_tools(
            self,
            messages: list,
            tools: List[dict],
            tool_choice: Optional[str] = "auto",
            model: Optional[str] = "gpt-4.1",
            temperature: Optional[float] = 0.0,
            top_p: Optional[float] = 1.0,
            max_tokens: Optional[int] = 32768,
        ):
            headers = {
                "Authorization": f"Bearer {get_copilot_token()}",
                "User-Agent": "GitHub-Copilot-Client/1.0",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Editor-Version": "vscode/1.85.0",
                "Editor-Plugin-Version": "copilot/1.155.0",
            }
            try:
                response = requests.post(
                    "https://api.githubcopilot.com/chat/completions",
                    headers=headers,
                    json={
                        "model": model,
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "tools": tools,
                        "tool_choice": tool_choice,
                    },
                ).json()
                result = response["choices"][0]["message"]
                tool_calls = result.get("tool_calls", [])
                content = result.get("content", None)
                content = content if content is not None else ""
                content = content.strip()
                return GenericOutputs(
                    content=content,
                    tool_calls=tool_calls,
                )
            except Exception as e:
                return GenericOutputs(
                    content="",
                    tool_calls=[],
                    error=str(e),
                )


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


class GenericResult(BaseModel):
    output: Any = Field(default=None)
    error: Optional[str] = Field(default=None)
    system: Optional[str] = Field(default=None)
    meta: Optional[dict] = Field(default=dict())

    def __add__(self, other: "GenericResult"):
        def comb(a, b):
            if a and b:
                return a + b
            return a or b

        return GenericResult(
            output=comb(self.output, other.output),
            error=comb(self.error, other.error),
            system=comb(self.system, other.system),
        )

    def __str__(self):
        result = ""
        if self.output:
            result += f"{self.output}\n"
        if self.error:
            result += f"Error: {self.error}\n"
        if self.meta:
            meta = self.meta.copy()
            meta = {
                k: v for k, v in meta.items() if v is not None and not k.startswith("_")
            }
            result += f"Meta: {meta}\n"
        return result.strip() if result else "No output or error."


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

    def to_dict(self) -> dict:
        """Convert message to dictionary format"""
        message = {"role": self.role}
        if self.tool_calls is not None:
            message["tool_calls"] = [
                tool_call.model_dump() for tool_call in self.tool_calls
            ]
        if self.name is not None:
            message["name"] = self.name
        if self.tool_call_id is not None:
            message["tool_call_id"] = self.tool_call_id
        if self.content is not None:
            message["content"] = self.content
        return message

    @classmethod
    def user_message(cls, content: str):
        """Create a user message"""
        return cls(role=Role.USER, content=content)

    @classmethod
    def system_message(cls, content: str):
        """Create a system message"""
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def assistant_message(
        cls,
        content: Optional[str] = None,
    ):
        """Create an assistant message"""
        return cls(role=Role.ASSISTANT, content=content)

    @classmethod
    def tool_message(cls, content: str, name, tool_call_id: str):
        """Create a tool message"""
        return cls(
            role=Role.TOOL,
            content=content,
            name=name,
            tool_call_id=tool_call_id,
        )

    @classmethod
    def from_tool_calls(
        cls,
        tool_calls: List[Any],
        content: Union[str, List[str]] = "",
        **kwargs,
    ):
        """Create ToolCallsMessage from raw tool calls.

        Args:
            tool_calls: Raw tool calls from LLM
            content: Optional message content
        """
        formatted_calls = [
            {"id": call.id, "function": call.function.model_dump(), "type": "function"}
            for call in tool_calls
        ]
        return cls(
            role=Role.ASSISTANT,
            content=content,
            tool_calls=formatted_calls,
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


_BASH_DESCRIPTION = """
Execute a bash command in the terminal.
* Long running commands: For commands that may run indefinitely, it should be run in the background and the output should be redirected to a file, e.g. command = `python3 app.py > server.log 2>&1 &`.
* Interactive: If a bash command returns exit code `-1`, this means the process is not yet finished. The assistant must then send a second call to terminal with an empty `command` (which will retrieve any additional logs), or it can send additional text (set `command` to the text) to STDIN of the running process, or it can send command=`ctrl+c` to interrupt the process.
* Timeout: If a command execution result says "Command timed out. Sending SIGINT to the process", the assistant should retry running the command in the background.
* Pay attention to the whitespace in the command, it should be escaped properly. Don't add extra whitespace in the command like `(`, `)`, `&`, etc.
"""


class BashSession:
    _started: bool
    _process: asyncio.subprocess.Process

    command: str = "/bin/bash"
    _output_delay: float = 0.2  # seconds
    _timeout: float = 120.0  # seconds
    _sentinel: str = "<<exit>>"

    def __init__(self):
        self._started = False
        self._timed_out = False

    async def start(self):
        if self._started:
            return

        self._process = await asyncio.create_subprocess_shell(
            self.command,
            preexec_fn=os.setsid,
            shell=True,
            bufsize=0,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        self._started = True

    def stop(self):
        """Terminate the bash shell."""
        if not self._started:
            raise GenericError(
                "Bash session has not been started. Please start the session before stopping it."
            )
        if self._process.returncode is not None:
            return
        self._process.terminate()

    def status(self) -> str:
        """Check the status of the bash shell."""
        if self._process.returncode is not None:
            return "exited"
        return "running"

    async def run(self, command: str):
        """Execute a command in the bash shell."""
        if not self._started:
            raise GenericError(
                "Bash session has not been started. Please start the session before running a command."
            )
        if self._process.returncode is not None:
            raise GenericError(
                f"Bash session has already exited with returncode {self._process.returncode}. Please restart the session."
            )
        if self._timed_out:
            raise GenericError(
                f"Bash session has timed out after {self._timeout} seconds. Please restart the session."
            )

        # we know these are not None because we created the process with PIPEs
        assert self._process.stdin
        assert self._process.stdout
        assert self._process.stderr

        # send command to the process
        self._process.stdin.write(
            command.encode() + f"; echo '{self._sentinel}'\n".encode()
        )
        await self._process.stdin.drain()

        output_lines = []
        error_lines = []

        stdout_done = asyncio.Event()
        stderr_triggered = asyncio.Event()

        async def read_stdout():
            while True:
                line = await self._process.stdout.readline()
                if not line:
                    break
                decoded = line.decode()
                if self._sentinel in decoded:
                    break
                output_lines.append(decoded.rstrip("\n"))
            stdout_done.set()

        async def read_stderr():
            while True:
                line = await self._process.stderr.readline()
                if not line:
                    break
                decoded = line.decode().rstrip("\n")
                if decoded:
                    error_lines.append(decoded)
                    stderr_triggered.set()
                    break  # you can also continue collecting all stderr lines here

        try:
            async with asyncio.timeout(self._timeout):
                # 启动两个任务
                stdout_task = asyncio.create_task(read_stdout())
                stderr_task = asyncio.create_task(read_stderr())

                done, pending = await asyncio.wait(
                    [stdout_task, stderr_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # 判断：是stderr先触发还是stdout先结束
                if stderr_triggered.is_set():
                    # 错误优先处理
                    stdout_task.cancel()
                    await asyncio.gather(stdout_task, return_exceptions=True)
                else:
                    # 等待 stderr 正常结束
                    stderr_task.cancel()
                    await asyncio.gather(stderr_task, return_exceptions=True)

        except asyncio.TimeoutError:
            raise GenericError(f"Command timed out after {self._timeout} seconds")

        output = "\n".join(output_lines)
        error = "\n".join(error_lines)

        if output.endswith("\n"):
            output = output[:-1]

        if error.endswith("\n"):
            error = error[:-1]

        return GenericResult(
            output=output,
            error=error,
        )


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
    session: BashSession = None

    async def execute(self, command: str) -> str:
        if self.session is None:
            # if session is None, we create a new session
            self.session = BashSession()
            await self.session.start()
        elif self.session.status() == "exited":
            # if session has exited, we create a new session
            self.session = BashSession()
            await self.session.start()

        if command is not None:
            result = await self.session.run(command)
            result.meta["command"] = command
            return result

        raise GenericError("No command provided. Please provide a command to execute.")


class LocalFileOperator:
    """File operations implementation for local filesystem."""

    encoding: str = "utf-8"

    async def read_file(self, path: str) -> str:
        """Read content from a local file."""
        try:
            return Path(path).read_text(encoding=self.encoding)
        except Exception as e:
            raise GenericError(f"Failed to read file {path}: {str(e)}")

    async def write_file(self, path: str, content: str) -> None:
        """Write content to a local file."""
        try:
            Path(path).write_text(content, encoding=self.encoding)
        except Exception as e:
            raise GenericError(f"Failed to write to file {path}: {str(e)}")

    async def is_directory(self, path: str) -> bool:
        """Check if path points to a directory."""
        return Path(path).is_dir()

    async def exists(self, path: str) -> bool:
        """Check if path exists."""
        return Path(path).exists()

    async def run_command(
        self, cmd: str, timeout: Optional[float] = 120.0
    ) -> Tuple[int, str, str]:
        """Run a shell command locally."""
        process = await asyncio.create_subprocess_shell(
            cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
            return (
                process.returncode or 0,
                stdout.decode(),
                stderr.decode(),
            )
        except asyncio.TimeoutError as exc:
            try:
                process.kill()
            except ProcessLookupError:
                pass

            return (
                -1,
                "",
                f"Command '{cmd}' timed out after {timeout} seconds",
            )


SNIPPET_LINES: int = 4
MAX_RESPONSE_LEN: int = 16000
TRUNCATED_MESSAGE: str = (
    "<response clipped><NOTE>To save on context only part of this file has been shown to you. "
    "You should retry this tool after you have searched inside the file with `grep -n` "
    "in order to find the line numbers of what you are looking for.</NOTE>"
)

_EDITOR_DESCRIPTION = """
Custom editing tool for viewing, creating and editing files
* State is persistent across command calls and discussions with the user
* If `path` is a file, `view` displays the result of applying `cat -n`. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep
* The `create` command cannot be used if the specified `path` already exists as a file
* The `undo_edit` command will revert the last edit made to the file at `path`

Notes for using the `str_replace` command:
* The `old_str` parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!
* If the `old_str` parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in `old_str` to make it unique
* The `new_str` parameter should contain the edited lines that should replace the `old_str`
"""


def maybe_truncate(
    content: str, truncate_after: Optional[int] = MAX_RESPONSE_LEN
) -> str:
    """Truncate content and append a notice if content exceeds the specified length."""
    if not truncate_after or len(content) <= truncate_after:
        return content
    return content[:truncate_after] + TRUNCATED_MESSAGE


class EditorTool(GenericTool):
    """Add a tool to ask human for help."""

    name: str = "editor"
    description: str = _EDITOR_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "command": {
                "description": "The commands to run. Allowed options are: `view`, `create`, `str_replace`, `insert`, `undo_edit`.",
                "enum": ["view", "create", "str_replace", "insert", "undo_edit"],
                "type": "string",
            },
            "path": {
                "description": "Absolute path to file or directory.",
                "type": "string",
            },
            "file_text": {
                "description": "Required parameter of `create` command, with the content of the file to be created.",
                "type": "string",
            },
            "old_str": {
                "description": "Required parameter of `str_replace` command containing the string in `path` to replace.",
                "type": "string",
            },
            "new_str": {
                "description": "Optional parameter of `str_replace` command containing the new string (if not given, no string will be added). Required parameter of `insert` command containing the string to insert.",
                "type": "string",
            },
            "insert_line": {
                "description": "Required parameter of `insert` command. The `new_str` will be inserted AFTER the line `insert_line` of `path`.",
                "type": "integer",
            },
            "view_range": {
                "description": "Optional parameter of `view` command when `path` points to a file. If none is given, the full file is shown. If provided, the file will be shown in the indicated line number range, e.g. [11, 12] will show lines 11 and 12. Indexing at 1 to start. Setting `[start_line, -1]` shows all lines from `start_line` to the end of the file.",
                "items": {"type": "integer"},
                "type": "array",
            },
        },
        "required": ["command", "path"],
    }
    _file_history: DefaultDict[Path, List[str]] = defaultdict(list)

    async def execute(
        self,
        command: str,
        path: str,
        file_text: str | None = None,
        view_range: list[int] | None = None,
        old_str: str | None = None,
        new_str: str | None = None,
        insert_line: int | None = None,
    ) -> str:
        operator = LocalFileOperator()

        await self.validate_path(command, Path(path), operator)

        if command == "view":
            result = await self.view(path, view_range, operator)
        elif command == "create":
            if file_text is None:
                raise GenericError(
                    "Parameter `file_text` is required for command: create"
                )
            await operator.write_file(path, file_text)
            result = self._file_history[path].append(file_text)
        elif command == "str_replace":
            if old_str is None:
                raise GenericError(
                    "Parameter `old_str` is required for command: str_replace"
                )
            result = await self.str_replace(path, old_str, new_str, operator)
        elif command == "insert":
            if insert_line is None:
                raise GenericError(
                    "Parameter `insert_line` is required for command: insert"
                )
            if new_str is None:
                raise GenericError(
                    "Parameter `new_str` is required for command: insert"
                )
            result = await self.insert(path, insert_line, new_str, operator)
        elif command == "undo_edit":
            result = await self.undo_edit(path, operator)
        else:
            raise GenericError(
                f"Unrecognized command {command}. The allowed commands are: view, create, str_replace, insert, undo_edit"
            )

        result = (
            result
            if isinstance(result, GenericResult)
            else GenericResult(output=result)
        )
        result.meta["_command"] = command
        result.meta["_path"] = path
        result.meta["_content"] = await operator.read_file(path)
        return result

    async def validate_path(
        self, command: str, path: Path, operator: LocalFileOperator = None
    ) -> None:
        """Validate path and command combination based on execution environment."""
        # Check if path is absolute
        if not path.is_absolute():
            raise GenericError(
                f"The path {path} is not an absolute path. Please provide an absolute path."
            )

        # Only check if path exists for non-create commands
        if command != "create":
            if not await operator.exists(path):
                raise GenericError(
                    f"The path {path} does not exist. Please provide a valid path."
                )

            # Check if path is a directory
            is_dir = await operator.is_directory(path)
            if is_dir and command != "view":
                raise GenericError(
                    f"The path {path} is a directory and the command `{command}` cannot be used on directories. Use `view` command to list its contents."
                )

        # Check if file exists for create command
        elif command == "create":
            exists = await operator.exists(path)
            if exists:
                raise GenericError(
                    f"The path {path} already exists. Cannot create a file with the same name."
                )

    async def view(
        self,
        path: str,
        view_range: Optional[List[int]] = None,
        operator: LocalFileOperator = None,
    ):
        """Display file or directory content."""
        # Determine if path is a directory
        is_dir = await operator.is_directory(path)

        if is_dir:
            # Directory handling
            if view_range:
                raise GenericError(
                    "The `view_range` parameter is not allowed when `path` points to a directory."
                )
            return await self._view_directory(path, operator)
        else:
            # File handling
            return await self._view_file(path, operator, view_range)

    @staticmethod
    async def _view_directory(path: str, operator: LocalFileOperator):
        """Display directory contents."""
        find_cmd = f"find {path} -maxdepth 2 -not -path '*/\\.*'"

        # Execute command using the operator
        returncode, stdout, stderr = await operator.run_command(find_cmd)

        if not stderr:
            stdout = (
                f"Here's the files and directories up to 2 levels deep in {path}, "
                f"excluding hidden items:\n{stdout}\n"
            )

        return GenericResult(
            output=stdout.strip(),
            error=stderr.strip() if stderr else None,
        )

    async def _view_file(
        self,
        path: str,
        operator: LocalFileOperator,
        view_range: Optional[List[int]] = None,
    ):
        """Display file content, optionally within a specified line range."""
        # Read file content
        file_content = await operator.read_file(path)
        init_line = 1

        # Apply view range if specified
        if view_range:
            if len(view_range) != 2 or not all(isinstance(i, int) for i in view_range):
                raise GenericError(
                    f"Invalid `view_range`: {view_range}. It should be a list of two integers."
                )

            file_lines = file_content.split("\n")
            n_lines_file = len(file_lines)
            init_line, final_line = view_range

            # Validate view range
            if init_line < 1 or init_line > n_lines_file:
                raise GenericError(
                    f"Invalid `view_range`: {view_range}. Its first element `{init_line}` should be within the range of lines of the file: {[1, n_lines_file]}"
                )
            if final_line > n_lines_file:
                raise GenericError(
                    f"Invalid `view_range`: {view_range}. Its second element `{final_line}` should not exceed the number of lines in the file: {n_lines_file}"
                )
            if final_line != -1 and final_line < init_line:
                raise GenericError(
                    f"Invalid `view_range`: {view_range}. Its second element `{final_line}` should be larger or equal than its first `{init_line}`"
                )

            # Apply range
            if final_line == -1:
                file_content = "\n".join(file_lines[init_line - 1 :])
            else:
                file_content = "\n".join(file_lines[init_line - 1 : final_line])

        return GenericResult(
            output=self._make_output(file_content, str(path), init_line=init_line)
        )

    async def str_replace(
        self,
        path: str,
        old_str: str,
        new_str: Optional[str] = None,
        operator: LocalFileOperator = None,
    ):
        """Replace a unique string in a file with a new string."""
        # Read file content and expand tabs
        file_content = (await operator.read_file(path)).expandtabs()
        old_str = old_str.expandtabs()
        new_str = new_str.expandtabs() if new_str is not None else ""

        # Check if old_str is unique in the file
        occurrences = file_content.count(old_str)
        if occurrences == 0:
            raise GenericError(
                f"No occurrences of old_str `{old_str}` found in {path}. Please ensure it is present in the file."
            )
        elif occurrences > 1:
            # Find line numbers of occurrences
            file_content_lines = file_content.split("\n")
            lines = [
                idx + 1
                for idx, line in enumerate(file_content_lines)
                if old_str in line
            ]
            raise GenericError(
                f"Multiple occurrences of old_str `{old_str}` found in {path} at lines {lines}. Please ensure it is unique."
            )

        # Replace old_str with new_str
        new_file_content = file_content.replace(old_str, new_str)

        # Write the new content to the file
        await operator.write_file(path, new_file_content)

        # Save the original content to history
        self._file_history[path].append(file_content)

        # Create a snippet of the edited section
        replacement_line = file_content.split(old_str)[0].count("\n")
        start_line = max(0, replacement_line - SNIPPET_LINES)
        end_line = replacement_line + SNIPPET_LINES + new_str.count("\n")
        snippet = "\n".join(new_file_content.split("\n")[start_line : end_line + 1])

        # Prepare the success message
        success_msg = f"The file {path} has been edited. "
        success_msg += self._make_output(
            snippet, f"a snippet of {path}", start_line + 1
        )
        success_msg += "Review the changes and make sure they are as expected. Edit the file again if necessary."

        return GenericResult(
            output=success_msg,
        )

    async def insert(
        self,
        path: str,
        insert_line: int,
        new_str: str,
        operator: LocalFileOperator = None,
    ):
        """Insert text at a specific line in a file."""
        # Read and prepare content
        file_text = (await operator.read_file(path)).expandtabs()
        new_str = new_str.expandtabs()
        file_text_lines = file_text.split("\n")
        n_lines_file = len(file_text_lines)

        # Validate insert_line
        if insert_line < 0 or insert_line > n_lines_file:
            raise GenericError(
                f"Invalid `insert_line`: {insert_line}. It should be within the range of lines of the file: {[0, n_lines_file]}"
            )

        # Perform insertion
        new_str_lines = new_str.split("\n")
        new_file_text_lines = (
            file_text_lines[:insert_line]
            + new_str_lines
            + file_text_lines[insert_line:]
        )

        # Create a snippet for preview
        snippet_lines = (
            file_text_lines[max(0, insert_line - SNIPPET_LINES) : insert_line]
            + new_str_lines
            + file_text_lines[insert_line : insert_line + SNIPPET_LINES]
        )

        # Join lines and write to file
        new_file_text = "\n".join(new_file_text_lines)
        snippet = "\n".join(snippet_lines)

        await operator.write_file(path, new_file_text)
        self._file_history[path].append(file_text)

        # Prepare success message
        success_msg = f"The file {path} has been edited. "
        success_msg += self._make_output(
            snippet,
            "a snippet of the edited file",
            max(1, insert_line - SNIPPET_LINES + 1),
        )
        success_msg += "Review the changes and make sure they are as expected (correct indentation, no duplicate lines, etc). Edit the file again if necessary."

        return GenericResult(
            output=success_msg,
        )

    async def undo_edit(self, path: str, operator: LocalFileOperator = None):
        """Revert the last edit made to a file."""
        if not self._file_history[path]:
            raise GenericError(
                f"No edit history found for {path}. Please make sure you have edited the file before."
            )

        old_text = self._file_history[path].pop()
        await operator.write_file(path, old_text)

        return GenericResult(
            output=f"Last edit to {path} undone successfully. {self._make_output(old_text, str(path))}"
        )

    def _make_output(
        self,
        file_content: str,
        file_descriptor: str,
        init_line: int = 1,
        expand_tabs: bool = True,
    ) -> str:
        """Format file content for display with line numbers."""
        file_content = maybe_truncate(file_content)
        if expand_tabs:
            file_content = file_content.expandtabs()

        # Add line numbers to each line
        file_content = "\n".join(
            [
                f"{i + init_line:6}\t{line}"
                for i, line in enumerate(file_content.split("\n"))
            ]
        )

        return (
            f"Here's the result of running `cat -n` on {file_descriptor}:\n"
            + file_content
            + "\n"
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


_PLANNER_DESCRIPTION = """
A planning tool that allows the agent to create and manage plans for solving complex/designer tasks.
The tool provides functionality for creating plans, updating plan steps, and tracking progress.

Notes:
- Creating structured plans for task completion
- Selecting appropriate tools and approaches for each step
- Executing steps methodically while monitoring progress
- Adapting plans when encountering unexpected challenges
- Providing regular updates on task status
- Use `todo.md` to track the progress of the plan.

Key capabilities include:
- `create`: Create a new plan with a unique ID, title, and steps.
- `update`: Update an existing plan's title or steps.
- `list`: List all available plans with their statuses.
- `get`: Retrieve details of a specific plan by ID.
- `set_active`: Set a plan as the active plan for further operations.
- `mark_step`: Mark a specific step in the active plan with a status and optional notes.
- `delete`: Delete a plan by ID.
"""


class PlannerTool(GenericTool):
    """Add a tool to ask human for help."""

    name: str = "planner"
    description: str = _PLANNER_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "command": {
                "description": "The command to execute. Available commands: create, update, list, get, set_active, mark_step, delete.",
                "enum": [
                    "create",
                    "update",
                    "list",
                    "get",
                    "set_active",
                    "mark_step",
                    "delete",
                ],
                "type": "string",
            },
            "plan_id": {
                "description": "Unique identifier for the plan. Required for update, set_active, and delete commands. Optional for get and mark_step (uses active plan if not specified).",
                "type": "string",
            },
            "title": {
                "description": "Title for the plan. Required for create command, optional for update command.",
                "type": "string",
            },
            "steps": {
                "description": "List of plan steps. Required for create command, optional for update command.",
                "type": "array",
                "items": {"type": "string"},
            },
            "step_index": {
                "description": "Index of the step to update (0-based). Required for mark_step command.",
                "type": "integer",
            },
            "step_status": {
                "description": "Status to set for a step. Used with mark_step command.",
                "enum": ["not_started", "in_progress", "completed", "blocked"],
                "type": "string",
            },
            "step_notes": {
                "description": "Additional notes for a step. Optional for mark_step command.",
                "type": "string",
            },
        },
        "required": ["command"],
        "dependencies": {
            "create": {
                "required": ["title", "steps"],
            },
            "update": {
                "required": ["plan_id"],
                "oneOf": [
                    {"required": ["title"]},
                    {"required": ["steps"]},
                ],
            },
            "set_active": {
                "required": ["plan_id"],
            },
            "mark_step": {
                "required": ["step_index"],
            },
        },
    }

    plans: dict = {}  # Dictionary to store plans by plan_id
    _current_plan_id: Optional[str] = None  # Track the current active plan

    async def execute(
        self,
        command: str,
        plan_id: Optional[str] = None,
        title: Optional[str] = None,
        steps: Optional[List[str]] = None,
        step_index: Optional[int] = None,
        step_status: Optional[
            Literal["not_started", "in_progress", "completed", "blocked"]
        ] = None,
        step_notes: Optional[str] = None,
    ) -> str:
        if command == "create":
            return self._create_plan(title, steps)
        elif command == "update":
            return self._update_plan(plan_id, title, steps)
        elif command == "list":
            return self._list_plans()
        elif command == "get":
            return self._get_plan(plan_id)
        elif command == "set_active":
            return self._set_active_plan(plan_id)
        elif command == "mark_step":
            return self._mark_step(plan_id, step_index, step_status, step_notes)
        elif command == "delete":
            return self._delete_plan(plan_id)
        else:
            raise GenericError(
                f"Unrecognized command: {command}. Allowed commands are: create, update, list, get, set_active, mark_step, delete"
            )

    def _create_plan(self, title: Optional[str], steps: Optional[List[str]]):
        """Create a new plan with the given ID, title, and steps."""

        while True:
            plan_id = uuid.uuid4().hex[:8]  # Generate a random 8-character plan ID
            if plan_id not in self.plans:
                break

        if not title:
            raise GenericError("Parameter `title` is required for command: create")

        if (
            not steps
            or not isinstance(steps, list)
            or not all(isinstance(step, str) for step in steps)
        ):
            raise GenericError(
                "Parameter `steps` is required and must be a non-empty list of strings for command: create"
            )

        # Create a new plan with initialized step statuses
        plan = {
            "plan_id": plan_id,
            "title": title,
            "steps": steps,
            "step_statuses": ["not_started"] * len(steps),
            "step_notes": [""] * len(steps),
        }

        self.plans[plan_id] = plan
        self._current_plan_id = plan_id  # Set as active plan

        return GenericResult(
            output=f"Plan created successfully with ID: {plan_id}\n\n{self._format_plan(plan)}"
        )

    def _update_plan(
        self, plan_id: Optional[str], title: Optional[str], steps: Optional[List[str]]
    ):
        """Update an existing plan with new title or steps."""
        if not plan_id:
            raise GenericError("Parameter `plan_id` is required for command: update")

        if plan_id not in self.plans:
            raise GenericError(
                f"No plan found with ID: {plan_id}. Use 'create' to create a new plan."
            )

        plan = self.plans[plan_id]

        if title:
            plan["title"] = title

        if steps:
            if not isinstance(steps, list) or not all(
                isinstance(step, str) for step in steps
            ):
                raise GenericError(
                    "Parameter `steps` must be a list of strings for command: update"
                )

            # Preserve existing step statuses for unchanged steps
            old_steps = plan["steps"]
            old_statuses = plan["step_statuses"]
            old_notes = plan["step_notes"]

            # Create new step statuses and notes
            new_statuses = []
            new_notes = []

            for i, step in enumerate(steps):
                # If the step exists at the same position in old steps, preserve status and notes
                if i < len(old_steps) and step == old_steps[i]:
                    new_statuses.append(old_statuses[i])
                    new_notes.append(old_notes[i])
                else:
                    new_statuses.append("not_started")
                    new_notes.append("")

            plan["steps"] = steps
            plan["step_statuses"] = new_statuses
            plan["step_notes"] = new_notes

        return GenericResult(
            output=f"Plan updated successfully: {plan_id}\n\n{self._format_plan(plan)}"
        )

    def _list_plans(self):
        """List all available plans."""
        if not self.plans:
            return GenericResult(
                output="No plans available. Create a plan with the 'create' command."
            )

        output = "Available plans:\n"
        for plan_id, plan in self.plans.items():
            current_marker = " (active)" if plan_id == self._current_plan_id else ""
            completed = sum(
                1 for status in plan["step_statuses"] if status == "completed"
            )
            total = len(plan["steps"])
            progress = f"{completed}/{total} steps completed"
            output += f"• {plan_id}{current_marker}: {plan['title']} - {progress}\n"

        return GenericResult(output=output)

    def _get_plan(self, plan_id: Optional[str]):
        """Get details of a specific plan."""
        if not plan_id:
            # If no plan_id is provided, use the current active plan
            if not self._current_plan_id:
                raise GenericError(
                    "No active plan. Please specify a plan_id or set an active plan."
                )
            plan_id = self._current_plan_id

        if plan_id not in self.plans:
            raise GenericError(
                f"No plan found with ID: {plan_id}. Use 'list' to see available plans."
            )

        plan = self.plans[plan_id]
        return GenericResult(
            output=self._format_plan(plan),
        )

    def _set_active_plan(self, plan_id: Optional[str]):
        """Set a plan as the active plan."""
        if not plan_id:
            raise GenericError(
                "Parameter `plan_id` is required for command: set_active"
            )

        if plan_id not in self.plans:
            raise GenericError(
                f"No plan found with ID: {plan_id}. Use 'list' to see available plans."
            )

        self._current_plan_id = plan_id
        return GenericResult(
            output=f"Plan '{plan_id}' is now the active plan.\n\n{self._format_plan(self.plans[plan_id])}"
        )

    def _mark_step(
        self,
        plan_id: Optional[str],
        step_index: Optional[int],
        step_status: Optional[str],
        step_notes: Optional[str],
    ):
        """Mark a step with a specific status and optional notes."""
        if not plan_id:
            # If no plan_id is provided, use the current active plan
            if not self._current_plan_id:
                raise GenericError(
                    "No active plan. Please specify a plan_id or set an active plan."
                )
            plan_id = self._current_plan_id

        if plan_id not in self.plans:
            raise GenericError(
                f"No plan found with ID: {plan_id}. Use 'list' to see available plans."
            )

        if step_index is None:
            raise GenericError(
                "Parameter `step_index` is required for command: mark_step"
            )

        plan = self.plans[plan_id]

        if step_index < 0 or step_index >= len(plan["steps"]):
            raise GenericError(
                f"Invalid `step_index`: {step_index}. Valid indices range from 0 to {len(plan['steps']) - 1}."
            )

        if step_status and step_status not in [
            "not_started",
            "in_progress",
            "completed",
            "blocked",
        ]:
            raise GenericError(
                f"Invalid `step_status`: {step_status}. Valid statuses are: not_started, in_progress, completed, blocked"
            )

        if step_status:
            plan["step_statuses"][step_index] = step_status

        if step_notes:
            plan["step_notes"][step_index] = step_notes

        return GenericResult(
            output=f"Step {step_index} in plan '{plan_id}' marked as '{step_status}' with notes: '{step_notes}'.\n\n{self._format_plan(plan)}",
        )

    def _delete_plan(self, plan_id: Optional[str]):
        """Delete a plan."""
        if not plan_id:
            raise GenericError("Parameter `plan_id` is required for command: delete")

        if plan_id not in self.plans:
            raise GenericError(
                f"No plan found with ID: {plan_id}. Use 'list' to see available plans."
            )

        del self.plans[plan_id]

        # If the deleted plan was the active plan, clear the active plan
        if self._current_plan_id == plan_id:
            self._current_plan_id = None

        return GenericResult(output=f"Plan '{plan_id}' has been deleted successfully.")

    def _format_plan(self, plan: Dict) -> str:
        """Format a plan for display."""
        output = f"Plan: {plan['title']} (ID: {plan['plan_id']})\n"
        output += "=" * len(output) + "\n\n"

        # Calculate progress statistics
        total_steps = len(plan["steps"])
        completed = sum(1 for status in plan["step_statuses"] if status == "completed")
        in_progress = sum(
            1 for status in plan["step_statuses"] if status == "in_progress"
        )
        blocked = sum(1 for status in plan["step_statuses"] if status == "blocked")
        not_started = sum(
            1 for status in plan["step_statuses"] if status == "not_started"
        )

        output += f"Progress: {completed}/{total_steps} steps completed "
        if total_steps > 0:
            percentage = (completed / total_steps) * 100
            output += f"({percentage:.1f}%)\n"
        else:
            output += "(0%)\n"

        output += f"Status: {completed} completed, {in_progress} in progress, {blocked} blocked, {not_started} not started\n\n"
        output += "Steps:\n"

        # Add each step with its status and notes
        for i, (step, status, notes) in enumerate(
            zip(plan["steps"], plan["step_statuses"], plan["step_notes"])
        ):
            status_symbol = {
                "not_started": "[ ]",
                "in_progress": "[→]",
                "completed": "[✓]",
                "blocked": "[!]",
            }.get(status, "[ ]")

            output += f"{i}. {status_symbol} {step}\n"
            if notes:
                output += f"   Notes: {notes}\n"

        return output


_NOTIFYHUMAN_DESCRIPTION = """
Use this tool to send a message to user without requiring a response. Use for acknowledging receipt of messages, providing progress updates, reporting task completion, or explaining changes in approach.
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

    async def execute(self, status: str):
        return GenericResult(
            output=f"The interaction has been completed with status: {status}",
        )


SYSTEM_PROMPT = """
You are Manus, a highly capable AI assistant designed to handle any task presented by the user. You have access to a variety of tools that you can invoke to efficiently complete complex objectives.

You excel at the following tasks:
1. Information gathering, fact-checking, and documentation
2. Data processing, analysis, and visualization
3. Writing multi-chapter articles and in-depth research reports
4. Using programming to solve various problems beyond development
5. Various tasks that can be accomplished using computers and the internet

You are operating in an agent loop, iteratively completing tasks through these steps:
1. Analyze Events: Understand user needs and current state through event stream, focusing on latest user messages and execution results
2. Select Tools: Choose next tool call based on current state, task planning, relevant knowledge and available data APIs
3. Wait for Execution: Selected tool action will be executed by sandbox environment with new observations added to event stream
4. Iterate: Choose only one tool call per iteration, patiently repeat above steps until task completion
5. Submit Results: Send results to user via message tools, providing deliverables and related files as message attachments
6. Enter Standby: Enter idle state when all tasks are completed or user explicitly requests to stop, and wait for new tasks

You can use the following tools to complete tasks:
1. `ask_human`: Request additional input.
2. `bash`: Execute shell commands. 
3. `python`: Run Python scripts or calculations.
4. `editor`: Create or modify text files
5. `planner`: Generate and update task plans.
6. `notify_human`: Notify human some messages or results without waiting for human response.
7. `terminate`: Enter idle mode once all tasks are completed or user requests stop.

During task execution, you must follow these rules:

<info_rules>
- Information priority: authoritative data from datasource API > model's internal knowledge
- Conduct searches step by step: search multiple attributes of single entity separately, process multiple entities one by one
</info_rules>

<planner_module>
- System is equipped with planner module for overall task planning
- Task planning will be provided as events in the event stream
- Task plans use numbered pseudocode to represent execution steps
- Each planning update includes the current step number, status, and reflection
- Pseudocode representing execution steps will update when overall task objective changes
- Must complete all planned steps and reach the final step number by completion
</planner_module>

<file_rules>
- Use file tools for reading, writing, appending, and editing to avoid string escape issues in shell commands
- Actively save intermediate results and store different types of reference information in separate files
- When merging text files, must use append mode of file writing tool to concatenate content to target file
- Strictly follow requirements in <writing_rules>, and avoid using list formats in any files
</file_rules>

<shell_rules>
- Avoid commands requiring confirmation; actively use -y or -f flags for automatic confirmation
- Avoid commands with excessive output; save to files when necessary
- Chain multiple commands with && operator to minimize interruptions
- Use pipe operator to pass command outputs, simplifying operations
- Use non-interactive `bc` for simple calculations, Python for complex math; never calculate mentally
- Use `uptime` command when users explicitly request sandbox status check or wake-up
</shell_rules>

<coding_rules>
- Must save code to files before execution; direct code input to interpreter commands is forbidden
- Write Python code for complex mathematical calculations and analysis
- Use search tools to find solutions when encountering unfamiliar problems
- For index.html referencing local resources, package everything into a zip file and provide it as a message attachment
</coding_rules>

<writing_rules>
- Write content in continuous paragraphs using varied sentence lengths for engaging prose; avoid list formatting
- Use prose and paragraphs by default; only employ lists when explicitly requested by users
- All writing must be highly detailed with a minimum length of several thousand words, unless user explicitly specifies length or format requirements
- When writing based on references, actively cite original text with sources and provide a reference list with URLs at the end
- For lengthy documents, first save each section as separate draft files, then append them sequentially to create the final document
- During final compilation, no content should be reduced or summarized; the final length must exceed the sum of all individual draft files
</writing_rules>

<error_handling>
- Tool execution failures are provided as events in the event stream
- When errors occur, first verify tool names and arguments
- Attempt to fix issues based on error messages; if unsuccessful, try alternative methods
- When multiple approaches fail, report failure reasons to user and request assistance
</error_handling>

<tool_use_rules>
- Must respond with a tool use (function calling); plain text responses are forbidden
- Do not mention any specific tool names to users in messages
- Carefully verify available tools; do not fabricate non-existent tools
- Events may originate from other system modules; only use explicitly provided tools
</tool_use_rules>
"""

DOING_PROMPT = """
Your workspace directory is `{workspace}`. Use it to store any files you create or edit during the task.
"""

CODING_PROMPT = """
Your workspace directory is `{workspace}`. 
The temporary directory is `{workspace}/.do/cache`. Use it to store any files you create or edit during the task.
The package index file is `{workspace}/.do/INDEX.md`. Please check it if you need any information about the files in the workspace. 
The actual package folder is in `{workspace}/src/`. Please import it from there if you need to use it in your code.
"""

ACTION_PROMPT = """
Based on current state, what's your next action?
Choose the most efficient path forward:
1. Is the plan sufficient, or does it need refinement?
2. Can you execute the next step immediately?
3. If you need any help, please use `ask_human` to get the answer from human.
4. Is the task complete? If so, use `terminate` right away.

Be concise in your reasoning, generate a detailed plan, check the current step and then select the appropriate tool or combination of tools or action as next step.
"""


class ManusAgent(GenericAgent):
    """Do Agent for processing data and performing basic analysis."""

    name: str = "Manus"
    description: str = (
        "A versatile agent that can solve various tasks using multiple tools, including Python, Bash, and Editor Tools."
    )

    available_tools: ToolCollection = ToolCollection(
        AskHumanTool(),
        BashTool(),
        PythonTool(),
        EditorTool(),
        PlannerTool(),
        NotifyHumanTool(),
        TerminateTool(),
    )
    content: str = Field(default="")
    tool_calls: List[ToolCall] = Field(default_factory=list)
    memory: Memory = Field(default_factory=Memory)
    state: AgentState = AgentState.IDLE
    gpt: Any = GPTModel()
    current_step: int = 1
    max_steps: int = 80

    def __init__(self, system_prompt, action_prompt):
        super().__init__()
        self._system_prompt = system_prompt
        self._action_prompt = action_prompt
        self._action_message = Message.user_message(content=self._action_prompt)

    async def execute(self, instruction: str, **kwargs):
        self.state = AgentState.RUNNING
        self.current_step = 1
        system_message = Message.system_message(content=self._system_prompt)
        user_content = f"### Task\n\n- **Instruction**:\n{instruction}\n\n ### Inputs\n\n ```json\n{json.dumps(kwargs, indent=2, ensure_ascii=False)}\n```"
        user_message = Message.user_message(content=user_content)

        self.memory = Memory(
            messages=[
                system_message,
                user_message,
            ]
        )
        self.content = ""
        self.tool_calls = []

        while self.current_step <= self.max_steps and self.state != AgentState.IDLE:
            resp = self.gpt.ask_tools(
                messages=[self.memory.to_dict_list()] + [self._action_message.to_dict()],
                tools=self.available_tools.to_params(),
                tool_choice=ToolChoice.AUTO,
            )
            self.tool_calls = [ToolCall(**tc) for tc in resp.tool_calls]
            self.content = resp.content

            logging.info(f"🔷 {'━'*10} Step {self.current_step} {'━'*10} 🔷")
            logging.info(f"🛠️ Tool Calls: {self.tool_calls}")
            logging.info(
                f"📝 Content: {self.content if self.content else '🚫 No content'}"
            )

            assistant_msg = (
                Message.from_tool_calls(
                    content=self.content, tool_calls=self.tool_calls
                )
                if self.tool_calls
                else Message.assistant_message(self.content)
            )
            self.memory.add_message(assistant_msg)

            for tool_call in self.tool_calls:
                if tool_call.function.name == "terminate":
                    logging.info("🧯 Terminating the agent as requested. Goodbye! 👋")
                    self.state = AgentState.IDLE
                    break

                result = await self.available_tools.execute_tool(tool_call)
                logging.info(
                    f"✅ 🎯 Tool 『{tool_call.function.name}』 completed its mission! 📦 {result}"
                )
                tool_msg = Message.tool_message(
                    content=str(result),
                    tool_call_id=tool_call.id,
                    name=tool_call.function.name,
                )
                self.memory.add_message(tool_msg)

            self.current_step += 1


class CLI:
    def __init__(self):
        package_folder = os.path.abspath(os.path.dirname(__file__))
        if not os.path.exists(f"{package_folder}/.do/cache"):
            os.makedirs(f"{package_folder}/.do/cache")
        self.code_manus = ManusAgent(
            system_prompt=SYSTEM_PROMPT
            + CODING_PROMPT.format(workspace=package_folder),
            action_prompt=ACTION_PROMPT,
        )
        self.do_manus = ManusAgent(
            system_prompt=SYSTEM_PROMPT
            + DOING_PROMPT.format(workspace=os.path.abspath(os.getcwd())),
            action_prompt=ACTION_PROMPT,
        )

    def __call__(self, instruction: str, **kwargs):
        asyncio.run(self.code_manus.execute(instruction, **kwargs))

    def do(self, instruction: str, **kwargs):
        asyncio.run(self.do_manus.execute(instruction, **kwargs))

    def build(self, **kwargs):
        asyncio.run(
            self.code_manus.execute(
                instruction="""
Build a high-level, human-readable reference index for this Python repository. Follow the instructions below:

Scope: Scan all .py and .ini files of provided folder in the project. if the folder is not provided, use the current working directory.

For each file:
* File Path: Include the relative file path as a section heading.
* Classes and Functions:
    - List all classes and functions.
    - Include full signatures (e.g., def func(arg1: int) -> str:).
    - For each item, provide a concise description summarizing its purpose.
* Dependencies:
    - List all internal dependencies (i.e., imports from within the same codebase).
    - List all external dependencies (i.e., third-party or standard library modules).

Output Format:
* Use Markdown.
* Update the result to `.do/INDEX.md` in workspace folder. Please create the file if it does not exist.
* Use clear formatting (e.g., ### for file names, bullet points for classes/functions/dependencies).

Goal:
Provide a structured and easy-to-navigate reference that helps developers quickly understand the layout, functionality, and dependencies of the codebase.
""",
                **kwargs,
            )
        )


if __name__ == "__main__":
    fire.Fire(CLI)
