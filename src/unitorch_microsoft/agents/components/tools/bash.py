# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import sys
import asyncio
import subprocess
from unitorch_microsoft.agents.components import (
    GenericTool,
    GenericResult,
    GenericError,
)

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
                    if any(
                        err in decoded.lower() for err in ["err", "failed", "exception"]
                    ):
                        # If stderr contains an error, we set the triggered event
                        # and stop reading further stderr lines.
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
            },
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
