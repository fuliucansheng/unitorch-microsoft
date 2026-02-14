# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import json
import sys
import uuid
import copy
import fire
import glob
import urllib.parse
import logging
import subprocess
import tempfile
import asyncio
import gradio as gr
import importlib_resources
from PIL import Image
from pydantic import BaseModel, Field
from typing import Any, List, Literal, Optional, Union
from unitorch.utils import read_file
from unitorch.cli.webuis import (
    create_element,
    create_accordion,
    create_row,
    create_column,
    create_tab,
    create_tabs,
    create_flex_layout,
    create_group,
    create_tab,
    create_tabs,
    create_blocks,
)
from unitorch_microsoft.agents.components.picasso import (
    PicassoImageTool,
    PicassoInternalTool,
    PicassoHtmlTool,
)
from unitorch_microsoft.agents.utils.github_copilot import GPTModel
from unitorch_microsoft.agents.utils.utils_ui import (
    format_tool_for_logs,
    format_tool_for_preview,
)
from unitorch_microsoft.agents.components import (
    GenericAgent,
    GenericResult,
    GenericTool,
    ToolCall,
    ToolChoice,
    ToolCollection,
    AgentState,
    Memory,
    Message,
)
from unitorch_microsoft.agents.components.tools import (
    PythonTool,
    BashTool,
    AskHumanTool,
    NotifyHumanTool,
    GPT4FormatTool,
    EditorTool,
    PlannerTool,
    BrowserUseTool,
    WebSearchTool,
    TerminateTool,
    CheckImageTool,
)

SYSTEM_PROMPT = """
You are a Designer AI Assistant, built to solve a wide range of visual design tasks presented by the user. You have access to a variety of tools that you can invoke to efficiently complete complex objectives. 

**Important**: 
1. You should follow <picasso_rules> & <designer_rules> strictly to complete the design tasks.

You are operating in an agent loop, iteratively completing tasks through these steps:
1. Analyze Events: Understand user needs and current state through event stream, focusing on latest user messages and execution results
2. Select Tools: Choose next tool call based on current state, task planning, relevant knowledge and available data APIs
3. Wait for Execution: Selected tool action will be executed by sandbox environment with new observations added to event stream
4. Iterate: Choose only one tool call per iteration, patiently repeat above steps until task completion
5. Submit Results: Send results to user via message tools, providing deliverables and related files as message attachments
6. Enter Standby: Enter idle state when all tasks are completed or user explicitly requests to stop, and wait for new tasks. use `terminate` to enter idle state if current task is completed.

You can use the following tools to complete tasks:
1. `ask_human`: Request additional input or suggest temporary manual browser control to the user. (message tools)
2. `bash`: Execute shell commands. 
3. `python`: Run Python scripts or calculations
4. `editor`: Create or modify text files
5. `picasso_internal_tool`: Process & generate any image with provided prompt & images.
6. `picasso_html_tool`: Render HTML content to an image and the feedback will be provided about the result.
7. `notify_human`: Notify user some messages/progress or results without waiting for user's response. (message tools)
8. `check_image_tool`: Analysis the image with the provided prompt or mark the image to the priority for view.
10. `terminate`: Enter idle mode once all tasks are completed or user requests stop

During task execution, you must follow these rules:

<info_rules>
- Information priority: authoritative data from datasource API > model's internal knowledge
- Prefer dedicated search tools over browser access to search engine result pages
- Conduct searches step by step: search multiple attributes of single entity separately, process multiple entities one by one
</info_rules>

<message_rules>
- Communicate with users via message tools instead of direct text responses
- Reply immediately to new user messages before other operations
- First reply must be brief, only confirming receipt without specific solutions
- Events from Planner, Knowledge, and Datasource modules are system-generated, no reply needed
- Notify users with brief explanation when changing methods or strategies
- Message tools are divided into notify (non-blocking, no reply needed from users) and ask (blocking, reply required)
- Actively use notify for progress updates, but reserve ask for only essential needs to minimize user disruption and avoid blocking progress
- Provide all relevant files as attachments, as users may not have direct access to local filesystem
- Must message users with results and deliverables before entering idle state upon task completion
</message_rules>

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

<picasso_rules>
- The overall design result must convey a premium and tech-savvy feel.
- Use `bash` to download images from URLs before processing; do not use URLs directly in image related tools.
- The image size could be found in the event stream after the image path which is always the absolute path.
- Use `picasso_internal_tool` to generate/edit/process any images instead of any other tools.
- Use `picasso_html_tool` to render HTML content to an image with Tailwind CSS, including text, images, and styles. Design Requirements as follows:
    * Set `auto_refine_steps` to at least 20 for high quality result. 
    * Don't check & refine the design from `picasso_html_tool` if the feedback is good.
</picasso_rules>

<designer_rules>
- Use `bash` or `python` to get the padding pixel numbers. Print results to logs. Never calculate mentally.
- Use `check_image_tool` to analyze the input & generated image based on the provided prompt first.
- Ignore minor issues such as typography refinements, small spacing or padding changes, subtle adjustments to color, opacity, or shadows, and slight alignment tweaks that do not materially impact the overall visual balance.
- If a product image is provided, check the main product image size, ratio, color, etc, and whether it needs to be processed (padding, fitting, editing, etc).
- If a background image is provided, check the background image size, ratio, color, etc, and whether it needs to be processed (padding, fitting, editing, etc). Resize it to fit the designed canvas size (could be done in `picasso_html_tool`) if necessary.
- If a logo image is provided, check the logo image size, ratio, color, etc, and whether it needs to be removed the background.
    * Removing the background of the logo image is preferable if the background is not pure white or transparent.
    * Don't change the logo image in any way except removing the background.
- Don't crop the image (except plain background), use `padding` or `fitting` to adjust the image to the expected size. 
- Don't edit the image in-place, always create a new file after processing the image.
- Don't design the image in a resolution more than 2048x2048. Resize the image (keeping the ratio) and save it to a new file when the image is larger than this resolution.
- Use `python` to resize the result to the expected size after finalize the design if user specifies it.
</designer_rules>

<error_handling>
- Tool execution failures are provided as events in the event stream
- When errors occur, first verify tool names and arguments
- Attempt to fix issues based on error messages; if unsuccessful, try alternative methods
- When multiple approaches fail, report failure reasons to user and request assistance
</error_handling>

<tool_use_rules>
- Do not mention any specific tool names to users in messages
- Carefully verify available tools; do not fabricate non-existent tools
- Events may originate from other system modules; only use explicitly provided tools
- Output the reasoning process and tool selection to the event stream
- Use one tool call per step; do not chain multiple tool calls in a single step
</tool_use_rules>

Your workspace directory is `{workspace}`. Use it to store any files you create or edit during the task.
"""

ACTION_PROMPT = """
Based on current state, what's your next action?
Choose the most efficient path forward:
1. Are the improvments from the Review resolved? If not, select the proper tool to resolve the feedback step by step.
2. What's your next step? Can you execute the next step immediately?
3. If you need any help, please use `ask_human` to get the answer from human.
4. Is the task complete? If so, use `terminate` right away.

Don't do duplicated work, and don't repeat the same step again.
Be concise in your reasoning, generate a detailed plan, check the current step and then select the appropriate tool or combination of tools or action as next step.
"""

PREVIEW_PROMPT = """
You are an intelligent summarizer. Your task is to analyze the current messages and generate a brief, concise and informative task summary for the user.
The summary should:
* Clearly explain the current progress and state based on the messages.
* Preserve all important numbers, files, and any information of this task as the final result for user's task.
* Format the summary in Markdown for better readability but not in a code block.
* For any files mentioned in the summary, include them using appropriate Markdown formatting:
    - Use absolute paths for all the files.
    - If the file is an image, embed it as: ![image]({0})
    - If the file is a video, embed it as: <video src="{0}" controls></video>
    - For other file types, format the content using a Markdown code block (e.g., ```txt). Provide a link to download the file if it's too large to display inline. the url to download the file is {0}.
"""

PREVIEW_PROMPT = PREVIEW_PROMPT.format("/gradio_api/file=<absolute-path-to-file>")


class DesignerAgent(GenericAgent):
    """Do Agent for processing data and performing basic analysis."""

    name: str = "Designer"
    description: str = (
        "A versatile agent that can solve various tasks using multiple tools, including Python, Bash, Editor, and Browser Tools."
    )

    available_tools: ToolCollection = ToolCollection(
        AskHumanTool(mode="ui"),
        BashTool(),
        PythonTool(),
        # BrowserUseTool(),
        EditorTool(),
        # PlannerTool(),
        # WebSearchTool(),
        PicassoInternalTool(),
        PicassoHtmlTool(),
        NotifyHumanTool(),
        CheckImageTool(),
        TerminateTool(),
    )
    content: str = Field(default="")
    tool_calls: List[ToolCall] = Field(default_factory=list)
    memory: Memory = Field(default_factory=Memory)
    state: AgentState = AgentState.IDLE
    gpt: Any = GPTModel()
    current_step: int = 1
    max_steps: int = 500

    def __init__(self, system_prompt, action_prompt, preview_prompt):
        super().__init__()
        self._system_prompt = system_prompt
        self._action_prompt = action_prompt
        self._preview_prompt = preview_prompt
        self._action_message = Message.user_message(content=self._action_prompt)
        self._question = ""
        self._answer = ""
        self._stop = False
        self._logs = []
        self._previews = []
        self._name = ""
        self._user_inputs = {}
        self._saved_records = {}

    def set_answer(self, answer):
        self._answer = answer

    def get_question(self):
        return self._question

    def stop(self):
        self._stop = True

    def get_step_state(self, step):
        """Get the current step."""
        step = max(1, step)
        step = min(step, len(self._previews))
        if len(self._previews) == 0:
            return ""
        return "\n\n".join(self._previews[step - 1])

    def records(self):
        names = list(self._saved_records.keys())
        return [[n] for n in names]

    def save(self):
        if self.state != AgentState.IDLE and self._name != "":
            logging.warning(
                "Cannot save the agent state while it is running. Please stop the agent first."
            )
            return self.records()
        self._saved_records[self._name] = {
            "user_inputs": copy.deepcopy(self._user_inputs),
            "current_step": self.current_step,
            "logs": copy.deepcopy(self._logs),
            "previews": copy.deepcopy(self._previews),
        }
        return self.records()

    def delete(self):
        if self._name in self._saved_records:
            del self._saved_records[self._name]
        else:
            logging.warning(f"Record {self._name} not found.")
        return self.records()

    def load(self, name):
        record = self._saved_records.get(name, None)
        self._name = name
        self._user_inputs = record.get("user_inputs", {})
        self.current_step = record.get("current_step", 1)
        self._logs = record.get("logs", [])
        self._previews = record.get("previews", [[]])
        if len(self._previews) == 0:
            self._previews = [[]]
        return (
            self._user_inputs,
            "\n\n".join(self._logs),
            "\n\n".join(self._previews[self.current_step - 1]),
            self.current_step,
        )

    async def execute(self, instruction: str, **kwargs):
        while self.state != AgentState.IDLE:
            logging.info(
                f"Current Agent state: {self.state}, Need to wait for a few seconds before start processing."
            )
            await asyncio.sleep(30)

        if not os.path.exists("/tmp/unitorch_microsoft/agents/designer"):
            os.makedirs("/tmp/unitorch_microsoft/agents/designer")

        workspace = tempfile.mkdtemp(dir="/tmp/unitorch_microsoft/agents/designer")
        logging.info(f"Workspace created at: {workspace}")

        self._name = os.path.basename(workspace)
        self._user_inputs = {
            "text": instruction,
            **kwargs,
        }

        self.state = AgentState.RUNNING
        self.current_step = 1
        system_message = Message.system_message(
            content=self._system_prompt.format(workspace=workspace)
        )
        preview_message = Message.system_message(content=self._preview_prompt)

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

        self._logs = [
            f"## Starting Task\n",
            f"- **Instruction**: `{instruction}`",
            f"- **Task Parameters**:\n```json\n{json.dumps(kwargs, indent=2, ensure_ascii=False)}\n```",
        ]
        self._previews = []

        yield "\n\n".join(self._logs), self.current_step

        while self.current_step <= self.max_steps and self.state != AgentState.IDLE:
            num_messages = len(self.memory.messages)
            resp = self.gpt.ask_tools(
                messages=self.memory.to_dict_list() + [self._action_message.to_dict()],
                tools=self.available_tools.to_params(),
                tool_choice=ToolChoice.REQUIRED,
            )
            self.tool_calls = [ToolCall(**tc) for tc in resp.tool_calls]
            self.content = resp.content

            logging.info(f"🔷 {'━'*10} Step {self.current_step} {'━'*10} 🔷")
            logging.info(f"🛠️ Tool Calls: {self.tool_calls}")
            logging.info(
                f"📝 Content: {self.content if self.content else '🚫 No content'}"
            )

            self._logs += [
                f"### 🔷 Step {self.current_step}",
            ]
            self._previews.append([])

            if self.content:
                self._logs += [
                    f"- 📝 **Content**: {self.content}",
                ]

            assistant_msg = (
                Message.from_tool_calls(
                    content=self.content, tool_calls=self.tool_calls
                )
                if self.tool_calls
                else Message.assistant_message(self.content)
            )
            self.memory.add_message(assistant_msg)

            for tool_call in self.tool_calls:
                result = await self.available_tools.execute_tool(tool_call)
                if tool_call.function.name == "ask_human":
                    logging.info(f"🔔 Asking human for input: {result.meta}")
                    self._question = result.meta.get("question", "")
                    while not self._answer:
                        await asyncio.sleep(2)
                        result.meta["answer"] = self._answer
                        result.output = self._answer
                    self._question, self._answer = "", ""

                logging.info(
                    f"✅ 🎯 Tool 『{tool_call.function.name}』 completed its mission! 📦 {result}"
                )
                self._logs.append(format_tool_for_logs(tool_call))
                self._previews[-1].append(format_tool_for_preview(tool_call, result))

                tool_msg = Message.tool_message(
                    content=str(result),
                    tool_call_id=tool_call.id,
                    name=tool_call.function.name,
                    images=result.images,
                )
                self.memory.add_message(tool_msg)

                yield "\n\n".join(self._logs), self.current_step

                if tool_call.function.name == "terminate":
                    logging.info("🧯 Terminating the agent as requested. Goodbye! 👋")
                    self.state = AgentState.IDLE
                    break

            yield "\n\n".join(self._logs), self.current_step

            self.current_step += 1

            if self._stop:
                logging.info("🛑 Task stop requested by user. Stopping the agent.")
                self.state = AgentState.IDLE
                self._stop = False
                break

        num_messages = len(self.memory.messages)
        preview_messages = Memory(
            messages=[preview_message]
            + self.memory.get_recent_messages(num_messages - 1)
        ).to_dict_list()
        preview_messages = [
            {
                "role": msg["role"] if msg["role"] != "tool" else "user",
                "content": msg["content"],
            }
            for msg in preview_messages
            if msg.get("content", "")
        ]
        preview_content = self.gpt.ask(
            messages=preview_messages,
        ).content

        self._previews.append([preview_content])
        logging.info(f"📜 Preview generated: {preview_content}")
        yield "\n\n".join(self._logs), self.current_step


def cli_main(host: str = "0.0.0.0", port: int = 7050):
    designer = DesignerAgent(
        system_prompt=SYSTEM_PROMPT,
        action_prompt=ACTION_PROMPT,
        preview_prompt=PREVIEW_PROMPT,
    )

    title = gr.Markdown(
        "# <div style='margin-top:10px; text-align: center'>Unitorch Microsoft Designer Agent</div>"
    )

    instruction = gr.MultimodalTextbox(
        sources=["upload"],
        file_count="multiple",
        label="Instruction",
        lines=5,
        show_label=False,
        placeholder="Please input your instruction here, you can also upload files for more inputs.",
        stop_btn=True,
        interactive=True,
    )

    hidden = gr.Textbox(interactive=False, visible=False)
    examples = gr.Examples(
        examples=["1"] * 50,
        inputs=hidden,
        label="Examples",
    )

    save_button = gr.Button(
        "Save",
        variant="primary",
        visible=True,
    )
    delete_button = gr.Button(
        "Delete",
        variant="secondary",
        visible=True,
    )

    logs = gr.Markdown(
        label="Logs",
        height=800,
        line_breaks=True,
        container=True,
    )
    preview = gr.Markdown(
        label="Preview",
        height=800,
        line_breaks=True,
        container=True,
    )
    slider = gr.Slider(
        minimum=1,
        maximum=designer.max_steps + 1,
        step=1,
        value=1,
        label="Step",
        scale=2,
    )
    prev_button = gr.Button(
        "Previous",
        variant="primary",
    )
    next_button = gr.Button(
        "Next",
        variant="secondary",
    )
    question = gr.Textbox(label="Question", interactive=False, lines=2)
    answer = gr.MultimodalTextbox(
        sources=["upload"],
        file_count="multiple",
        label="Answer",
        lines=2,
        placeholder="Please input your answer here, you can also upload files.",
        interactive=True,
    )
    timer = gr.Timer(3)

    demo = create_blocks(
        title,
        instruction,
        create_row(
            create_column(examples.dataset, scale=3),
            create_row(delete_button, save_button),
        ),
        create_row(
            create_column(
                logs,
                create_row(question, answer),
            ),
            create_column(
                preview,
                create_row(
                    slider,
                    prev_button,
                    next_button,
                ),
            ),
        ),
        timer,
        hidden,
    )

    demo.__enter__()

    async def _go(ins):
        async for logs, step in designer.execute(ins["text"], files=ins["files"]):
            yield logs, step
            await asyncio.sleep(2)

    instruction.submit(
        _go,
        inputs=[instruction],
        outputs=[logs, slider],
    )

    instruction.stop(
        designer.stop,
        outputs=[],
    )

    def _ans(ans):
        res = ans["text"]
        if ans["files"]:
            res += "\n" + "\n".join([f"[File]({file})" for file in ans["files"]])
        designer.set_answer(res)

    answer.submit(
        _ans,
        inputs=[answer],
        outputs=[answer],
    )

    timer.tick(
        fn=designer.get_question,
        outputs=[question],
    )

    slider.change(
        lambda x: designer.get_step_state(x),
        inputs=[slider],
        outputs=[preview],
    )

    prev_button.click(
        lambda x: x - 1,
        inputs=[slider],
        outputs=[slider],
    )
    next_button.click(
        lambda x: x + 1,
        inputs=[slider],
        outputs=[slider],
    )

    examples.create()

    hidden.change(
        designer.load,
        inputs=[hidden],
        outputs=[instruction, logs, preview, slider],
    )

    save_button.click(
        lambda: gr.Dataset(samples=designer.save()),
        inputs=[],
        outputs=[examples.dataset],
    )
    delete_button.click(
        lambda: gr.Dataset(
            samples=designer.delete(),
        ),
        inputs=[],
        outputs=[examples.dataset],
    )

    demo.load(
        lambda: gr.Dataset(samples=designer.records()),
        inputs=[],
        outputs=[examples.dataset],
    )

    demo.__exit__()

    css_file = os.path.join(
        os.path.join(importlib_resources.files("unitorch"), "cli/assets/style.css")
    )
    demo.theme_css = read_file(css_file)
    demo.title = "Unitorch Microsoft Designer Agent"

    demo.launch(
        server_name=host,
        server_port=port,
        favicon_path=os.path.join(
            importlib_resources.files("unitorch"), "cli/assets/icon.png"
        ),
        allowed_paths=["/"],
    )


if __name__ == "__main__":
    fire.Fire(cli_main)
