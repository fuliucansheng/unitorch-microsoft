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
import frontmatter
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
    PicassoLayoutTool,
)
from unitorch_microsoft.agents.utils.chatgpt import GPTModel

# from unitorch_microsoft.agents.utils.github_copilot import GPTModel
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
2. Improvments from the `review_image_tool` must be resolved before finishing the design task. Except the result from `picasso_layout_tool`, which could be the final result of the design task without any further modifications.

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
6. `picasso_layout_tool`: Generate a poster image with some title/images assets, prefer to use `picasso_layout_tool` to generate a poster/layout image than `picasso_html_tool`.
7. `picasso_html_tool`: Render HTML content to an image.
8. `notify_human`: Notify user some messages/progress or results without waiting for user's response. (message tools)
9. `check_image_tool`: Analysis the image with the provided prompt or mark the image to the priority for view.
10. `terminate`: Enter idle mode once all tasks are completed or user requests stop
11. `review_image_tool`: Check the designed image and give feedback from a professional visual designer perspective.

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
- Use `bash` to download images from URLs before processing; do not use URLs directly in image related tools.
- The image size could be found in the event stream after the image path which is always the absolute path.
- Use `picasso_internal_tool` to generate/edit/process any images instead of any other tools. It can also process the generated image from other tools like `picasso_html_tool` or `picasso_layout_tool`.
- Use `picasso_layout_tool` to generate a poster/layout image is preferred than `picasso_html_tool`. Don't use this tool if the user does not provide any product or logo image.
- Use `picasso_html_tool` to render HTML content to image, including text, images, and styles using Tailwind CSS
</picasso_rules>

<designer_rules>
- Use `bash` or `python` to get the padding pixel numbers. Print results to logs. Never calculate mentally.
- If the user provides a product or background, it must be retained in the final result. padding the input as background is preferred than generating a new background.
- Use `picasso_layout_tool` to generate a poster/layout image is preferred than `picasso_html_tool`.
- All the Call-To-Action elements must be aligned to each other from both style, position and size, such as buttons, text, and images.
- The logo or other image assets may needs to be `remove_background` before overlaying it on the background. Be careful about this.
- Use `check_image_tool` to analyze the input & generated image based on the provided prompt or set it priority for view.
- Don't edit the image in-place, always create a new file after processing the image.
- Don't design the image in a resolution more than 1536x1536. Resize the image (don't change the ratio) and save it to a new file when the image is larger than this resolution.
- Follow professional visual and design principles to ensure high-quality output:
    * Ensure the overall composition is clear, engaging, and emotionally resonant.
    * Maintain a strong aesthetic presence and creative originality.
    * Avoid issues such as text, product, or any key elements overlapping (including the elements from background from visual view), misalignment, or visual clutter.
    * Typography: Establish a clear hierarchy with legible, well-paired fonts.
    * Color Harmony: Use a cohesive color palette and apply contrast effectively to highlight key elements.
    * Layout & Spacing: Structure the layout logically with proper alignment—including strict horizontal and vertical alignment of elements—and sufficient whitespace. Element sizes should be well considered to ensure clear hierarchy and legibility.
    * Visual Hierarchy: Ensure key elements are visually distinct and easy to identify.
    * Aspect Ratio & Dimensions: Ensure the design is appropriately sized for its intended platform (e.g., print, mobile, social media).
    * Product Integration: Integrate product images seamlessly without distortion or loss of focus. Ensure the product's size and proportion within the final composition are appropriate and balanced, so it neither overwhelms nor gets visually lost.
- Use `review_image_tool` to check the designed image and get feedbacks from a professional visual designer. Except the result from `picasso_layout_tool`.
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

REVIEW_PROMPT = """
You are a professional visual designer and experienced design critic.

Your task is to evaluate the final designed image based on the provided messages. Don't need to check the logo correctness, relevance or any color if it's provided by user.

When reviewing visual design work (e.g., posters, promotional graphics, UI screens), your evaluation must address

1. Is the overall composition visually appealing and professionally executed?
2. Does the design clearly communicate its intended purpose or message?
3. Does the aesthetic reflect a modern and refined visual style?
4. Are there flaws such as text, product or any key elements overlap (including the elements from background from visual view), inconsistent spacing, imbalanced sizing or misaligned elements?
5. Are horizontal and vertical alignments precise and intentional?
6. What specific, actionable improvements can enhance the visual outcome?
7. Is the product image preserved and integrated effectively into the design if a product is provided?
8. Does the image meet the user's requirements and expectations?
9. Does the image has any issues from the visual design perspective to be improved?

If flaws are identified, provide a clear, actionable suggestions to guide refinement. Please focus on the high-priority issues that can significantly enhance the design quality.
Don't need to mention the good parts of the design, just focus on the flaws and how to improve them.

If the overall looks excellent, you can just say "The design looks excellent, no further improvements needed.".
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


class ReviewImageTool(GenericTool):
    name: str = "review_image_tool"
    description: str = """
Use this tool to check the designed image and give feedback from a professional visual designer perspective.

Parameters:
- `image`: The image path for checking.
"""

    parameters: dict = {
        "type": "object",
        "properties": {
            "image": {
                "type": "string",
                "description": "The image path for checking or marking.",
            },
        },
        "required": ["image"],
    }
    gpt: Any = GPTModel()
    system_message: Message = None
    user_message: Message = None

    def setup(self, system_prompt, user_inputs):
        self.system_message = Message.system_message(content=system_prompt)
        files = user_inputs.get("files", [])
        images = []
        for file in files:
            try:
                im = Image.open(file)
                images.append(
                    {
                        "path": file,
                        "width": im.width,
                        "height": im.height,
                        "priority": "high",
                    }
                )
            except Exception as e:
                logging.warning(f"Failed to open image {file}: {e}")
                images.append(
                    {"path": file, "width": None, "height": None, "priority": "low"}
                )
        self.user_message = Message.user_message(
            content=f"### Task\n\n- **Instruction**:\n{user_inputs.get('text', '')}\n\n ### Inputs\n\n ```json\n{json.dumps(user_inputs, indent=2, ensure_ascii=False)}\n```",
            images=images,
        )

    async def execute(
        self,
        image: str,
    ):
        """Execute the tool to check the image."""
        if image is None or not os.path.exists(image):
            raise ValueError("image is required and must exist.")
        msg = Message.assistant_message(
            content=f"Here is the image for review: {image}",
            images=[{"path": image, "width": None, "height": None, "priority": "high"}],
        )
        resp = self.gpt.ask(
            messages=Memory(
                messages=[
                    self.system_message,
                    self.user_message,
                    msg,
                ]
            ).to_dict_list(),
        )
        if not resp.content:
            raise ValueError("No content returned from GPT-4 model.")
        return GenericResult(
            output=f"Review result: {resp.content}",
            images=[{"path": image, "width": None, "height": None, "priority": "low"}],
        )


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
        PicassoLayoutTool(),
        NotifyHumanTool(),
        CheckImageTool(),
        TerminateTool(),
        ReviewImageTool(),
    )
    content: str = Field(default="")
    tool_calls: List[ToolCall] = Field(default_factory=list)
    memory: Memory = Field(default_factory=Memory)
    state: AgentState = AgentState.IDLE
    gpt: Any = GPTModel()
    current_step: int = 1
    max_steps: int = 500

    def __init__(self, system_prompt, action_prompt, review_prompt, preview_prompt):
        super().__init__()
        self._system_prompt = system_prompt
        self._action_prompt = action_prompt
        self._review_prompt = review_prompt
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
        self.available_tools.get_tool("review_image_tool").setup(
            system_prompt=self._review_prompt,
            user_inputs=self._user_inputs,
        )

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
        review_prompt=REVIEW_PROMPT,
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
