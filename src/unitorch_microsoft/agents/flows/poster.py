# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import json
import logging
from pydantic import BaseModel, Field
from typing import Any, List, Literal, Optional, Union
from unitorch_microsoft.agents.utils.github_copilot import GPTModel
from unitorch_microsoft.agents.components import (
    GenericTool,
    GenericResult,
    GenericError,
    GenericResult,
    GenericAgent,
    ToolCall,
    ToolChoice,
    ToolCollection,
    AgentState,
    Memory,
    Message,
)
from unitorch_microsoft.agents.components.tools import TerminateTool
from unitorch_microsoft.agents.components.picasso.html import PicassoHtmlTool
from unitorch_microsoft.agents.components.picasso import get_picasso_temp_dir

SYSTEM_PROMPT = """
You are a professional graphic designer. You need to create a poster image based on the user's prompt with html_to_image tool. 

* You may need multiple iterations to refine the html code, by checking the previous generated image to ensure it meets the requirements finally.
* Only use one tool call in each step.

You can use the following tools to complete tasks:
1. `picasso_html_tool`: Render HTML content to an image.
2. `terminate`: The designed image is perfect, and you can terminate the process.

Design the image with Tailwind CSS, including text, images, and styles. Follow these guidelines to ensure your poster designs are visually consistent, pixel-perfect, and responsive to layout constraints:
* Overall Style:
    * Sophisticated, futuristic, high-tech aesthetic.
    * Flat design with a card-based layout; harmonious card colors, spacing, and alignment with the overall theme.
        * Choose a color scheme suited to the content (e.g., Morandi palette, advanced gray tones, Memphis, Mondrian).
        * No gradients. Maintain high text-background contrast for readability.
        * Avoid plain white backgrounds unless explicitly requested.
        * Avoid ordinary or tacky designs.
        * Treat the page as one unified container — no nested cards or isolated container blocks.
* Layout Guidelines:
    * Avoid overlap, clutter, emptiness, overflow, or truncation.
    * Color palette must be harmonious and consistent with the theme; avoid harsh tones.
    * Place content in the visual focal area. If minimal content, enlarge fonts slightly and center the card.
    * For multiple cards, ensure a clear arrangement, alignment, and proportionate sizing with minimal unused white space. No overlapping or cluttering of cards.
    * All elements must be fully visible within the page boundaries with proper size. Adjust font sizes, card/image/video dimensions, spacing, and positioning to maintain balance.

When validating the designed image, consider:
1. Is the overall composition visually appealing and professionally executed?
2. Does the design clearly communicate its intended purpose or message?
3. Does the aesthetic reflect a modern and refined visual style?
4. Are there flaws such as text, product or any key elements overlap (including the elements from background from visual view), inconsistent spacing, imbalanced sizing or misaligned elements?
5. Are horizontal and vertical alignments precise and intentional?
6. What specific, actionable improvements can enhance the visual outcome?
7. Is the product image preserved and integrated effectively into the design if a product is provided?
8. Does the image meet the user's requirements and expectations?
9. Does the image has any issues from the visual design perspective to be improved?

If flaws are identified, refine it or create a new one right now. Please focus on the high-priority issues that can significantly enhance the design quality.
If the overall looks prefect, use the `terminate` tool to end the process.
"""


class PosterAgentFlow(GenericAgent):
    """An agent flow to create poster with Picasso & GPT-4.

    The agent can use the following tools:
    - `picasso_html_tool`: Create a poster using Picasso based on the provided prompt.
    - `terminate_tool`: Terminate the agent when the task is completed.

    The agent follows these steps:
    1. Receives a prompt describing the desired poster.
    2. Uses the `picasso_html_tool` to generate a poster based on the prompt.
    3. Reviews the generated poster and makes any necessary adjustments.
    4. Uses the `terminate_tool` to end the process when satisfied with the result.

    The agent maintains a memory of previous interactions to improve its performance over time.
    """

    name: str = "poster_agent_flow"
    description: str = """
You are a poster design assistant. Your task is to create visually appealing posters based on user prompts.
Use the tools at your disposal to generate and refine poster designs.
"""

    available_tools: ToolCollection = ToolCollection(
        PicassoHtmlTool(),
        TerminateTool(),
    )
    content: str = Field(default="")
    tool_calls: List[ToolCall] = Field(default_factory=list)
    memory: Memory = Field(default_factory=Memory)
    state: AgentState = AgentState.IDLE
    gpt: Any = GPTModel()
    current_step: int = 1
    max_steps: int = 20

    async def execute(
        self, prompt: str, width: int = 1024, height: int = 1024
    ) -> GenericResult:
        """Execute the poster design flow.

        Args:
            prompt (str): The user prompt describing the desired poster.

        Returns:
            GenericResult: The result of the poster design process, including the final image and any relevant information.
        """
        system_message = Message.system_message(content=SYSTEM_PROMPT)
        user_message = Message.user_message(
            content=f"""
Create a poster image with the following requirements:
* Poster size: {width}x{height} pixels.
* Poster content: {prompt}
* Follow the design guidelines and Tailwind CSS guidelines provided.            
""",
        )
        designed_html, designed_image = None, None
        for _ in range(self.max_steps):
            if designed_image is not None:
                assistant_message = Message.assistant_message(
                    content=f"""
The previous design is as follows. The HTML code is:
{designed_html}
The designed image is: {designed_image}
You need to check if the designed image meets the requirements. If not, refine the HTML code or generate a new design.
""",
                    images=[
                        {
                            "path": designed_image,
                            "width": None,
                            "height": None,
                            "priority": "high",
                        }
                    ],
                )
            else:
                assistant_message = None

            for _ in range(3):
                resp = self.gpt.ask_tools(
                    messages=Memory(
                        messages=[
                            system_message,
                            user_message,
                        ]
                        + ([assistant_message] if assistant_message is not None else [])
                    ).to_dict_list(),
                    tools=self.available_tools.to_params(),
                    tool_choice=ToolChoice.REQUIRED,
                )
                tool_calls = [ToolCall(**tc) for tc in resp.tool_calls]
                if len(tool_calls) == 1:
                    break
                print("Poster Tool calls:", tool_calls)
            tool_call = tool_calls[0]
            if tool_call.function.name == "terminate":
                logging.info("🧯 Terminating the agent as requested. Goodbye! 👋")
                self.state = AgentState.IDLE
                break
            args = json.loads(tool_calls[0].function.arguments)
            raw_html = args.get("raw_html", None)
            viewport = args.get("viewport", None)
            result = await self.available_tools.get_tool("picasso_html_tool").execute(
                action="html_to_image",
                raw_html=raw_html,
                viewport=viewport,
            )
            designed_image = result.meta["_image"]
            designed_html = raw_html
            logging.info(f"Designed image: {designed_image}")

        return GenericResult(
            output=f"Designed poster image with prompt: {prompt}. Final designed image: {designed_image}.",
            images={
                "path": designed_image,
                "width": width,
                "height": height,
                "priority": "low",
            },
            meta={
                "_width": width,
                "_height": height,
                "_prompt": prompt,
                "_image": designed_image,
                "_html": designed_html,
            },
        )


_POSTER_DESCRIPTION = """
Use the picasso_poster_tool to create visually appealing posters, banners, or social media cards with the provided prompt.
* The prompt should describe the details of all the elements, color, styles and etc except layout of the poster very clearly and concisely.
* All the assets (e.g. product image, logo image) should be provided as absolute file paths.
"""


class PosterAgentFlowTool(GenericTool):
    name: str = "poster_agent_flow_tool"
    description: str = _POSTER_DESCRIPTION
    parameters: str = {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "The detailed prompt to create a poster including all the elements, assets, color, styles and etc except layout.",
            },
            "width": {
                "type": "integer",
                "description": "The width of the poster image to be created.",
                "default": 1024,
            },
            "height": {
                "type": "integer",
                "description": "The height of the poster image to be created.",
                "default": 1024,
            },
        },
        "required": [
            "prompt",
            "width",
            "height",
        ],
    }
    agent: Any = PosterAgentFlow()

    async def execute(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
    ) -> GenericResult:

        try:
            result = await self.agent.execute(
                prompt=prompt,
                width=width,
                height=height,
            )
        except Exception as e:
            return GenericError(message=f"Failed to create poster: {str(e)}")

        return result
