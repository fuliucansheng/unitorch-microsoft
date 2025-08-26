# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import asyncio
import time
import logging
import markdownify
import tempfile
from typing import Optional, Any
from pydantic import BaseModel, Field
from browser_use import Browser as BrowserUseBrowser
from browser_use import BrowserProfile as BrowserUseProfile
from unitorch_microsoft.chatgpt.papyrus import get_gpt4_response
from unitorch_microsoft.agents.components import (
    GenericTool,
    GenericResult,
    GenericError,
)


_BROWSERUSE_TEMP_DIR = "/tmp/browser_use"
_BROWSERUSE_TEMP_SCREENSHOT_DIR = f"{_BROWSERUSE_TEMP_DIR}/screenshot"

if not os.path.exists(_BROWSERUSE_TEMP_DIR):
    os.makedirs(_BROWSERUSE_TEMP_DIR)

if not os.path.exists(_BROWSERUSE_TEMP_SCREENSHOT_DIR):
    os.makedirs(_BROWSERUSE_TEMP_SCREENSHOT_DIR)

_BROWSERUSE_DESCRIPTION = """\
A powerful browser automation tool that allows interaction with web pages through various actions.
* This tool provides commands for controlling a browser session, navigating web pages, and extracting information
* It maintains state across calls, keeping the browser session alive until explicitly closed
* Use this when you need to browse websites, fill forms, click buttons, extract content, or perform web searches
* Each action requires specific parameters as defined in the tool's dependencies

Notes:
* BrowserState will be returned after each action, which includes the current URL, title, and clickable elements. You need to take more actions if you want to get more content.
* Screenshot will be taken after each action and returned for review to take next step.
* When using element indices, refer to the numbered elements shown in the current BrowserState and Screenshot.
* Don't take more than one action at a time, as the browser state will change after each action.

Key capabilities include:
* `go_to_url`: Navigate to a specific URL in current browser tab
* `click_element`: Click on a specific element by its index in current browser tab
* `input_text`: Input text into a specific element by its index in current browser tab
* `scroll_down`/`scroll_up`: Scroll the page down or up in current browser tab
* `get_dropdown_options`: Get options from a dropdown element by its index in current browser tab
* `select_dropdown_option`: Select an option from a dropdown element by its index and text in current browser tab
* `go_back`: Navigate back to the previous page in current browser tab
* `wait`: Wait for a specified number of seconds in current browser tab
* `extract_content`: Extract content from the current page based on a specific goal in current browser tab
* `switch_tab`: Switch to a specific tab by its ID
* `open_tab`: Open a new tab with a specific URL
* `close_tab`: Close the current tab
"""


class BrowserUseTool(GenericTool):
    """Add a tool to ask human for help."""

    name: str = "browser_use"
    description: str = _BROWSERUSE_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "go_to_url",
                    "click_element",
                    "input_text",
                    "scroll_down",
                    "scroll_up",
                    "get_dropdown_options",
                    "select_dropdown_option",
                    "go_back",
                    "wait",
                    "extract_content",
                    "switch_tab",
                    "open_tab",
                    "close_tab",
                ],
                "description": "The browser action to perform in the current browser tab. Please switch to the correct tab before performing the action.",
            },
            "url": {
                "type": "string",
                "description": "URL for 'go_to_url' or 'open_tab' actions",
            },
            "index": {
                "type": "integer",
                "description": "Element index for 'click_element', 'input_text', 'get_dropdown_options', or 'select_dropdown_option' actions",
            },
            "text": {
                "type": "string",
                "description": "Text for 'input_text', or 'select_dropdown_option' actions",
            },
            "tab_id": {
                "type": "integer",
                "description": "Tab ID for 'switch_tab' action",
            },
            "goal": {
                "type": "string",
                "description": "Extraction goal for 'extract_content' action",
            },
            "seconds": {
                "type": "integer",
                "description": "Seconds to wait for 'wait' action",
            },
        },
        "required": ["action"],
        "dependencies": {
            "go_to_url": ["url"],
            "click_element": ["index"],
            "input_text": ["index", "text"],
            "switch_tab": ["tab_id"],
            "open_tab": ["url"],
            "scroll_down": [],
            "scroll_up": [],
            "get_dropdown_options": ["index"],
            "select_dropdown_option": ["index", "text"],
            "go_back": [],
            "wait": ["seconds"],
            "extract_content": ["goal"],
        },
    }

    # config = BrowserConfig(
    #     # cdp_url="ws://127.0.0.1:9222/devtools/browser/9331ed5c-57a0-40b5-aebd-eac2fb1d7c78",
    #     # headless=True,
    #     # disable_security=True,
    #     # browser_binary_path="/usr/bin/google-chrome",
    #     # browser_binary_path="C:/Program Files/Google/Chrome/Application/chrome.exe",
    #     # extra_browser_args=['--user-data-dir=C:/Users/decu/AppData/Local/Google/Chrome/User Data'],
    # )

    browser: Optional[BrowserUseBrowser] = Field(default=None, exclude=True)
    context: Optional[Any] = Field(default=None, exclude=True)

    async def setup(self):
        self.browser = BrowserUseBrowser(
            executable_path="/usr/bin/google-chrome",
            browser_profile=BrowserUseProfile(viewport={"width": 1920, "height": 1080}),
        )
        self.context = await self.browser.new_context()

    async def go_to_url(self, url: str):
        page = await self.context.get_current_page()
        await page.goto(url)
        await page.wait_for_load_state()

    async def go_back(self):
        await self.context.go_back()

    async def click_element(self, index):
        element = await self.context.get_dom_element_by_index(index)
        if not element:
            return None
        download_path = await self.context._click_element_node(element)
        return download_path if download_path else None

    async def input_text(self, index, text):
        element = await self.context.get_dom_element_by_index(index)
        if not element:
            return None
        await self.context._input_text_element_node(element, text)

    async def scroll_window(self, direction: str = "down"):
        if direction not in ["up", "down"]:
            raise ValueError("Direction must be 'up' or 'down'")
        direction = 1 if direction == "down" else -1
        amount = 0
        if self.context.browser_profile.window_size:
            amount = amount or self.context.browser_profile.window_size.get("height", 0)
        if self.context.browser_profile.viewport:
            amount = amount or self.context.browser_profile.viewport.get("height", 0)
        amount = amount // 2
        await self.context.execute_javascript(
            f"window.scrollBy(0, {direction * amount});"
        )

    async def get_dropdown_options(self, index: int):
        element = await self.context.get_dom_element_by_index(index)
        if not element:
            return None
        page = await self.context.get_current_page()
        options = await page.evaluate(
            """
            (xpath) => {
                const select = document.evaluate(xpath, document, null,
                    XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
                if (!select) return null;
                return Array.from(select.options).map(opt => ({
                    text: opt.text,
                    value: opt.value,
                    index: opt.index
                }));
            }
            """,
            element.xpath,
        )
        return options

    async def select_dropdown_option(self, index: int, option):
        element = await self.context.get_dom_element_by_index(index)
        if not element:
            return None
        page = await self.context.get_current_page()
        await page.select_option(
            element.xpath,
            label=option.get("text"),
        )

    async def extract_content(self, goal):
        page = await self.context.get_current_page()
        await page.wait_for_load_state()
        raw_html = await page.content()
        content = markdownify.markdownify(raw_html)
        prompt = f"""\
        Your task is to extract the content of the page. You will be given a page and a goal, and you should extract all relevant information around this goal from the page. If the goal is vague, summarize the page. Respond in json format. Return the full urls of images or videos if the goal need.
        Extraction goal: {goal}

        Page content:
        {content}
        """
        result = get_gpt4_response(prompt)
        return result

    async def switch_tab(self, tab_id):
        self.context.switch_tab(tab_id)
        page = await self.context.get_current_page()
        await page.wait_for_load_state()

    async def open_tab(self, url: str):
        await self.context.create_new_tab(url)

    async def close_tab(self):
        await self.context.close_current_tab()

    async def wait(self, seconds: int):
        time.sleep(seconds)

    async def screenshot(self):
        page = await self.context.get_current_page()
        path = tempfile.NamedTemporaryFile(
            suffix=".png",
            dir=_BROWSERUSE_TEMP_SCREENSHOT_DIR,
        ).name
        await page.screenshot(path=path)
        return path

    async def execute(
        self,
        action: str,
        url: Optional[str] = None,
        index: Optional[int] = None,
        text: Optional[str] = None,
        tab_id: Optional[int] = None,
        goal: Optional[str] = None,
        seconds: Optional[int] = None,
    ) -> str:
        if not self.browser or not self.context:
            await self.setup()

        message, error = "", ""
        try:
            if action == "go_to_url":
                await self.go_to_url(url)
                message = f"Navigated to {url} successfully."
            elif action == "click_element":
                await self.click_element(index)
                message = f"Clicked element at index {index} successfully."
            elif action == "input_text":
                await self.input_text(index, text)
                message = (
                    f"Input text '{text}' into element at index {index} successfully."
                )
            elif action == "scroll_down":
                await self.scroll_window("down")
                message = "Scrolled down the window successfully."
            elif action == "scroll_up":
                await self.scroll_window("up")
                message = "Scrolled up the window successfully."
            elif action == "get_dropdown_options":
                await self.get_dropdown_options(index)
                message = f"Retrieved dropdown options for element at index {index}."
            elif action == "select_dropdown_option":
                await self.select_dropdown_option(index, {"text": text})
                message = f"Selected dropdown option '{text}' at index {index}."
            elif action == "go_back":
                await self.go_back()
                message = "Navigated back successfully."
            elif action == "wait":
                await self.wait(seconds)
                message = f"Waited for {seconds} seconds."
            elif action == "extract_content":
                content = await self.extract_content(goal)
                message = f"Extracted content based on goal '{goal}': {content}"
            elif action == "switch_tab":
                await self.switch_tab(tab_id)
                message = f"Switched to tab with ID {tab_id} successfully."
            elif action == "open_tab":
                await self.open_tab(url)
                message = f"Opened new tab with URL {url} successfully."
            elif action == "close_tab":
                await self.close_tab()
                message = "Closed the current tab successfully."
            else:
                message = f"Unknown action: {action}"
        except Exception as e:
            error = f"Error executing action '{action}': {str(e)}"

        state = await self.context.get_state_summary(False)
        state_info = {
            "url": state.url,
            "title": state.title,
            "tabs": [tab.model_dump() for tab in state.tabs],
            "help": "[0], [1], [2], etc., represent clickable indices corresponding to the elements listed. Clicking on these indices will navigate to or interact with the respective content behind them.",
            "interactive_elements": (
                state.element_tree.clickable_elements_to_string()
                if state.element_tree
                else ""
            ),
            "scroll_info": {
                "pixels_above": getattr(state, "pixels_above", 0),
                "pixels_below": getattr(state, "pixels_below", 0),
            },
        }
        screenshot = await self.screenshot()

        logging.info(
            f"Browser state after action '{action}', url: {state.url}, title: {state.title}. Screenshot saved at {screenshot}."
        )

        result = GenericResult(
            output=f"Message: {message}. Browser Latest State: {state_info}.",
            error=error,
            images={"path": screenshot},
            meta={
                "_action": action,
                "_url": state.url,
                "_goal": goal,
                "_screenshot": screenshot,
            },
        )
        return result
