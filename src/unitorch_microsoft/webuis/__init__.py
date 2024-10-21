# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import gradio as gr
from unitorch.utils import is_diffusers_available
from unitorch.cli import CoreConfigureParser
from unitorch.cli import register_webui
from unitorch.cli.webuis import SimpleWebUI
from unitorch_microsoft.webuis.detection import DetectionWebUI
from unitorch_microsoft.webuis.llm import LLMWebUI
from unitorch_microsoft.webuis.picasso import PicassoWebUI
from unitorch_microsoft.webuis.classification import ClassificationWebUI
from unitorch_microsoft.webuis.segmentation import SegmentationWebUI
from unitorch_microsoft.webuis.selection import SelectionWebUI
from unitorch_microsoft.webuis.tools import ToolsWebUI
from unitorch_microsoft.webuis.chatgpt import ChatGPTWebUI

if is_diffusers_available():
    from unitorch_microsoft.webuis.diffusion import DiffusionWebUI
