# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import gradio as gr
import unitorch
from unitorch.utils.decorators import replace
from unitorch.utils import is_diffusers_available
from unitorch.cli import CoreConfigureParser
from unitorch.cli import register_webui
from unitorch.cli.webuis import SimpleWebUI
