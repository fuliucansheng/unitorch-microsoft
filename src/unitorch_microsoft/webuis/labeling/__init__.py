# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import re
import pandas as pd
import gradio as gr
from unitorch.cli import CoreConfigureParser
from unitorch.cli import register_webui
from unitorch.cli.webuis import (
    create_element,
    create_accordion,
    create_row,
    create_column,
    create_group,
    create_tab,
    create_tabs,
    create_blocks,
)
from unitorch.cli.webuis import SimpleWebUI
import unitorch_microsoft.webuis.labeling.classification
