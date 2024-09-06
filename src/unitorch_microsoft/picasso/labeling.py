# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import io
import re
import socket
import requests
import tempfile
import hashlib
import subprocess
import pandas as pd
import gradio as gr
from PIL import Image
from collections import Counter
from torch.hub import download_url_to_file
from unitorch import get_temp_home
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
from unitorch_microsoft.webuis.labeling.classification import (
    GenericClassificationLabelingWebUI,
)


@register_webui("microsoft/picasso/labeling/classification")
class PicassoClassificationLabelingWebUI(SimpleWebUI):
    def __init__(
        self,
        config: CoreConfigureParser,
    ):
        self._config = config
        config.set_default_section("microsoft/picasso/labeling/classification")
        sections = config.getoption("sections", [])
        names = config.getoption("names", [])
        webuis = [
            GenericClassificationLabelingWebUI(config, section, name)
            for name, section in zip(names, sections)
        ]
        iface = gr.TabbedInterface(
            [webui.iface for webui in webuis],
            tab_names=[webui.iname for webui in webuis],
        )
        super().__init__(config, iname="Picasso Classification Labeling", iface=iface)
