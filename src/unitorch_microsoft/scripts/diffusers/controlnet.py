# Copyright (c) MICROSOFT.
# Licensed under the MIT License.
import os
import io
import re
import json
import base64
import zipfile
import logging
import hashlib
import torch
import numpy as np
import pandas as pd
from transformers.utils import is_remote_url
from unitorch.utils import load_weight
from unitorch.cli import CoreConfigureParser, GenericScript
from unitorch.cli import register_script
from unitorch_microsoft import cached_path


@register_script("microsoft/script/diffusers/controlnet")
class ControlNetScript(GenericScript):
    def __init__(self, config: CoreConfigureParser):
        self.config = config

    def launch(self, **kwargs):
        config = self.config
        config.set_default_section("microsoft/script/diffusers/controlnet")

        checkpoint_file = config.getoption("checkpoint_file", None)
        output_file = config.getoption("output_file", "./pytorch_model.bin")

        if checkpoint_file is not None:
            state_dict = load_weight(checkpoint_file)
            state_dict = {
                k.replace("controlnet.", ""): v
                for k, v in state_dict.items()
                if k.startswith("controlnet.")
            }
            torch.save(state_dict, output_file)
