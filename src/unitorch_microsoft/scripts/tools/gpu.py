# Copyright (c) MICROSOFT.
# Licensed under the MIT License.
import os
import json
import re
import random
import torch
import logging
import requests
import pandas as pd
from typing import Optional

from unitorch.cli import CoreConfigureParser, GenericScript
from unitorch.cli import register_script, cached_path


@register_script("microsoft/script/tools/gpu")
class GPUScript(GenericScript):
    def __init__(self, config: CoreConfigureParser):
        super().__init__(config)
        self.config = config

    def launch(self):
        num_gpus = torch.cuda.device_count()
        tensors = [torch.rand(10240, 10240).cuda(i) for i in range(num_gpus)]

        while True:
            for tensor in tensors:
                _ = torch.matmul(tensor, tensor)
