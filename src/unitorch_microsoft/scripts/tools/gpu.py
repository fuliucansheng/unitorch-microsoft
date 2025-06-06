# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import json
import re
import fire
import random
import torch
import logging
import requests
import pandas as pd
from typing import Optional


def launch():
    num_gpus = torch.cuda.device_count()
    tensors = [torch.rand(10240, 10240).cuda(i) for i in range(num_gpus)]

    while True:
        for tensor in tensors:
            _ = torch.matmul(tensor, tensor)


if __name__ == "__main__":
    fire.Fire(launch)
