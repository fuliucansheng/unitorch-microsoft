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


@register_script("microsoft/omnigpt/script/intents/phrase")
class PhraseScript(GenericScript):
    def __init__(self, config: CoreConfigureParser):
        self.config = config

    def launch(self, **kwargs):
        config = self.config
        config.set_default_section("microsoft/omnigpt/script/intents/phrase")

        data_file = config.getoption("data_file", None)
        chunksize = config.getoption("chunksize", 100)
        threshold = config.getoption("threshold", 0.01)
        output_file = config.getoption("output_file", "./output.txt")

        data = pd.read_csv(
            data_file,
            names=["phrase", "emb"],
            sep="\t",
            chunksize=chunksize,
            quoting=3,
            header=None,
            iterator=True,
            low_memory=True,
        )

        results, selected = [], []

        def check(new_emb):
            nonlocal selected
            new_emb = np.array(new_emb)
            if len(selected) == 0:
                selected = new_emb.reshape(1, -1)
            else:
                min_dist = np.min(np.linalg.norm(selected - new_emb, axis=1))
                if min_dist >= threshold:
                    selected = np.vstack([selected, new_emb.reshape(1, -1)])
                else:
                    return False
            return True

        for chunk in data:
            for _, row in chunk.iterrows():
                emb = np.array(list(map(float, row["emb"].split(" "))))
                if check(emb):
                    results.append(row["phrase"])

        results = pd.DataFrame(results, columns=["phrase"])
        results.to_csv(
            output_file,
            sep="\t",
            index=False,
            quoting=3,
            header=None,
        )
