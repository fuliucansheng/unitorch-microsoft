# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import json
import re
import random
import logging
import requests
import pandas as pd
from typing import Optional

from unitorch.cli import CoreConfigureParser, GenericScript
from unitorch.cli import register_script, cached_path


@register_script("microsoft/script/tools/eda")
class EDAScript(GenericScript):
    def __init__(self, config: CoreConfigureParser):
        super().__init__(config)
        self.config = config

    def launch(self):
        config = self.config
        config.set_default_section("microsoft/script/tools/eda")
        data_file = config.getoption("data_file", None)
        names = config.getoption("names", None)
        if isinstance(names, str) and names.strip() == "*":
            names = None
        if isinstance(names, str):
            names = re.split(r"[,;]", names)
            names = [n.strip() for n in names]

        data = pd.read_csv(
            data_file,
            names=names,
            sep="\t",
            quoting=3,
            header=None if names is None else "infer",
        )

        dtypes = data.dtypes.to_dict()

        for k, v in dtypes.items():
            if v == "int64" or v == "float64":
                print(
                    f"Column: {k} | Dtype: {v} ｜ Mean: {data[k].mean()} ｜ Median: {data[k].median()} ｜ Min: {data[k].min()} ｜ Max: {data[k].max()} ｜ Std: {data[k].std()} ｜ Unique: {data[k].nunique()} ｜ Missing: {data[k].isna().sum()}"
                )
            else:
                print(
                    f"Column: {k} | Dtype: {v} ｜ Unique: {data[k].nunique()} ｜ Missing: {data[k].isna().sum()}"
                )
