# Copyright (c) MICROSOFT.
# Licensed under the MIT License.
import re
import torch
import tempfile
import logging
import subprocess
import pandas as pd
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from sklearn.metrics import roc_auc_score
from unitorch.models import GenericOutputs
from unitorch.cli import (
    hf_endpoint_url,
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
)
from unitorch.cli import CoreConfigureParser, GenericScript
from unitorch.cli import register_script


@register_script("microsoft/script/interrogator/classifier")
class ClipInterrogatorScript(GenericScript):
    def __init__(self, config: CoreConfigureParser):
        self.config = config

    def launch(self, **kwargs):
        config = self.config

        config.set_default_section("microsoft/script/interrogator/classifier")

        data_file = config.getoption("data_file", None)
        names = config.getoption("names", None)
        if isinstance(names, str) and names.strip() == "*":
            names = None
        if isinstance(names, str):
            names = re.split(r"[,;]", names)
            names = [n.strip() for n in names]

        image_col = config.getoption("image_col", None)
        label_col = config.getoption("label_col", None)
        device = config.getoption("device", "cpu")

        data = pd.read_csv(
            data_file,
            names=names,
            sep="\t",
            quoting=3,
            header=None,
        )

        assert image_col in data.columns, f"Column {image_col} not found in data."
        assert label_col in data.columns, f"Column {label_col} not found in data."

        labels = set(data[label_col].tolist())
        for label in labels:
            new_data = data.copy()
            new_data[label_col] = (new_data[label_col] == label).astype(int)
            new_data_file = tempfile.NamedTemporaryFile(suffix=".tsv").name
            new_data = new_data[[image_col, label_col]]
            new_data.to_csv(new_data_file, sep="\t", index=False, header=False)
            logging.info(f"Processing label {label}: do_reverse=False")
            process = subprocess.Popen(
                [
                    "unitorch-launch",
                    "configs/interrogator/clip.ini",
                    "--data_file",
                    new_data_file,
                    "--device",
                    str(device),
                    "--do_reverse",
                    "False",
                ],
            )
            process.wait()
            logging.info(f"Processing label {label}: do_reverse=True")
            process = subprocess.Popen(
                [
                    "unitorch-launch",
                    "configs/interrogator/clip.ini",
                    "--data_file",
                    new_data_file,
                    "--device",
                    str(device),
                    "--do_reverse",
                    "True",
                ],
            )
            process.wait()
            logging.info(f"Processing label {label}: done")
