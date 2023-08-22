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
import numpy as np
import pandas as pd
from unitorch.cli import CoreConfigureParser, GenericScript
from unitorch.cli import register_script
from unitorch_microsoft import cached_path

@register_script("microsoft/script/chatgpt/dv3")
class DV3Script(GenericScript):
    def __init__(self, config: CoreConfigureParser):
        self.config = config

    def launch(self, **kwargs):
        config = self.config
        config.set_default_section("microsoft/script/chatgpt/dv3")
        data_file = config.getoption("data_file", None)
        prompt_file = config.getoption("prompt_file", None)
        prompt_text = config.getoption("prompt_text", None)
        output_file = config.getoption("output_file", "./output.txt")
        assert data_file is not None and (prompt_file is not None or prompt_text is not None)

        if prompt_file is not None:
            prompt_file = cached_path(prompt_file)

        names = config.getoption("names", None)
        index_col = config.getoption("index_col", None)
        chunksize = config.getoption("chunksize", 1000)
        max_tokens = config.getoption("max_tokens", 200)
        temperature = config.getoption("temperature", 0)
        top_p = config.getoption("top_p", 1)
        freq = config.getoption("freq", 20)
        escapechar = config.getoption("escapechar", "\\")
        replace_items = config.getoption("replace_items", {"#endl#": "\n"})

        if names is None:
            data = pd.read_csv(
                data_file,
                names=names,
                sep="\t",
                chunksize=chunksize,
                quoting=3,
                header="infer",
                iterator=True,
                low_memory=True,
            )
        else:
            if isinstance(names, str):
                names = re.split(r"[,;]", names)
                names = [n.strip() for n in names]
            data = pd.read_csv(
                data_file,
                names=names,
                sep="\t",
                chunksize=chunksize,
                quoting=3,
                header=None,
                iterator=True,
                low_memory=True,
            )

        if prompt_file is not None:
            prompt = open(prompt_file, "r").read()
        else:
            prompt = prompt_text

        for k, v in replace_items.items():
            prompt = prompt.replace(k, v)

        def processing(row):
            res = {
                "prompt": prompt.format(**row.to_dict()),
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "_batch_request_metadata": {
                    "ConversationId": str(row[index_col]),
                },
            }
            return json.dumps(res)

        start_index = 1
        if index_col is None:
            index_col = "__index__"

        for i, _data in enumerate(data):
            if index_col == "__index__":
                _data["__index__"] = range(start_index, start_index + len(_data))
            assert index_col in _data.columns
            _data.fillna("", inplace=True)
            _data["jsonl"] = _data.apply(processing, axis=1)

            _data["jsonl"].to_csv(
                output_file,
                sep="\t",
                index=False,
                header=None,
                mode="a",
                quoting=3,
                escapechar=escapechar,
            )
            start_index += len(_data)
            logging.info(f"partition {i} processed finish.")


