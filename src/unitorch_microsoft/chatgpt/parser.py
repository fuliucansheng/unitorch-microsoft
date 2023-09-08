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


@register_script("microsoft/script/chatgpt/parser")
class ParserScript(GenericScript):
    def __init__(self, config: CoreConfigureParser):
        self.config = config

    def launch(self, **kwargs):
        config = self.config
        config.set_default_section("microsoft/script/chatgpt/parser")
        data_file = config.getoption("data_file", None)
        output_file = config.getoption("output_file", "./output.txt")
        assert data_file is not None and os.path.exists(data_file)

        chunksize = config.getoption("chunksize", 1000)
        freq = config.getoption("freq", 20)
        input_escapechar = config.getoption("input_escapechar", None)
        output_escapechar = config.getoption("output_escapechar", None)
        result_tags = config.getoption("result_tags", None)
        result_sep = config.getoption("result_sep", ";")
        output_header = config.getoption("output_header", False)
        output_tokens = config.getoption("output_tokens", True)
        replace_items = config.getoption(
            "replace_items", {"\n": " ", "\t": " ", "\r": " "}
        )

        data = pd.read_csv(
            data_file,
            names=["response"],
            sep="\t",
            chunksize=chunksize,
            quoting=3,
            header=None,
            iterator=True,
            low_memory=True,
            escapechar=input_escapechar,
        )
        if isinstance(result_tags, str):
            result_tags = re.split(r"[,;]", result_tags)
            result_tags = [tag.strip() for tag in result_tags]

        for i, _data in enumerate(data):
            _data.fillna("", inplace=True)
            _data["response"] = _data.response.map(lambda r: json.loads(r))
            _data["index"] = _data.response.map(
                lambda r: r["request_metadata"]["ConversationId"]
            )
            _data["model"] = _data.response.map(lambda r: r["response"]["model"])
            _data["results"] = _data.response.map(lambda r: r["response"]["choices"])
            _data["answer"] = _data.results.map(
                lambda r: r[0]["text"] if len(r) > 0 else ""
            )
            _data["prompt_tokens"] = _data.response.map(
                lambda r: r["response"]["usage"]["prompt_tokens"]
            )
            _data["completion_tokens"] = _data.response.map(
                lambda r: r["response"]["usage"]["completion_tokens"]
            )
            _data["total_tokens"] = _data.response.map(
                lambda r: r["response"]["usage"]["total_tokens"]
            )

            for k, v in replace_items.items():
                _data["answer"] = _data["answer"].map(lambda ans: ans.replace(k, v))

            _columns = ["index", "model", "answer"]
            for tag in result_tags:
                pattern = re.compile(r"<{0}>(.*?)</{0}>".format(tag), re.DOTALL)
                _data[f"result_{tag}"] = _data.answer.map(
                    lambda ans: result_sep.join(pattern.findall(ans))
                )
                _columns.append(f"result_{tag}")

            if output_tokens:
                _columns += ["prompt_tokens", "completion_tokens", "total_tokens"]

            _data[_columns].to_csv(
                output_file,
                sep="\t",
                index=False,
                header=output_header,
                mode="a",
                quoting=3,
                escapechar=output_escapechar,
            )
            logging.info(f"partition {i} processed finish.")
