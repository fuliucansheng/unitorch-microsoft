# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import io
import re
import json
import base64
import requests
import zipfile
import logging
import hashlib
import numpy as np
import pandas as pd
from transformers.utils import is_remote_url
from unitorch.cli import CoreConfigureParser, GenericScript
from unitorch.cli import register_script
from unitorch_microsoft import cached_path

@register_script("microsoft/script/chatgpt/gpt4o_v")
class GPT4O_VScript(GenericScript):
    def __init__(self, config: CoreConfigureParser):
        self.config = config

    def launch(self, **kwargs):
        config = self.config
        config.set_default_section("microsoft/script/chatgpt/gpt4o_v")
        data_file = config.getoption("data_file", None)
        system_prompt = config.getoption("system_prompt", None)
        prompt_file = config.getoption("prompt_file", None)
        prompt_text = config.getoption("prompt_text", None)
        output_file = config.getoption("output_file", "./output.txt")
        if prompt_file is not None and prompt_file.strip() == "":
            prompt_file = None
        assert data_file is not None and os.path.exists(data_file)
        assert prompt_file is not None or prompt_text is not None

        if prompt_file is not None:
            if is_remote_url(prompt_file):
                prompt_file = cached_path(prompt_file)
            else:
                prompt_file = cached_path(prompt_file)

        names = config.getoption("names", None)
        index_col = config.getoption("index_col", None)
        images = config.getoption("images", None)
        chunksize = config.getoption("chunksize", 1000)
        max_tokens = config.getoption("max_tokens", 200)
        temperature = config.getoption("temperature", 0)
        top_p = config.getoption("top_p", 1)
        presence_penalty = config.getoption("presence_penalty", 0.1)
        frequency_penalty = config.getoption("frequency_penalty", 0.1)
        input_escapechar = config.getoption("input_escapechar", None)
        output_escapechar = config.getoption("output_escapechar", None)
        output_header = config.getoption("output_header", False)
        replace_items = config.getoption("replace_items", {"#endl#": "\n"})

        if isinstance(names, str) and names.strip() == "*":
            names = None

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
                escapechar=input_escapechar,
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
                escapechar=input_escapechar,
            )

        if prompt_file is not None:
            prompt = open(prompt_file, "r", encoding="utf-8").read()
        else:
            prompt = prompt_text

        for k, v in replace_items.items():
            prompt = prompt.replace(k, v)

        if isinstance(images, str):
            images = re.split(r"[,;]", images)

        def processing(row):
            content = [
                {
                    "type": "text",
                    "data": prompt.format(**row.to_dict()),
                },
            ]
            
            def read_image(image):
                if is_remote_url(image):
                    return requests.get(image, timeout=300).content
                if os.path.exists(image):
                    return open(image, "rb").read()
                return base64.b64decode(image)

            if isinstance(images, list):
                content += [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64.b64encode(read_image(row[image])).decode()}",
                        }
                    }
                    for image in images
                ]

            messages = [
                {
                    "role": "user",
                    "content": content,
                }
            ]

            if system_prompt is not None and system_prompt.strip() != "":
                messages = [
                    {
                        "role": "system",
                        "content": system_prompt,
                    }
                ] + messages

            res = {
                "messages": messages,
                "stream": False,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
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
                header=output_header,
                mode="a",
                quoting=3,
                escapechar=output_escapechar,
            )
            start_index += len(_data)
            logging.info(f"partition {i} processed finish.")