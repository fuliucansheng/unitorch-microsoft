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

try:
    from openai import AzureOpenAI
except ImportError:
    raise ImportError(
        "Please install the openai package by running `pip install openai`"
    )


def get_respone(
    prompt,
    system_prompt: Optional[
        str
    ] = "You are an AI assistant that helps people find information.",
    api_endpoint: Optional[str] = None,
    api_deploy_name: Optional[str] = None,
    api_key: Optional[str] = None,
):
    client = AzureOpenAI(
        azure_endpoint=api_endpoint, api_key=api_key, api_version="2023-07-01-preview"
    )

    try:
        response = client.chat.completions.create(
            model=api_deploy_name,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )
        result = response.choices[0].message.content.strip()
    except Exception as e:
        print(e)
        result = ""

    return result


@register_script("microsoft/script/chatgpt/azure")
class AzureChatGPT(GenericScript):
    def __init__(self, config: CoreConfigureParser):
        super().__init__(config)
        self.config = config

    def launch(self):
        config = self.config
        config.set_default_section("microsoft/script/chatgpt/azure")
        input_file = config.getoption("input_file", None)
        prompt_file = config.getoption("prompt_file", None)
        system_prompt = config.getoption(
            "system_prompt",
            "You are an AI assistant that helps people find information.",
        )
        output_file = config.getoption("output_file", "./output.txt")

        prompt_file = cached_path(prompt_file)

        if not os.path.exists(input_file):
            raise ValueError(f"data file {input_file} not found")

        names = config.getoption("names", None)
        chunksize = config.getoption("chunksize", 10)
        freq = config.getoption("freq", 10)

        # Azure API
        api_endpoint = config.getoption("api_endpoint", None)
        api_key = config.getoption("api_key", None)
        api_deploy_name = config.getoption("api_deploy_name", None)

        input_escapechar = config.getoption("input_escapechar", None)
        output_escapechar = config.getoption("output_escapechar", None)
        result_tags = config.getoption("result_tags", None)
        result_sep = config.getoption("result_sep", " #;# ")

        replace_items = {"\n": " ", "\t": " ", "\r": " ", r"\s+": " "}

        if names is None:
            data = pd.read_csv(
                input_file,
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
                input_file,
                names=names,
                sep="\t",
                chunksize=chunksize,
                quoting=3,
                header=None,
                iterator=True,
                low_memory=True,
                escapechar=input_escapechar,
            )

        prompt = open(prompt_file, "r", encoding="utf-8").read()

        for i, _data in enumerate(data):
            _data["prompt"] = _data.apply(lambda x: prompt.format(**x), axis=1)
            _data["answer"] = _data["prompt"].apply(
                lambda x: get_respone(
                    x,
                    system_prompt=system_prompt,
                    api_endpoint=api_endpoint,
                    api_key=api_key,
                    api_deploy_name=api_deploy_name,
                )
            )

            for k, v in replace_items.items():
                _data["prompt"] = _data["prompt"].map(lambda x: re.sub(k, v, x))
                _data["answer"] = _data["answer"].map(lambda x: re.sub(k, v, x))

            if result_tags is not None:
                if isinstance(result_tags, str):
                    result_tags = re.split(r"[,;]", result_tags)
                    result_tags = [t.strip() for t in result_tags]
                for result_tag in result_tags:
                    pattern = re.compile(
                        r"<{0}>(.*?)</{0}>".format(result_tag), re.DOTALL
                    )
                    _data[f"answer_{result_tag}"] = _data["answer"].map(
                        lambda ans: result_sep.join(pattern.findall(ans))
                    )

            _data.to_csv(
                output_file,
                sep="\t",
                index=False,
                header=i == 0,
                mode="a",
                quoting=3,
                escapechar=output_escapechar,
            )

            if (i + 1) % freq == 0:
                logging.info(f"partition {i} processed finish.")
