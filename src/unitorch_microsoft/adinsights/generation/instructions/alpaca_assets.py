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
from transformers.utils import is_remote_url
from unitorch.cli import CoreConfigureParser, GenericScript
from unitorch.cli import register_script
from unitorch_microsoft import cached_path


@register_script("microsoft/script/adinsights/generation/instructions/alpaca/sahara/v1")
class AlpacaSaharaV1Script(GenericScript):
    def __init__(self, config: CoreConfigureParser):
        self.config = config

    def launch(self, **kwargs):
        config = self.config
        config.set_default_section(
            "microsoft/script/adinsights/generation/instructions/alpaca/sahara/v1"
        )
        input_file = config.getoption("input_file", None)
        names = config.getoption("names", "*")
        prompt_file = config.getoption("prompt_file", None)
        prompt_text = config.getoption("prompt_text", None)
        output_file = config.getoption("output_file", "./output.txt")

        if prompt_file is not None and prompt_file.strip() == "":
            prompt_file = None

        assert prompt_file is not None or prompt_text is not None

        if prompt_file is not None:
            if is_remote_url(prompt_file):
                prompt_file = cached_path(prompt_file)
            else:
                prompt_file = cached_path(prompt_file)

        instruction_col = config.getoption("instruction_col", "instruction")
        input_col = config.getoption("input_col", "input")
        output_col = config.getoption("output_col", "output")
        chunk_size = config.getoption("chunk_size", 5)
        max_tokens = config.getoption("max_tokens", 2000)
        temperature = config.getoption("temperature", 0)
        top_p = config.getoption("top_p", 1)

        input_escapechar = config.getoption("input_escapechar", None)
        output_escapechar = config.getoption("output_escapechar", None)

        output_header = config.getoption("output_header", False)
        replace_items = config.getoption("replace_items", {"#endl#": "\n"})

        if isinstance(names, str) and names.strip() == "*":
            names = None
        if isinstance(names, str):
            names = re.split(r"[,;]", names)
            names = [n.strip() for n in names]
        input = pd.read_csv(
            input_file,
            names=names,
            sep="\t",
            quoting=3,
            header="infer" if names is None else None,
            escapechar=input_escapechar,
        )

        if prompt_file is not None:
            prompt_text = open(prompt_file, "r", encoding="utf-8").read()

        for k, v in replace_items.items():
            prompt_text = prompt_text.replace(k, v)

        def processing(prompt, _index):
            res = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stop": "<|im_end|>",
                "_batch_request_metadata": {
                    "ConversationId": str(_index),
                },
            }
            return json.dumps(res)

        input["rownum"] = range(input.shape[0])
        input["group"] = input["rownum"] // chunk_size
        input["chunk"] = (input["rownum"] % chunk_size) + 1
        num_groups = input["group"].max() + 1
        output = []
        for group in range(num_groups):
            chunk = input[input["group"] == group]
            content = "\n###\n".join(
                chunk.apply(
                    lambda x: f"{x['chunk']}. Instruction: {x[instruction_col]}\n{x['chunk']}. Input: {x[input_col]}\n{x['chunk']}. Output: {x[output_col]}",
                    axis=1,
                )
            )
            content = f"###\n{content}\n###\n"
            output.append(
                [group, prompt_text.format(num_tasks=len(chunk), content=content)]
            )

        output = pd.DataFrame(output, columns=["groupnum", "prompt"])
        output["jsonl"] = output.apply(
            lambda x: processing(x["prompt"], x["groupnum"]), axis=1
        )

        output["jsonl"].to_csv(
            output_file,
            sep="\t",
            index=False,
            header=output_header,
            quoting=3,
            escapechar=output_escapechar,
        )


@register_script("microsoft/script/adinsights/generation/instructions/alpaca/sahara/v2")
class AlpacaSaharaV2Script(GenericScript):
    def __init__(self, config: CoreConfigureParser):
        self.config = config

    def launch(self, **kwargs):
        config = self.config
        config.set_default_section(
            "microsoft/script/adinsights/generation/instructions/alpaca/sahara/v2"
        )
        input_file = config.getoption("input_file", None)
        names = config.getoption("names", "*")
        prompt_file = config.getoption("prompt_file", None)
        prompt_text = config.getoption("prompt_text", None)
        output_file = config.getoption("output_file", "./output.txt")

        if prompt_file is not None and prompt_file.strip() == "":
            prompt_file = None

        assert prompt_file is not None or prompt_text is not None

        if prompt_file is not None:
            if is_remote_url(prompt_file):
                prompt_file = cached_path(prompt_file)
            else:
                prompt_file = cached_path(prompt_file)

        instruction_col = config.getoption("instruction_col", "instruction")
        input_col = config.getoption("input_col", "input")
        output_col = config.getoption("output_col", "output")
        max_tokens = config.getoption("max_tokens", 2000)
        temperature = config.getoption("temperature", 0)
        top_p = config.getoption("top_p", 1)

        input_escapechar = config.getoption("input_escapechar", None)
        output_escapechar = config.getoption("output_escapechar", None)

        output_header = config.getoption("output_header", False)
        replace_items = config.getoption("replace_items", {"#endl#": "\n"})

        if isinstance(names, str) and names.strip() == "*":
            names = None
        if isinstance(names, str):
            names = re.split(r"[,;]", names)
            names = [n.strip() for n in names]
        input = pd.read_csv(
            input_file,
            names=names,
            sep="\t",
            quoting=3,
            header="infer" if names is None else None,
            escapechar=input_escapechar,
        )

        if prompt_file is not None:
            prompt_text = open(prompt_file, "r", encoding="utf-8").read()

        for k, v in replace_items.items():
            prompt_text = prompt_text.replace(k, v)

        def processing(prompt, _index):
            res = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stop": "<|im_end|>",
                "_batch_request_metadata": {
                    "ConversationId": str(_index),
                },
            }
            return json.dumps(res)

        def generate_prompt(instruction, input, output):
            assets = re.split(r"[,;]", output)
            indexs = list(range(1, len(assets) + 1))
            content = "\n###\n".join(
                [
                    f"{i}. Instruction: {instruction}\n{i}. Input: {input} \n{i}. Output: {asset}"
                    for i, asset in zip(indexs, assets)
                ]
            )
            content = f"###\n{content}\n###\n"
            return prompt_text.format(num_tasks=len(assets), content=content)

        output = input.copy()
        output["rownum"] = range(output.shape[0])
        output["prompt"] = output.apply(
            lambda x: generate_prompt(x[instruction_col], x[input_col], x[output_col]),
            axis=1,
        )
        output["jsonl"] = output.apply(
            lambda x: processing(x["prompt"], x["rownum"]), axis=1
        )

        output["jsonl"].to_csv(
            output_file,
            sep="\t",
            index=False,
            header=output_header,
            quoting=3,
            escapechar=output_escapechar,
        )
