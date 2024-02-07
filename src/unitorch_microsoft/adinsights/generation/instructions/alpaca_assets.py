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

prompt_v1 = """
<|im_start|>system
You are an AI assistant that helps people find information.
<|im_end|>
<|im_start|>user
You are asked to come up with a set of {num_tasks} diverse task instructions. These task instructions will be given to a LLM model and we will evaluate the LLM model for completing the instructions.

Here are the requirements:
1. Try not to repeat the verb for each instruction to maximize diversity.
2. The language used for the instruction also should be diverse. For example, you should combine questions with imperative instrucitons.
3. The type of instructions should be diverse. The list should include diverse types of tasks like open-ended generation, classification, editing, etc.
2. A LLM language model should be able to complete the instruction. For example, do not ask the assistant to create any visual or audio output. For another example, do not ask the assistant to wake you up at 5pm or set a reminder because it cannot perform any action.
3. The instructions should be in English.
4. The instructions should be 1 to 2 sentences long. Either an imperative sentence or a question is permitted.
5. You should generate an appropriate input to the instruction. The input field should contain a specific example provided for the instruction. It should involve realistic data and should not contain simple placeholders. The input should provide substantial content to make the instruction challenging but should ideally not exceed 100 words.
6. Not all instructions require input. For example, when a instruction asks about some general information, "what is the highest peak in the world", it is not necssary to provide a specific context. In this case, we simply put "<noinput>" in the input field.
7. The output should be an appropriate response to the instruction and the input. Make sure the output is less than 100 words.

List of {num_tasks} tasks:

{content}

Please generate the instructions for each task line by line in <result></result>. Use the following format:
<result>
    #1. <instruction 1>
    #2. <instruction 2>
    #3. <instruction 3>
    #4. <instruction 4>
</result>
<|im_end|>
<|im_start|>assistant
"""


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
        output_file = config.getoption("output_file", "./output.txt")

        instruction_col = config.getoption("instruction_col", "instruction")
        input_col = config.getoption("input_col", "input")
        output_col = config.getoption("output_col", "output")
        chunk_size = config.getoption("chunk_size", 5)
        max_tokens = config.getoption("max_tokens", 2000)
        temperature = config.getoption("temperature", 0)
        top_p = config.getoption("top_p", 1)

        input_escapechar = config.getoption("input_escapechar", None)
        output_escapechar = config.getoption("output_escapechar", "\\")

        output_header = config.getoption("output_header", False)

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

        def processing(prompt, _index):
            res = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
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
                [group, prompt_v1.format(num_tasks=len(chunk), content=content)]
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


prompt_v2 = """
<|im_start|>system
You are an AI assistant that helps people find information.
<|im_end|>
<|im_start|>user
You are asked to come up with a set of {num_tasks} diverse task instructions. These task instructions will be given to a LLM model and we will evaluate the LLM model for completing the instructions.

Here are the requirements:
1. Try not to repeat the verb for each instruction to maximize diversity.
2. The language used for the instruction also should be diverse. For example, you should combine questions with imperative instrucitons.
3. The type of instructions should be diverse. The list should include diverse types of tasks like open-ended generation, classification, editing, etc.
2. A LLM language model should be able to complete the instruction. For example, do not ask the assistant to create any visual or audio output. For another example, do not ask the assistant to wake you up at 5pm or set a reminder because it cannot perform any action.
3. The instructions should be in English.
4. The instructions should be 1 to 2 sentences long. Either an imperative sentence or a question is permitted.
5. You should generate an appropriate input to the instruction. The input field should contain a specific example provided for the instruction. It should involve realistic data and should not contain simple placeholders. The input should provide substantial content to make the instruction challenging but should ideally not exceed 100 words.
6. Not all instructions require input. For example, when a instruction asks about some general information, "what is the highest peak in the world", it is not necssary to provide a specific context. In this case, we simply put "<noinput>" in the input field.
7. The output should be an appropriate response to the instruction and the input. Make sure the output is less than 100 words.

List of {num_tasks} tasks:

{content}

Please generate the instructions for each task line by line in <result></result>. Use the following format:
<result>
    #1. <instruction 1>
    #2. <instruction 2>
    #3. <instruction 3>
    #4. <instruction 4>
</result>
<|im_end|>
<|im_start|>assistant
"""


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
        output_file = config.getoption("output_file", "./output.txt")

        instruction_col = config.getoption("instruction_col", "instruction")
        input_col = config.getoption("input_col", "input")
        output_col = config.getoption("output_col", "output")
        max_tokens = config.getoption("max_tokens", 2000)
        temperature = config.getoption("temperature", 0)
        top_p = config.getoption("top_p", 1)

        input_escapechar = config.getoption("input_escapechar", None)
        output_escapechar = config.getoption("output_escapechar", "\\")

        output_header = config.getoption("output_header", False)

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

        def processing(prompt, _index):
            res = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
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
            return prompt_v2.format(num_tasks=len(assets), content=content)

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
