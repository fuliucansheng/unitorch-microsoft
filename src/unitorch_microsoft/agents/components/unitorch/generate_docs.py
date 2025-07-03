# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import re
import sys
import fire
import glob
import json
import importlib_resources
import importlib.metadata as importlib_metadata
import frontmatter
import subprocess
import tempfile
import logging
import pandas as pd
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch_microsoft import cached_path
from unitorch_microsoft.chatgpt.papyrus import (
    get_gpt4_response,
    get_gpt4_tools_response,
    get_gpt4_chat_response,
    get_gpt_image_response,
)


model_doc_example = """
---
title: model & task brief introduction
description: model functionality in details
config file: path/to/config.ini
---

# Instruction

## Training

### Dataset

Describe the dataset used for training in detail, if applicable.

### Command

```bash
unitorch-train path/to/config.ini --section@option value
```

## Inference

### Dataset

Describe the dataset used for inference in detail, if applicable.

### Command

```bash
unitorch-infer path/to/config.ini --section@option value
```

### More Instructions

If there are more instructions or details about the model or task, please include them here.

"""

tool_doc_example = """
---
title: tool brief introduction
description: tool functionality in details
config file: path/to/config.ini
---

# Instruction

## Usage

### Command with `unitorch-launch`

```bash
unitorch-launch path/to/config.ini --section@option value
```

### Command with `python3.10 -m unitorch_microsoft.path.to.module`

```bash
python3.10 -m unitorch_microsoft.path.to.module --param value
```

"""


def format_doc(doc):
    if "<!-- MACHINE_GENERATED -->" in doc.content:
        human, machine = doc.content.split("<!-- MACHINE_GENERATED -->", 1)
        return human.strip(), machine.strip()
    else:
        return doc.content.strip(), None


def generate_doc(file, is_model: bool = True):
    if is_model:
        example = model_doc_example
    else:
        example = tool_doc_example

    doc = frontmatter.load(file)
    human, machine = format_doc(doc)

    if machine is not None:
        return

    prompt = f"""
    You are a skilled technical writer. Your task is to create clear, comprehensive, and well-structured documentation for the following { 'model' if is_model else 'tool' } using the provided content as your source.
    The documentation should:
    * Be written in **Markdown** format.
    * Follow the structure and formatting style of the provided example.
    * Include all key details necessary for new users to understand and effectively use the { 'model' if is_model else 'tool' }.
    * Cover aspects such as functionality, installation or usage instructions, configuration options, input/output format, and any other relevant technical or practical considerations.

    Content to document:
    {human}
    Example documentation structure:
    {example}
    Please generate the documentation in Markdown format and wrap your final output within the following tags <result></result>.
    """

    response = get_gpt4_response(prompt=prompt)
    result = re.search(r"<result>(.*?)</result>", response, re.DOTALL)
    if result:
        result_content = result.group(1).strip()
        if not result_content:
            logging.warning(f"Generated documentation for {file} is empty.")
            return

        new_doc = frontmatter.loads(result_content)
        new_doc.content = human + "\n\n<!-- MACHINE_GENERATED -->\n\n" + new_doc.content
        frontmatter.dump(new_doc, file)
        logging.info(f"Generated documentation for {file}.")


def generate():
    pkg_folder = importlib_resources.files("unitorch_microsoft")
    model_docs = glob.glob(
        f"{pkg_folder}/agents/components/unitorch/models/*.md", recursive=True
    )
    tool_docs = glob.glob(
        f"{pkg_folder}/agents/components/unitorch/tools/*.md", recursive=True
    )
    for file in model_docs:
        try:
            generate_doc(file, is_model=True)
        except Exception as e:
            logging.error(f"Failed to generate documentation for {file}: {e}")
    for file in tool_docs:
        try:
            generate_doc(file, is_model=False)
        except Exception as e:
            logging.error(f"Failed to generate documentation for {file}: {e}")


if __name__ == "__main__":
    fire.Fire(generate)
