# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import re
import json
import logging
import fire
import pandas as pd
from unitorch_microsoft import cached_path


def main(
    data_file: str,
    output_file: str = "./output.txt",
    prompt_file: str = None,
    prompt_text: str = None,
    system_prompt: str = None,
    names: str = None,
    index_col: str = None,
    chunksize: int = 1000,
    max_tokens: int = 200,
    temperature: float = 0,
    top_p: float = 1,
    presence_penalty: float = 0.1,
    frequency_penalty: float = 0.1,
    input_escapechar: str = None,
    output_escapechar: str = None,
    output_header: bool = False,
):
    assert os.path.exists(data_file), f"data_file not found: {data_file}"
    assert prompt_file is not None or prompt_text is not None, \
        "Either prompt_file or prompt_text must be provided."

    if prompt_file is not None:
        prompt_file = cached_path(prompt_file)
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt = f.read()
    else:
        prompt = prompt_text

    prompt = prompt.replace("#endl#", "\n")

    if isinstance(names, str) and names.strip() == "*":
        names = None
    elif isinstance(names, str):
        names = [n.strip() for n in re.split(r"[,;]", names)]

    data = pd.read_csv(
        data_file,
        names=names,
        sep="\t",
        chunksize=chunksize,
        quoting=3,
        header="infer" if names is None else None,
        iterator=True,
        low_memory=True,
        escapechar=input_escapechar,
    )

    if index_col is None:
        index_col = "__index__"

    def build_request(row):
        messages = [{"role": "user", "content": prompt.format(**row.to_dict())}]
        if system_prompt and system_prompt.strip():
            messages = [{"role": "system", "content": system_prompt}] + messages
        return json.dumps({
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
        })

    start_index = 1
    for i, chunk in enumerate(data):
        if index_col == "__index__":
            chunk["__index__"] = range(start_index, start_index + len(chunk))
        assert index_col in chunk.columns
        chunk.fillna("", inplace=True)
        chunk["jsonl"] = chunk.apply(build_request, axis=1)
        chunk["jsonl"].to_csv(
            output_file,
            sep="\t",
            index=False,
            header=output_header and i == 0,
            mode="a",
            quoting=3,
            escapechar=output_escapechar,
        )
        start_index += len(chunk)
        logging.info(f"Partition {i} processed.")


if __name__ == "__main__":
    fire.Fire(main)
