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
    names: str = None,
    index_col: str = None,
    chunksize: int = 1000,
    max_tokens: int = 200,
    temperature: float = 0,
    presence_penalty: float = 0,
    frequency_penalty: float = 0,
    top_p: float = 1,
    stop: str = None,
    freq: int = 20,
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
        return json.dumps({
            "prompt": prompt.format(**row.to_dict()),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "top_p": top_p,
            "stop": stop,
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
        if (i + 1) % freq == 0:
            logging.info(f"Partition {i} processed.")


if __name__ == "__main__":
    fire.Fire(main)
