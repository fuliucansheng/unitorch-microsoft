# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import re
import json
import logging
import fire
import pandas as pd


def main(
    data_file: str,
    output_file: str = "./output.txt",
    chunksize: int = 1000,
    freq: int = 20,
    input_escapechar: str = None,
    output_escapechar: str = None,
    result_tags: str = None,
    result_sep: str = ";",
    output_header: bool = False,
    output_tokens: bool = True,
):
    assert os.path.exists(data_file), f"data_file not found: {data_file}"

    tag_list = None
    if isinstance(result_tags, str):
        tag_list = [t.strip() for t in re.split(r"[,;]", result_tags)]

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

    for i, chunk in enumerate(data):
        chunk.fillna("", inplace=True)
        chunk["response"] = chunk["response"].map(json.loads)

        null_mask = chunk["response"].map(lambda r: r.get("response") is None)
        if null_mask.any():
            logging.warning(f"Partition {i}: {null_mask.sum()} null responses skipped.")
        chunk = chunk[~null_mask]

        chunk["index"] = chunk["response"].map(
            lambda r: r["request_metadata"]["ConversationId"]
        )
        chunk["model"] = chunk["response"].map(lambda r: r["response"]["model"])
        chunk["answer"] = chunk["response"].map(
            lambda r: r["response"]["choices"][0]["text"] if r["response"]["choices"] else ""
        )
        chunk["prompt_tokens"] = chunk["response"].map(
            lambda r: r["response"]["usage"]["prompt_tokens"]
        )
        chunk["completion_tokens"] = chunk["response"].map(
            lambda r: r["response"]["usage"]["completion_tokens"]
        )
        chunk["total_tokens"] = chunk["response"].map(
            lambda r: r["response"]["usage"]["total_tokens"]
        )

        for old, new in {"\n": " ", "\t": " ", "\r": " "}.items():
            chunk["answer"] = chunk["answer"].str.replace(old, new, regex=False)

        columns = ["index", "model", "answer"]
        if tag_list:
            for tag in tag_list:
                pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL)
                chunk[f"result_{tag}"] = chunk["answer"].map(
                    lambda ans, p=pattern: result_sep.join(p.findall(ans))
                )
                columns.append(f"result_{tag}")

        if output_tokens:
            columns += ["prompt_tokens", "completion_tokens", "total_tokens"]

        chunk[columns].to_csv(
            output_file,
            sep="\t",
            index=False,
            header=output_header and i == 0,
            mode="a",
            quoting=3,
            escapechar=output_escapechar,
        )
        if (i + 1) % freq == 0:
            logging.info(f"Partition {i} processed.")


if __name__ == "__main__":
    fire.Fire(main)
