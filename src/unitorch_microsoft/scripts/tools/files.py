# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import re
import fire
import logging
import pandas as pd
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union


def list_folder(
    folder: str,
    output_file: str,
    extensions: Optional[str] = "jpg,png",
    recursive: bool = False,
):
    if not os.path.exists(folder):
        raise ValueError(f"folder {folder} not found.")

    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    if extensions is not None:
        extensions = re.split(r"[;,]", extensions)
        extensions = [e.strip() for e in extensions]
    else:
        extensions = []

    data_files = []
    if recursive:
        for root, _, files in os.walk(folder):
            data_files += [
                os.path.join(root, f) for f in files if f.split(".")[-1] in extensions
            ]
    else:
        data_files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.split(".")[-1] in extensions
        ]
    output = pd.DataFrame({"file": data_files})
    output.to_csv(
        output_file,
        index=False,
        sep="\t",
        header=None,
        quoting=3,
    )


def filter_file(
    input_file: str,
    output_file: str,
    filter_column: str,
    filter_values: Union[str, List[str]],
    names: Optional[Union[str, List[str]]] = None,
    chunksize: int = 1000,
    freq: int = 20,
    input_escapechar: Optional[str] = None,
    output_escapechar: Optional[str] = None,
    output_header: bool = False,
):
    if isinstance(filter_values, str):
        filter_values = re.split(r"[;,]", filter_values)
        filter_values = [v.strip() for v in filter_values]

    if isinstance(names, str):
        names = re.split(r"[;,]", names)
        names = [n.strip() for n in names]

    data = pd.read_csv(
        input_file,
        names=names,
        sep="\t",
        chunksize=chunksize,
        quoting=3,
        header="infer" if names is None else None,
        escapechar=input_escapechar,
        iterator=True,
        low_memory=True,
    )

    assert filter_column in data.columns, f"{filter_column} not in {data.columns}"
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    for i, _data in enumerate(data):
        _data = _data[
            _data[filter_column].map(lambda x: not any([v in x for v in filter_values]))
        ]
        _data.to_csv(
            output_file,
            sep="\t",
            index=False,
            quoting=3,
            header=output_header and i == 0,
            mode="a",
            escapechar=output_escapechar,
        )
        if (i + 1) % freq == 0:
            logging.info(f"partition {i} processed finish.")


def sample_file(
    input_file: str,
    output_file: str,
    sample_size: int = 1000,
    names: Optional[Union[str, List[str]]] = None,
    input_escapechar: Optional[str] = None,
    output_escapechar: Optional[str] = None,
):
    if isinstance(names, str):
        names = re.split(r"[;,]", names)
        names = [n.strip() for n in names]

    data = pd.read_csv(
        input_file,
        names=names,
        sep="\t",
        quoting=3,
        header="infer" if names is None else None,
        escapechar=input_escapechar,
    )

    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    if sample_size > len(data):
        sample_size = len(data)
    sampled_data = data.sample(n=sample_size, random_state=42)
    sampled_data.to_csv(
        output_file,
        sep="\t",
        index=False,
        quoting=3,
        header=True if names is None else False,
        escapechar=output_escapechar,
    )


if __name__ == "__main__":
    fire.Fire()
