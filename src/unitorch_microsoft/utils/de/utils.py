# Copyright (c) MICROSOFT.
# Licensed under the MIT License.
import os
import logging


def convert_weight_to_ranker_string(
    name,
    value,
    num_row,
    num_col,
):
    logging.info(
        f"Convert to weight to DE Ranker format. Before: {name} - shape: {value.shape} - target: ({num_row}, {num_col})"
    )
    if len(value.shape) == 0:
        value = value.unsqueeze(0).unsqueeze(0)
    if len(value.shape) == 1:
        value = value.unsqueeze(0)
    if len(value.shape) == 2:
        logging.info(
            f"Convert to weight to DE Ranker format. The Tensor {name} is transposed."
        )
        value = value.transpose(0, 1)
    sh = value.shape
    if sh[0] != num_row:
        value = value.transpose(0, 1)
        sh = value.shape
    assert sh[0] == num_row
    assert sh[1] == num_col
    ret = "; ".join([" ".join(map(lambda x: str(float(x)), v)) for v in value])
    return ret
