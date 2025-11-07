# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import logging
import fire
import torch
from typing import Optional


def launch(
    checkpoint1: Optional[str] = None,
    checkpoint2: Optional[str] = None,
    checkpoint3: Optional[str] = None,
    checkpoint4: Optional[str] = None,
    checkpoint5: Optional[str] = None,
    checkpoint6: Optional[str] = None,
    checkpoint7: Optional[str] = None,
    checkpoint8: Optional[str] = None,
    checkpoint9: Optional[str] = None,
    checkpoint10: Optional[str] = None,
    output_folder: str = "output",
):
    checkpoints = [
        checkpoint1,
        checkpoint2,
        checkpoint3,
        checkpoint4,
        checkpoint5,
        checkpoint6,
        checkpoint7,
        checkpoint8,
        checkpoint9,
        checkpoint10,
    ]
    checkpoints = list(filter(lambda x: x is not None, checkpoints))

    output_checkpoint = os.path.join(output_folder, "pytorch_model.bin")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    assert len(checkpoints) > 0 and output_checkpoint is not None

    checkpoints = [torch.load(c, map_location="cpu") for c in checkpoints]

    num_checkpoints = len(checkpoints)
    final_checkpoint = {k: v / num_checkpoints for k, v in checkpoints[0].items()}
    keys = list(final_checkpoint.keys())
    for checkpoint in checkpoints[1:]:
        assert keys == list(checkpoint.keys())
        for key in keys:
            final_checkpoint[key] += checkpoint[key] / num_checkpoints

    torch.save(final_checkpoint, output_checkpoint)
    logging.info(f"Save final checkpoint to {output_checkpoint}")

if __name__ == "__main__":
    fire.Fire(launch)