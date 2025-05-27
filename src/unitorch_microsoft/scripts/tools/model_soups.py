# Copyright (c) MICROSOFT.
# Licensed under the MIT License.
import os
import logging
import torch

from unitorch.cli import GenericScript, CoreConfigureParser
from unitorch.cli import register_script


@register_script("microsoft/script/tools/model_soups")
class ModelSoupsScript(GenericScript):
    def __init__(self, config: CoreConfigureParser):
        self.config = config

    def launch(self, **kwargs):
        config = self.config
        config.set_default_section("microsoft/script/tools/model_soups")
        checkpoint1 = config.getoption("checkpoint1", None)
        checkpoint2 = config.getoption("checkpoint2", None)
        checkpoint3 = config.getoption("checkpoint3", None)
        checkpoint4 = config.getoption("checkpoint4", None)
        checkpoint5 = config.getoption("checkpoint5", None)
        checkpoint6 = config.getoption("checkpoint6", None)
        checkpoint7 = config.getoption("checkpoint7", None)
        checkpoint8 = config.getoption("checkpoint8", None)
        checkpoint9 = config.getoption("checkpoint9", None)
        checkpoint10 = config.getoption("checkpoint10", None)

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

        output_folder = config.getoption("output_folder", None)
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
