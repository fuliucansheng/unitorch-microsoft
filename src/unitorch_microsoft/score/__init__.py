# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import numpy as np
from unitorch.cli import add_default_section_for_init, register_score
from unitorch.cli.models import ClassificationOutputs, ClassificationTargets
from unitorch.cli.score import Score


@register_score("microsoft/score/pearsonr_corr")
class PearsonrCorrScore(Score):
    def __init__(
        self,
    ):
        super().__init__()

    @classmethod
    @add_default_section_for_init("microsoft/score/pearsonr_corr")
    def from_core_configure(cls, config, **kwargs):
        pass

    def forward(
        self,
        outputs: ClassificationOutputs,
        targets: ClassificationTargets,
    ):
        if isinstance(outputs, ClassificationOutputs):
            outputs = outputs.outputs
        if isinstance(targets, ClassificationTargets):
            targets = targets.targets

        outputs = outputs.view(-1)
        targets = targets.view(-1)

        assert outputs.numel() == targets.numel()

        return np.corrcoef(targets, outputs).sum() / 2 - 1
