# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import torch
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.scores import (
    accuracy_score,
)
from unitorch.cli import add_default_section_for_init, register_score
from unitorch.cli.models import (
    ModelOutputs,
    ModelTargets,
    ClassificationOutputs,
    ClassificationTargets,
    RankingOutputs,
    RankingTargets,
    GenerationOutputs,
    GenerationTargets,
    DetectionOutputs,
    DetectionTargets,
    SegmentationOutputs,
    SegmentationTargets,
    LossOutputs,
)
from unitorch.cli.scores import Score

@register_score("microsoft/score/acc")
class AccuracyScore(Score):
    def __init__(self, gate: Optional[float] = 0.5):
        super().__init__()
        self.gate = gate

    @classmethod
    @add_default_section_for_init("microsoft/score/acc")
    def from_core_configure(cls, config, **kwargs):
        pass

    def forward(
        self,
        outputs: Union[ClassificationOutputs, GenerationOutputs, SegmentationOutputs],
        targets: Union[ClassificationTargets, GenerationTargets, SegmentationTargets],
    ):
        if isinstance(outputs, GenerationOutputs):
            outputs = outputs.sequences
            outputs = outputs.view(-1, outputs.size(-1))

        if isinstance(targets, GenerationTargets):
            targets = targets.refs
            targets = targets.view(-1)

        if isinstance(outputs, SegmentationOutputs):
            outputs = outputs.outputs
            outputs = torch.cat([t.view(-1, t.size(-1)) for t in outputs])

        if isinstance(targets, SegmentationTargets):
            targets = targets.targets
            targets = torch.cat([t.view(-1) for t in targets])

        if isinstance(outputs, ClassificationOutputs):
            outputs = outputs.outputs

        if isinstance(targets, ClassificationTargets):
            targets = targets.targets

        if outputs.numel() == targets.numel():
            outputs = outputs.view(-1) > self.gate
            targets = targets.view(-1)

        if outputs.dim() > 1:
            outputs = outputs.argmax(dim=-1)

        assert outputs.numel() == targets.numel(), (
            f"Outputs and targets must have the same number of elements, "
            f"but got {outputs.numel()} and {targets.numel()}"
        )

        outputs = outputs.view(-1)
        targets = targets.view(-1)

        return accuracy_score(targets, outputs)