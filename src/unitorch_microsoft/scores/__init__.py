# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import numpy as np
from unitorch.scores import (
    bleu_score,
    rouge1_score,
    rouge2_score,
    rougel_score,
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

@register_score("microsoft/score/bleu")
class BleuScore(Score):
    def __init__(
        self,
        ignore_tokens: Optional[List[int]] = [0, 1],
    ):
        super().__init__()
        self.ignore_tokens = ignore_tokens

    @classmethod
    @add_default_section_for_init("microsoft/score/bleu")
    def from_core_configure(cls, config, **kwargs):
        pass

    def forward(
        self,
        outputs: GenerationOutputs,
        targets: GenerationTargets,
    ):
        if isinstance(outputs, GenerationOutputs):
            outputs = outputs.sequences
        if isinstance(targets, GenerationTargets):
            targets = targets.refs
        return bleu_score(
            targets.long(),
            outputs.long(),
            ignore_tokens=self.ignore_tokens,
        )


@register_score("microsoft/score/rouge1")
class Rouge1Score(Score):
    def __init__(
        self,
        ignore_tokens: Optional[List[int]] = [0, 1],
    ):
        super().__init__()
        self.ignore_tokens = ignore_tokens

    @classmethod
    @add_default_section_for_init("microsoft/score/rouge1")
    def from_core_configure(cls, config, **kwargs):
        pass

    def forward(
        self,
        outputs: GenerationOutputs,
        targets: GenerationTargets,
    ):
        if isinstance(outputs, GenerationOutputs):
            outputs = outputs.sequences
        if isinstance(targets, GenerationTargets):
            targets = targets.refs

        return rouge1_score(
            targets.long(),
            outputs.long(),
            ignore_tokens=self.ignore_tokens,
        )["f1"]


@register_score("microsoft/score/rouge2")
class Rouge2Score(Score):
    def __init__(
        self,
        ignore_tokens: Optional[List[int]] = [0, 1],
    ):
        super().__init__()
        self.ignore_tokens = ignore_tokens

    @classmethod
    @add_default_section_for_init("microsoft/score/rouge2")
    def from_core_configure(cls, config, **kwargs):
        pass

    def forward(
        self,
        outputs: GenerationOutputs,
        targets: GenerationTargets,
    ):
        if isinstance(outputs, GenerationOutputs):
            outputs = outputs.sequences
        if isinstance(targets, GenerationTargets):
            targets = targets.refs

        return rouge2_score(
            targets.long(),
            outputs.long(),
            ignore_tokens=self.ignore_tokens,
        )["f1"]


@register_score("microsoft/score/rougel")
class RougelScore(Score):
    def __init__(
        self,
        ignore_tokens: Optional[List[int]] = [0, 1],
    ):
        super().__init__()
        self.ignore_tokens = ignore_tokens

    @classmethod
    @add_default_section_for_init("microsoft/score/rougel")
    def from_core_configure(cls, config, **kwargs):
        pass

    def forward(
        self,
        outputs: GenerationOutputs,
        targets: GenerationTargets = None,
    ):
        if isinstance(outputs, GenerationOutputs):
            outputs = outputs.sequences
        if isinstance(targets, GenerationTargets):
            targets = targets.refs

        return rougel_score(
            targets.long(),
            outputs.long(),
            ignore_tokens=self.ignore_tokens,
        )["f1"]