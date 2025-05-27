# Copyright (c) MICROSOFT.
# Licensed under the MIT License.
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
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
