# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import pyarrow as pa
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.cli import (
    add_default_section_for_init,
    add_default_section_for_function,
    register_process,
)
from unitorch.cli import WriterMixin, WriterOutputs
from unitorch.cli.models import ClassificationOutputs, EmbeddingOutputs, ACT2FN


class PreProcessor:
    def __init__(
        self,
        map_dict: Optional[Dict[str, str]] = None,
    ):
        self.map_dict = map_dict if map_dict is not None else {}

    @classmethod
    @add_default_section_for_init("microsoft/process")
    def from_core_configure(cls, config, **kwargs):
        pass
