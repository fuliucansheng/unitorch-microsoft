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


class ClassificationProcessor:
    """Processor for classification models."""

    def __init__(
        self,
        act_fn: Optional[str] = None,
        return_scores: Optional[bool] = False,
    ):
        """
        Initialize the ClassificationProcessor.

        Args:
            act_fn (Optional[str]): Activation function to apply to the model outputs.
            return_scores (Optional[bool]): Whether to return the scores in addition to the predictions.
        """
        self.act_fn = ACT2FN.get(act_fn, None)
        self.return_scores = return_scores

    @classmethod
    @add_default_section_for_init("microsoft/process/classification")
    def from_core_configure(cls, config, **kwargs):
        """
        Create a ClassificationProcessor instance from core configuration.

        Args:
            config: Configuration object.
            **kwargs: Additional keyword arguments.

        Returns:
            ClassificationProcessor: The initialized ClassificationProcessor instance.
        """
        pass

    @register_process("microsoft/postprocess/classification/embedding/string")
    def _embedding_string(
        self,
        outputs: EmbeddingOutputs,
    ):
        """
        Postprocess the embedding outputs as string representations.

        Args:
            outputs (EmbeddingOutputs): Outputs from the embedding model.

        Returns:
            WriterOutputs: Processed outputs with string representations of embeddings.
        """
        results = outputs.to_pandas()
        assert results.shape[0] == 0 or results.shape[0] == outputs.embedding.shape[0]

        embedding = outputs.embedding.numpy()
        if embedding.ndim > 2:
            embedding = embedding.reshape(embedding.size(0), -1)
        results["embedding"] = embedding.tolist()

        results["embedding"] = results["embedding"].map(
            lambda x: " ".join([str(f"{i:.6f}") for i in x])
        )

        embedding1 = outputs.embedding1.numpy()
        if embedding1.size > 0:
            if embedding1.ndim > 2:
                embedding1 = embedding1.reshape(embedding1.size(0), -1)
            results["embedding1"] = embedding1.tolist()
            results["embedding1"] = results["embedding1"].map(
                lambda x: " ".join([str(f"{i:.6f}") for i in x])
            )

        embedding2 = outputs.embedding2.numpy()
        if embedding2.size > 0:
            if embedding2.ndim > 2:
                embedding2 = embedding2.reshape(embedding2.size(0), -1)
            results["embedding2"] = embedding2.tolist()
            results["embedding2"] = results["embedding2"].map(
                lambda x: " ".join([str(f"{i:.6f}") for i in x])
            )

        embedding3 = outputs.embedding3.numpy()
        if embedding3.size > 0:
            if embedding3.ndim > 2:
                embedding3 = embedding3.reshape(embedding3.size(0), -1)
            results["embedding3"] = embedding3.tolist()
            results["embedding3"] = results["embedding3"].map(
                lambda x: " ".join([str(f"{i:.6f}") for i in x])
            )

        embedding4 = outputs.embedding4.numpy()
        if embedding4.size > 0:
            if embedding4.ndim > 2:
                embedding4 = embedding4.reshape(embedding4.size(0), -1)
            results["embedding4"] = embedding4.tolist()
            results["embedding4"] = results["embedding4"].map(
                lambda x: " ".join([str(f"{i:.6f}") for i in x])
            )

        return WriterOutputs(results)
