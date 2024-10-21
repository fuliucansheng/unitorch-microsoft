# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import re
import torch
import pandas as pd
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.models import GenericOutputs
from unitorch.utils import pop_value, nested_dict_value, read_file, read_json_file

from unitorch.cli import (
    add_default_section_for_init,
    add_default_section_for_function,
)
from unitorch.cli import CoreConfigureParser, GenericScript
from unitorch.cli import register_script
from unitorch_microsoft import cached_path
from unitorch_microsoft.models.bletchley import pretrained_bletchley_v1_infos
from unitorch_microsoft.models.bletchley.modeling_v1 import BletchleyForTextPretrain
from unitorch_microsoft.models.bletchley.processing_v1 import BletchleyProcessor


class BletchleyForTextInterrogator(BletchleyForTextPretrain):
    def __init__(
        self,
        query_config_type: str,
        doc_config_type: str,
        phrase_file: str,
        projection_dim: Optional[int] = 1024,
        max_seq_length: Optional[int] = 120,
        use_query_encoder: Optional[bool] = True,
        weight_path: Optional[Union[str, List[str]]] = None,
        device: Optional[Union[str, int]] = "cpu",
    ):
        super().__init__(
            query_config_type,
            doc_config_type,
            projection_dim=projection_dim,
        )
        self.processor = BletchleyProcessor(max_seq_length=max_seq_length)

        self._device = "cpu" if device == "cpu" else int(device)

        self.from_pretrained(weight_path)
        self.to(device=self._device)

        if os.path.exists(phrase_file):
            intent_phrases = pd.read_csv(phrase_file, header=None)[0].tolist()

        self.use_query_encoder = use_query_encoder
        self.intent_phrases = intent_phrases

        if self.use_query_encoder:
            self.intent_phrases_embeds = self.get_query_embeds(intent_phrases)
        else:
            self.intent_phrases_embeds = self.get_doc_embeds(intent_phrases)

    @classmethod
    @add_default_section_for_init(
        "microsoft/omnigpt/pipeline/intents/interrogator/bletchley"
    )
    def from_core_configure(
        cls,
        config,
        pretrained_weight_path: Optional[str] = None,
        device: Optional[str] = "cpu",
        **kwargs,
    ):
        config.set_default_section(
            "microsoft/omnigpt/pipeline/intents/interrogator/bletchley"
        )
        pretrained_config_type = config.getoption("pretrained_config_type", "0.8B")
        query_config_type = config.getoption(
            "query_config_type", pretrained_config_type
        )
        doc_config_type = config.getoption("doc_config_type", pretrained_config_type)
        projection_dim = config.getoption("projection_dim", 1024)
        max_seq_length = config.getoption("max_seq_length", 120)
        use_query_encoder = config.getoption("use_query_encoder", True)
        device = config.getoption("device", device)

        phrase_file = config.getoption("phrase_file", None)
        if phrase_file is not None:
            phrase_file = cached_path(phrase_file)

        pretrained_weight_path = pretrained_weight_path or config.getoption(
            "pretrained_weight_path", None
        )
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_bletchley_v1_infos, pretrained_config_type),
            check_none=False,
        )

        inst = cls(
            query_config_type=query_config_type,
            doc_config_type=doc_config_type,
            phrase_file=phrase_file,
            projection_dim=projection_dim,
            max_seq_length=max_seq_length,
            use_query_encoder=use_query_encoder,
            weight_path=weight_path,
            device=device,
            **kwargs,
        )

        return inst

    def get_query_embeds(
        self,
        texts: Union[str, List[str]],
        max_batch_size: Optional[int] = 512,
    ):
        if isinstance(texts, str):
            texts = [texts]

        inputs = [
            self.processor._text_classification(text, prefix="query_").dict()
            for text in texts
        ]
        keys = inputs[0].keys()
        inputs = {key: torch.stack([inp[key] for inp in inputs]) for key in keys}
        results = []
        for i in range(0, len(texts), max_batch_size):
            query_outputs = self.query_encoder(
                input_ids=inputs["query_input_ids"][i : i + max_batch_size].to(
                    self._device
                ),
                attention_mask=inputs["query_attention_mask"][
                    i : i + max_batch_size
                ].to(self._device),
            )
            query_embeds = self.query_projection(query_outputs[:, 0])
            query_embeds = query_embeds / query_embeds.norm(dim=-1, keepdim=True)
            results.append(query_embeds.cpu())
        return torch.cat(results, dim=0)

    def get_doc_embeds(
        self,
        texts: Union[str, List[str]],
        max_batch_size: Optional[int] = 512,
    ):
        if isinstance(texts, str):
            texts = [texts]

        inputs = [
            self.processor._text_classification(text, prefix="doc_").dict()
            for text in texts
        ]
        keys = inputs[0].keys()
        inputs = {key: torch.stack([inp[key] for inp in inputs]) for key in keys}
        results = []
        for i in range(0, len(texts), max_batch_size):
            doc_outputs = self.doc_encoder(
                input_ids=inputs["doc_input_ids"][i : i + max_batch_size].to(
                    self._device
                ),
                attention_mask=inputs["doc_attention_mask"][i : i + max_batch_size].to(
                    self._device
                ),
            )
            doc_embeds = self.doc_projection(doc_outputs[:, 0])
            doc_embeds = doc_embeds / doc_embeds.norm(dim=-1, keepdim=True)
            results.append(doc_embeds.cpu())
        return torch.cat(results, dim=0)

    def rank_top(self, phrases, phrase_pairs_embeds, reverse=True):
        batch, num = len(phrases), len(phrases[0])
        if self.use_query_encoder:
            phrase_embeds = self.get_query_embeds(phrases)
        else:
            phrase_embeds = self.get_doc_embeds(phrases)

        scores = torch.einsum(
            "bpi,bi->bp",
            phrase_embeds.reshape(batch, num, -1),
            phrase_pairs_embeds.reshape(batch, -1),
        )
        if reverse:
            score, index = scores.max(dim=1)
        else:
            score, index = scores.min(dim=1)
        return [phrases[i][index[i]] for i in range(batch)], score

    def get_neighbor(self, phrases, topk=1024):
        if self.use_query_encoder:
            phrase_embeds = self.get_query_embeds(phrases)
        else:
            phrase_embeds = self.get_doc_embeds(phrases)

        scores = torch.einsum("bi,pj->bp", phrase_embeds, self.intent_phrases_embeds)
        _, index = scores.topk(topk, dim=1)
        return [[self.intent_phrases[j] for j in index[i]] for i in range(len(phrases))]

    def interrogator(
        self, phrases, phrase_pairs, min_count=3, max_count=10, reverse=False
    ):
        if self.use_query_encoder:
            phrase_pairs_embeds = self.get_doc_embeds(phrase_pairs)
        else:
            phrase_pairs_embeds = self.get_query_embeds(phrase_pairs)
        best_scores = torch.zeros(len(phrases), dtype=torch.float32)

        neighbors = self.get_neighbor(phrases)
        phrases = [phrase for phrase in phrases]

        for i in range(max_count):
            new_phrases = [
                [phrase + ", " + nbr for nbr in neighbor]
                for phrase, neighbor in zip(phrases, neighbors)
            ]
            new_phrases, scores = self.rank_top(
                new_phrases, phrase_pairs_embeds, reverse=reverse
            )
            if i < min_count:
                phrases = new_phrases
                best_scores = scores
            else:
                for j in range(len(phrases)):
                    if scores[j] > best_scores[j]:
                        phrases[j] = new_phrases[j]
                        best_scores[j] = scores[j]
        return phrases, best_scores.tolist()

    def get_positive_phrases(self, phrases, phrase_pairs, min_count=3, max_count=10):
        return self.interrogator(
            phrases,
            phrase_pairs,
            min_count=min_count,
            max_count=max_count,
            reverse=True,
        )

    def get_negative_phrases(self, phrases, phrase_pairs, min_count=3, max_count=10):
        return self.interrogator(
            phrases,
            phrase_pairs,
            min_count=min_count,
            max_count=max_count,
            reverse=False,
        )


@register_script("microsoft/omnigpt/script/intents/interrogator/bletchley")
class BletchleyInterrogatorScript(GenericScript):
    def __init__(self, config: CoreConfigureParser):
        self.config = config

    def launch(self, **kwargs):
        config = self.config

        pipe = BletchleyForTextInterrogator.from_core_configure(config)

        config.set_default_section(
            "microsoft/omnigpt/script/intents/interrogator/bletchley"
        )

        data_file = config.getoption("data_file", None)
        names = config.getoption("names", None)
        if isinstance(names, str) and names.strip() == "*":
            names = None
        if isinstance(names, str):
            names = re.split(r"[,;]", names)
            names = [n.strip() for n in names]

        text_col = config.getoption("text_col", "phrase")
        pair_col = config.getoption("pair_col", "phrase")

        data = pd.read_csv(
            data_file,
            names=names,
            sep="\t",
            quoting=3,
            header=None,
        )

        assert text_col in data.columns and pair_col in data.columns

        min_count = config.getoption("min_count", 3)
        max_count = config.getoption("max_count", 10)

        positive_phrases, positive_scores = pipe.get_positive_phrases(
            data[text_col], data[pair_col], min_count, max_count
        )
        data["positive_phrases"] = positive_phrases
        data["positive_scores"] = positive_scores
        negative_phrases, negative_scores = pipe.get_negative_phrases(
            data[text_col], data[pair_col], min_count, max_count
        )
        data["negative_phrases"] = negative_phrases
        data["negative_scores"] = negative_scores

        output_file = config.getoption("output_file", "./output.txt")
        data.to_csv(
            output_file,
            sep="\t",
            index=False,
            quoting=3,
            header=None,
        )
