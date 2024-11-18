# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import re
import torch
import pandas as pd
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from sklearn.metrics import roc_auc_score
from unitorch.models import GenericOutputs
from unitorch.models.clip import (
    ClipForPretrain as _ClipForPretrain,
    ClipProcessor,
)
from unitorch.utils import pop_value, nested_dict_value, read_file, read_json_file

from unitorch.cli import (
    hf_endpoint_url,
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
)
from unitorch.cli import CoreConfigureParser, GenericScript
from unitorch.cli import register_script
from unitorch.cli.models.clip import pretrained_clip_infos


class ClipInterrogatorPipeline(_ClipForPretrain):
    def __init__(
        self,
        config_path: str,
        vocab_path: str,
        merge_path: str,
        vision_config_path: str,
        max_seq_length: Optional[int] = 77,
        weight_path: Optional[Union[str, List[str]]] = None,
        state_dict: Optional[Dict[str, Any]] = None,
        device: Optional[Union[str, int]] = "cpu",
    ):
        projection_dim = nested_dict_value(
            read_json_file(config_path), "projection_dim"
        )
        super().__init__(
            config_path=config_path,
            projection_dim=projection_dim,
        )
        self.processor = ClipProcessor(
            vocab_path=vocab_path,
            merge_path=merge_path,
            vision_config_path=vision_config_path,
            max_seq_length=max_seq_length,
        )
        self._device = "cpu" if device == "cpu" else int(device)

        self.from_pretrained(weight_path, state_dict=state_dict)
        self.to(device=self._device)

        artists = read_file(
            cached_path(
                hf_endpoint_url(
                    "/datasets/fuliucansheng/hubfiles/raw/main/clip-interrogator/artists.txt"
                )
            ),
            lines=True,
        )
        flavors = read_file(
            cached_path(
                hf_endpoint_url(
                    "/datasets/fuliucansheng/hubfiles/raw/main/clip-interrogator/flavors.txt"
                )
            ),
            lines=True,
        )
        mediums = read_file(
            cached_path(
                hf_endpoint_url(
                    "/datasets/fuliucansheng/hubfiles/raw/main/clip-interrogator/mediums.txt"
                )
            ),
            lines=True,
        )
        movements = read_file(
            cached_path(
                hf_endpoint_url(
                    "/datasets/fuliucansheng/hubfiles/raw/main/clip-interrogator/movements.txt"
                )
            ),
            lines=True,
        )
        negative = read_file(
            cached_path(
                hf_endpoint_url(
                    "/datasets/fuliucansheng/hubfiles/raw/main/clip-interrogator/negative.txt"
                )
            ),
            lines=True,
        )
        sites = read_file(
            cached_path(
                hf_endpoint_url(
                    "/datasets/fuliucansheng/hubfiles/raw/main/clip-interrogator/sites.txt"
                )
            ),
            lines=True,
        )

        artists = [f"by {a}" for a in artists] + [f"inspired by {a}" for a in artists]
        trendings = (
            sites
            + [f"trending on {s}" for s in sites]
            + [f"featured on {s}" for s in sites]
            + [f"{s} contest winner" for s in sites]
        )

        self.positive_labels = artists + flavors + mediums + movements + trendings
        self.positive_artists_labels = artists
        self.positive_flavors_labels = flavors
        self.positive_mediums_labels = mediums
        self.positive_movements_labels = movements
        self.positive_trendings_labels = trendings
        self.negative_labels = negative

        self.eval()

        self.positive_labels_embeds = self.get_text_embeds(self.positive_labels)

    @classmethod
    @add_default_section_for_init("microsoft/omnilora/interrogator/clip")
    def from_core_configure(
        cls,
        config,
        pretrained_name: str = None,
        config_path: Optional[str] = None,
        vocab_path: Optional[str] = None,
        merge_path: Optional[str] = None,
        vision_config_path: Optional[str] = None,
        pretrained_weight_path: Optional[str] = None,
        device: Optional[str] = "cpu",
        **kwargs,
    ):
        config.set_default_section("microsoft/omnilora/interrogator/clip")
        pretrained_name = config.getoption("pretrained_name", pretrained_name)

        config_path = config.getoption("config_path", config_path)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_clip_infos, pretrained_name, "config"),
        )
        config_path = cached_path(config_path)

        vocab_path = config.getoption("vocab_path", vocab_path)
        vocab_path = pop_value(
            vocab_path,
            nested_dict_value(pretrained_clip_infos, pretrained_name, "vocab"),
        )
        vocab_path = cached_path(vocab_path)

        merge_path = config.getoption("merge_path", merge_path)
        merge_path = pop_value(
            merge_path,
            nested_dict_value(pretrained_clip_infos, pretrained_name, "merge"),
        )
        merge_path = cached_path(merge_path)

        vision_config_path = config.getoption("vision_config_path", vision_config_path)
        vision_config_path = pop_value(
            vision_config_path,
            nested_dict_value(pretrained_clip_infos, pretrained_name, "vision_config"),
        )
        vision_config_path = cached_path(vision_config_path)

        max_seq_length = config.getoption("max_seq_length", 77)
        device = config.getoption("device", device)
        pretrained_weight_path = config.getoption(
            "pretrained_weight_path", pretrained_weight_path
        )
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_clip_infos, pretrained_name, "weight"),
            check_none=False,
        )
        inst = cls(
            config_path,
            vocab_path,
            merge_path,
            vision_config_path,
            max_seq_length=max_seq_length,
            weight_path=weight_path,
            device=device,
        )

        return inst

    def get_image_embeds(self, images: List[Image.Image], max_batch_size=128):
        inputs = [self.processor.image_classification(image) for image in images]
        keys = inputs[0].keys()
        inputs = {
            k: torch.stack([i[k] for i in inputs]).to(device=self._device) for k in keys
        }
        results = []
        for i in range(0, len(inputs["pixel_values"]), max_batch_size):
            vision_outputs = self.vision_model(
                pixel_values=inputs["pixel_values"][i : i + max_batch_size]
            )
            image_embeds = self.visual_projection(vision_outputs[1])
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            results.append(image_embeds.cpu())
        return torch.cat(results, dim=0)

    def get_text_embeds(self, texts: Union[str, List[str]], max_batch_size=128):
        if isinstance(texts, str):
            texts = [texts]
        inputs = [self.processor.text_classification(text) for text in texts]
        keys = inputs[0].keys()
        inputs = {
            k: torch.stack([i[k] for i in inputs]).to(device=self._device) for k in keys
        }
        results = []
        for i in range(0, len(inputs["input_ids"]), max_batch_size):
            text_outputs = self.text_model(
                input_ids=inputs["input_ids"][i : i + max_batch_size],
                attention_mask=inputs["attention_mask"][i : i + max_batch_size],
                position_ids=inputs["position_ids"][i : i + max_batch_size],
            )
            text_embeds = self.text_projection(text_outputs[1])
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            results.append(text_embeds.cpu())
        return torch.cat(results, dim=0)

    def get_score(self, image_embeds, text, labels=None):
        text_embeds = self.get_text_embeds(text)
        scores = torch.einsum("te,be->tb", text_embeds, image_embeds)
        assert labels is not None
        return roc_auc_score(labels, scores[0].tolist())

    def rank_top(
        self,
        image_embeds,
        texts=[],
        topk=1,
        reverse=True,
        text_embeds=None,
        labels=None,
    ):
        if text_embeds is None:
            text_embeds = self.get_text_embeds(texts)
        scores = torch.einsum("te,be->tb", text_embeds, image_embeds)
        assert labels is not None
        scores = [roc_auc_score(labels, s.tolist()) for s in scores]
        return sorted(list(zip(texts, scores)), key=lambda x: x[1], reverse=reverse)[
            :topk
        ]

    def chain(
        self,
        image_embeds,
        phrases,
        caption=None,
        min_count=8,
        max_count=32,
        reverse=True,
        labels=None,
    ):
        if caption is None:
            caption = self.rank_top(
                image_embeds, phrases, topk=1, reverse=reverse, labels=labels
            )[0][0]
        best_prompt, best_score = caption, self.get_score(
            image_embeds, caption, labels=labels
        )
        if best_prompt in phrases:
            phrases.remove(best_prompt)

        prompt, score = best_prompt, best_score

        for i in range(max_count):
            new_prompt, new_score = self.rank_top(
                image_embeds,
                [f"{prompt}, {p}" for p in phrases],
                topk=1,
                reverse=reverse,
                labels=labels,
            )[0]

            label = new_prompt[len(prompt) + 2 :]
            phrases.remove(label)

            if (reverse and new_score > best_score) or (
                not reverse and new_score < best_score
            ):
                best_prompt, best_score = new_prompt, new_score
            if (
                i < min_count
                or (reverse and new_score < score)
                or (not reverse and new_score > score)
            ):
                prompt, score = new_prompt, new_score
            else:
                break

        return best_prompt

    def get_best_prompt(self, image_embeds, min_count=8, max_count=32, labels=None):
        positive_labels = self.rank_top(
            image_embeds,
            texts=self.positive_labels,
            topk=1024,
            text_embeds=self.positive_labels_embeds,
            labels=labels,
        )
        positive_labels = list(map(lambda x: x[0], positive_labels))

        best_prompt = self.chain(
            image_embeds,
            positive_labels,
            min_count=min_count,
            max_count=max_count,
            labels=labels,
        )
        return best_prompt

    def get_negative_prompt(self, image_embeds, max_count=3, labels=None):
        positive_flavors_labels_embeds = self.positive_labels_embeds[
            len(self.positive_artists_labels) : len(self.positive_artists_labels)
            + len(self.positive_flavors_labels)
        ]
        negative_labels = self.rank_top(
            image_embeds,
            texts=self.positive_flavors_labels,
            topk=max_count,
            reverse=False,
            text_embeds=positive_flavors_labels_embeds,
            labels=labels,
        )
        negative_labels = list(map(lambda x: x[0], negative_labels))
        negative_labels += self.negative_labels
        negative_prompt = self.chain(
            image_embeds,
            negative_labels,
            max_count=max_count,
            reverse=False,
            labels=labels,
        )
        return negative_prompt

    @torch.no_grad()
    def __call__(
        self,
        images: List[Image.Image],
        labels: List[int],
    ):
        image_embeds = self.get_image_embeds(images)
        best_prompt = self.get_best_prompt(image_embeds, labels=labels)
        negative_prompt = self.get_negative_prompt(image_embeds, labels=labels)
        return GenericOutputs(
            best_prompt=best_prompt,
            negative_prompt=negative_prompt,
        )


@register_script("microsoft/omnilora/script/interrogator/clip")
class ClipInterrogatorScript(GenericScript):
    def __init__(self, config: CoreConfigureParser):
        self.config = config

    def launch(self, **kwargs):
        config = self.config

        pipe = ClipInterrogatorPipeline.from_core_configure(config)

        config.set_default_section("microsoft/omnilora/script/interrogator/clip")

        data_file = config.getoption("data_file", None)
        names = config.getoption("names", None)
        if isinstance(names, str) and names.strip() == "*":
            names = None
        if isinstance(names, str):
            names = re.split(r"[,;]", names)
            names = [n.strip() for n in names]

        image_col = config.getoption("image_col", None)
        label_col = config.getoption("label_col", None)
        do_reverse = config.getoption("do_reverse", False)

        data = pd.read_csv(
            data_file,
            names=names,
            sep="\t",
            quoting=3,
            header=None,
        )

        assert image_col in data.columns, f"Column {image_col} not found in data."
        assert label_col in data.columns, f"Column {label_col} not found in data."

        images = [Image.open(image).convert("RGB") for image in data[image_col]]
        if do_reverse:
            labels = [1 - int(label) for label in data[label_col]]
        else:
            labels = [int(label) for label in data[label_col]]

        results = pipe(images=images, labels=labels)

        print("Best Prompt:", results.best_prompt)
        print("Negative Prompt:", results.negative_prompt)
