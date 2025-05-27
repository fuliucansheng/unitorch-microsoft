# Copyright (c) MICROSOFT.
# Licensed under the MIT License.
import re
import torch
import numpy as np
import pandas as pd
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.models import GenericOutputs
from unitorch.utils import pop_value, nested_dict_value, read_file, read_json_file

from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
)
from unitorch_microsoft.models.bletchley.modeling_v3 import (
    BletchleyForPretrain as _BletchleyForPretrain,
)
from unitorch_microsoft.models.bletchley.processing_v3 import BletchleyProcessor

try:
    import faiss
except ImportError:
    raise ImportError(
        "Please install faiss to use BletchleyImageTaggerSelectionPipeline. "
        "You can install it with `pip install faiss-cpu`."
    )

pretrained_bletchley_infos = {
    "0.8B": "https://huggingface.co/datasets/fuliucansheng/unitorchblobfuse/resolve/main/models/bletchley/v3/pytorch_model.base.bin",
    "2.5B": "https://huggingface.co/datasets/fuliucansheng/unitorchblobfuse/resolve/main/models/bletchley/v3/pytorch_model.large.bin",
}


class BletchleyImageTaggerSelectionPipeline(_BletchleyForPretrain):
    def __init__(
        self,
        config_type: str,
        phrase_file: str,
        max_seq_length: Optional[int] = 120,
        projection_dim: Optional[int] = None,
        output_embed_dim: Optional[int] = None,
        weight_path: Optional[Union[str, List[str]]] = None,
        lora_weight_path: Optional[Union[str, List[str]]] = None,
        lora_weight: Optional[float] = 1.0,
        lora_alpha: Optional[float] = 32.0,
        state_dict: Optional[Dict[str, Any]] = None,
        device: Optional[Union[str, int]] = "cpu",
    ):
        if projection_dim is None:
            projection_dim = 1024 if config_type == "2.5B" else 768
        super().__init__(
            config_type=config_type,
            projection_dim=projection_dim,
            output_embed_dim=output_embed_dim,
        )
        self.processor = BletchleyProcessor(
            max_seq_length=max_seq_length,
        )
        self._device = "cpu" if device == "cpu" else int(device)

        self.from_pretrained(weight_path, state_dict=state_dict)

        if lora_weight_path is not None:
            self.load_lora_weights(
                lora_weight_path,
                lora_weights=lora_weight,
                lora_alphas=lora_alpha,
                save_base_state=False,
            )

        self.to(device=self._device)

        self.phrases = pd.read_csv(
            phrase_file,
            names=["phrase"],
            header=None,
            sep="\t",
            quoting=3,
        ).drop_duplicates()
        self.phrases["__index__"] = range(len(self.phrases))

        phrase_embeds = self.get_phrase_embeds(self.phrases["phrase"])
        faiss_emb_dim = (
            projection_dim if self.output_projection is None else output_embed_dim
        )

        self.faiss_index = faiss.IndexFlatIP(faiss_emb_dim)
        for emb in phrase_embeds:
            self.faiss_index.add(np.array(emb).reshape(1, -1))

    @classmethod
    @add_default_section_for_init("microsoft/pipeline/selection/image/tagger")
    def from_core_configure(
        cls,
        config,
        config_type: Optional[str] = "0.8B",
        phrase_file: Optional[str] = None,
        projection_dim: Optional[int] = None,
        output_embed_dim: Optional[int] = None,
        pretrained_weight_path: Optional[str] = None,
        pretrained_lora_weight_path: Optional[str] = None,
        device: Optional[str] = "cpu",
        **kwargs,
    ):
        config.set_default_section("microsoft/pipeline/selection/image/tagger")
        config_type = config.getoption("config_type", config_type)

        phrase_file = config.getoption("phrase_file", phrase_file)
        projection_dim = config.getoption("projection_dim", projection_dim)
        output_embed_dim = config.getoption("output_embed_dim", output_embed_dim)
        max_seq_length = config.getoption("max_seq_length", 77)
        device = config.getoption("device", device)
        pretrained_weight_path = config.getoption(
            "pretrained_weight_path", pretrained_weight_path
        )
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_bletchley_infos, config_type),
            check_none=False,
        )

        lora_weight_path = config.getoption(
            "pretrained_lora_weight_path", pretrained_lora_weight_path
        )
        lora_weight = config.getoption("pretrained_lora_weight", 1.0)
        lora_alpha = config.getoption("pretrained_lora_alpha", 32.0)

        inst = cls(
            config_type,
            phrase_file=phrase_file,
            projection_dim=projection_dim,
            output_embed_dim=output_embed_dim,
            max_seq_length=max_seq_length,
            weight_path=weight_path,
            lora_weight_path=lora_weight_path,
            lora_weight=lora_weight,
            lora_alpha=lora_alpha,
            device=device,
        )

        return inst

    @torch.no_grad()
    def get_image_embeds(self, image: Image.Image):
        inputs = self.processor._image_classification(image)
        inputs = {
            k: v.unsqueeze(0) if v is not None else v for k, v in inputs.dict().items()
        }
        inputs = {
            k: v.to(device=self._device) if v is not None else v
            for k, v in inputs.items()
        }
        image_outputs = self.image_encoder(inputs["images"])
        image_embeds = self.image_projection(image_outputs[:, 0])
        if self.output_projection is not None:
            image_embeds = self.output_projection(image_embeds)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        return image_embeds

    @torch.no_grad()
    def get_phrase_embeds(self, texts: Union[str, List[str]], max_batch_size=128):
        if isinstance(texts, str):
            texts = [texts]
        inputs = [self.processor._text_classification(text).dict() for text in texts]
        keys = inputs[0].keys()
        inputs = {
            k: torch.stack([i[k] for i in inputs]).to(device=self._device) for k in keys
        }
        results = []
        for i in range(0, len(inputs["input_ids"]), max_batch_size):
            text_outputs = self.text_encoder(
                inputs["input_ids"][i : i + max_batch_size],
                inputs["attention_mask"][i : i + max_batch_size],
            )
            text_embeds = self.text_projection(text_outputs[:, 0])
            if self.output_projection is not None:
                text_embeds = self.output_projection(text_embeds)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            results.append(text_embeds.cpu())
        return torch.cat(results, dim=0)

    @torch.no_grad()
    @add_default_section_for_function("microsoft/pipeline/selection/image/tagger")
    def __call__(
        self,
        image,
        topk: Optional[int] = 10,
    ):
        assert image is not None, "image must be provided"
        image_embeds = self.get_image_embeds(image)
        image_embeds = image_embeds.cpu().numpy().reshape(1, -1)
        dists, indices = self.faiss_index.search(image_embeds, topk)

        dists_maps = {i: d for i, d in zip(indices[0], dists[0])}
        results = self.phrases[self.phrases["__index__"].isin(dists_maps.keys())]
        results["similarity"] = results["__index__"].map(dists_maps)
        results.sort_values("similarity", inplace=True, ascending=False)
        result = {p: s for p, s in zip(results["phrase"], results["similarity"])}
        return result
