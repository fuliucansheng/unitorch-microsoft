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
from unitorch_microsoft.models.bletchley.modeling_v1 import (
    BletchleyForPretrain as _BletchleyForPretrain,
)
from unitorch_microsoft.models.bletchley.modeling_v1 import (
    get_bletchley_text_config,
    get_bletchley_image_config,
)
from unitorch_microsoft.models.bletchley.processing_v1 import BletchleyProcessor

try:
    import faiss
except ImportError:
    raise ImportError(
        "Please install faiss to use BletchleyAli1688ImageSelectionPipeline. "
        "You can install it with `pip install faiss-cpu`."
    )

pretrained_bletchley_infos = {
    "0.3B": "https://unitorchazureblob.blob.core.windows.net/shares/models/bletchley/v1/pytorch_model.0.3B.bin",
    "0.8B": "https://unitorchazureblob.blob.core.windows.net/shares/models/bletchley/v1/pytorch_model.0.8B.bin",
    "2.5B": "https://unitorchazureblob.blob.core.windows.net/shares/models/bletchley/v1/pytorch_model.2.5B.bin",
}


class BletchleyAli1688ImageSelectionPipeline(_BletchleyForPretrain):
    def __init__(
        self,
        config_type: str,
        data_file: str,
        names: Optional[Union[str, List[str]]] = "*",
        emb_col: Optional[str] = None,
        show_cols: Optional[Union[str, List[str]]] = None,
        image_cols: Optional[Union[str, List[str]]] = None,
        max_seq_length: Optional[int] = 120,
        weight_path: Optional[Union[str, List[str]]] = None,
        state_dict: Optional[Dict[str, Any]] = None,
        device: Optional[Union[str, int]] = "cpu",
    ):
        projection_dim = get_bletchley_text_config(config_type).global_vector_size
        super().__init__(
            config_type=config_type,
            projection_dim=projection_dim,
        )
        self.processor = BletchleyProcessor(
            max_seq_length=max_seq_length,
        )
        self._device = "cpu" if device == "cpu" else int(device)

        self.from_pretrained(weight_path, state_dict=state_dict)
        self.to(device=self._device)

        if isinstance(names, str) and names == "*":
            names = None
        if isinstance(names, str):
            names = re.split(r"[,;]", names)
            names = [n.strip() for n in names]

        self.dataset = pd.read_csv(
            data_file,
            names=names,
            header="infer" if names is None else None,
            sep="\t",
            quoting=3,
        )
        self.dataset["__index__"] = range(len(self.dataset))
        assert (
            emb_col is not None and emb_col in self.dataset.columns
        ), f"emb_col {emb_col} not found in dataset"

        if show_cols is not None:
            if isinstance(show_cols, str):
                show_cols = re.split(r"[,;]", show_cols)
                show_cols = [n.strip() for n in show_cols]
            assert all(
                [col in self.dataset.columns for col in show_cols]
            ), f"show_cols {show_cols} not found in dataset"
        else:
            show_cols = list(self.dataset.columns)
            show_cols.remove("__index__")
            show_cols.remove(emb_col)

        self.show_cols = show_cols

        if image_cols is not None:
            if isinstance(image_cols, str):
                image_cols = re.split(r"[,;]", image_cols)
                image_cols = [n.strip() for n in image_cols]
            assert all(
                [col in self.dataset.columns for col in image_cols]
            ), f"image_cols {image_cols} not found in dataset"
            self.image_cols = image_cols
            for image_col in image_cols:
                self.dataset[image_col] = self.dataset[image_col].map(
                    lambda x: f'<img src="{x}" width="100%">'
                )

        self.faiss_index = faiss.IndexFlatIP(projection_dim)
        for _, row in self.dataset.iterrows():
            emb = row[emb_col]
            if isinstance(emb, str):
                emb = re.split(r"[,; ]", emb)
                emb = [float(e) for e in emb]
            self.faiss_index.add(np.array(emb).reshape(1, -1))

    @classmethod
    @add_default_section_for_init("microsoft/china/pipeline/ali1688")
    def from_core_configure(
        cls,
        config,
        config_type: Optional[str] = "2.5B",
        data_file: Optional[str] = None,
        names: Optional[Union[str, List[str]]] = "*",
        emb_col: Optional[str] = None,
        show_cols: Optional[Union[str, List[str]]] = None,
        pretrained_weight_path: Optional[str] = None,
        device: Optional[str] = "cpu",
        **kwargs,
    ):
        config.set_default_section("microsoft/china/pipeline/ali1688")
        config_type = config.getoption("config_type", config_type)

        data_file = config.getoption("data_file", data_file)
        names = config.getoption("names", names)
        emb_col = config.getoption("emb_col", emb_col)
        show_cols = config.getoption("show_cols", show_cols)
        image_cols = config.getoption("image_cols", None)
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

        inst = cls(
            config_type,
            data_file=data_file,
            names=names,
            emb_col=emb_col,
            show_cols=show_cols,
            image_cols=image_cols,
            max_seq_length=max_seq_length,
            weight_path=weight_path,
            device=device,
        )

        return inst

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
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        return image_embeds

    def get_text_embeds(self, text: str):
        inputs = self.processor._text_classification(text)
        inputs = {
            k: v.unsqueeze(0) if v is not None else v for k, v in inputs.dict().items()
        }
        inputs = {
            k: v.to(device=self._device) if v is not None else v
            for k, v in inputs.items()
        }
        text_outputs = self.text_encoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        text_embeds = self.text_projection(text_outputs[:, 0])
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        return text_embeds

    @torch.no_grad()
    @add_default_section_for_function("microsoft/china/pipeline/ali1688")
    def __call__(
        self,
        text: Optional[str] = None,
        image: Optional[Image.Image] = None,
        topk: Optional[int] = 50,
    ):
        assert text is not None or image is not None, "text or image must be provided"
        if text is not None:
            text_embeds = self.get_text_embeds(text)
            text_embeds = text_embeds.cpu().numpy().reshape(1, -1)
            dists, indices = self.faiss_index.search(text_embeds, topk)
        else:
            image_embeds = self.get_image_embeds(image)
            image_embeds = image_embeds.cpu().numpy().reshape(1, -1)
            dists, indices = self.faiss_index.search(image_embeds, topk)

        dists_maps = {i: d for i, d in zip(indices[0], dists[0])}
        results = self.dataset[self.dataset["__index__"].isin(dists_maps.keys())]
        results["similarity"] = results["__index__"].map(dists_maps)
        results.sort_values("similarity", inplace=True, ascending=False)
        return results[self.show_cols + ["similarity"]]
