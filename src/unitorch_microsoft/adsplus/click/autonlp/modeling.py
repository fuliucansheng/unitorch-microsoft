# Copyright (c) MICROSOFT.
# Licensed under the MIT License.
import torch
import torch.nn as nn
from torch import autocast
from unitorch.models import GenericModel
from unitorch.cli import (
    add_default_section_for_init,
    add_default_section_for_function,
    register_model,
)
from unitorch.cli.models import ClassificationOutputs

from unitorch_microsoft.adsplus.click.autonlp.modeling_utils import NASADRModel


@register_model("microsoft/adsplus/click/classification/autonlp")
class TribertForClassification(GenericModel):
    prefix_keys_in_state_dict = {
        "^(?!model\.)": "model.",
    }

    def __init__(
        self,
        max_n_letters=20,
    ):
        super().__init__()
        self.model = NASADRModel()
        self.max_n_letters = max_n_letters

    @classmethod
    @add_default_section_for_init("microsoft/adsplus/click/classification/autonlp")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/adsplus/click/classification/autonlp")
        max_n_letters = config.getoption("max_n_letters", 20)
        inst = cls(max_n_letters=max_n_letters)
        weight_path = config.getoption("pretrained_weight_path", None)
        if weight_path is not None:
            inst.from_pretrained(weight_path)
        return inst

    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(
        self,
        query_input_ids,
        query_attention_mask,
        ads_input_ids,
        ads_attention_mask,
        doc_input_ids,
        doc_attention_mask,
    ):
        batch_size = query_input_ids.size(0)

        query_count = (
            query_input_ids.view(batch_size, -1, self.max_n_letters).ne(0).sum(dim=-1)
        )
        doc_count = (
            doc_input_ids.view(batch_size, -1, self.max_n_letters).ne(0).sum(dim=-1)
        )
        ads_count = (
            ads_input_ids.view(batch_size, -1, self.max_n_letters).ne(0).sum(dim=-1)
        )
        query_position_ids = torch.arange(
            query_input_ids.size(1) // self.max_n_letters,
            dtype=torch.long,
            device=query_input_ids.device,
        )
        doc_position_ids = torch.arange(
            doc_input_ids.size(1) // self.max_n_letters,
            dtype=torch.long,
            device=doc_input_ids.device,
        )
        ads_position_ids = torch.arange(
            ads_input_ids.size(1) // self.max_n_letters,
            dtype=torch.long,
            device=ads_input_ids.device,
        )
        logits, _, _, _ = self.model(
            query_input_ids,
            doc_input_ids,
            ads_input_ids,
            query_position_ids,
            doc_position_ids,
            ads_position_ids,
            query_attention_mask,
            doc_attention_mask,
            ads_attention_mask,
            query_count,
            doc_count,
            ads_count,
        )
        logits += 8.8673
        return ClassificationOutputs(outputs=logits.unsqueeze(-1))
