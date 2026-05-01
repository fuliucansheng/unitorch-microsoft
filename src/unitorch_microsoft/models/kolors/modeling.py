# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import gc
import torch
from torch import autocast
from transformers import CLIPConfig, CLIPModel
from typing import Optional

from unitorch.models import GenericModel
from unitorch.models.kolors.modeling import CrossModel
from unitorch.utils import nested_dict_value, pop_value
from unitorch.cli import (
    add_default_section_for_init,
    cached_path,
    register_model,
)
from unitorch.cli.models import ClassificationOutputs
from unitorch.cli.models.kolors import pretrained_kolors_infos


@register_model("microsoft/model/classification/kolors/mps")
class KolorsMPSModel(GenericModel):
    """CLIP-based multi-modal preference scoring model."""

    def __init__(self, config_path: str):
        super().__init__()
        self.config = CLIPConfig.from_json_file(config_path)
        self.model = CLIPModel(self.config)
        self.cross_model = CrossModel(dim=1024, layer_num=4, heads=16)

    @classmethod
    @add_default_section_for_init("microsoft/model/classification/kolors/mps")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/model/classification/kolors/mps")
        pretrained_name = config.getoption("pretrained_name", "kolors-mps-overall")
        config_path = pop_value(
            config.getoption("config_path", None),
            nested_dict_value(pretrained_kolors_infos, pretrained_name, "config"),
        )
        config_path = cached_path(config_path)

        inst = cls(config_path=config_path)

        weight_path = pop_value(
            config.getoption("pretrained_weight_path", None),
            nested_dict_value(pretrained_kolors_infos, pretrained_name, "weight"),
            check_none=False,
        )
        if weight_path is not None:
            inst.from_pretrained(weight_path)

        return inst

    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        condition_input_ids: torch.Tensor,
        condition_attention_mask: torch.Tensor,
        condition_position_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> ClassificationOutputs:
        text_outputs = self.model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        text_features = self.model.text_projection(text_outputs.last_hidden_state)
        text_pooled_features = self.model.text_projection(text_outputs.pooler_output)

        image_outputs = self.model.vision_model(pixel_values=pixel_values)
        image_features = self.model.visual_projection(image_outputs.last_hidden_state)

        condition_outputs = self.model.text_model(
            input_ids=condition_input_ids,
            attention_mask=condition_attention_mask,
            position_ids=condition_position_ids,
        )
        condition_features = self.model.text_projection(condition_outputs.last_hidden_state)

        # Build cross-attention mask from text-condition similarity
        sim = torch.einsum("b i d, b j d -> b j i", text_features, condition_features)
        sim = torch.max(sim, dim=1, keepdim=True)[0]
        sim = sim / sim.max()
        mask = torch.where(sim > 0.01, 0.0, float("-inf"))          # (B, 1, 77)
        mask = mask.expand(-1, image_features.shape[1], -1)          # (B, 257, 77)

        cross_features = self.cross_model(image_features, text_features, mask)[:, 0]

        text_embeds = text_pooled_features / text_pooled_features.norm(dim=-1, keepdim=True)
        cross_embeds = cross_features / cross_features.norm(dim=-1, keepdim=True)
        scores = (1 + torch.sum(text_embeds * cross_embeds, dim=-1, keepdim=True)) / 2

        return ClassificationOutputs(outputs=scores)
