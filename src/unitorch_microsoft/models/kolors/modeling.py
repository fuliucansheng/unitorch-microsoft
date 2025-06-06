# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import torch
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast
from transformers import CLIPModel
from transformers import CLIPConfig
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.models import GenericModel, GenericOutputs
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models.kolors.modeling import CrossModel
from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
    register_model,
)
from unitorch.cli.models import ClassificationOutputs, LossOutputs
from unitorch.cli.models.kolors import pretrained_kolors_infos


@register_model("microsoft/model/classification/kolors/mps")
class KolorsMPSModel(GenericModel):
    """CLIP model for classification."""

    def __init__(self, config_path: str):
        """
        Initialize the ClipForClassification model.

        Args:
            config_path (str): The path to the model configuration file.
            projection_dim (int, optional): The dimension of the projection head. Defaults to 512.
            num_classes (int, optional): The number of output classes. Defaults to 1.
            freeze_base_model (bool, optional): Whether to freeze the base model. Defaults to True.
            gradient_checkpointing (bool, optional): Whether to use gradient checkpointing. Defaults to False.
        """
        super().__init__()
        self.config = CLIPConfig.from_json_file(config_path)
        self.model = CLIPModel(self.config)
        self.cross_model = CrossModel(dim=1024, layer_num=4, heads=16)

    @classmethod
    @add_default_section_for_init("microsoft/model/classification/kolors/mps")
    def from_core_configure(cls, config, **kwargs):
        """
        Create an instance of ClipForClassification from a core configuration.

        Args:
            config: The core configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            ClipForClassification: An instance of the ClipForClassification model.
        """
        config.set_default_section("microsoft/model/classification/kolors/mps")
        pretrained_name = config.getoption("pretrained_name", "kolors-mps-overall")
        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_kolors_infos, pretrained_name, "config"),
        )

        config_path = cached_path(config_path)

        inst = cls(
            config_path=config_path,
        )
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            pretrained_weight_path,
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
    ):
        """
        Perform a forward pass through the model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            pixel_values (torch.Tensor): Input pixel values.
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            position_ids (torch.Tensor, optional): Position IDs. Defaults to None.

        Returns:
            ClassificationOutputs: The classification outputs.
        """
        text_outputs = self.model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        text_features = self.model.text_projection(text_outputs[0])
        text_pooled_features = self.model.text_projection(text_outputs[1])

        image_outputs = self.model.vision_model(pixel_values=pixel_values)
        image_features = self.model.visual_projection(image_outputs[0])

        condition_outputs = self.model.text_model(
            input_ids=condition_input_ids,
            attention_mask=condition_attention_mask,
            position_ids=condition_position_ids,
        )
        condition_features = self.model.text_projection(condition_outputs[0])

        sim_text_condition = torch.einsum(
            "b i d, b j d -> b j i", text_features, condition_features
        )
        sim_text_condition = torch.max(sim_text_condition, dim=1, keepdim=True)[0]
        sim_text_condition = sim_text_condition / sim_text_condition.max()
        mask = torch.where(sim_text_condition > 0.01, 0, float("-inf"))  # B*1*77

        mask = mask.repeat(1, image_features.shape[1], 1)  # B*257*77
        cross_features = self.cross_model(image_features, text_features, mask)[:, 0]

        text_embeds = text_pooled_features / text_pooled_features.norm(
            dim=-1, keepdim=True
        )
        cross_embeds = cross_features / cross_features.norm(dim=-1, keepdim=True)
        scores = torch.sum(text_embeds * cross_embeds, dim=-1, keepdim=True)
        scores = (1 + scores) / 2

        return ClassificationOutputs(outputs=scores)
