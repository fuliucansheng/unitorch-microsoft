# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from torch import autocast
from PIL import Image
from open_clip.factory import image_transform_v2, PreprocessCfg
from open_clip.tokenizer import get_clean_fn
from transformers import PreTrainedTokenizerFast
from peft import LoraConfig
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.models import GenericModel
from unitorch.utils import read_json_file
from unitorch.models.peft import (
    GenericPeftModel,
    PeftModelForSequenceClassification,
    PeftWeightLoaderMixin,
)
from unitorch.cli import cached_path
from unitorch.cli import (
    add_default_section_for_init,
    add_default_section_for_function,
    register_model,
    register_process,
)
from unitorch.cli import WriterMixin, WriterOutputs
from unitorch.cli.models import TensorsInputs, ClassificationOutputs

os.environ.setdefault("HOME", "/home/decu")


class TuringMMV3Model(GenericModel):
    def __init__(
        self,
        pretrained_weight_path: str,
    ):
        super().__init__()
        self.model = open_clip.create_model(
            model_name="ViT-B-16-SigLIP-i18n-256", pretrained=pretrained_weight_path
        )

    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        image_embeds = self.model.encode_image(pixel_values, normalize=True)
        text_embeds = self.model.encode_text(input_ids, normalize=True)
        return (text_embeds, image_embeds)


@register_model("microsoft/picasso/model/turingmm/v3")
class TuringMMV3ForMatching(GenericModel):
    def __init__(
        self,
        pretrained_weight_path: str,
    ):
        super().__init__()
        self.model = open_clip.create_model(
            model_name="ViT-B-16-SigLIP-i18n-256", pretrained=pretrained_weight_path
        )
        self.classifier = nn.Linear(1, 1)
        self.classifier.weight.data.fill_(5.0)

    @classmethod
    @add_default_section_for_init("microsoft/picasso/model/turingmm/v3")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/picasso/model/turingmm/v3")
        pretrained_weight_path = config.getoption(
            "pretrained_weight_path",
            "https://huggingface.co/datasets/fuliucansheng/unitorchblobfuse/resolve/main/models/pytorch_model.turingmm_v3.bin",
        )
        pretrained_weight_path = cached_path(pretrained_weight_path)
        inst = cls(
            pretrained_weight_path=pretrained_weight_path,
        )

        return inst

    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(
        self,
        input_ids: torch.Tensor = None,
        pixel_values: torch.Tensor = None,
    ):
        image_embeds = self.model.encode_image(pixel_values, normalize=True)
        text_embeds = self.model.encode_text(input_ids, normalize=True)
        scores = torch.sum(text_embeds * image_embeds, dim=-1, keepdim=True)

        outputs = self.classifier(scores)
        return ClassificationOutputs(outputs=outputs)


@register_model("microsoft/picasso/model/turingmm/v3/lora")
class TuringMMV3LoraForMatching(GenericPeftModel, PeftWeightLoaderMixin):
    prefix_keys_in_state_dict = {
        "^(?!peft_model\.base_model\.model\.).*": "peft_model.base_model.model."
    }
    replace_keys_in_state_dict = {
        "out_proj.weight": "out_proj.base_layer.weight",
        "out_proj.bias": "out_proj.base_layer.bias",
    }
    modules_to_save_checkpoints = ["lora", "classifier"]
    replace_keys_in_peft_state_dict = {
        ".weight": ".base_layer.weight",
        ".bias": ".base_layer.bias",
    }

    def __init__(
        self,
        pretrained_weight_path: str,
        lora_r: Optional[int] = 16,
        lora_alpha: Optional[int] = 32,
        lora_dropout: Optional[float] = 0.05,
        fan_in_fan_out: Optional[bool] = True,
        target_modules: Optional[Union[List[str], str]] = ["out_proj"],
    ):
        super().__init__()
        self.peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            target_modules=target_modules,
        )
        self.peft_model = PeftModelForSequenceClassification(
            TuringMMV3Model(pretrained_weight_path=pretrained_weight_path),
            self.peft_config,
        )
        self.classifier = nn.Linear(1, 1)
        self.classifier.weight.data.fill_(5.0)

    @classmethod
    @add_default_section_for_init("microsoft/picasso/model/turingmm/v3/lora")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/picasso/model/turingmm/v3/lora")
        lora_r = config.getoption("lora_r", 16)
        lora_alpha = config.getoption("lora_alpha", 32)
        lora_dropout = config.getoption("lora_dropout", 0.05)
        fan_in_fan_out = config.getoption("fan_in_fan_out", True)
        target_modules = config.getoption("target_modules", ["out_proj"])
        pretrained_weight_path = config.getoption(
            "pretrained_weight_path",
            "https://huggingface.co/datasets/fuliucansheng/unitorchblobfuse/resolve/main/models/pytorch_model.turingmm_v3.bin",
        )

        pretrained_weight_path = cached_path(pretrained_weight_path)
        inst = cls(
            pretrained_weight_path=pretrained_weight_path,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            target_modules=target_modules,
        )
        pretrained_lora_weight_path = config.getoption(
            "pretrained_lora_weight_path", None
        )
        pretrained_lora_weight = config.getoption("pretrained_lora_weight", 1.0)
        pretrained_lora_alpha = config.getoption("pretrained_lora_alpha", 32.0)
        if pretrained_lora_weight_path is not None:
            inst.load_lora_weights(
                pretrained_lora_weight_path,
                lora_weights=pretrained_lora_weight,
                lora_alphas=pretrained_lora_alpha,
                save_base_state=False,
            )

        return inst

    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
    ):
        text_embeds, image_embeds = self.peft_model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            return_dict=False,
        )
        scores = torch.sum(text_embeds * image_embeds, dim=-1, keepdim=True)

        outputs = self.classifier(scores)
        return ClassificationOutputs(outputs=outputs)


class TuringMMV3Processor:
    def __init__(
        self,
    ):
        self.preprocess = image_transform_v2(
            PreprocessCfg(
                **{
                    "size": 256,
                    "mean": [0.5, 0.5, 0.5],
                    "std": [0.5, 0.5, 0.5],
                    "interpolation": "bicubic",
                    "resize_mode": "squash",
                }
            ),
            is_train=False,
        )
        # special_tokens = read_json_file(cached_path("https://huggingface.co/timm/ViT-B-16-SigLIP-i18n-256/resolve/main/special_tokens_map.json"))
        # tokenizer_config = read_json_file(cached_path("https://huggingface.co/timm/ViT-B-16-SigLIP-i18n-256/resolve/main/tokenizer_config.json"))
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=cached_path(
                "https://huggingface.co/timm/ViT-B-16-SigLIP-i18n-256/resolve/main/tokenizer.json"
            ),
            eos_token="</s>",
            pad_token="</s>",
            unk_token="<unk>",
        )
        self.clean_fn = get_clean_fn("whitespace")

    @classmethod
    @add_default_section_for_init("microsoft/picasso/process/turingmm/v3")
    def from_core_configure(cls, config, **kwargs):
        pass

    @register_process("microsoft/picasso/process/turingmm/v3/classification")
    def _classification(self, text, image):
        pixel_values = self.preprocess(image)
        input_ids = self.tokenizer.encode_plus(
            self.clean_fn(text),
            return_tensors="pt",
            max_length=64,
            padding="max_length",
            truncation=True,
        ).input_ids.squeeze(0)
        # input_ids = self.tokenizer(text, context_length=64).squeeze(0)
        return TensorsInputs(
            input_ids=input_ids,
            pixel_values=pixel_values,
        )
