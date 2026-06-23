# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from PIL import Image
from typing import Any, List, Optional, Sequence, Union

from unitorch.utils import nested_dict_value, pop_value
from unitorch.models import GenericModel
from unitorch.models.siglip import SiglipProcessor as _SiglipProcessor
from unitorch.cli import cached_path, config_defaults_init, register_model, register_process
from unitorch.cli.models import ClassificationOutputs, TensorInputs
from unitorch.cli.models.siglip import pretrained_siglip_infos
from unitorch_microsoft.models.siglip.modeling import SiglipForMatchingV2


RUBY_PRETRAINED_NAMES = [
    "siglip2-so400m-patch14-384",
    # "siglip2-large-patch16-512",
    # "siglip2-so400m-patch16-512",
    "siglip2-so400m-patch14-384",
    "siglip2-large-patch16-512",
    # "siglip2-so400m-patch16-512",
    "siglip2-so400m-patch14-384",
    "siglip2-large-patch16-512",
    "siglip2-so400m-patch16-512",
]

RUBY_LORA_WEIGHT_PATHS = [
    [
        "https://huggingface.co/datasets/fuliucansheng/unitorchblobfuse/resolve/main/models/adsplus/lora/siglip/pytorch_model.v2.lora4.badcrop.2506.bin",
        "https://unitorchazureblob.blob.core.windows.net/shares/models/adsplus/lora/siglip/pytorch_model.v2.lora4.badcrop.2602.ruby.c1.bin",
        "https://unitorchazureblob.blob.core.windows.net/shares/models/adsplus/lora/siglip/pytorch_model.v2.lora4.badcrop.2602.ruby.c2.bin",
        "https://unitorchazureblob.blob.core.windows.net/shares/models/adsplus/lora/siglip/pytorch_model.v2.lora4.badcrop.2605.ruby.r1.bin",
        "https://unitorchazureblob.blob.core.windows.net/shares/models/adsplus/lora/siglip/pytorch_model.v2.lora4.badcrop.2605.ruby.r4.bin",
    ],
    # [
    #     "https://unitorchazureblob.blob.core.windows.net/shares/models/adsplus/lora/siglip/pytorch_model.v2.lora4.badcrop.2605.ruby.r2.bin",
    #     "https://unitorchazureblob.blob.core.windows.net/shares/models/adsplus/lora/siglip/pytorch_model.v2.lora4.badcrop.2605.ruby.r5.bin",
    # ],
    # [
    #     "https://unitorchazureblob.blob.core.windows.net/shares/models/adsplus/lora/siglip/pytorch_model.v2.lora4.badcrop.2605.ruby.r3.bin",
    #     "https://unitorchazureblob.blob.core.windows.net/shares/models/adsplus/lora/siglip/pytorch_model.v2.lora4.badcrop.2605.ruby.r6.bin",
    # ],
    [
        "https://huggingface.co/datasets/fuliucansheng/unitorchblobfuse/resolve/main/models/adsplus/lora/siglip/pytorch_model.v2.lora4.badcrop.2506.bin",
        "https://unitorchazureblob.blob.core.windows.net/shares/models/adsplus/lora/siglip/pytorch_model.v2.lora4.badcrop.2602.ruby.c1.bin",
        "https://unitorchazureblob.blob.core.windows.net/shares/models/adsplus/lora/siglip/pytorch_model.v2.lora4.badcrop.2602.ruby.c2.bin",
        "https://unitorchazureblob.blob.core.windows.net/shares/models/adsplus/lora/siglip/pytorch_model.v2.lora4.badcrop.2605.ruby.r1.bin",
        "https://unitorchazureblob.blob.core.windows.net/shares/models/adsplus/lora/siglip/pytorch_model.v2.lora4.badcrop.2606.ruby.r1.bin",
    ],
    [
        "https://unitorchazureblob.blob.core.windows.net/shares/models/adsplus/lora/siglip/pytorch_model.v2.lora4.badcrop.2605.ruby.r2.bin",
        "https://unitorchazureblob.blob.core.windows.net/shares/models/adsplus/lora/siglip/pytorch_model.v2.lora4.badcrop.2606.ruby.r2.bin",
    ],
    # [
    #     "https://unitorchazureblob.blob.core.windows.net/shares/models/adsplus/lora/siglip/pytorch_model.v2.lora4.badcrop.2605.ruby.r3.bin",
    #     "https://unitorchazureblob.blob.core.windows.net/shares/models/adsplus/lora/siglip/pytorch_model.v2.lora4.badcrop.2606.ruby.r3.bin",
    # ],
    [
        "https://huggingface.co/datasets/fuliucansheng/unitorchblobfuse/resolve/main/models/adsplus/lora/siglip/pytorch_model.v2.lora4.badcrop.2506.bin",
        "https://unitorchazureblob.blob.core.windows.net/shares/models/adsplus/lora/siglip/pytorch_model.v2.lora4.badcrop.2602.ruby.c1.bin",
        "https://unitorchazureblob.blob.core.windows.net/shares/models/adsplus/lora/siglip/pytorch_model.v2.lora4.badcrop.2602.ruby.c2.bin",
        "https://unitorchazureblob.blob.core.windows.net/shares/models/adsplus/lora/siglip/pytorch_model.v2.lora4.badcrop.2605.ruby.r1.bin",
        "https://unitorchazureblob.blob.core.windows.net/shares/models/adsplus/lora/siglip/pytorch_model.v2.lora4.badcrop.2605.ruby.r4.bin",
        "https://unitorchazureblob.blob.core.windows.net/shares/models/adsplus/lora/siglip/pytorch_model.v2.lora4.badcrop.2606.ruby.r4.bin",
    ],
    [
        "https://unitorchazureblob.blob.core.windows.net/shares/models/adsplus/lora/siglip/pytorch_model.v2.lora4.badcrop.2605.ruby.r2.bin",
        "https://unitorchazureblob.blob.core.windows.net/shares/models/adsplus/lora/siglip/pytorch_model.v2.lora4.badcrop.2605.ruby.r5.bin",
        "https://unitorchazureblob.blob.core.windows.net/shares/models/adsplus/lora/siglip/pytorch_model.v2.lora4.badcrop.2606.ruby.r5.bin",
    ],
    [
        "https://unitorchazureblob.blob.core.windows.net/shares/models/adsplus/lora/siglip/pytorch_model.v2.lora4.badcrop.2605.ruby.r3.bin",
        "https://unitorchazureblob.blob.core.windows.net/shares/models/adsplus/lora/siglip/pytorch_model.v2.lora4.badcrop.2605.ruby.r6.bin",
        "https://unitorchazureblob.blob.core.windows.net/shares/models/adsplus/lora/siglip/pytorch_model.v2.lora4.badcrop.2606.ruby.r6.bin",
    ],
]
# MODEL_WEIGHTS = [0.053814054, 0, 0, 0.119015119, 0.195228195, 0, 0.054886055, 0.337091337, 0.239965240]
MODEL_WEIGHTS = [0.053814054, 0.119015119, 0.195228195, 0.054886055, 0.337091337, 0.239965240]
THRESHOLD = 0.494638283

RUBY_LORA_WEIGHTS = [[1.0] * len(paths) for paths in RUBY_LORA_WEIGHT_PATHS]
RUBY_LABELS = ["bad cropped, cut off, mutilated"]


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return value
    return [value]


def _broadcast(value: Any, size: int, name: str, default: Any = None) -> List[Any]:
    if value is None:
        return [default for _ in range(size)]
    if not isinstance(value, (list, tuple)):
        return [value for _ in range(size)]

    value = list(value)
    if len(value) == size:
        return value
    if len(value) == 1:
        return value * size
    raise ValueError(f"{name} length should be 1 or {size}, got {len(value)}.")


def _broadcast_nested(value: Any, size: int, name: str, default: Any = None) -> List[Any]:
    if value is None:
        return [default for _ in range(size)]
    if not isinstance(value, (list, tuple)):
        return [value for _ in range(size)]

    value = list(value)
    if len(value) == size:
        return value
    if len(value) == 1:
        return value * size
    return [value for _ in range(size)]


def _cached_siglip_path(pretrained_name: str, key: str, value: Optional[str] = None):
    path = pop_value(
        value,
        nested_dict_value(pretrained_siglip_infos, pretrained_name, key),
    )
    return cached_path(path)


def _build_siglip_model(
    pretrained_name: str,
    labels: Sequence[str],
    max_seq_length: int,
    freeze_base_model: bool,
    gradient_checkpointing: bool,
    config_path: Optional[str] = None,
    vocab_path: Optional[str] = None,
    vision_config_path: Optional[str] = None,
    pretrained_weight_path: Optional[Union[str, List[str]]] = None,
    pretrained_lora_weight_path: Optional[Union[str, List[str]]] = None,
    pretrained_lora_weight: Optional[Union[float, List[float]]] = None,
    pretrained_lora_alpha: Optional[Union[float, List[float]]] = 32.0,
) -> SiglipForMatchingV2:
    config_path = _cached_siglip_path(pretrained_name, "config", config_path)
    vocab_path = _cached_siglip_path(pretrained_name, "vocab", vocab_path)
    vision_config_path = _cached_siglip_path(
        pretrained_name, "vision_config", vision_config_path
    )

    model = SiglipForMatchingV2(
        config_path=config_path,
        freeze_base_model=freeze_base_model,
        gradient_checkpointing=gradient_checkpointing,
        labels=list(labels),
        vocab_path=vocab_path,
        vision_config_path=vision_config_path,
        max_seq_length=max_seq_length,
    )

    weight_path = pop_value(
        pretrained_weight_path,
        nested_dict_value(pretrained_siglip_infos, pretrained_name, "weight"),
        check_none=False,
    )
    if weight_path is not None:
        model.from_pretrained(weight_path)

    if pretrained_lora_weight_path is not None:
        model.load_lora_weights(
            pretrained_lora_weight_path,
            lora_weights=pretrained_lora_weight,
            lora_alphas=pretrained_lora_alpha,
            save_base_state=False,
        )

    return model


@register_model("microsoft/picasso/model/bad_crop/ruby/v2")
class BadCropRubyV2Model(GenericModel):
    def __init__(
        self,
        models: Sequence[SiglipForMatchingV2],
        model_weights: Sequence[float],
    ):
        super().__init__()
        if len(models) == 0:
            raise ValueError("models should not be empty.")
        if len(models) != len(model_weights):
            raise ValueError("models and model_weights should have the same length.")

        self.models = nn.ModuleList(models)
        weights = torch.tensor(model_weights, dtype=torch.float32)
        self.register_buffer("model_weights", weights, persistent=False)

    @classmethod
    @config_defaults_init("microsoft/picasso/model/bad_crop/ruby/v2")
    def from_config(cls, config, **kwargs):
        config.set_default_section("microsoft/picasso/model/bad_crop/ruby/v2")

        pretrained_names = _as_list(
            config.getoption("pretrained_names", RUBY_PRETRAINED_NAMES)
        )
        model_size = len(pretrained_names)
        labels = config.getoption("labels", RUBY_LABELS)
        max_seq_lengths = _broadcast(
            config.getoption("max_seq_lengths", config.getoption("max_seq_length", 48)),
            model_size,
            "max_seq_lengths",
        )
        freeze_base_model = config.getoption("freeze_base_model", True)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)

        model_weights = _broadcast(
            config.getoption("model_weights", MODEL_WEIGHTS),
            model_size,
            "model_weights",
        )
        model_weights = [float(weight) for weight in model_weights]

        config_paths = _broadcast(
            config.getoption("config_paths", config.getoption("config_path", None)),
            model_size,
            "config_paths",
        )
        vocab_paths = _broadcast(
            config.getoption("vocab_paths", config.getoption("vocab_path", None)),
            model_size,
            "vocab_paths",
        )
        vision_config_paths = _broadcast(
            config.getoption(
                "vision_config_paths", config.getoption("vision_config_path", None)
            ),
            model_size,
            "vision_config_paths",
        )
        pretrained_weight_paths = _broadcast(
            config.getoption(
                "pretrained_weight_paths",
                config.getoption("pretrained_weight_path", None),
            ),
            model_size,
            "pretrained_weight_paths",
        )
        pretrained_lora_weight_paths = _broadcast_nested(
            config.getoption(
                "pretrained_lora_weight_paths",
                config.getoption("pretrained_lora_weight_path", RUBY_LORA_WEIGHT_PATHS),
            ),
            model_size,
            "pretrained_lora_weight_paths",
        )
        pretrained_lora_weights = _broadcast_nested(
            config.getoption(
                "pretrained_lora_weights",
                config.getoption("pretrained_lora_weight", RUBY_LORA_WEIGHTS),
            ),
            model_size,
            "pretrained_lora_weights",
        )
        pretrained_lora_alphas = _broadcast_nested(
            config.getoption(
                "pretrained_lora_alphas",
                config.getoption("pretrained_lora_alpha", 32.0),
            ),
            model_size,
            "pretrained_lora_alphas",
        )

        models = []
        for idx, pretrained_name in enumerate(pretrained_names):
            models.append(
                _build_siglip_model(
                    pretrained_name=pretrained_name,
                    labels=labels,
                    max_seq_length=int(max_seq_lengths[idx]),
                    freeze_base_model=freeze_base_model,
                    gradient_checkpointing=gradient_checkpointing,
                    config_path=config_paths[idx],
                    vocab_path=vocab_paths[idx],
                    vision_config_path=vision_config_paths[idx],
                    pretrained_weight_path=pretrained_weight_paths[idx],
                    pretrained_lora_weight_path=pretrained_lora_weight_paths[idx],
                    pretrained_lora_weight=pretrained_lora_weights[idx],
                    pretrained_lora_alpha=pretrained_lora_alphas[idx],
                )
            )

        return cls(
            models=models,
            model_weights=model_weights,
        )

    def forward(self, **kwargs):
        weighted_outputs = None
        for idx, (model, weight) in enumerate(zip(self.models, self.model_weights), 1):
            if weight.item() == 0.0:
                continue

            pixel_values = kwargs.get(f"pixel_values_{idx}", None)
            if pixel_values is None:
                pixel_values = kwargs.get("pixel_values", None)
            if pixel_values is None:
                raise ValueError(f"pixel_values_{idx} is required.")

            outputs = model(pixel_values=pixel_values).outputs
            outputs = torch.sigmoid(outputs)
            weight = weight.to(device=outputs.device, dtype=outputs.dtype)
            weighted_outputs = (
                outputs * weight
                if weighted_outputs is None
                else weighted_outputs + outputs * weight
            )

        if weighted_outputs is None:
            raise ValueError("At least one model weight should be non-zero.")
        return ClassificationOutputs(outputs=weighted_outputs)


class BadCropRubyV2Processor:
    def __init__(
        self,
        pretrained_names: Sequence[str],
        vocab_paths: Sequence[str],
        vision_config_paths: Sequence[str],
        max_seq_lengths: Sequence[int],
    ):
        self.processors = []
        self.processor_ids = []
        processor_cache = {}
        for pretrained_name, vocab_path, vision_config_path, max_seq_length in zip(
            pretrained_names, vocab_paths, vision_config_paths, max_seq_lengths
        ):
            cache_key = (pretrained_name, vocab_path, vision_config_path, max_seq_length)
            if cache_key not in processor_cache:
                processor_cache[cache_key] = _SiglipProcessor(
                    vocab_path=vocab_path,
                    vision_config_path=vision_config_path,
                    max_seq_length=max_seq_length,
                )
            self.processor_ids.append(cache_key)
            self.processors.append(processor_cache[cache_key])

    @classmethod
    @config_defaults_init("microsoft/picasso/process/bad_crop/ruby/v2")
    def from_config(cls, config, **kwargs):
        config.set_default_section("microsoft/picasso/process/bad_crop/ruby/v2")
        pretrained_names = _as_list(
            config.getoption("pretrained_names", RUBY_PRETRAINED_NAMES)
        )
        model_size = len(pretrained_names)
        max_seq_lengths = _broadcast(
            config.getoption("max_seq_lengths", config.getoption("max_seq_length", 48)),
            model_size,
            "max_seq_lengths",
        )
        vocab_paths = _broadcast(
            config.getoption("vocab_paths", config.getoption("vocab_path", None)),
            model_size,
            "vocab_paths",
        )
        vision_config_paths = _broadcast(
            config.getoption(
                "vision_config_paths", config.getoption("vision_config_path", None)
            ),
            model_size,
            "vision_config_paths",
        )

        vocab_paths = [
            _cached_siglip_path(pretrained_name, "vocab", vocab_path)
            for pretrained_name, vocab_path in zip(pretrained_names, vocab_paths)
        ]
        vision_config_paths = [
            _cached_siglip_path(pretrained_name, "vision_config", vision_config_path)
            for pretrained_name, vision_config_path in zip(
                pretrained_names, vision_config_paths
            )
        ]

        return {
            "pretrained_names": pretrained_names,
            "vocab_paths": vocab_paths,
            "vision_config_paths": vision_config_paths,
            "max_seq_lengths": [int(length) for length in max_seq_lengths],
        }

    @register_process(
        "microsoft/picasso/process/bad_crop/ruby/v2/image_classification"
    )
    def _image_classification(self, image: Union[Image.Image, str]):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        tensors = {}
        outputs_cache = {}
        for idx, (processor_id, processor) in enumerate(
            zip(self.processor_ids, self.processors), 1
        ):
            if processor_id not in outputs_cache:
                outputs_cache[processor_id] = processor.image_classification(image=image)
            tensors[f"pixel_values_{idx}"] = outputs_cache[processor_id].pixel_values
        return TensorInputs(**tensors)
