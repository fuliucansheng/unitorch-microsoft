# Copyright (c) MICROSOFT.
# Licensed under the MIT License.
import json
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from diffusers import AutoencoderKL
from unitorch.models import (
    GenericModel,
    GenericOutputs,
    QuantizationConfig,
    QuantizationMixin,
)
from unitorch.utils import (
    pop_value,
    nested_dict_value,
    is_bfloat16_available,
    is_cuda_available,
)
from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
    register_model,
)
from unitorch.cli.models import DiffusionOutputs, LossOutputs
from unitorch.cli.models.diffusers import (
    pretrained_stable_infos,
    pretrained_stable_extensions_infos,
    load_weight,
)


@register_model("microsoft/model/diffusers/vae")
class VAEForDiffusion(GenericModel):
    prefix_keys_in_state_dict = {
        "^encoder.*": "vae.",
        "^decoder.*": "vae.",
        "^post_quant_conv.*": "vae.",
        "^quant_conv.*": "vae.",
    }

    replace_keys_in_state_dict = {
        "\.query\.": ".to_q.",
        "\.key\.": ".to_k.",
        "\.value\.": ".to_v.",
        "\.proj_attn\.": ".to_out.0.",
    }

    def __init__(
        self,
        config_path: str,
        patch_size: Optional[int] = 32,
        stride: Optional[int] = 16,
        use_fp16: Optional[bool] = True,
        use_bf16: Optional[bool] = False,
        use_lpips: Optional[bool] = False,
    ):
        super().__init__()
        config_dict = json.load(open(config_path))
        self.vae = AutoencoderKL.from_config(config_dict)
        self.patch_size = patch_size
        self.stride = stride
        self.use_lpips = use_lpips

        self.use_dtype = torch.float16 if use_fp16 else torch.float32
        self.use_dtype = (
            torch.bfloat16 if use_bf16 and is_bfloat16_available() else self.use_dtype
        )

        if self.use_lpips:
            try:
                import lpips
            except ImportError:
                raise ImportError(
                    "Please install lpips to use VAEForDiffusion with LPIPS loss. "
                    "You can install it with `pip install lpips`."
                )

            self.lpips_model = lpips.LPIPS(net="alex")
            self.lpips_model.requires_grad_(False)
            self.lpips_model.eval()

    @classmethod
    @add_default_section_for_init("microsoft/model/diffusers/vae")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/model/diffusers/vae")
        pretrained_name = config.getoption("pretrained_name", "stable-v1.5")
        pretrained_infos = nested_dict_value(pretrained_stable_infos, pretrained_name)

        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_infos, "vae", "config"),
        )
        config_path = cached_path(config_path)

        patch_size = config.getoption("patch_size", 32)
        stride = config.getoption("stride", 16)
        use_fp16 = config.getoption("use_fp16", True)
        use_bf16 = config.getoption("use_bf16", False)
        use_lpips = config.getoption("use_lpips", False)

        inst = cls(
            config_path,
            patch_size=patch_size,
            stride=stride,
            use_fp16=use_fp16,
            use_bf16=use_bf16,
            use_lpips=use_lpips,
        )

        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_infos, "vae", "weight"),
            check_none=False,
        )
        if weight_path is not None:
            inst.from_pretrained(weight_path)

        return inst

    def patch_mse_loss(self, y_true, y_pred):
        y_true = y_true.unfold(2, self.patch_size, self.stride).unfold(
            3, self.patch_size, self.stride
        )
        y_pred = y_pred.unfold(2, self.patch_size, self.stride).unfold(
            3, self.patch_size, self.stride
        )
        y_true = y_true.contiguous().view(y_true.size(0), -1)
        y_pred = y_pred.contiguous().view(y_pred.size(0), -1)
        return F.mse_loss(y_pred, y_true, reduction="none").mean(1)

    @torch.no_grad()
    def patch_lpips_loss(self, y_true, y_pred):
        y_true = y_true.unfold(2, self.patch_size, self.stride).unfold(
            3, self.patch_size, self.stride
        )
        y_pred = y_pred.unfold(2, self.patch_size, self.stride).unfold(
            3, self.patch_size, self.stride
        )
        y_true = y_true.contiguous().view(
            y_true.size(0), y_true.size(1), -1, self.patch_size, self.patch_size
        )
        y_pred = y_pred.contiguous().view(
            y_pred.size(0), y_pred.size(1), -1, self.patch_size, self.patch_size
        )
        y_true_patches = (
            y_true.permute(2, 0, 1, 3, 4)
            .contiguous()
            .view(-1, y_true.size(1), self.patch_size, self.patch_size)
        )
        y_pred_patches = (
            y_pred.permute(2, 0, 1, 3, 4)
            .contiguous()
            .view(-1, y_pred.size(1), self.patch_size, self.patch_size)
        )
        lpips_loss = self.lpips_model(y_true_patches, y_pred_patches).mean(dim=1)
        lpips_loss = torch.where(
            torch.isfinite(lpips_loss),
            lpips_loss,
            torch.tensor(0.0, device=lpips_loss.device),
        )
        return lpips_loss.mean()

    def forward(self, pixel_values):
        with autocast(
            device_type=("cuda" if torch.cuda.is_available() else "cpu"),
            dtype=self.use_dtype,
        ):
            latents = self.vae.encode(pixel_values).latent_dist.sample()
            outputs = self.vae.decode(latents).sample

            loss = self.patch_mse_loss(pixel_values, outputs).mean()
            if self.use_lpips:
                lpips_loss = self.patch_lpips_loss(pixel_values, outputs)
                loss += lpips_loss * 0.1
            return LossOutputs(loss=loss)
