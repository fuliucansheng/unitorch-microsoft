# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import io
import gc
import base64
import torch
import asyncio
from PIL import Image
from pydantic import BaseModel
from typing import Any, Dict, List, Optional, Union
from fastapi import APIRouter
from unitorch.utils import is_remote_url
from unitorch.models.clip import ClipForMatching as _ClipForMatching
from unitorch.models.clip import ClipProcessor
from unitorch.utils import pop_value, nested_dict_value
from unitorch.cli import (
    cached_path,
    config_defaults_init,
    config_defaults_method,
    register_fastapi,
)
from unitorch.cli import Config, GenericFastAPI
from unitorch.cli.models.clip import (
    pretrained_clip_infos,
    pretrained_clip_extensions_infos,
)


class ClipForMatchingPipeline(_ClipForMatching):
    def __init__(
        self,
        config_path: str,
        vocab_path: str,
        merge_path: str,
        vision_config_path: str,
        projection_dim: Optional[int] = 512,
        max_seq_length: Optional[int] = 512,
        weight_path: Optional[Union[str, List[str]]] = None,
        state_dict: Optional[Dict[str, Any]] = None,
        enable_cpu_offload: Optional[bool] = True,
        device: Optional[Union[str, int]] = "cpu",
    ):
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
        self._enable_cpu_offload = enable_cpu_offload
        if not self._enable_cpu_offload and self._device != "cpu":
            self.to(device=self._device)
        self.eval()

    @classmethod
    @config_defaults_init("microsoft/dlis/pipeline/clip/matching")
    def from_config(
        cls,
        config,
        pretrained_name: Optional[str] = None,
        config_path: Optional[str] = None,
        vocab_path: Optional[str] = None,
        merge_path: Optional[str] = None,
        vision_config_path: Optional[str] = None,
        pretrained_weight_path: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        config.set_default_section("microsoft/dlis/pipeline/clip/matching")
        pretrained_name = pretrained_name or config.getoption(
            "pretrained_name", "clip-vit-base-patch16"
        )

        config_path = config_path or config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_clip_infos, pretrained_name, "config"),
        )
        config_path = cached_path(config_path)

        vocab_path = vocab_path or config.getoption("vocab_path", None)
        vocab_path = pop_value(
            vocab_path,
            nested_dict_value(pretrained_clip_infos, pretrained_name, "vocab"),
        )
        vocab_path = cached_path(vocab_path)

        merge_path = merge_path or config.getoption("merge_path", None)
        merge_path = pop_value(
            merge_path,
            nested_dict_value(pretrained_clip_infos, pretrained_name, "merge"),
        )
        merge_path = cached_path(merge_path)

        vision_config_path = vision_config_path or config.getoption(
            "vision_config_path", None
        )
        vision_config_path = pop_value(
            vision_config_path,
            nested_dict_value(pretrained_clip_infos, pretrained_name, "vision_config"),
        )
        vision_config_path = cached_path(vision_config_path)

        projection_dim = config.getoption("projection_dim", 512)
        max_seq_length = config.getoption("max_seq_length", 512)
        enable_cpu_offload = config.getoption("enable_cpu_offload", True)
        device = config.getoption("device", "cpu") if device is None else device
        pretrained_weight_path = pretrained_weight_path or config.getoption(
            "pretrained_weight_path", None
        )
        weight_path = pop_value(
            pretrained_weight_path,
            nested_dict_value(pretrained_clip_infos, pretrained_name, "weight"),
            check_none=False,
        )

        inst = cls(
            config_path,
            vocab_path=vocab_path,
            merge_path=merge_path,
            vision_config_path=vision_config_path,
            projection_dim=projection_dim,
            max_seq_length=max_seq_length,
            weight_path=weight_path,
            enable_cpu_offload=enable_cpu_offload,
            device=device,
        )

        return inst

    @torch.no_grad()
    @config_defaults_method("microsoft/dlis/pipeline/clip/matching")
    def __call__(
        self,
        text: str,
        image: Image.Image,
        max_seq_length: Optional[int] = 77,
    ):
        if self._enable_cpu_offload:
            self.to(self._device)
        inputs = self.processor.classification(
            text=text,
            image=image,
            max_seq_length=max_seq_length,
        )
        inputs = {k: v.unsqueeze(0) if v is not None else v for k, v in inputs.items()}
        inputs = {
            k: v.to(device=self._device) if v is not None else v
            for k, v in inputs.items()
        }
    
        outputs = super().forward(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            position_ids=inputs["position_ids"],
            pixel_values=inputs["pixel_values"],
        )
        scores = outputs.sigmoid().squeeze(0)
        if self._enable_cpu_offload:
            self.to("cpu")
            torch.cuda.empty_cache()
        return scores[0].item()


@register_fastapi("microsoft/dlis/fastapi/clip/matching")
class ClipForMatchingFastAPI(GenericFastAPI):
    def __init__(self, config: Config):
        self.config = config
        config.set_default_section("microsoft/dlis/fastapi/clip/matching")
        router = config.getoption("router", "/microsoft/dlis/fastapi/clip/matching")
        self._pipe = None
        self._router = APIRouter(prefix=router)
        self._router.add_api_route("/generate", self.generate, methods=["POST"])
        self._router.add_api_route("/status", self.status, methods=["GET"])
        self._router.add_api_route("/start", self.start, methods=["GET"])
        self._router.add_api_route("/stop", self.stop, methods=["GET"])
        self._lock = asyncio.Lock()

    @property
    def router(self):
        return self._router

    def start(self, pretrained_name: str = "clip-vit-base-patch16"):
        self._pipe = ClipForMatchingPipeline.from_config(
            self.config,
            pretrained_name=pretrained_name,
        )
        return "start success"

    def stop(self):
        self._pipe.to("cpu")
        del self._pipe
        gc.collect()
        torch.cuda.empty_cache()
        self._pipe = None
        return "stop success"

    def status(self):
        return "running" if self._pipe is not None else "stopped"

    async def generate(
        self,
        payload: "ClipMatchingPayload",
    ):
        assert self._pipe is not None
        image_bytes = base64.b64decode(payload.image)
        image = Image.open(io.BytesIO(image_bytes))
        async with self._lock:
            result = self._pipe(
                payload.text,
                image,
                max_seq_length=payload.max_seq_length,
            )

        return {"score": result}


class ClipMatchingPayload(BaseModel):
    text: str
    image: str  # base64-encoded image
    max_seq_length: Optional[int] = 77
