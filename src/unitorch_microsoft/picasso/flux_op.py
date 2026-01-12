# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

"""
Requires:
    pip install paddlepaddle-gpu "paddleocr<3.0"
"""

import os
import re
import cv2
import io
import fire
import glob
import torch
import json
import base64
import logging
import hashlib
import requests
import numpy as np
import pandas as pd
from torch import autocast
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from multiprocessing import Process, Queue
from diffusers.utils import numpy_to_pil
from diffusers.pipelines import (
    FluxPipeline,
    FluxImg2ImgPipeline,
    FluxInpaintPipeline,
    FluxFillPipeline,
)

from paddleocr import PaddleOCR

from unitorch.utils import is_remote_url
from unitorch.utils import (
    pop_value,
    nested_dict_value,
    read_file,
    read_json_file,
    is_bfloat16_available,
)
from unitorch.models import GenericOutputs
from unitorch.models.diffusers import GenericStableFluxModel
from unitorch.models.diffusers import StableFluxProcessor
from unitorch.cli import CoreConfigureParser
from unitorch.cli import (
    cached_path,
    register_fastapi,
    add_default_section_for_init,
    add_default_section_for_function,
)
from unitorch.cli.models.diffusers import (
    pretrained_stable_infos,
    pretrained_stable_extensions_infos,
    load_weight,
)
from unitorch_microsoft.models.bletchley.pipeline_v3 import (
    BletchleyForMatchingV2Pipeline as BletchleyV3ForMatchingV2Pipeline,
)
import unitorch_microsoft.models.diffusers


class StableFluxForImageInpaintingFastAPIPipeline(GenericStableFluxModel):
    def __init__(
        self,
        config_path: str,
        text_config_path: str,
        text2_config_path: str,
        vae_config_path: str,
        scheduler_config_path: str,
        vocab_path: str,
        merge_path: str,
        vocab2_path: str,
        quant_config_path: Optional[str] = None,
        max_seq_length: Optional[int] = 77,
        max_seq_length2: Optional[int] = 256,
        pad_token: Optional[str] = "<|endoftext|>",
        weight_path: Optional[Union[str, List[str]]] = None,
        state_dict: Optional[Dict[str, Any]] = None,
        lora_checkpoints: Optional[Union[str, List[str]]] = None,
        lora_weights: Optional[Union[float, List[float]]] = 1.0,
        lora_alphas: Optional[Union[float, List[float]]] = 32,
        device: Optional[Union[str, int]] = "cpu",
        enable_cpu_offload: Optional[bool] = False,
    ):
        super().__init__(
            config_path=config_path,
            text_config_path=text_config_path,
            text2_config_path=text2_config_path,
            vae_config_path=vae_config_path,
            scheduler_config_path=scheduler_config_path,
            quant_config_path=quant_config_path,
        )
        self.processor = StableFluxProcessor(
            vocab_path=vocab_path,
            merge_path=merge_path,
            vocab2_path=vocab2_path,
            vae_config_path=vae_config_path,
            max_seq_length=max_seq_length,
            max_seq_length2=max_seq_length2,
            pad_token=pad_token,
        )
        self._device = "cpu" if device == "cpu" else int(device)

        self.from_pretrained(weight_path, state_dict=state_dict)

        self.eval()

        self.pipeline = FluxFillPipeline(
            vae=self.vae,
            text_encoder=self.text,
            text_encoder_2=self.text2,
            transformer=self.transformer,
            scheduler=self.scheduler,
            tokenizer=None,
            tokenizer_2=None,
        )
        self.pipeline.set_progress_bar_config(disable=True)

        if lora_checkpoints is not None:
            self.load_lora_weights(
                lora_checkpoints,
                lora_weights=lora_weights,
                lora_alphas=lora_alphas,
                save_base_state=False,
            )

        self._enable_cpu_offload = enable_cpu_offload

        if self._enable_cpu_offload and self._device != "cpu":
            self.pipeline.enable_model_cpu_offload(self._device)
        else:
            self.to(device=self._device)

    @classmethod
    def from_core_configure(
        cls,
        config,
        pretrained_name: Optional[str] = "stable-flux-dev-fill",
        config_path: Optional[str] = None,
        text_config_path: Optional[str] = None,
        text2_config_path: Optional[str] = None,
        vae_config_path: Optional[str] = None,
        scheduler_config_path: Optional[str] = None,
        vocab_path: Optional[str] = None,
        merge_path: Optional[str] = None,
        vocab2_path: Optional[str] = None,
        quant_config_path: Optional[str] = None,
        pretrained_weight_folder: Optional[str] = None,
        pretrained_weight_path: Optional[str] = None,
        device: Optional[str] = None,
        enable_cpu_offload: Optional[bool] = False,
        pretrained_lora_names: Optional[Union[str, List[str]]] = None,
        pretrained_lora_weights_path: Optional[Union[str, List[str]]] = None,
        pretrained_lora_weights: Optional[Union[float, List[float]]] = None,
        pretrained_lora_alphas: Optional[Union[float, List[float]]] = None,
        **kwargs,
    ):
        pretrained_infos = nested_dict_value(pretrained_stable_infos, pretrained_name)

        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_infos, "transformer", "config"),
        )
        config_path = cached_path(config_path)

        text_config_path = pop_value(
            text_config_path,
            nested_dict_value(pretrained_infos, "text", "config"),
        )
        text_config_path = cached_path(text_config_path)

        text2_config_path = pop_value(
            text2_config_path,
            nested_dict_value(pretrained_infos, "text2", "config"),
        )
        text2_config_path = cached_path(text2_config_path)

        vae_config_path = pop_value(
            vae_config_path,
            nested_dict_value(pretrained_infos, "vae", "config"),
        )
        vae_config_path = cached_path(vae_config_path)

        scheduler_config_path = pop_value(
            scheduler_config_path,
            nested_dict_value(pretrained_infos, "scheduler"),
        )
        scheduler_config_path = cached_path(scheduler_config_path)

        vocab_path = pop_value(
            vocab_path,
            nested_dict_value(pretrained_infos, "text", "vocab"),
        )
        vocab_path = cached_path(vocab_path)

        merge_path = pop_value(
            merge_path,
            nested_dict_value(pretrained_infos, "text", "merge"),
        )
        merge_path = cached_path(merge_path)

        vocab2_path = pop_value(
            vocab2_path,
            nested_dict_value(pretrained_infos, "text2", "vocab"),
        )
        vocab2_path = cached_path(vocab2_path)

        if quant_config_path is not None:
            quant_config_path = cached_path(quant_config_path)

        max_seq_length = 77
        max_seq_length2 = 256
        pad_token = "<|endoftext|>"
        weight_path = pretrained_weight_path
        device = 0 if torch.cuda.is_available() else "cpu"
        enable_cpu_offload = enable_cpu_offload

        state_dict = None
        if weight_path is None and pretrained_weight_folder is not None:
            state_dict = [
                load_weight(
                    glob.glob(f"{pretrained_weight_folder}/transformer/*.safetensors"),
                    prefix_keys={"": "transformer."},
                ),
                load_weight(
                    glob.glob(f"{pretrained_weight_folder}/text_encoder/*.safetensors"),
                    prefix_keys={"": "text."},
                ),
                load_weight(
                    glob.glob(
                        f"{pretrained_weight_folder}/text_encoder_2/*.safetensors"
                    ),
                    prefix_keys={"": "text2."},
                ),
                load_weight(
                    glob.glob(f"{pretrained_weight_folder}/vae/*.safetensors"),
                    prefix_keys={"": "vae."},
                ),
            ]
        elif weight_path is None and pretrained_infos is not None:
            state_dict = [
                load_weight(
                    nested_dict_value(pretrained_infos, "transformer", "weight"),
                    prefix_keys={"": "transformer."},
                ),
                load_weight(
                    nested_dict_value(pretrained_infos, "text", "weight"),
                    prefix_keys={"": "text."},
                ),
                load_weight(
                    nested_dict_value(pretrained_infos, "text2", "weight"),
                    prefix_keys={"": "text2."},
                ),
                load_weight(
                    nested_dict_value(pretrained_infos, "vae", "weight"),
                    prefix_keys={"": "vae."},
                ),
            ]

        pretrained_lora_names = pretrained_lora_names
        pretrained_lora_weights = pretrained_lora_weights or 1.0
        pretrained_lora_alphas = pretrained_lora_alphas or 32.0

        if (
            isinstance(pretrained_lora_names, str)
            and pretrained_lora_weights_path is None
        ):
            pretrained_lora_weights_path = nested_dict_value(
                pretrained_stable_extensions_infos,
                pretrained_lora_names,
                "lora",
                "weight",
            )
        elif (
            isinstance(pretrained_lora_names, list)
            and pretrained_lora_weights_path is None
        ):
            pretrained_lora_weights_path = [
                nested_dict_value(
                    pretrained_stable_extensions_infos, name, "lora", "weight"
                )
                for name in pretrained_lora_names
            ]
            assert len(pretrained_lora_weights_path) == len(pretrained_lora_weights)
            assert len(pretrained_lora_weights_path) == len(pretrained_lora_alphas)

        lora_weights_path = pretrained_lora_weights_path

        inst = cls(
            config_path=config_path,
            text_config_path=text_config_path,
            text2_config_path=text2_config_path,
            vae_config_path=vae_config_path,
            scheduler_config_path=scheduler_config_path,
            vocab_path=vocab_path,
            merge_path=merge_path,
            vocab2_path=vocab2_path,
            quant_config_path=quant_config_path,
            pad_token=pad_token,
            max_seq_length=max_seq_length,
            max_seq_length2=max_seq_length2,
            weight_path=weight_path,
            state_dict=state_dict,
            lora_checkpoints=lora_weights_path,
            lora_weights=pretrained_lora_weights,
            lora_alphas=pretrained_lora_alphas,
            device=device,
            enable_cpu_offload=enable_cpu_offload,
        )
        return inst

    @torch.no_grad()
    @autocast(
        device_type=("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=(torch.bfloat16 if is_bfloat16_available() else torch.float32),
    )
    def __call__(
        self,
        text: str,
        image: Image.Image,
        mask_image: Image.Image,
        neg_text: Optional[str] = "",
        width: Optional[int] = None,
        height: Optional[int] = None,
        guidance_scale: Optional[float] = 30.0,
        strength: Optional[float] = 1.0,
        num_timesteps: Optional[int] = 50,
        seed: Optional[int] = 1123,
    ):
        if width is None or height is None:
            width, height = image.size
        width = width // 16 * 16
        height = height // 16 * 16
        image = image.resize((width, height), resample=Image.LANCZOS)
        mask_image = mask_image.resize((width, height), resample=Image.LANCZOS)

        text_inputs = self.processor.text2image_inputs(
            text,
            negative_prompt=neg_text,
        )
        image_inputs = self.processor.inpainting_inputs(image, mask_image)
        inputs = {**text_inputs, **image_inputs}
        self.seed = seed

        inputs = {k: v.unsqueeze(0) if v is not None else v for k, v in inputs.items()}
        inputs = {
            k: v.to(device=self.device) if v is not None else v
            for k, v in inputs.items()
        }

        prompt_outputs = self.get_prompt_outputs(
            input_ids=inputs.get("input_ids"),
            input2_ids=inputs.get("input2_ids"),
            attention_mask=inputs.get("attention_mask"),
            attention2_mask=inputs.get("attention2_mask"),
            enable_cpu_offload=self._enable_cpu_offload,
            cpu_offload_device=self._device,
        )

        prompt_embeds = prompt_outputs.prompt_embeds
        pooled_prompt_embeds = prompt_outputs.pooled_prompt_embeds

        outputs = self.pipeline(
            image=inputs["pixel_values"],
            mask_image=inputs["pixel_masks"],
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            width=inputs["pixel_values"].size(-1),
            height=inputs["pixel_values"].size(-2),
            generator=torch.Generator(device=self.pipeline.device).manual_seed(
                self.seed
            ),
            num_inference_steps=num_timesteps,
            guidance_scale=guidance_scale,
            strength=strength,
            output_type="np.array",
        )

        images = torch.from_numpy(outputs.images)
        images = numpy_to_pil(images.cpu().numpy())
        return images[0]


def pil_to_base64(image: Image.Image, format: str = "PNG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{base64_str}"


def __out_processing_image(image, ratio, max_size=2048):
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    width, height = image.size

    longest_side = max_size
    shortest_side = (
        int(longest_side * ratio) if ratio < 1 else int(longest_side / ratio)
    )
    size = (longest_side, shortest_side) if ratio > 1 else (shortest_side, longest_side)

    scale = min(size[0] / width, size[1] / height)
    if scale > 1:
        size = (int(size[0] // scale), int(size[1] // scale))
    if size[0] < 512:
        size = (512, int(size[1] * 512 / size[0]))
    if size[1] < 512:
        size = (int(size[0] * 512 / size[1]), 512)

    size = (size[0] // 8 * 8, size[1] // 8 * 8)

    scale = min(size[0] / width, size[1] / height)

    new_width = int(width * scale)
    new_height = int(height * scale)

    image = image.resize(
        (new_width // 8 * 8, new_height // 8 * 8), resample=Image.LANCZOS
    )

    im_width, im_height = image.size

    mask = Image.new("L", (size[0], size[1]), 255)
    black = Image.new("RGB", (im_width, im_height), (0, 0, 0))
    mask.paste(black, ((size[0] - im_width) // 2, (size[1] - im_height) // 2))
    new_image = Image.new("RGB", (size[0], size[1]), (255, 255, 255))
    new_image.paste(image, ((size[0] - im_width) // 2, (size[1] - im_height) // 2))

    return new_image, mask


def get_outpainted_image(im1, im2, ratio):
    im1, ms = __out_processing_image(im1, float(ratio))
    ms = ms.resize(im2.size)
    ms = ImageOps.invert(ms).convert("L")
    im2 = im2.convert("RGBA")
    white_layer = Image.new("RGBA", im2.size, (255, 255, 255, 255))
    im2.paste(white_layer, (0, 0), ms)
    return im2


def outpainting(
    data_file: str,
    cache_dir: str,
    image_col: str,
    names: Union[str, List[str]],
    prompt_col: Optional[str] = None,
    prompt_text: Optional[str] = None,
    pretrained_name: Optional[str] = "stable-flux-dev-fill",
    guidance_scale: Optional[float] = 30.0,
    num_timesteps: Optional[int] = 30,
    seed: Optional[int] = 1123,
    lora_name: Optional[str] = "stable-flux-lora-ms-dev-fill-simple",
    lora_weight: Optional[float] = 0.2,
    lora_alpha: Optional[float] = 32.0,
    strength: Optional[float] = 0.95,
    padding_max_ratio: Optional[float] = 0.4,
    ratios: Optional[Union[str, List[float]]] = [0.5, 1.0, 2.0],
    http_url: str = "http://0.0.0.0:11230/?file={0}",
    pretrained_weight_folder: Optional[str] = None,
    enable_cpu_offload: Optional[bool] = False,
    enable_filters: Optional[bool] = True,
):
    if isinstance(names, str) and names.strip() == "*":
        names = None
    if isinstance(names, str):
        names = re.split(r"[,;]", names)
        names = [n.strip() for n in names]

    data = pd.read_csv(
        data_file,
        names=names,
        sep="\t",
        quoting=3,
        header=None,
    )

    if isinstance(ratios, str):
        ratios = re.split(r"[,;]", ratios)
        ratios = [float(n.strip()) for n in ratios]
    if isinstance(ratios, float):
        ratios = [ratios]

    os.makedirs(cache_dir, exist_ok=True)

    assert image_col in data.columns, f"Column {image_col} not found in data."

    output_file = f"{cache_dir}/output.txt"

    if os.path.exists(output_file):
        uniques = pd.read_csv(
            output_file,
            names=[prompt_col, image_col, "ratio", "result"],
            sep="\t",
            quoting=3,
            header=None,
        )
        uniques = set((uniques[prompt_col] + " - " + uniques[image_col]).tolist())

        data = data[
            ~data.apply(
                lambda x: (prompt_text if prompt_text is not None else x[prompt_col])
                + " - "
                + x[image_col]
                in uniques,
                axis=1,
            )
        ]

    pipe = StableFluxForImageInpaintingFastAPIPipeline.from_core_configure(
        config=CoreConfigureParser(),
        pretrained_name=pretrained_name,
        pretrained_weight_folder=pretrained_weight_folder,
        pretrained_lora_names=[lora_name] if lora_name is not None else None,
        pretrained_lora_weights=[lora_weight] if lora_weight is not None else None,
        pretrained_lora_alphas=[lora_alpha] if lora_alpha is not None else None,
        device=0 if torch.cuda.is_available() else "cpu",
        enable_cpu_offload=enable_cpu_offload,
    )

    if enable_filters:
        filter1 = BletchleyV3ForMatchingV2Pipeline.from_core_configure(
            config=CoreConfigureParser(),
            config_type="2.5B",
            pretrained_weight_path="https://huggingface.co/datasets/fuliucansheng/unitorchblobfuse/resolve/main/models/bletchley/v3/pytorch_model.large.bin",
            pretrained_lora_weight_path="https://huggingface.co/datasets/fuliucansheng/unitorchblobfuse/resolve/main/models/adsplus/lora/bletchley/pytorch_model.v3.2.5B.lora4.watermark.2410.bin",
            label_dict={
                "watermark": "watermarked, no watermark signature, brand logo",
            },
            act_fn="sigmoid",
            device=0 if torch.cuda.is_available() else "cpu",
        )
        filter2 = PaddleOCR(
            use_angle_cls=True,
            lang="ch",
            use_gpu=torch.cuda.is_available(),
            show_log=False,
        )
    else:
        filter1 = None
        filter2 = None

    def check_result(im1, im2, ra):
        if not enable_filters:
            return True
        res = get_outpainted_image(im1, im2, ra)
        pass1 = filter1(res)["watermark"] < 0.9
        pass2 = True
        try:
            outputs = filter2.ocr(np.array(res), det=True, cls=False)
            if outputs is None or len(outputs) < 1:
                pass
            elif outputs[0] is None or len(outputs[0]) < 1:
                pass
            else:
                pass2 = False
        except:
            pass
        return pass1 and pass2

    writer = open(output_file, "a+")

    for _, row in data.iterrows():
        prompt = prompt_text if prompt_text is not None else row[prompt_col]
        if http_url is not None:
            url = http_url.format(row[image_col])
            doc = requests.get(url, timeout=600)
            raw_image = Image.open(io.BytesIO(doc.content)).convert("RGB")
        else:
            # image = Image.open(image).convert("RGB")
            raw_image = Image.open(row[image_col])

        for ratio in ratios:
            ratio2 = ratio

            if padding_max_ratio is not None:
                _ratio = raw_image.size[0] / raw_image.size[1]
                if ratio > _ratio and ratio > (1 + padding_max_ratio) * _ratio:
                    ratio = (1 + padding_max_ratio) * _ratio
                if ratio < _ratio and ratio < _ratio / (1 + padding_max_ratio):
                    ratio = _ratio / (1 + padding_max_ratio)

            image, mask_image = __out_processing_image(raw_image, ratio)
            image_np = np.array(image.convert("RGB"))
            mask_np = np.array(mask_image.convert("L")).astype(np.uint8)

            _, binary_mask = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
            inpainted_image = cv2.inpaint(image_np, binary_mask, 10, cv2.INPAINT_TELEA)
            latent_image = Image.fromarray(inpainted_image)
            result = pipe(
                text=prompt,
                image=latent_image,
                mask_image=mask_image,
                guidance_scale=guidance_scale,
                num_timesteps=num_timesteps,
                strength=strength,
                seed=seed,
            )

            raw_width, raw_height = raw_image.size
            if raw_width / raw_height > ratio:
                result = result.resize(
                    (raw_width, int(raw_width / ratio)), resample=Image.LANCZOS
                )
            else:
                result = result.resize(
                    (int(raw_height * ratio), raw_height), resample=Image.LANCZOS
                )

            if ratio == ratio2:
                if check_result(raw_image, result, ratio2):
                    writer.write(
                        "\t".join(
                            [prompt, row[image_col], str(ratio), pil_to_base64(result)]
                        )
                        + "\n"
                    )
                    writer.flush()
                continue

            ratio = ratio2
            image, mask_image = __out_processing_image(result, ratio)
            image_np = np.array(image.convert("RGB"))
            mask_np = np.array(mask_image.convert("L")).astype(np.uint8)

            _, binary_mask = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
            inpainted_image = cv2.inpaint(image_np, binary_mask, 10, cv2.INPAINT_TELEA)
            latent_image = Image.fromarray(inpainted_image)
            result = pipe(
                text=prompt,
                image=latent_image,
                mask_image=mask_image,
                guidance_scale=guidance_scale,
                num_timesteps=num_timesteps,
                strength=strength,
                seed=seed,
            )
            raw_width, raw_height = image.size
            if raw_width / raw_height > ratio:
                result = result.resize(
                    (raw_width, int(raw_width / ratio)), resample=Image.LANCZOS
                )
            else:
                result = result.resize(
                    (int(raw_height * ratio), raw_height), resample=Image.LANCZOS
                )

            if check_result(raw_image, result, ratio2):
                writer.write(
                    "\t".join(
                        [prompt, row[image_col], str(ratio), pil_to_base64(result)]
                    )
                    + "\n"
                )
                writer.flush()


if __name__ == "__main__":
    fire.Fire(outpainting)
