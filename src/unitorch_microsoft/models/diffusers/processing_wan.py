# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import cv2
import torch
import json
import numpy as np
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from transformers import CLIPTokenizer, T5Tokenizer, CLIPImageProcessor
from torchvision.transforms import (
    Resize,
    CenterCrop,
    RandomCrop,
    Lambda,
    ToTensor,
    Normalize,
    Compose,
    RandomHorizontalFlip,
)
from diffusers.image_processor import VaeImageProcessor
from unitorch.models import (
    HfTextClassificationProcessor,
    HfImageClassificationProcessor,
    GenericOutputs,
)
from unitorch.utils import pop_value, nested_dict_value
from unitorch.cli import (
    add_default_section_for_init,
    add_default_section_for_function,
    register_process,
)
from unitorch.cli.models import (
    TensorsInputs,
)
from unitorch.cli.models.diffusers import pretrained_stable_infos
from unitorch_microsoft import cached_path


class WanProcessor(HfTextClassificationProcessor):
    def __init__(
        self,
        vocab_path: str,
        vae_config_path: Optional[str] = None,
        max_seq_length: Optional[int] = 77,
        position_start_id: Optional[int] = 0,
        video_size: Optional[Tuple[int, int]] = (832, 480),
    ):
        tokenizer = T5Tokenizer(
            vocab_file=vocab_path,
        )

        tokenizer.bos_token_id = 0
        tokenizer.cls_token = tokenizer.convert_ids_to_tokens(0)
        tokenizer.sep_token = tokenizer.eos_token
        tokenizer.sep_token_id = tokenizer.eos_token_id

        HfTextClassificationProcessor.__init__(
            self,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            position_start_id=position_start_id,
        )

        if video_size is not None:
            self.video_size = (
                video_size
                if isinstance(video_size, tuple)
                else (video_size, video_size)
            )
            self.video_size = (video_size[0] // 16 * 16, video_size[1] // 16 * 16)
        else:
            self.video_size = None
        self.divisor = 16

        if self.video_size is not None:
            self.frame_processor = Compose(
                [
                    CenterCrop(size=(self.video_size[1], self.video_size[0])),
                    ToTensor(),
                    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
        else:
            self.frame_processor = None

        if vae_config_path is not None:
            vae_config_dict = json.load(open(vae_config_path))
            vae_scale_factor = 2 ** (
                len(vae_config_dict.get("block_out_channels", [])) - 1
            )
            self.vae_image_processor = VaeImageProcessor(
                vae_scale_factor=vae_scale_factor
            )
        else:
            self.vae_image_processor = None

    @classmethod
    @add_default_section_for_init("microsoft/process/diffusion/wan")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/process/diffusion/wan")
        pretrained_name = config.getoption("pretrained_name", "wan-v2.2-i2v-14b")
        pretrained_infos = nested_dict_value(pretrained_stable_infos, pretrained_name)

        vocab_path = config.getoption("vocab_path", None)
        vocab_path = pop_value(
            vocab_path,
            nested_dict_value(pretrained_infos, "text", "vocab"),
        )
        vocab_path = cached_path(vocab_path)

        vae_config_path = config.getoption("vae_config_path", None)
        vae_config_path = pop_value(
            vae_config_path,
            nested_dict_value(pretrained_infos, "vae", "config"),
        )
        vae_config_path = cached_path(vae_config_path)

        return {
            "vocab_path": vocab_path,
            "vae_config_path": vae_config_path,
        }

    def get_video_frames(self, video: Union[cv2.VideoCapture, str]):
        if isinstance(video, str):
            video = cv2.VideoCapture(video)

        if isinstance(video, cv2.VideoCapture):
            frames = []
            while video.isOpened():
                ret, frame = video.read()
                if not ret:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                frames.append(pil_img)
        else:
            frames = video

        return frames

    @register_process("microsoft/process/diffusion/wan/text2video")
    def text2video(
        self,
        prompt: str,
        video: Union[cv2.VideoCapture, str, List[Image.Image]],
        max_seq_length: Optional[int] = None,
    ):
        frames = self.get_video_frames(video)
        print(f"debug video frames: {len(frames)}")

        pixel_values = []
        for frame in frames:
            if self.frame_processor is not None:
                width, height = frame.size
                scale = max(self.video_size[0] / width, self.video_size[1] / height)
                frame = frame.resize(
                    (round(width * scale), round(height * scale)),
                    resample=Image.LANCZOS,
                )
                pixel_frame = self.frame_processor(frame)
                pixel_values.append(pixel_frame)
            else:
                raise ValueError(
                    "frame_processor is None, please set video_size to process video"
                )
        pixel_values = torch.stack(pixel_values, dim=0)
        pixel_values = pixel_values.permute(1, 0, 2, 3)

        prompt_outputs = self.classification(prompt, max_seq_length=max_seq_length)

        return TensorsInputs(
            pixel_values=pixel_values,
            input_ids=prompt_outputs.input_ids,
            attention_mask=prompt_outputs.attention_mask,
        )

    @register_process("microsoft/process/diffusion/wan/image2video")
    def image2video(
        self,
        prompt: str,
        video: Union[cv2.VideoCapture, str, List[Image.Image]],
        image: Optional[Union[Image.Image, str]] = None,
        max_seq_length: Optional[int] = None,
    ):
        frames = self.get_video_frames(video)

        outputs = self.text2video(
            prompt=prompt,
            video=frames,
            max_seq_length=max_seq_length,
        )
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        if image is None:
            image = frames[0].convert("RGB")

        width, height = image.size
        scale = max(self.video_size[0] / width, self.video_size[1] / height)
        image = image.resize(
            (round(width * scale), round(height * scale)), resample=Image.LANCZOS
        )
        image = CenterCrop(size=(self.video_size[1], self.video_size[0]))(image)

        vae_pixel_values = self.vae_image_processor.preprocess(image)[0]

        return TensorsInputs(
            pixel_values=outputs.pixel_values,
            input_ids=outputs.input_ids,
            attention_mask=outputs.attention_mask,
            vae_pixel_values=vae_pixel_values,
        )

    @register_process("microsoft/process/diffusion/wan/text2video/inputs")
    def text2video_inputs(
        self,
        prompt: str,
        negative_prompt: Optional[str] = "",
        max_seq_length: Optional[int] = None,
    ):
        prompt_outputs = self.classification(prompt, max_seq_length=max_seq_length)
        negative_prompt_outputs = self.classification(
            negative_prompt, max_seq_length=max_seq_length
        )

        return TensorsInputs(
            input_ids=prompt_outputs.input_ids,
            attention_mask=prompt_outputs.attention_mask,
            negative_input_ids=negative_prompt_outputs.input_ids,
            negative_attention_mask=negative_prompt_outputs.attention_mask,
        )

    @register_process("microsoft/process/diffusion/wan/image2video/inputs")
    def image2video_inputs(
        self,
        prompt: str,
        image: Union[Image.Image, str],
        negative_prompt: Optional[str] = "",
        max_seq_length: Optional[int] = None,
        keep_original_ratio: Optional[bool] = False,
    ):  # keep original ratio
        if isinstance(image, str):
            image = Image.open(image)
        image = image.convert("RGB")

        if keep_original_ratio:
            max_area = self.video_size[0] * self.video_size[1]
            aspect_ratio = image.height / image.width
            height = (
                round(np.sqrt(max_area * aspect_ratio)) // self.divisor * self.divisor
            )
            width = (
                round(np.sqrt(max_area / aspect_ratio)) // self.divisor * self.divisor
            )
            image = image.resize((width, height), resample=Image.LANCZOS)
        else:
            width, height = image.size
            scale = max(self.video_size[0] / width, self.video_size[1] / height)
            image = image.resize(
                (round(width * scale), round(height * scale)), resample=Image.LANCZOS
            )
            image = CenterCrop(size=(self.video_size[1], self.video_size[0]))(image)

        vae_pixel_values = self.vae_image_processor.preprocess(image)[0]
        text_outputs = self.text2video_inputs(
            prompt=prompt,
            negative_prompt=negative_prompt,
            max_seq_length=max_seq_length,
        )
        return TensorsInputs(
            input_ids=text_outputs.input_ids,
            attention_mask=text_outputs.attention_mask,
            negative_input_ids=text_outputs.negative_input_ids,
            negative_attention_mask=text_outputs.negative_attention_mask,
            vae_pixel_values=vae_pixel_values,
        )
