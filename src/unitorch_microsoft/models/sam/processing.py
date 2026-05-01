# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import torch
import random
import numpy as np
from PIL import Image, ImageEnhance
from typing import List, Tuple, Union, Optional
from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize, Compose
from transformers import SamImageProcessor
from transformers.image_utils import to_numpy_array, ChannelDimension
from transformers.image_transforms import to_channel_dimension_format
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models import HfImageClassificationProcessor, GenericOutputs
from unitorch.cli import (
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
    register_process,
)
from unitorch.cli.models import SegmentationOutputs, TensorInputs
from unitorch.cli.models.sam import pretrained_sam_infos


def cv_random_flip(img, label):
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label


def random_crop(image, label):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    border = int(min(image_width, image_height) * 0.1)
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1,
        (image_height - crop_win_height) >> 1,
        (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1,
    )
    return image.crop(random_region), label.crop(random_region)


def random_rotate(image, label, angle=15):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-angle, angle)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
    return image, label


def color_enhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def random_gaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))


def random_pepper(img, N=0.0015):
    img = np.array(img)
    noiseNum = int(N * img.shape[0] * img.shape[1])
    for i in range(noiseNum):
        randX = random.randint(0, img.shape[0] - 1)
        randY = random.randint(0, img.shape[1] - 1)
        if random.randint(0, 1) == 0:
            img[randX, randY] = 0
        else:
            img[randX, randY] = 255
    return Image.fromarray(img)


def get_random_points(mask: Image.Image, num_points: int):
    mask = np.array(mask.convert("L"))
    non_zeros = np.argwhere(mask > 0)
    if len(non_zeros) < num_points:
        return np.array(non_zeros)
    return np.array(random.choices(non_zeros, k=num_points))


def get_mask_box(mask: Image.Image):
    width, height = mask.size
    mask = np.array(mask.convert("L"))
    non_zeros = np.argwhere(mask > 0)
    if len(non_zeros) == 0:
        return (0, 0, width, height)
    return (
        np.min(non_zeros[:, 1]),
        np.min(non_zeros[:, 0]),
        np.max(non_zeros[:, 1]),
        np.max(non_zeros[:, 0]),
    )


class SamProcessor:
    def __init__(
        self,
        vision_config_path: str,
    ):
        """
        Initializes a SwinProcessor for image classification tasks.

        Args:
            vision_config_path (str): The path to the SamImageProcessor configuration file.
        """
        self.vision_processor = SamImageProcessor.from_json_file(vision_config_path)

    @classmethod
    @add_default_section_for_init("microsoft/process/sam")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/process/sam")
        pretrained_name = config.getoption("pretrained_name", "sam-vit-base")
        vision_config_path = config.getoption("vision_config_path", None)
        vision_config_path = pop_value(
            vision_config_path,
            nested_dict_value(pretrained_sam_infos, pretrained_name, "vision_config"),
        )

        vision_config_path = cached_path(vision_config_path)

        return {
            "vision_config_path": vision_config_path,
        }

    @register_process("microsoft/process/sam/segmentation")
    def _segmentation(
        self,
        image: Union[Image.Image, str],
        mask: Union[Image.Image, str],
        # num_points: Optional[int] = 1,
    ):
        if isinstance(image, str):
            image = Image.open(image)

        if isinstance(mask, str):
            mask = Image.open(mask)

        image, mask = cv_random_flip(image, mask)
        image, mask = random_rotate(image, mask)
        image = color_enhance(image)
        mask = random_pepper(mask)

        pixel_results = self.vision_processor(image)
        pixel_values = torch.tensor(pixel_results.get("pixel_values")[0])
        original_sizes = pixel_results.get("original_sizes")[0]  # h, w
        reshaped_input_sizes = pixel_results.get("reshaped_input_sizes")[0]  # h, w
        height, width = reshaped_input_sizes
        # mask = mask.resize((width, height), resample=Image.LANCZOS).convert("L")
        # pixel_targets = torch.zeros_like(pixel_values[0]).float()
        # pixel_targets[:height, :width] = torch.tensor(np.array(mask)).float() / 255.0

        mask = mask.convert("L").resize((width, height), resample=Image.LANCZOS)
        pixel_targets = torch.zeros_like(pixel_values[0]).float()
        pixel_targets[:height, :width] = torch.tensor(np.array(mask)).long() / 255.0

        # input_points = get_random_points(mask, random.randint(1, num_points))
        # input_boxes = torch.tensor([get_mask_box(mask)]).float()
        input_boxes = torch.tensor([(0, 0, width, height)]).float()

        return TensorInputs(
            pixel_values=pixel_values,
            # input_points=torch.tensor(input_points),
            input_boxes=input_boxes,
            pixel_targets=pixel_targets,
            reshaped_input_sizes=torch.tensor(reshaped_input_sizes),
        )
