# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import math
import random
import numpy as np
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from PIL import Image, ImageOps, ImageFile, ImageFilter, ImageChops, ImageDraw
from random import randint, shuffle, choice
from unitorch.cli import (
    add_default_section_for_init,
    add_default_section_for_function,
    register_process,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_roi_bbox(img):
    gray = img.convert("L")
    bw = gray.point(lambda x: 0 if x < 128 else 255, "1")
    inverted = ImageChops.invert(bw)
    bbox = inverted.getbbox()
    if bbox is None:
        bbox = (0, 0, img.width, img.height)
    return bbox


def get_pad_size(w, h, roi_w, roi_h, min_ratio, max_ratio, cateid):
    furnitures = [
        3367,
        3877,
        3388,
        2857,
        3879,
        3878,
        3389,
        3814,
        2866,
        1652,
        2529,
        1692,
        671,
        660,
        3725,
        3726,
        1588,
        1507,
        1667,
        1672,
        1566,
        1769,
        1793,
        3788,
        3488,
        3371,
        3094,
        3797,
        1596,
        402,
        3372,
        3370,
        2856,
        1707,
        3376,
        4382,
        3458,
        1638,
        1723,
        769,
        2241,
        1048,
        3690,
        3973,
        3061,
        373,
        3731,
        165,
        1729,
        2206,
        3364,
        3365,
        3366,
        3733,
        708,
        1685,
        1563,
        387,
        2250,
        458,
        1613,
        3770,
        2408,
        4245,
        3771,
        1738,
        3760,
        3373,
        3705,
        2425,
        3706,
        2292,
        1696,
        1524,
        4059,
        652,
        3809,
        3777,
        1552,
        1786,
        3375,
        4292,
        3986,
        2253,
        1753,
        3774,
        3773,
        150,
        199,
        217,
        3369,
        1513,
        1627,
        1767,
        658,
        2616,
    ]
    if cateid not in furnitures and (w / roi_h >= 1.5):
        min_ratio = 1.0
    pad_height = max(math.ceil(roi_w / min_ratio), max(h, math.ceil(1.2 * roi_h)))
    if pad_height * max_ratio > w:
        pad_width = math.ceil(pad_height * max_ratio)
    else:
        pad_width = w
        pad_height = math.ceil(pad_width / max_ratio)
    return pad_width, pad_height


def get_pad_image(img, left, up, right, down):
    w, h = img.size
    pad_image = Image.new("RGB", (w + left + right, h + up + down), (255, 255, 255))
    pad_image.paste(img, (left, up))
    return pad_image


class PicassoDiffusionProcessor:
    """
    Processor for image-related operations.
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (512, 512),
    ):
        """
        Initializes a new instance of the ImageProcessor.
        """
        self.image_size = (
            image_size if isinstance(image_size, tuple) else (image_size, image_size)
        )

    @classmethod
    @add_default_section_for_init("microsoft/picasso/diffusion/process")
    def from_core_configure(cls, config, **kwargs):
        """
        Creates a new instance of the ImageProcessor using the configuration from the core.

        Args:
            config: The configuration object.
            **kwargs: Additional keyword arguments.

        Returns:
            An instance of the ImageProcessor.
        """
        pass

    @register_process("microsoft/picasso/diffusion/process/padding")
    def _padding(
        self,
        image: Image.Image,
        cateid: int,
    ):
        """
        Detects edges in the image using the Canny algorithm.

        Args:
            image (Image.Image): The image to detect edges in.

        Returns:
            The image with detected edges as a PIL Image object.
        """
        w, h = image.size
        new_w, new_h = self.image_size
        roi = get_roi_bbox(image)
        roi_w, roi_h = roi[2] - roi[0], roi[3] - roi[1]

        pad_w, pad_h = get_pad_size(w, h, roi_w, roi_h, 1.5, new_w / new_h, cateid)
        up, left = (pad_h - h) // 2, (pad_w - w) // 2
        down, right = pad_h - h - up, pad_w - w - left
        result = get_pad_image(image, left, up, right, down).resize((new_w, new_h))
        return result

    @register_process("microsoft/picasso/diffusion/process/random_mask")
    def _random_mask(
        self,
        image: Union[Image.Image, str],
        ratios: Optional[List[float]] = [0.8, 1.0],
        mask_full_image: Optional[bool] = False,
    ):
        ratio = np.random.choice(ratios)
        im_shape = image.size
        mask = Image.new("L", im_shape, 0)
        draw = ImageDraw.Draw(mask)
        size = (
            random.randint(0, int(im_shape[0] * ratio)),
            random.randint(0, int(im_shape[1] * ratio)),
        )
        # use this to always mask the whole image
        if mask_full_image:
            size = (int(im_shape[0] * ratio), int(im_shape[1] * ratio))
        limits = (im_shape[0] - size[0] // 2, im_shape[1] - size[1] // 2)
        center = (
            random.randint(size[0] // 2, limits[0]),
            random.randint(size[1] // 2, limits[1]),
        )
        draw_type = random.randint(0, 1)
        if draw_type == 0 or mask_full_image:
            draw.rectangle(
                (
                    center[0] - size[0] // 2,
                    center[1] - size[1] // 2,
                    center[0] + size[0] // 2,
                    center[1] + size[1] // 2,
                ),
                fill=255,
            )
        else:
            draw.ellipse(
                (
                    center[0] - size[0] // 2,
                    center[1] - size[1] // 2,
                    center[0] + size[0] // 2,
                    center[1] + size[1] // 2,
                ),
                fill=255,
            )
        return mask

    @register_process("microsoft/picasso/diffusion/process/random_lama_mask")
    def _random_lama_mask(
        self,
        image: Union[Image.Image, str],
        max_angle: Optional[int] = 180,
        max_length: Optional[int] = 60,
        max_width: Optional[int] = 20,
        min_times: Optional[int] = 0,
        max_times: Optional[int] = 10,
        choice_probs: Optional[List[float]] = None,
    ):
        choices = ["line", "circle", "square"]
        if choice_probs is None:
            choice_probs = [0.38, 0.24, 0.38]
        im_shape = image.size
        mask = Image.new("L", im_shape, 0)
        draw = ImageDraw.Draw(mask)
        for t in range(random.randint(min_times, max_times + 1)):
            choice_type = np.random.choice(choices, p=choice_probs)
            for _ in range(1 + random.randint(0, 5)):
                if choice_type == "line":
                    start = (
                        random.randint(0, im_shape[0]),
                        random.randint(0, im_shape[1]),
                    )
                    angle = random.randint(0, max_angle)
                    if t % 2 == 0:
                        angle = 360 - angle
                    length = 10 + random.randint(0, max_length)
                    width = 5 + random.randint(0, max_width)
                    end = (
                        start[0] + int(math.cos(math.radians(angle)) * length),
                        start[1] + int(math.sin(math.radians(angle)) * length),
                    )
                    draw.line([start, end], fill=255, width=width)
                elif choice_type == "circle":
                    center = (
                        random.randint(0, im_shape[0]),
                        random.randint(0, im_shape[1]),
                    )
                    radius = random.randint(0, max_length)
                    draw.ellipse(
                        (
                            center[0] - radius,
                            center[1] - radius,
                            center[0] + radius,
                            center[1] + radius,
                        ),
                        fill=255,
                    )
                elif choice_type == "square":
                    center = (
                        random.randint(0, im_shape[0]),
                        random.randint(0, im_shape[1]),
                    )
                    length = random.randint(0, max_length)
                    draw.rectangle(
                        (
                            center[0] - length,
                            center[1] - length,
                            center[0] + length,
                            center[1] + length,
                        ),
                        fill=255,
                    )

        return mask

    @register_process("microsoft/picasso/diffusion/process/random_object_mask")
    def _random_object_mask(
        self,
        obj_mask: Image.Image,
        choice_probs: Optional[List[float]] = None,
        mask_background: Optional[bool] = False,
    ):
        inv_obj_mask = ImageOps.invert(obj_mask)
        im_shape = obj_mask.size
        mask = Image.new("L", im_shape, 0)
        draw = ImageDraw.Draw(mask)
        choices = ["object", "circle", "square"]
        if choice_probs is None:
            choice_probs = [0.6, 0.2, 0.2]
        choice_type = np.random.choice(choices, p=choice_probs)
        x1, y1, x2, y2 = obj_mask.getbbox()
        if choice_type == "object":
            if not mask_background:
                mask.paste(obj_mask, (0, 0), obj_mask)
            else:
                mask.paste(inv_obj_mask, (0, 0), inv_obj_mask)
        if choice_type == "circle":
            if not mask_background:
                center = (
                    random.randint(x1, x2),
                    random.randint(y1, y2),
                )
                min_radius = max(
                    x2 - center[0], y2 - center[1], center[0] - x1, center[1] - y1
                )
                radius = random.randint(
                    max(x2 - center[0], y2 - center[1], center[0] - x1, center[1] - y1),
                    max(
                        min_radius + 1,
                        min(
                            center[0],
                            center[1],
                            im_shape[0] - center[0],
                            im_shape[1] - center[1],
                        ),
                    ),
                )
                draw.ellipse(
                    (
                        center[0] - radius,
                        center[1] - radius,
                        center[0] + radius,
                        center[1] + radius,
                    ),
                    fill=255,
                )
            else:
                regions = [
                    [(0, x1), (0, y1)],
                    [(0, x1), (y2, im_shape[1])],
                    [(x2, im_shape[0]), (0, y1)],
                    [(x2, im_shape[0]), (y2, im_shape[1])],
                    [(x1, x2), (0, y1)],
                    [(x1, x2), (y2, im_shape[1])],
                    [(0, x1), (y1, y2)],
                    [(x2, im_shape[0]), (y1, y2)],
                ]
                centers = [
                    (random.randint(*r[0]), random.randint(*r[1])) for r in regions
                ]
                radiuss = [
                    (
                        random.randint(
                            min(
                                20,
                                c[0] - r[0][0],
                                r[0][1] - c[0],
                                c[1] - r[1][0],
                                r[1][1] - c[1],
                            ),
                            min(
                                c[0] - r[0][0],
                                r[0][1] - c[0],
                                c[1] - r[1][0],
                                r[1][1] - c[1],
                            ),
                        )
                    )
                    for c, r in zip(centers, regions)
                ]
                # filter the centers that are too close to the object
                for center, radius in zip(centers, radiuss):
                    if radius < 10:
                        continue
                    draw.ellipse(
                        (
                            center[0] - radius,
                            center[1] - radius,
                            center[0] + radius,
                            center[1] + radius,
                        ),
                        fill=255,
                    )
        if choice_type == "square":
            if not mask_background:
                center = (
                    random.randint(x1, x2),
                    random.randint(y1, y2),
                )
                min_length = max(
                    x2 - center[0], y2 - center[1], center[0] - x1, center[1] - y1
                )
                length = random.randint(
                    max(x2 - center[0], y2 - center[1], center[0] - x1, center[1] - y1),
                    max(
                        min_length + 1,
                        min(
                            center[0],
                            center[1],
                            im_shape[0] - center[0],
                            im_shape[1] - center[1],
                        ),
                    ),
                )
                draw.rectangle(
                    (
                        center[0] - length,
                        center[1] - length,
                        center[0] + length,
                        center[1] + length,
                    ),
                    fill=255,
                )
            else:
                regions = [
                    [(0, x1), (0, y1)],
                    [(0, x1), (y2, im_shape[1])],
                    [(x2, im_shape[0]), (0, y1)],
                    [(x2, im_shape[0]), (y2, im_shape[1])],
                    [(x1, x2), (0, y1)],
                    [(x1, x2), (y2, im_shape[1])],
                    [(0, x1), (y1, y2)],
                    [(x2, im_shape[0]), (y1, y2)],
                ]
                centers = [
                    (random.randint(*r[0]), random.randint(*r[1])) for r in regions
                ]
                lengths = [
                    (
                        random.randint(
                            min(
                                20,
                                c[0] - r[0][0],
                                r[0][1] - c[0],
                                c[1] - r[1][0],
                                r[1][1] - c[1],
                            ),
                            min(
                                c[0] - r[0][0],
                                r[0][1] - c[0],
                                c[1] - r[1][0],
                                r[1][1] - c[1],
                            ),
                        )
                    )
                    for c, r in zip(centers, regions)
                ]
                # filter the centers that are too close to the object
                for center, length in zip(centers, lengths):
                    if length < 20:
                        continue
                    draw.rectangle(
                        (
                            center[0] - length,
                            center[1] - length,
                            center[0] + length,
                            center[1] + length,
                        ),
                        fill=255,
                    )
        return mask

    @register_process("microsoft/picasso/diffusion/process/outpainting/random_mask")
    def _outpainting_random_mask(
        self,
        image: Union[Image.Image, str],
        lower_ratio: Optional[float] = 0.1,
        upper_ratio: Optional[float] = 0.5,
        blur_radius: Optional[float] = 10.0,
    ):
        width, height = image.size
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)

        is_width = random.choice([True, False])
        mask_ratio = random.uniform(lower_ratio, upper_ratio)
        if is_width:
            mask_size = int(width * mask_ratio)
            left = mask_size // 2
            right = width - left
            draw.rectangle((0, 0, left, height), fill=255)
            draw.rectangle((right, 0, width, height), fill=255)
        else:
            mask_size = int(height * mask_ratio)
            top = mask_size // 2
            bottom = height - top
            draw.rectangle((0, 0, width, top), fill=255)
            draw.rectangle((0, bottom, width, height), fill=255)

        mask = mask.filter(ImageFilter.GaussianBlur(blur_radius))

        return mask

    @register_process("microsoft/picasso/diffusion/process/dominate_color")
    def _dominate_color(
        self,
        image: Image.Image,
        topk: Optional[int] = 5,
        return_image: Optional[bool] = False,
    ):
        image = image.convert("RGB")
        quantized_image = image.quantize(colors=topk, method=2)
        palette = quantized_image.getpalette()
        palette_colors = [tuple(palette[i : i + 3]) for i in range(0, len(palette), 3)]

        color_counts = Counter(quantized_image.getdata())
        topk_colors = [
            palette_colors[color] for color, count in color_counts.most_common(topk)
        ]
        if return_image:
            stripe_height = image.height // len(topk_colors)
            new_image = Image.new("RGB", image.size)
            for i, color in enumerate(topk_colors):
                new_image.paste(
                    color, (0, i * stripe_height, image.width, (i + 1) * stripe_height)
                )
            return new_image

        return topk_colors
