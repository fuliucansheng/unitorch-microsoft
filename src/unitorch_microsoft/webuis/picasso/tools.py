# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import io
import torch
import random
import gc
import cv2
import math
import numpy as np
import gradio as gr
from PIL import (
    Image,
    ImageFilter,
    ImageOps,
    ImageEnhance,
    ImageFile,
    ImageChops,
    ImageDraw,
)
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from unitorch.utils import is_opencv_available
from unitorch.cli import CoreConfigureParser, GenericWebUI
from unitorch.cli import register_webui
from unitorch.cli.webuis import (
    create_element,
    create_accordion,
    create_row,
    create_column,
    create_group,
    create_tab,
    create_tabs,
    create_blocks,
    create_pretrain_layout,
)
from unitorch.cli.webuis import SimpleWebUI

ImageFile.LOAD_TRUNCATED_IMAGES = True


class CopyWebUI(SimpleWebUI):
    def __init__(self, config: CoreConfigureParser):
        # create elements
        product_image = create_element("image", "Product Image")
        background_image = create_element("image", "Background Image")
        height = create_element(
            "slider", "Height", min_value=0, max_value=1024, step=1, default=0
        )
        width = create_element(
            "slider", "Width", min_value=0, max_value=1024, step=1, default=0
        )
        threshold = create_element(
            "slider", "White Threshold", min_value=0, max_value=255, step=1, default=200
        )
        generate = create_element("button", "Generate")
        output_image = create_element("image", "Output Image")

        left = create_column(
            create_row(product_image, background_image),
            height,
            width,
            threshold,
            generate,
        )
        right = create_column(output_image)
        iface = create_blocks(create_row(left, right))

        # create events
        iface.__enter__()

        generate.click(
            fn=self.copy,
            inputs=[product_image, background_image, height, width, threshold],
            outputs=[output_image],
            trigger_mode="once",
        )

        iface.__exit__()

        super().__init__(config, iname="Copy", iface=iface)

    def copy(
        self,
        product,
        background,
        height: Optional[int] = 0,
        width: Optional[int] = 0,
        threshold: Optional[int] = 200,
    ):
        if height > 0 and width > 0:
            product = product.resize((width, height), resample=Image.LANCZOS)
            background = background.resize((width, height), resample=Image.LANCZOS)
        if product.size != background.size:
            product = product.resize(background.size, resample=Image.LANCZOS)

        product = product.convert("RGBA")
        background = background.convert("RGBA")
        product_data = product.getdata()
        new_product_data = []
        for item in product_data:
            if item[0] > threshold and item[1] > threshold and item[2] > threshold:
                new_product_data.append((255, 255, 255, 0))
            else:
                new_product_data.append(item)
        product.putdata(new_product_data)
        background.paste(product, (0, 0), product)
        return background


class InpaintWebUI(SimpleWebUI):
    def __init__(self, config: CoreConfigureParser):
        # create elements
        image = create_element("image_editor", "Image")
        mask = create_element("image", "Mask")
        radius = create_element(
            "slider", "Radius", min_value=0, max_value=50, step=1, default=3
        )
        generate = create_element("button", "Generate")
        output_image = create_element("image", "Output Image")

        left = create_column(create_row(image, mask), radius, generate)
        right = create_column(output_image)
        iface = create_blocks(create_row(left, right))

        # create events
        iface.__enter__()
        image.change(fn=self.composite_images, inputs=[image], outputs=[mask])

        generate.click(
            fn=self.smooth_mask,
            inputs=[image, mask, radius],
            outputs=[output_image],
            trigger_mode="once",
        )

        iface.__exit__()

        super().__init__(config, iname="Inpaint", iface=iface)

    def composite_images(self, images):
        if images is None:
            return None
        layers = images["layers"]
        if len(layers) == 0:
            return None
        image = layers[0]
        for i in range(1, len(layers)):
            image = Image.alpha_composite(image, layers[i])
        image = image.convert("L")
        image = image.point(lambda p: p < 5 and 255)
        image = ImageOps.invert(image)
        return image

    def smooth_mask(self, image, mask, radius):
        image_np = np.array(image["background"].convert("RGB"))
        mask_np = np.array(mask.convert("L")).astype(np.uint8)

        _, binary_mask = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
        inpainted_image = cv2.inpaint(image_np, binary_mask, radius, cv2.INPAINT_TELEA)
        result_image = Image.fromarray(inpainted_image)
        return result_image


class OutpaintWebUI(SimpleWebUI):
    def __init__(self, config: CoreConfigureParser):
        # create elements
        image1 = create_element("image", "Image")
        ratio1 = create_element("radio", "Ratio", values=[0.5, 1, 1.9, 2], default=1)
        generate1 = create_element("button", "Generate")

        image2 = create_element("image", "Image")
        crop_w = create_element(
            "slider", "Crop Width", min_value=0, max_value=1024, step=1, default=0
        )
        crop_h = create_element(
            "slider", "Crop Height", min_value=0, max_value=1024, step=1, default=0
        )
        mask_side = create_element(
            "radio", "Mask Width/Height", values=["width", "height"], default="width"
        )
        mask_ratio = create_element(
            "slider", "Mask Ratio", min_value=0, max_value=1, step=0.01, default=0.5
        )
        generate2 = create_element("button", "Generate")

        output_image = create_element("image", "Output Image")
        output_smooth_image = create_element("image", "Output Smooth Image")
        output_mask = create_element("image", "Output Mask")

        tab1 = create_tab(image1, ratio1, generate1, name="Infer")
        tab2 = create_tab(
            image2,
            create_row(crop_w, crop_h),
            create_row(mask_side, mask_ratio),
            generate2,
            name="Train",
        )
        left = create_tabs(tab1, tab2)

        right = create_column(output_image, output_smooth_image, output_mask)
        iface = create_blocks(create_row(left, right))

        # create events
        iface.__enter__()

        generate1.click(
            fn=self.process_infer,
            inputs=[image1, ratio1],
            outputs=[output_image, output_smooth_image, output_mask],
            trigger_mode="once",
        )

        generate2.click(
            fn=self.process_train,
            inputs=[image2, crop_w, crop_h, mask_side, mask_ratio],
            outputs=[output_image, output_smooth_image, output_mask],
            trigger_mode="once",
        )

        iface.__exit__()

        super().__init__(config, iname="Outpaint", iface=iface)

    def process_infer(self, image, ratio):
        size_dict = {
            "1": (1024, 1024),
            "0.5": (1024, 2048),
            "1.9": (2048, 1072),
            "2": (2048, 1024),
        }

        width, height = image.size
        size = size_dict[str(ratio)]
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

        image_np = np.array(new_image.convert("RGB"))
        mask_np = np.array(mask.convert("L")).astype(np.uint8)

        _, binary_mask = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
        inpainted_image = cv2.inpaint(image_np, binary_mask, 10, cv2.INPAINT_TELEA)
        result_image = Image.fromarray(inpainted_image)

        return new_image, result_image, mask

    def process_train(self, image, crop_w, crop_h, mask_side, mask_ratio):
        width, height = image.size
        size = (crop_w, crop_h)
        left = (width - size[0]) // 2
        top = (height - size[1]) // 2
        left, top = max(0, left), max(0, top)
        right = left + size[0]
        bottom = top + size[1]
        right, bottom = min(width, right), min(height, bottom)
        image = image.crop((left, top, right, bottom))

        width, height = image.size
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)

        is_width = mask_side == "width"
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

        image.paste((255, 255, 255), mask=mask)

        image_np = np.array(image.convert("RGB"))
        mask_np = np.array(mask.convert("L")).astype(np.uint8)

        _, binary_mask = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
        inpainted_image = cv2.inpaint(image_np, binary_mask, 10, cv2.INPAINT_TELEA)
        result_image = Image.fromarray(inpainted_image)

        return image, result_image, mask


class BlurWebUI(SimpleWebUI):
    def __init__(self, config: CoreConfigureParser):
        # create elements
        image = create_element("image", "Image")
        radius = create_element(
            "slider", "Radius", min_value=0, max_value=50, step=1, default=15
        )
        generate = create_element("button", "Generate")
        output_image = create_element("image", "Output Image")

        left = create_column(image, radius, generate)
        right = create_column(output_image)
        iface = create_blocks(create_row(left, right))

        # create events
        iface.__enter__()

        generate.click(
            fn=self.blur,
            inputs=[image, radius],
            outputs=[output_image],
            trigger_mode="once",
        )

        iface.__exit__()

        super().__init__(config, iname="Blur", iface=iface)

    def blur(self, image, radius: Optional[int] = 15):
        blurred = image.filter(ImageFilter.GaussianBlur(radius))
        mask = Image.new("L", blurred.size, 0)
        mask_draw = Image.new("L", blurred.size, 0)
        mask_draw.paste(
            255,
            (
                image.width // 4,
                image.height // 4,
                3 * image.width // 4,
                3 * image.height // 4,
            ),
        )
        mask = mask_draw.filter(ImageFilter.GaussianBlur(50))
        return Image.composite(image, blurred, mask)


class BrightnessWebUI(SimpleWebUI):
    def __init__(self, config: CoreConfigureParser):
        # create elements
        image = create_element("image", "Image")
        factor = create_element(
            "slider", "Factor", min_value=-1, max_value=1, step=0.01, default=0
        )
        generate = create_element("button", "Generate")
        output_image = create_element("image", "Output Image")

        left = create_column(image, factor, generate)
        right = create_column(output_image)
        iface = create_blocks(create_row(left, right))

        # create events
        iface.__enter__()

        generate.click(
            fn=self.brightness,
            inputs=[image, factor],
            outputs=[output_image],
            trigger_mode="once",
        )

        iface.__exit__()

        super().__init__(config, iname="Brightness", iface=iface)

    def brightness(self, image, factor: Optional[float] = 0):
        bright = ImageEnhance.Brightness(image).enhance(factor)
        w, h = bright.size
        mask = Image.new("L", (w, h), 255)
        for x in range(w):
            for y in range(h):
                dx = abs(x - w / 2)
                dy = abs(y - h / 2)
                d = np.sqrt(dx**2 + dy**2)
                mask.putpixel((x, y), int(255 * (1 - d / max(w, h))))
        return Image.composite(image, bright, mask)


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


class ControlNetWebUI(SimpleWebUI):
    def __init__(self, config: CoreConfigureParser):
        # create elements
        image = create_element("image", "Image")
        cate = create_element(
            "slider", "Cate", min_value=0, max_value=4559, step=1, default=1
        )
        width = create_element(
            "slider", "Width", min_value=0, max_value=1024, step=1, default=832
        )
        height = create_element(
            "slider", "Height", min_value=0, max_value=1024, step=1, default=384
        )
        generate = create_element("button", "Generate")
        output_image = create_element("image", "Output Image")

        left = create_column(image, cate, width, height, generate)
        right = create_column(output_image)
        iface = create_blocks(create_row(left, right))

        # create events
        iface.__enter__()

        generate.click(
            fn=self.serve,
            inputs=[image, cate, width, height],
            outputs=[output_image],
            trigger_mode="once",
        )

        iface.__exit__()

        super().__init__(config, iname="ControlNet", iface=iface)

    def serve(
        self,
        image: Image.Image,
        cateid: int,
        width: Optional[int] = 832,
        height: Optional[int] = 384,
    ):
        """
        Detects edges in the image using the Canny algorithm.

        Args:
            image (Image.Image): The image to detect edges in.

        Returns:
            The image with detected edges as a PIL Image object.
        """
        w, h = image.size
        new_w, new_h = width, height
        roi = get_roi_bbox(image)
        roi_w, roi_h = roi[2] - roi[0], roi[3] - roi[1]

        pad_w, pad_h = get_pad_size(w, h, roi_w, roi_h, 1.5, new_w / new_h, cateid)
        up, left = (pad_h - h) // 2, (pad_w - w) // 2
        down, right = pad_h - h - up, pad_w - w - left
        result = get_pad_image(image, left, up, right, down).resize(
            (new_w, new_h), resample=Image.LANCZOS
        )
        return result


class ToolsWebUI(SimpleWebUI):
    def __init__(self, config: CoreConfigureParser):
        webuis = [
            CopyWebUI(config),
            BlurWebUI(config),
            BrightnessWebUI(config),
            ControlNetWebUI(config),
            InpaintWebUI(config),
            OutpaintWebUI(config),
        ]
        iface = gr.TabbedInterface(
            [webui.iface for webui in webuis],
            tab_names=[webui.iname for webui in webuis],
        )
        super().__init__(config, iname="Tools", iface=iface)
