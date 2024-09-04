# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import re
import cv2
import hashlib
import numpy as np
import pandas as pd
from torch.hub import download_url_to_file
from unitorch.cli import CoreConfigureParser, GenericScript
from unitorch.cli import register_script


def zoom_in_effect(image, start_coords, end_coords, output_video, num_frames=60):
    """
    Args:
    - image: 输入图像（numpy数组）。
    - start_coords: 缩放起始区域的坐标 (x, y, w, h)。
    - end_coords: 缩放结束区域的坐标 (x, y, w, h)。
    - output_video: 输出视频文件路径。
    - num_frames: 视频帧数。
    """
    h, w, _ = image.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, 30.0, (w, h))

    def ease_in_out(t):
        """使用缓动函数（Ease In Out）来平滑过渡"""
        return t * t * (3 - 2 * t)

    for i in range(num_frames):
        t = i / (num_frames - 1)
        alpha = ease_in_out(t)  # 使用缓动函数调整 alpha

        # 计算当前帧的浮点缩放区域
        current_x = start_coords[0] * (1 - alpha) + end_coords[0] * alpha
        current_y = start_coords[1] * (1 - alpha) + end_coords[1] * alpha
        current_w = start_coords[2] * (1 - alpha) + end_coords[2] * alpha
        current_h = start_coords[3] * (1 - alpha) + end_coords[3] * alpha

        # 将浮点数转换为整数，并尽量保持精度
        zoomed_img = cv2.getRectSubPix(
            image,
            (int(current_w), int(current_h)),
            (current_x + current_w / 2, current_y + current_h / 2),
        )

        # 使用双线性插值进行缩放
        zoomed_img = cv2.resize(zoomed_img, (w, h), interpolation=cv2.INTER_LINEAR)

        # 写入视频帧
        out.write(zoomed_img)

    out.release()


@register_script("microsoft/picasso/script/video/opencv")
class OpenCVScript(GenericScript):
    def __init__(self, config: CoreConfigureParser):
        self.config = config

    def launch(self, **kwargs):
        config = self.config
        config.set_default_section("microsoft/picasso/script/video/opencv")

        data_file = config.getoption("data_file", None)
        names = config.getoption("names", None)
        if isinstance(names, str) and names.strip() == "*":
            names = None
        if isinstance(names, str):
            names = re.split(r"[,;]", names)
            names = [n.strip() for n in names]
        image_col = config.getoption("image_col", None)
        bboxes_col = config.getoption("bboxes_col", None)
        num_frames = config.getoption("num_frames", 60)
        output_folder = config.getoption("output_folder", "./output")

        os.makedirs(output_folder, exist_ok=True)

        data = pd.read_csv(
            data_file,
            names=names,
            sep="\t",
            quoting=3,
            header=None,
        )

        assert image_col in data.columns, f"Column {image_col} not found in data file"
        assert bboxes_col in data.columns, f"Column {bboxes_col} not found in data file"

        for _, row in data.iterrows():
            image_path = row[image_col]
            if image_path.startswith("http"):
                image_path = os.path.join(
                    output_folder, hashlib.md5(image_path.encode()).hexdigest() + ".jpg"
                )
                download_url_to_file(row[image_col], image_path)
            image_bboxes = row[bboxes_col]
            image = cv2.imread(image_path)
            h, w = image.shape[:2]
            start_coords = (0, 0, image.shape[1], image.shape[0])
            for image_bbox in image_bboxes.split(";"):
                bbox = image_bbox.split(",")
                x1, y1, x2, y2 = [int(b) for b in bbox]
                new_w, new_h = x2 - x1, y2 - y1
                ratio, new_ratio = w / h, new_w / new_h
                if new_ratio > ratio:
                    new_w, new_h = new_w, int(new_w / ratio)
                    x1, y1, x2, y2 = (
                        x1,
                        max(0, y1 - (new_h - (y2 - y1)) // 2),
                        x2,
                        min(h, y2 + (new_h - (y2 - y1)) // 2),
                    )
                else:
                    new_w, new_h = int(new_h * ratio), new_h
                    x1, y1, x2, y2 = (
                        max(0, x1 - (new_w - (x2 - x1)) // 2),
                        y1,
                        min(w, x2 + (new_w - (x2 - x1)) // 2),
                        y2,
                    )
                end_coords = (x1, y1, x2 - x1, y2 - y1)
                output_video = os.path.join(
                    output_folder, f"{os.path.basename(image_path)}_{image_bbox}.mp4"
                )
                zoom_in_effect(
                    image, start_coords, end_coords, output_video, num_frames=num_frames
                )
