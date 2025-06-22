# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import io
import decord
import imageio
import requests
import time
import base64
import json
import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from random import random
from PIL import Image, ImageOps, ImageFile, ImageFilter
from unitorch.utils import is_opencv_available
from unitorch.cli import (
    add_default_section_for_init,
    add_default_section_for_function,
    register_process,
)
from decord import cpu, gpu
decord.bridge.set_bridge("torch")
import torchvision

ImageFile.LOAD_TRUNCATED_IMAGES = True


class VideoProcessor:
    """
    Processor for image-related operations.
    """

    def __init__(
        self,
        video_type: Optional[str] = None,
        video_size: tuple = (256, 256),
        http_url: Optional[str] = None,
    ):
        """
        Initializes a new instance of the ImageProcessor.

        Args:
            image_type (Optional[str]): The type of the image. Defaults to None.
            image_size (tuple): The size of the image. Defaults to (256, 256).
            http_url (Optional[str]): The URL for fetching images. Defaults to None.
        """
        self.video_type = video_type
        self.video_size = video_size
        self.http_url = http_url

    @classmethod
    @add_default_section_for_init("microsoft/process/video")
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

    def _request_url(self, url):
        """
        Sends an HTTP request to the specified URL and returns the response.

        Args:
            url (str): The URL to request.

        Returns:
            The response received from the URL.
        """
        while True:
            try:
                doc = requests.get(url, timeout=600)
                return doc
            except:
                time.sleep(random() * 2)

    @register_process("microsoft/process/video/read")
    def _read(
        self,
        video,
        video_type=None,
    ):
        """
        Reads and processes an image.

        Args:
            image: The image to read and process.
            image_type (Optional[str]): The type of the image. Defaults to None.

        Returns:
            The processed image as a PIL Image object.
        """
        video_type = video_type if video_type is not None else self.video_type
        try:
            if video_type == "base64":
                video = io.BytesIO(base64.b64decode(video))

                frames = []
                with imageio.v3.imopen(
                    video, "r", plugin="pyav", format="mp4"
                ) as reader:
                    for frame in reader.iter():
                        img = Image.fromarray(frame)
                        frames.append(img)
                return frames

            if video_type == "hex":
                video = io.BytesIO(bytes.fromhex(video))

                frames = []
                with imageio.v3.imopen(
                    video, "r", plugin="pyav", format="mp4"
                ) as reader:
                    for frame in reader.iter():
                        img = Image.fromarray(frame)
                        frames.append(img)
                return frames

            if self.http_url is None:
                video = decord.VideoReader(
                    video, num_threads=0, ctx=cpu(0)
                )
                return video

            url = self.http_url.format(video)
            doc = self._request_url(url)
            if doc.status_code != 200 or doc.content == b"":
                raise ValueError(f"can't find the video {video}")

            video = io.BytesIO(doc.content)
            frames = []
            with imageio.v3.imopen(video, "r", plugin="pyav", format="mp4") as reader:
                for frame in reader.iter():
                    img = Image.fromarray(frame)
                    frames.append(img)
            return frames

        except Exception as e:
            logging.debug(f"core/process/video/read use fake video for {video} {e}")
            return []

    @register_process("microsoft/process/video/sample")
    def _sample(
        self,
        video: Union[decord.VideoReader, str, List[Image.Image]],
        freq: Optional[int] = 1,
        num: Optional[int] = None,
        mode: Optional[str] = "middle",
        target_fps: Optional[float] = None,
    ):
        """
        Samples frames from a video.

        Args:
            video: The video to sample frames from.
            freq (Optional[int]): The frequency of frames to sample. Defaults to 0.
            num (Optional[int]): The number of frames to sample. Defaults to None.
            mode (Optional[str]): The mode of sampling. Defaults to "random".
        Returns:
            A list of sampled frames.
        """
        if isinstance(video, str):
            video = decord.VideoReader(
                video, num_threads=0, ctx=cpu(0)
            )

        if isinstance(video, decord.VideoReader):
            fps = video.get_avg_fps()  # note that the fps here is float.
            if target_fps is not None:
                freq = round(fps / target_fps)
            print(f"video fps: {fps}, freq: {freq}, target_fps: {target_fps}")

        vlen = len(video)
        print(f"video length: {vlen}")
        indices = list(range(vlen))
        if freq > 1:
            indices = indices[:: (freq)]
        if num is None:
            num = len(indices)
        num = min(num, len(indices))
        print(f"indices: {indices}, num: {num}, mode: {mode}")

        if mode == "random":
            start = int(random() * (len(indices) - num))
            end = start + num
            frames = indices[start:end]
        elif mode == "first":
            frames = indices[:num]
        elif mode == "last":
            frames = indices[-num:]
        elif mode == "middle":
            start = max((0, (len(indices) - num) // 2))
            end = start + num
            frames = indices[start:end]
            print(f"middle frames: {frames}")
        else:
            raise ValueError(f"Unknown sampling mode: {mode}")
        
        samples = []
        if isinstance(video, decord.VideoReader):
            frames = video.get_batch(frames)
            frames = frames.permute(0, 3, 1, 2)
            print(f"frames shape: {frames.shape}")
            for frame in frames:
                frame = torchvision.transforms.functional.to_pil_image(frame).convert("RGB")
                print(f"frame size: {frame.size}")
                samples.append(frame)
        else:
            for i in frames:
                frame = video[i].convert("RGB")
                samples.append(frame)
        print(f"sampled frames count: {len(samples)}")
        return samples
    
