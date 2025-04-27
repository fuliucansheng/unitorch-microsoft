# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import io
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
import hashlib
import imageio
import math
from torch.hub import download_url_to_file
from PIL import Image, ImageOps, ImageFile, ImageFilter
from unitorch.utils import is_opencv_available
from unitorch.cli.models.image_utils import ImageProcessor
from unitorch.cli import (
    add_default_section_for_init,
    add_default_section_for_function,
    register_process,
)

# ImageFile.LOAD_TRUNCATED_IMAGES = True


def sample_frames(num_frames, vlen, sample="uniform", **kwargs):
    """
    num_frames: The number of frames to sample.
    vlen: The length of the video.
    sample: The sampling method.
        choices of frame_sample:
        - 'equally spaced': sample frames equally spaced
            e.g.,1s video has 30 frames, when 'es_interval'=8, we sample frames with spacing of 8
        - 'proportional': sample frames proportional to the length of the frames in one second
            e.g., 1s video has 30 frames, when 'prop_factor'=3, we sample frames with spacing of 30/3=10
        - 'random': sample frames randomly (not recommended)
        - 'uniform': sample frames uniformly (not recommended)
    kwargs["start_frame"]: The starting frame index. If it is not None, then it will be used as the starting frame index.
    """
    acc_samples = min(num_frames, vlen)
    if sample in ["rand", "uniform"]:
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == "rand":
            frame_idxs = [random.choice(range(x[0], x[1])) for x in ranges]
        elif kwargs["start_frame"] is not None:
            frame_idxs = [x[0] + kwargs["start_frame"] for x in ranges]
        elif sample == "uniform":
            frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]
    elif sample in ["equally spaced", "proportional"]:
        if sample == "equally spaced":
            raise NotImplementedError  # need to pass in the corresponding parameters
        else:
            interval = round(kwargs["fps"] / kwargs["sample_factor"])
            needed_frames = (acc_samples - 1) * interval

            if kwargs["start_frame"] is not None:
                start = kwargs["start_frame"]
            else:
                if vlen - needed_frames - 1 < 0:
                    start = 0
                else:
                    start = random.randint(0, vlen - needed_frames - 1)
            frame_idxs = np.linspace(
                start=start, stop=min(vlen - 1, start + needed_frames), num=acc_samples
            ).astype(int)
    elif sample == "middle":
        interval = round(kwargs["fps"] / kwargs["sample_factor"])
        needed_frames = (acc_samples - 1) * interval
        kwargs["start_frame"] = max(0, vlen // 2 - needed_frames // 2)
        frame_idxs = np.linspace(
            kwargs["start_frame"],
            min(kwargs["start_frame"] + needed_frames, vlen - 1),
            acc_samples,
            dtype=int,
        )

    elif sample == "fix":
        assert kwargs["sample_rate"] != None
        frame_idxs = [
            min(math.floor(vlen * i), vlen - 1) for i in kwargs["sample_rate"]
        ]
    else:
        raise NotImplementedError

    return frame_idxs


class VideoProcessor(ImageProcessor):
    """
    Processor for video-related operations.
    """

    def __init__(
        self,
        video_type: Optional[str] = None,
        sample_strategy: Optional[str] = None,
        http_url: Optional[str] = None,
        sample_frame_num: Optional[
            int
        ] = None,  # sample_frame_num=1, sample 1 frame from video
        start_frame: Optional[int] = None,  # start_frame=0, start from the first frame
        sample_rate: Optional[
            List[float]
        ] = None,  # [0.1,0.5,0.9] get 10%, 50%, 90% of the video
        sample_factor: Optional[int] = None,  # sample_factor=1 means 1 frame per second
    ):
        """
        Initializes a new instance of the ImageProcessor.

        Args:
            image_type (Optional[str]): The type of the image. Defaults to None.
            image_size (tuple): The size of the image. Defaults to (256, 256).
            http_url (Optional[str]): The URL for fetching images. Defaults to None.
        """
        super().__init__()
        self.video_type = video_type
        self.http_url = http_url
        self.sample_strategy = sample_strategy
        self.sample_frame_num = sample_frame_num
        self.start_frame = start_frame
        self.sample_rate = sample_rate
        self.sample_factor = sample_factor
        self.tmp_download_folder = "./tmp"
        if not os.path.exists(self.tmp_download_folder):
            os.makedirs(self.tmp_download_folder)

    @classmethod
    @add_default_section_for_init("microsoft/adinsights/process/video")
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

    def _download_video(self, video_path):
        """
        Downloads a video from the specified URL.

        Args:
            url (str): The URL to download the video from.

        Returns:
            The path to the downloaded video file.
        """

        name = video_path
        try:
            _, ext = os.path.splitext(video_path)
        except:
            ext = ".mp4"
        name = hashlib.md5(video_path.encode()).hexdigest() + ext
        name = os.path.join(self.tmp_download_folder, name)
        download_url_to_file(video_path, name, progress=False)
        return name

    @register_process("microsoft/adinsights/process/video/read")
    def _read(
        self,
        video,
        video_type=None,
        sample_strategy=None,
        sample_frame_num=None,
        start_frame=None,
        sample_rate=None,
        sample_factor=None,
    ):
        """
        Reads and processes an image.

        Args:
            image: The image to read and process.
            image_type (Optional[str]): The type of the image. Defaults to None.

        Returns:
            The processed video as a list of PIL Image objects.
        """

        def _read_video(path):
            reader = imageio.get_reader(path)
            meta_data = reader.get_meta_data()
            video_len = reader.count_frames()
            fps = meta_data["fps"]
            meta_info = {"fps": fps, "video_len": video_len}
            return reader, meta_info

        video_type = video_type if video_type is not None else self.video_type
        sample_strategy = (
            sample_strategy if sample_strategy is not None else self.sample_strategy
        )
        sample_frame_num = (
            sample_frame_num if sample_frame_num is not None else self.sample_frame_num
        )
        start_frame = start_frame if start_frame is not None else self.start_frame
        sample_rate = sample_rate if sample_rate is not None else self.sample_rate
        sample_factor = (
            sample_factor if sample_factor is not None else self.sample_factor
        )

        # try:
        if True:
            print(f"process video {video}")
            if video.startswith("http://") or video.startswith("https://"):
                video = self._download_video(video)

            reader, meta_info = _read_video(video)
            if sample_strategy != None:
                sample_kwags = {
                    "fps": meta_info["fps"],
                    "sample_factor": sample_factor,
                    "sample_rate": sample_rate,
                    "start_frame": start_frame,
                }
                samples = sample_frames(
                    num_frames=sample_frame_num,
                    vlen=meta_info["video_len"],
                    sample=sample_strategy,
                    **sample_kwags,
                )
            else:
                samples = list(range(meta_info["video_len"]))
            print(f"sample frames {samples}")

            frames = []
            for frame_id in samples:
                frame = reader.get_data(frame_id)
                frame = Image.fromarray(frame).convert("RGB")
                print(f"image size {frame.size}")
                frames.append(frame)

            return frames
        """
        except Exception as e:
            logging.error(f"Error reading video: {e}")
            logging.debug(f"core/process/video/read use fake image for {video}")
            return Image.new("RGB", self.image_size, (255, 255, 255))
        """
