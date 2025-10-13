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
import torchvision
import torch.nn as nn
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from random import random
import hashlib
import imageio
import math
import re
from torch.hub import download_url_to_file
from PIL import Image, ImageOps, ImageFile, ImageFilter
from unitorch.utils import is_opencv_available
from unitorch.cli.models.image_utils import ImageProcessor
from unitorch.cli import (
    add_default_section_for_init,
    add_default_section_for_function,
    register_process,
)
import decord
import fire
import time
from tqdm import tqdm


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
    assert num_frames != None and num_frames > 0
    assert vlen != None and vlen > 0
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
            assert (
                "fps" in kwargs
                and "sample_factor" in kwargs
                and kwargs["fps"] > 0
                and kwargs["sample_factor"] > 0
            )
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
        assert (
            "fps" in kwargs
            and "sample_factor" in kwargs
            and kwargs["fps"] > 0
            and kwargs["sample_factor"] > 0
        )
        interval = round(kwargs["fps"] / kwargs["sample_factor"])
        needed_frames = (acc_samples - 1) * interval
        fix_start = max(0, vlen // 2 - needed_frames // 2)
        frame_idxs = np.linspace(
            fix_start,
            min(fix_start + needed_frames, vlen - 1),
            acc_samples,
            dtype=int,
        )

    elif sample == "fix":
        assert "sample_rate" in kwargs and kwargs["sample_rate"] != None
        frame_idxs = [
            min(math.floor(vlen * i), vlen - 1) for i in kwargs["sample_rate"]
        ]
    elif sample == "middlefix":
        assert (
            "fps" in kwargs
            and "sample_factor" in kwargs
            and kwargs["fps"] > 0
            and kwargs["sample_factor"] > 0
        )
        interval = round(kwargs["fps"] / kwargs["sample_factor"])
        needed_frames = (acc_samples - 1) * interval
        fix_start = max(0, vlen // 2 - needed_frames // 2)
        middle_idxs = np.linspace(
            fix_start,
            min(fix_start + needed_frames, vlen - 1),
            acc_samples,
            dtype=int,
        )

        assert "sample_rate" in kwargs and kwargs["sample_rate"] != None
        vlensample = len(middle_idxs)
        idxs = [
            min(math.floor(vlensample * i), vlensample - 1)
            for i in kwargs["sample_rate"]
        ]
        frame_idxs = [middle_idxs[i] for i in idxs]
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
            Union[str, List[float]]
        ] = None,  # [0.1,0.5,0.9] get 10%, 50%, 90% of the video
        sample_factor: Optional[
            int
        ] = None,  # sample_factor=1 means 1 frame per second, same as target fps
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
        self.sample_rate = self.Tolist_float(sample_rate)
        self.sample_factor = sample_factor
        self.tmp_download_folder = "./tmp"

    def Tolist_float(self, sample_rate):
        if sample_rate != None and isinstance(sample_rate, str):
            sample_rates = re.split(r"[,;]", sample_rate)
            sample_rate = [float(i) for i in sample_rates]
        if sample_rate != None and not isinstance(sample_rate, list):
            sample_rate = [sample_rate]
        return sample_rate

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
        if not os.path.exists(self.tmp_download_folder):
            os.makedirs(self.tmp_download_folder, exist_ok=True)
        name = hashlib.md5(video_path.encode()).hexdigest() + ext
        name = os.path.join(self.tmp_download_folder, name)
        download_url_to_file(video_path, name, progress=False)
        return name

    def _read_video_imageio(self, video_path):
        reader = imageio.get_reader(video_path)
        meta_data = reader.get_meta_data()
        video_len = reader.count_frames()
        fps = meta_data["fps"]
        meta_info = {"fps": fps, "video_len": video_len}
        return reader, meta_info

    def _sample_video_imageio(self, samples, reader):
        frames = []
        for frame_id in samples:
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame).convert("RGB")
            frames.append(frame)

        return frames

    def _read_video_decord(self, video_path):
        from decord import cpu, gpu

        decord.bridge.set_bridge("torch")
        video_reader = decord.VideoReader(
            video_path, num_threads=0, ctx=cpu(0)
        )  # width,height//check #gpu only works for single process
        vlen = len(video_reader)
        fps = video_reader.get_avg_fps()  # note that the fps here is float.
        meta_info = {"fps": fps, "video_len": vlen}
        frame = video_reader[0]
        height, width, _ = frame.shape
        return video_reader, meta_info

    def _sample_video_decord(self, samples, reader):
        result = []
        frames = reader.get_batch(samples)
        # frames = reader.get_key_indices()
        frames = frames.permute(0, 3, 1, 2)

        for frame in frames:
            frame = torchvision.transforms.functional.to_pil_image(frame).convert("RGB")
            result.append(frame)

        return result

    @register_process("microsoft/adinsights/process/video/read")
    def _read(
        self,
        video,
        sample_strategy=None,
        sample_frame_num=None,
        start_frame=None,
        sample_rate=None,
        sample_factor=None,
        video_type=None,
        middle_offset=None,
    ):
        """
        Reads and processes an image.

        Args:
            image: The image to read and process.
            image_type (Optional[str]): The type of the image. Defaults to None.

        Returns:
            The processed video as a list of PIL Image objects.
        """

        video_type = video_type if video_type is not None else self.video_type
        sample_strategy = (
            sample_strategy if sample_strategy is not None else self.sample_strategy
        )
        sample_frame_num = (
            sample_frame_num if sample_frame_num is not None else self.sample_frame_num
        )
        start_frame = start_frame if start_frame is not None else self.start_frame
        sample_rate = sample_rate if sample_rate is not None else self.sample_rate
        sample_rate = self.Tolist_float(sample_rate)
        sample_factor = (
            sample_factor if sample_factor is not None else self.sample_factor
        )

        if sample_frame_num is None and sample_rate != None:
            sample_frame_num = len(sample_rate)

        assert sample_frame_num != None and sample_frame_num > 0
        assert sample_strategy != None, "sample_strategy is None"

        # print(
        #    f"sample_strategy: {sample_strategy} sample_rate: {sample_rate} sample_frame_num: {sample_frame_num} sample_factor: {sample_factor} start_frame: {start_frame}"
        # )
        def fix_param_samplerate(
            sample_strategy, sample_rate, middle_offset_in_duration, fps, video_len
        ):
            if (
                sample_strategy == "fix"
                and sample_rate == None
                and middle_offset_in_duration != None
            ):
                frame_offset = middle_offset_in_duration * fps
                middle_frame = video_len // 2
                start_frame = max(0, int(middle_frame - frame_offset))
                sample_rate = [
                    min(1.0, max(0.0, float(start_frame) / float(video_len)))
                ]
                print(
                    f"fix sample_rate: {sample_rate} for video_len: {video_len} fps: {fps} middle_offset_in_duration: {middle_offset_in_duration}"
                )
            return sample_rate

        try:
            # print(f"process video {video}")
            if video.startswith("http://") or video.startswith("https://"):
                video = self._download_video(video)

            reader, meta_info = self._read_video_decord(video)
            if sample_strategy != None:
                sample_kwags = {
                    "fps": meta_info["fps"],
                    "sample_factor": sample_factor,
                    "sample_rate": fix_param_samplerate(
                        sample_strategy,
                        sample_rate,
                        middle_offset,
                        meta_info["fps"],
                        meta_info["video_len"],
                    ),
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

            frames = self._sample_video_decord(samples, reader)

            return frames

        except Exception as e:
            logging.error(f"Error reading video: {e}")
            logging.error(f"core/process/video/read use fake image for {video}")
            return [Image.new("RGB", (256, 256), (255, 255, 255))]


def get_base64(image):
    # image = Image.open(image).convert("RGB")
    if np.all(np.array(image) == [255, 255, 255]):
        return None
    image_buffer = io.BytesIO()
    image.save(image_buffer, format="JPEG")
    image_buffer.seek(0)
    return base64.b64encode(image_buffer.getvalue()).decode()


def azure_login(connect_key, account_name, container_name):
    """
    intall required packages: pip3 install azure-storage-blob azure-identity
    """
    from azure.storage.blob import BlobServiceClient
    from azure.storage.blob import BlobClient, ContentSettings

    connect_str = (
        "DefaultEndpointsProtocol=https;AccountName="
        + account_name
        + ";AccountKey="
        + connect_key
        + ";EndpointSuffix=core.windows.net"
    )
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    container_client = blob_service_client.get_container_client(container_name)
    return container_client


def get_azureurl(data, container_client, savename, container_name, subfolder):
    from azure.storage.blob import BlobServiceClient
    from azure.storage.blob import BlobClient, ContentSettings

    if np.all(np.array(data) == [255, 255, 255]):
        return None

    img_bytes = io.BytesIO()
    data.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    # print(f"type of data: {type(data)} type of img_bytes: {type(img_bytes)}")
    try:
        remote_name = subfolder + "/" + savename
        # print("remote_name: ", remote_name)
        image_blob = container_client.get_blob_client(remote_name)
        image_blob.upload_blob(img_bytes, overwrite=True)
        url = f"https://{account_name}.blob.core.windows.net/i2v/{container_name}/{remote_name}"
        # print(url)
        return url
    except Exception as e:
        # print("upload img2azure failed {!r}".format(e))
        return None


def process_chunk(
    videos,
    chunk_start,
    chunk_size,
    process_id,
    cache_dir,
    num_processes,
    total_rows,
    sample_factor,
    sample_strategy,
    sample_frame_num,
    sample_rate,
    data_type,
    connect_key,
    account_name,
    subfolder,
    resize_image_height,
    resize_image_width,
    res_file,
    lock,
):
    if process_id == num_processes - 1:
        chunk_size = total_rows - chunk_start + 1
    chunks = videos[chunk_start : chunk_start + chunk_size]
    print(
        f"Worker {process_id} Processing rows: {chunk_start} to {chunk_start + chunk_size - 1} \n"
    )

    processor = VideoProcessor(
        sample_factor=sample_factor,
        sample_strategy=sample_strategy,
        sample_frame_num=sample_frame_num,
        sample_rate=sample_rate,
    )

    # res_file = os.path.join(cache_dir, f"proc_{process_id}.tsv")
    if data_type == "url":
        container_name = "videoproc"
        container_client = azure_login(connect_key, account_name, container_name)
    else:
        container_name = "videoproc"
        container_client = None

    write_str = ""
    for idx, video in enumerate(chunks):
        if idx % 30 == 0 and write_str != "":
            with lock:
                writer = open(res_file, "a+")
                writer.write(write_str)
                writer.flush()
                writer.close()
            write_str = ""
        frames = processor._read(video)
        for index, frame in enumerate(frames):
            if np.all(np.array(frame) == [255, 255, 255]):
                print(f"skip placeholder image for {video} at index {index}")
                continue
            if resize_image_height != None or resize_image_width != None:
                if resize_image_height != None and resize_image_width != None:
                    resize_image_size = (resize_image_width, resize_image_height)
                elif resize_image_width != None:
                    resize_image_size = (
                        resize_image_width,
                        int(frame.size[1] * resize_image_width / frame.size[0]),
                    )
                elif resize_image_height != None:
                    resize_image_size = (
                        int(frame.size[0] * resize_image_height / frame.size[1]),
                        resize_image_height,
                    )
                frame = frame.resize(resize_image_size)

            if data_type == "url":
                img_url = get_azureurl(
                    frame,
                    container_client,
                    video + f".{index}.png",
                    container_name,
                    subfolder,
                )
                if img_url != None:
                    write_str += video + f".{index}.png" + "\t" + img_url + "\n"
            else:
                base64_str = get_base64(frame)
                if base64_str != None:
                    write_str += video + f".{index}.png" + "\t" + base64_str + "\n"
    if write_str != "":
        with lock:
            writer = open(res_file, "a+")
            writer.write(write_str)
            writer.flush()
            writer.close()


def extract_frame(
    data_file,
    names,
    video_col="video",
    sample_strategy="middlefix",
    sample_frame_num=81,
    sample_factor=16,
    sample_rate=[0.1, 0.5, 0.9],
    cache_dir="output",
    data_type="base64",
    connect_key=None,
    account_name="i2v",
    subfolder="ExtractFrame",
    resize_image_height=None,
    resize_image_width=None,
):
    import re
    import pandas as pd
    import multiprocessing as mp

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
    os.makedirs(cache_dir, exist_ok=True)
    output_file = f"{cache_dir}/output.tsv"
    if os.path.exists(output_file):
        uniques = []
        with open(output_file, "r") as f:
            for line in f.readlines():
                videoid, emb = line.strip().split("\t")
                uniques.append(videoid)
        data = data[
            ~data.apply(
                lambda x: x[video_col] in uniques,
                axis=1,
            )
        ]
    print(f"Data loaded, total rows to move: {len(data)}")
    assert video_col in data.columns, f"Column {video_col} not found in data."
    videos = data[video_col].to_list()

    num_processes = mp.cpu_count()
    total_rows = len(videos)
    chunk_size = total_rows // num_processes
    start = time.time()
    print(f"need to process {total_rows} videos")

    if data_type == "url":
        assert connect_key != None, "connect_key is None"

    lock = mp.Lock()

    processes = []
    print(f"need to process {total_rows} videos")
    for i in range(num_processes):
        p = mp.Process(
            target=process_chunk,
            args=(
                videos,
                i * chunk_size,
                chunk_size,
                i,
                cache_dir,
                num_processes,
                total_rows,
                sample_factor,
                sample_strategy,
                sample_frame_num,
                sample_rate,
                data_type,
                connect_key,
                account_name,
                subfolder,
                resize_image_height,
                resize_image_width,
                output_file,
                lock,
            ),
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    end = time.time()
    print(f"latency: {end-start} samples: {total_rows}")


def check_size(
    videos,
    chunk_start,
    chunk_size,
    process_id,
    num_processes,
    total_rows,
    sample_factor,
    sample_strategy,
    sample_frame_num,
    sample_rate,
    res_file,
    lock,
):
    if process_id == num_processes - 1:
        chunk_size = total_rows - chunk_start + 1
    chunks = videos[chunk_start : chunk_start + chunk_size]
    print(
        f"Worker {process_id} Processing rows: {chunk_start} to {chunk_start + chunk_size - 1} \n"
    )

    processor = VideoProcessor(
        sample_factor=sample_factor,
        sample_strategy=sample_strategy,
        sample_frame_num=sample_frame_num,
        sample_rate=sample_rate,
    )

    write_str = ""
    for idx, video in enumerate(chunks):
        if idx % 30 == 0 and write_str != "":
            with lock:
                writer = open(res_file, "a+")
                writer.write(write_str)
                writer.flush()
                writer.close()
            write_str = ""
        frames = processor._read(video)
        # print(f"debug length of frames: {len(frames)} for video {video}")
        for index, frame in enumerate(frames):
            if np.all(np.array(frame) == [255, 255, 255]):
                print(f"skip placeholder image for {video} at index {index}")
                continue
            h = frame.size[1]
            w = frame.size[0]
            size = f"{w}x{h}"
            write_str += f"{video}\t{size}\n"
    if write_str != "":
        with lock:
            writer = open(res_file, "a+")
            writer.write(write_str)
            writer.flush()
            writer.close()


def read_video_size(
    data_file,
    names,
    video_col="video",
    sample_strategy="fix",
    sample_frame_num=1,
    sample_factor=1,
    sample_rate=[0.5],
    cache_dir="output",
):
    import re
    import pandas as pd
    import multiprocessing as mp

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
    os.makedirs(cache_dir, exist_ok=True)
    output_file = f"{cache_dir}/output.tsv"
    invalidcnt = 0
    totalcnt = 0
    if os.path.exists(output_file):
        uniques = []
        with open(output_file, "r") as f:
            for line in f.readlines():
                totalcnt += 1
                try:
                    videoid, size = line.strip().split("\t")
                    uniques.append(videoid)
                except Exception as e:
                    print(f"Error parsing line: {line.strip()} with error: {e}")
                    invalidcnt += 1
        data = data[
            ~data.apply(
                lambda x: x[video_col] in uniques,
                axis=1,
            )
        ]
    print(f"Read previous results, total rows: {totalcnt}, invalid rows: {invalidcnt}")
    print(f"Data loaded, total rows to move: {len(data)}")
    assert video_col in data.columns, f"Column {video_col} not found in data."
    videos = data[video_col].to_list()

    num_processes = mp.cpu_count()
    total_rows = len(videos)
    chunk_size = total_rows // num_processes
    start = time.time()
    print(f"need to process {total_rows} videos")

    lock = mp.Lock()

    processes = []
    print(f"need to process {total_rows} videos")
    for i in range(num_processes):
        p = mp.Process(
            target=check_size,
            args=(
                videos,
                i * chunk_size,
                chunk_size,
                i,
                num_processes,
                total_rows,
                sample_factor,
                sample_strategy,
                sample_frame_num,
                sample_rate,
                output_file,
                lock,
            ),
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    end = time.time()
    print(f"latency: {end-start} samples: {total_rows}")


def MergeCkpts(ckpt_1, ckpt_2, alpha, output):
    from unitorch_microsoft.adinsights.video.wan_official import (
        get_model_list,
        load_model,
    )
    from unitorch_microsoft.adinsights.video.wan_official_2 import merge_state_dicts

    def read_state(ckpt):
        try:
            state_dict = {}
            if os.path.isdir(ckpt):
                model_files = get_model_list(ckpt)
                for model_file in model_files:
                    state_dict.update(load_model(model_file))
                print(f"Loaded {len(state_dict)} parameters from {model_files}")
            else:
                if os.path.exists(ckpt):
                    state_dict = load_model(ckpt)
                    print(f"Loaded {len(state_dict)} parameters from {ckpt}")
            state_dict = (
                state_dict["state_dict"] if "state_dict" in state_dict else state_dict
            )
            for key, value in state_dict.items():
                print(f"Info state_dict from {ckpt} {key} {value.shape} {value.dtype}")
                break
            return state_dict
        except Exception as e:
            print(f"Error loading model from {ckpt}: {e}")
            return {}

    state_dict_1 = read_state(ckpt_1)
    check_key = None
    check_value = None
    for key, value in state_dict_1.items():
        print(f"state_dict_1 from {ckpt_1} {key} {value.shape} {value.dtype}")
        check_key = key
        check_value = value
        break

    state_dict_2 = read_state(ckpt_2)
    if check_key is not None and check_value is not None and check_key in state_dict_2:
        assert (
            state_dict_2[check_key].shape == check_value.shape
        ), f"Error: shape of {check_key} in {ckpt_2} is {state_dict_2[check_key].shape}, not match with {check_value.shape} in {ckpt_1}"
        # check value equals or not
        if not torch.all(state_dict_2[check_key] == check_value):
            print(f"Info: value of {check_key} in {ckpt_2} is not equal to {ckpt_1}")

    def save_state(
        state_dict, output_dir, max_shard_size="5GB", safe_serialization=False
    ):
        os.makedirs(output_dir, exist_ok=True)
        if safe_serialization:
            try:
                from safetensors.torch import save_file
            except ImportError:
                print(
                    "If you want to use `safe_serialization`, please `pip install safetensors`"
                )
                raise
        try:
            from huggingface_hub import split_torch_state_dict_into_shards
        except ImportError:
            print(
                "If you want to use `max_shard_size`, please `pip install huggingface_hub`"
            )
            raise
        weights_name = (
            "model.safetensors" if safe_serialization else "pytorch_model.bin"
        )
        if max_shard_size is not None:
            filename_pattern = weights_name.replace(".bin", "{suffix}.bin").replace(
                ".safetensors", "{suffix}.safetensors"
            )
            state_dict_split = split_torch_state_dict_into_shards(
                state_dict,
                filename_pattern=filename_pattern,
                max_shard_size=max_shard_size,
            )
        else:
            from collections import namedtuple

            StateDictSplit = namedtuple(
                "StateDictSplit", ["is_sharded", "filename_to_tensors"]
            )
            state_dict_split = StateDictSplit(
                is_sharded=False,
                filename_to_tensors={weights_name: list(state_dict.keys())},
            )
        # Save the model
        filename_to_tensors = state_dict_split.filename_to_tensors.items()
        for shard_file, tensors in tqdm(
            filename_to_tensors, desc="Saving checkpoint shards"
        ):
            shard = {tensor: state_dict[tensor].contiguous() for tensor in tensors}
            output_path = os.path.join(output_dir, shard_file)
            if safe_serialization:
                save_file(shard, output_path, metadata={"format": "pt"})
            else:
                torch.save(shard, output_path)

        # Save index if sharded
        if state_dict_split.is_sharded:
            index = {
                "metadata": state_dict_split.metadata,
                "weight_map": state_dict_split.tensor_to_filename,
            }
            save_index_file = (
                "model.safetensors.index.json"
                if safe_serialization
                else "pytorch_model.bin.index.json"
            )
            save_index_file = os.path.join(output_dir, save_index_file)
            with open(save_index_file, "w", encoding="utf-8") as f:
                content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                f.write(content)

    merged_state_dict = merge_state_dicts(state_dict_1, state_dict_2, alpha)
    if (
        check_key is not None
        and check_value is not None
        and check_key in merged_state_dict
    ):
        assert (
            merged_state_dict[check_key].shape == check_value.shape
        ), f"Error: shape of {check_key} in merged ckpt is {merged_state_dict[check_key].shape}, not match with {check_value.shape} in {ckpt_1}"
        # check value equals or not
        if not torch.all(merged_state_dict[check_key] == check_value):
            print(
                f"Info: value of {check_key} in merged ckpt is not equal to value in {ckpt_1}"
            )

    if len(merged_state_dict) > 0:
        save_state(
            merged_state_dict, output, max_shard_size="5GB", safe_serialization=False
        )
        print(f"Saved merged checkpoint to {output}")
    else:
        print("No parameters to save in merged checkpoint.")


def MergeCkpts(ckpt_1, ckpt_2, alpha, output):
    from unitorch_microsoft.adinsights.video.wan_official import (
        get_model_list,
        load_model,
    )
    from unitorch_microsoft.adinsights.video.wan_official_2 import merge_state_dicts

    def read_state(ckpt):
        try:
            state_dict = {}
            if os.path.isdir(ckpt):
                model_files = get_model_list(ckpt)
                for model_file in model_files:
                    state_dict.update(load_model(model_file))
                print(f"Loaded {len(state_dict)} parameters from {model_files}")
            else:
                if os.path.exists(ckpt):
                    state_dict = load_model(ckpt)
                    print(f"Loaded {len(state_dict)} parameters from {ckpt}")
            state_dict = (
                state_dict["state_dict"] if "state_dict" in state_dict else state_dict
            )
            for key, value in state_dict.items():
                print(f"Info state_dict from {ckpt} {key} {value.shape} {value.dtype}")
                break
            return state_dict
        except Exception as e:
            print(f"Error loading model from {ckpt}: {e}")
            return {}

    state_dict_1 = read_state(ckpt_1)
    check_key = None
    check_value = None
    for key, value in state_dict_1.items():
        print(f"state_dict_1 from {ckpt_1} {key} {value.shape} {value.dtype}")
        check_key = key
        check_value = value
        break

    state_dict_2 = read_state(ckpt_2)
    if check_key is not None and check_value is not None and check_key in state_dict_2:
        assert (
            state_dict_2[check_key].shape == check_value.shape
        ), f"Error: shape of {check_key} in {ckpt_2} is {state_dict_2[check_key].shape}, not match with {check_value.shape} in {ckpt_1}"
        # check value equals or not
        if not torch.all(state_dict_2[check_key] == check_value):
            print(f"Info: value of {check_key} in {ckpt_2} is not equal to {ckpt_1}")

    def save_state(
        state_dict, output_dir, max_shard_size="5GB", safe_serialization=False
    ):
        os.makedirs(output_dir, exist_ok=True)
        if safe_serialization:
            try:
                from safetensors.torch import save_file
            except ImportError:
                print(
                    "If you want to use `safe_serialization`, please `pip install safetensors`"
                )
                raise
        try:
            from huggingface_hub import split_torch_state_dict_into_shards
        except ImportError:
            print(
                "If you want to use `max_shard_size`, please `pip install huggingface_hub`"
            )
            raise
        weights_name = (
            "model.safetensors" if safe_serialization else "pytorch_model.bin"
        )
        if max_shard_size is not None:
            filename_pattern = weights_name.replace(".bin", "{suffix}.bin").replace(
                ".safetensors", "{suffix}.safetensors"
            )
            state_dict_split = split_torch_state_dict_into_shards(
                state_dict,
                filename_pattern=filename_pattern,
                max_shard_size=max_shard_size,
            )
        else:
            from collections import namedtuple

            StateDictSplit = namedtuple(
                "StateDictSplit", ["is_sharded", "filename_to_tensors"]
            )
            state_dict_split = StateDictSplit(
                is_sharded=False,
                filename_to_tensors={weights_name: list(state_dict.keys())},
            )
        # Save the model
        filename_to_tensors = state_dict_split.filename_to_tensors.items()
        for shard_file, tensors in tqdm(
            filename_to_tensors, desc="Saving checkpoint shards"
        ):
            shard = {tensor: state_dict[tensor].contiguous() for tensor in tensors}
            output_path = os.path.join(output_dir, shard_file)
            if safe_serialization:
                save_file(shard, output_path, metadata={"format": "pt"})
            else:
                torch.save(shard, output_path)

        # Save index if sharded
        if state_dict_split.is_sharded:
            index = {
                "metadata": state_dict_split.metadata,
                "weight_map": state_dict_split.tensor_to_filename,
            }
            save_index_file = (
                "model.safetensors.index.json"
                if safe_serialization
                else "pytorch_model.bin.index.json"
            )
            save_index_file = os.path.join(output_dir, save_index_file)
            with open(save_index_file, "w", encoding="utf-8") as f:
                content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                f.write(content)

    merged_state_dict = merge_state_dicts(state_dict_1, state_dict_2, alpha)
    if (
        check_key is not None
        and check_value is not None
        and check_key in merged_state_dict
    ):
        assert (
            merged_state_dict[check_key].shape == check_value.shape
        ), f"Error: shape of {check_key} in merged ckpt is {merged_state_dict[check_key].shape}, not match with {check_value.shape} in {ckpt_1}"
        # check value equals or not
        if not torch.all(merged_state_dict[check_key] == check_value):
            print(
                f"Info: value of {check_key} in merged ckpt is not equal to value in {ckpt_1}"
            )

    if len(merged_state_dict) > 0:
        save_state(
            merged_state_dict, output, max_shard_size="5GB", safe_serialization=False
        )
        print(f"Saved merged checkpoint to {output}")
    else:
        print("No parameters to save in merged checkpoint.")


if __name__ == "__main__":
    fire.Fire()
