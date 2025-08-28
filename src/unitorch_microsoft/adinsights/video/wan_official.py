import torch
from PIL import Image
import os
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import re
import pandas as pd
import hashlib
import json
from io import BytesIO
import requests
import numpy as np
import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander

import argparse
import random
import sys


def readimg(imagefile, cache_dir, max_area_str, return_bytes=True):
    try:
        if imagefile.startswith(("http://", "https://")):
            response = requests.get(imagefile)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(imagefile).convert("RGB")

        h, w = max_area_str.split("*")
        max_area = int(h) * int(w)

        aspect_ratio = image.height / image.width
        mod_value = 16
        height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        image = image.resize((width, height), resample=Image.LANCZOS)
        if return_bytes:
            return image
        else:
            name = hashlib.md5(imagefile.encode()).hexdigest() + ".jpg"
            name = os.path.join(cache_dir, name)
            image.save(name)
            return name
    except Exception as e:
        print(e)
        return None


def get_smaller_size(size_str1, size_str2):
    width1, height1 = map(int, size_str1.split("*"))
    width2, height2 = map(int, size_str2.split("*"))
    area1 = width1 * height1
    area2 = width2 * height2
    return size_str1 if area1 < area2 else size_str2


def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"

    # The default sampling steps are 40 for image-to-video tasks and 50 for text-to-video tasks.
    if args.sample_steps is None:
        args.sample_steps = 50
        if "i2v" in args.task:
            args.sample_steps = 40

    if args.sample_shift is None:
        args.sample_shift = 5.0
        if "i2v" in args.task and args.size in ["832*480", "480*832"]:
            args.sample_shift = 3.0
        elif "flf2v" in args.task or "vace" in args.task:
            args.sample_shift = 16

    # The default number of frames are 1 for text-to-image tasks and 81 for other tasks.
    if args.frame_num is None:
        args.frame_num = 1 if "t2i" in args.task else 81

    # T2I frame_num check
    if "t2i" in args.task:
        assert (
            args.frame_num == 1
        ), f"Unsupport frame_num {args.frame_num} for task {args.task}"

    args.base_seed = (
        args.base_seed if args.base_seed >= 0 else random.randint(0, sys.maxsize)
    )
    # Size check
    assert (
        args.size in SUPPORTED_SIZES[args.task]
    ), f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"


def _parse_args():
    from wan.utils.utils import cache_image, cache_video, str2bool

    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="t2v-14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.",
    )
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image.",
    )
    parser.add_argument(
        "--resize_max_area",
        type=str,
        default="1280*720",
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image.",
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="How many frames to sample from a image or video. The number should be 4n+1",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory.",
    )
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage.",
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.",
    )
    parser.add_argument(
        "--ring_size",
        type=int,
        default=1,
        help="The size of the ring attention parallelism in DiT.",
    )
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.",
    )
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.",
    )
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.",
    )
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated image or video to.",
    )
    parser.add_argument(
        "--src_video",
        type=str,
        default=None,
        help="The file of the source video. Default None.",
    )
    parser.add_argument(
        "--src_mask",
        type=str,
        default=None,
        help="The file of the source mask. Default None.",
    )
    parser.add_argument(
        "--src_ref_images",
        type=str,
        default=None,
        help="The file list of the source reference images. Separated by ','. Default None.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the image or video from.",
    )
    parser.add_argument(
        "--use_prompt_extend",
        action="store_true",
        default=False,
        help="Whether to use prompt extend.",
    )
    parser.add_argument(
        "--prompt_extend_method",
        type=str,
        default="local_qwen",
        choices=["dashscope", "local_qwen"],
        help="The prompt extend method to use.",
    )
    parser.add_argument(
        "--prompt_extend_model",
        type=str,
        default=None,
        help="The prompt extend model to use.",
    )
    parser.add_argument(
        "--prompt_extend_target_lang",
        type=str,
        default="zh",
        choices=["zh", "en"],
        help="The target language of prompt extend.",
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=43,
        help="The seed to use for generating the image or video.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="[image to video] The image to generate the video from.",
    )
    parser.add_argument(
        "--first_frame",
        type=str,
        default=None,
        help="[first-last frame to video] The image (first frame) to generate the video from.",
    )
    parser.add_argument(
        "--last_frame",
        type=str,
        default=None,
        help="[first-last frame to video] The image (last frame) to generate the video from.",
    )
    parser.add_argument(
        "--sample_solver",
        type=str,
        default="unipc",
        choices=["unipc", "dpm++"],
        help="The solver used to sample.",
    )
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="The sampling steps."
    )
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.",
    )
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=5.0,
        help="Classifier free guidance scale.",
    )
    parser.add_argument(
        "--data_file", type=str, default=None, help="Path to the input data file."
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory to cache images/videos and outputs.",
    )
    parser.add_argument(
        "--names",
        type=str,
        default=None,
        help="Column names for the input data file, separated by commas, or '*' for default.",
    )
    parser.add_argument(
        "--prompt_col",
        type=str,
        default=None,
        help="Name of the column containing prompts.",
    )
    parser.add_argument(
        "--camera_col",
        type=str,
        default="camera",
        help="Name of the column containing camera information.",
    )
    parser.add_argument(
        "--start_frame_col",
        type=str,
        default=None,
        help="Name of the column containing the start frame image.",
    )
    parser.add_argument(
        "--end_frame_col",
        type=str,
        default=None,
        help="Name of the column containing the end frame image.",
    )
    parser.add_argument(
        "--neg_prompt_col",
        type=str,
        default=None,
        help="Name of the column containing negative prompts.",
    )

    parser.add_argument(
        "--transformer_folder",
        type=str,
        default=None,
        help="point to the FTed ckpt folder",
    )
    parser.add_argument(
        "--z3_flag_disable",
        action="store_true",
        default=False,
        help="Disable Z3 flag",
    )
    parser.add_argument(
        "--enable_cm_adaln",
        action="store_true",
        default=False,
        help="Enable CM-ADALN",
    )

    args = parser.parse_args()

    _validate_args(args)

    return args


def generation(pipe, start_frame, prompt, camera, args):
    from wan.utils.utils import cache_image, cache_video, str2bool

    print(f"Process video gen for {start_frame}")
    image = readimg(
        start_frame, args.cache_dir, get_smaller_size(args.resize_max_area, args.size)
    )
    if image == None:
        return None
    print("finish read img")
    try:
        if args.enable_cm_adaln:
            video = pipe.generate(
                prompt,
                image,
                max_area=MAX_AREA_CONFIGS[args.size],
                frame_num=args.frame_num,
                shift=args.sample_shift,
                sample_solver=args.sample_solver,
                sampling_steps=args.sample_steps,
                guide_scale=args.sample_guide_scale,
                seed=args.base_seed,
                offload_model=args.offload_model,
                camera=camera,
            )
        else:
            video = pipe.generate(
                prompt,
                image,
                max_area=MAX_AREA_CONFIGS[args.size],
                frame_num=args.frame_num,
                shift=args.sample_shift,
                sample_solver=args.sample_solver,
                sampling_steps=args.sample_steps,
                guide_scale=args.sample_guide_scale,
                seed=args.base_seed,
                offload_model=args.offload_model,
            )

        name = hashlib.md5(start_frame.encode()).hexdigest() + f"_.mp4"
        name = os.path.join(args.cache_dir, name)
        cache_video(
            tensor=video[None],
            save_file=name,
            fps=16,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )
        return name
    except Exception as e:
        print(e)
        return None


def load_z3_model(ckpt_dir):
    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

    state_dict = get_fp32_state_dict_from_zero_checkpoint(
        ckpt_dir,
        exclude_frozen_parameters=True,
    )
    for key in state_dict.keys():
        print(f"Key: {key}, Shape: {state_dict[key].shape}")

    print(f"Loaded {len(state_dict)} parameters from {ckpt_dir}")

    return state_dict


def get_model_list(ckpt_folder):
    """
    Get a list of model files in the specified folder.
    """
    model_files = []
    for root, dirs, files in os.walk(ckpt_folder):
        for file in files:
            if file.endswith(".bin") or file.endswith(".safetensors"):
                model_files.append(os.path.join(root, file))
    return model_files


def load_model(file_path):
    if file_path.endswith("safetensors"):
        from safetensors.torch import load_file, safe_open

        state_dict = load_file(file_path)
    else:
        state_dict = torch.load(file_path, map_location="cpu")

    return state_dict


def check_state_dict(old_state_dict, state_dict):
    import time

    load_keys = []
    non_load_keys = []
    cnt = 0
    for key, value in state_dict.items():
        if key in old_state_dict and old_state_dict[key].shape == state_dict[key].shape:
            if cnt == 0:
                print(
                    f"Key {key} found in old state dict with matching shape, old dtype: {old_state_dict[key].dtype}, new dtype: {value.dtype}"
                )
                cnt += 1
            load_keys.append(key)
        else:
            print(f"Key {key} not found in old state dict or shape mismatch")
            non_load_keys.append(key)
    print(f"Total keys in state dict: {len(state_dict)}")
    print(f"Total keys in old state dict: {len(old_state_dict)}")
    load_percent = (
        len(load_keys) / len(old_state_dict) * 100
    )  # Calculate the percentage of loaded keys
    print(f"load percent: {load_percent}%")
    print(f"Non load keys in new weights: {list(non_load_keys)}")
    print(f"missing keys in old weights: {list(old_state_dict.keys() - load_keys)}")
    time.sleep(20)
    print(f"Check state dict complete {load_keys[20]}")
    old_val = old_state_dict[load_keys[20]]
    new_val = state_dict[load_keys[20]]
    print(f"shape diff: {old_val.shape} vs {new_val.shape}")
    print(f"old val: {old_val}")
    time.sleep(20)
    print(f"new val: {new_val}")
    time.sleep(20)


def prepare_pipeline(args):
    try:
        print("Prepare I2V pipeline")
        rank = int(os.getenv("RANK", 0))
        world_size = int(os.getenv("WORLD_SIZE", 1))
        local_rank = int(os.getenv("LOCAL_RANK", 0))
        device = local_rank
        cfg = WAN_CONFIGS[args.task]
        # load transformer state
        if args.transformer_folder is not None:
            state_dict = {}
            if not args.z3_flag_disable:
                state_dict = load_z3_model(args.transformer_folder)
            else:
                if os.path.isdir(args.transformer_folder):
                    model_files = get_model_list(args.transformer_folder)
                    for model_file in model_files:
                        state_dict.update(load_model(model_file))
                    print(f"Loaded {len(state_dict)} parameters from {model_files}")
                else:
                    if os.path.exists(args.transformer_folder):
                        state_dict = load_model(args.transformer_folder)
                        print(
                            f"Loaded {len(state_dict)} parameters from {args.transformer_folder}"
                        )
            state_dict = (
                state_dict["state_dict"] if "state_dict" in state_dict else state_dict
            )
            # check_state_dict(wan_i2v.model.state_dict(), state_dict)
        else:
            state_dict = None

        if args.enable_cm_adaln and state_dict is not None:
            checkpoint_state = state_dict
            wan_i2v = wan.WanI2V(
                config=cfg,
                checkpoint_dir=args.ckpt_dir,
                device_id=device,
                rank=rank,
                t5_fsdp=args.t5_fsdp,
                dit_fsdp=args.dit_fsdp,
                use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
                t5_cpu=args.t5_cpu,
                checkpoint_state=checkpoint_state,
            )
        else:
            checkpoint_state = None
            wan_i2v = wan.WanI2V(
                config=cfg,
                checkpoint_dir=args.ckpt_dir,
                device_id=device,
                rank=rank,
                t5_fsdp=args.t5_fsdp,
                dit_fsdp=args.dit_fsdp,
                use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
                t5_cpu=args.t5_cpu,
            )

            if args.transformer_folder is not None and not args.enable_cm_adaln:
                m, u = wan_i2v.model.load_state_dict(state_dict, strict=False)
                print(
                    f"load from transformer folder missing keys: {len(m)}, unexpected keys: {len(u)}"
                )
        return wan_i2v
    except Exception as e:
        print(f"Prepare I2V pipeline error {e}")
        return None


def image2video(args):
    if isinstance(args.names, str) and args.names.strip() == "*":
        names = None
    if isinstance(args.names, str):
        names = re.split(r"[,;]", args.names)
        names = [n.strip() for n in names]

    data = pd.read_csv(args.data_file, names=names, sep="\t", quoting=3, header=None)
    os.makedirs(args.cache_dir, exist_ok=True)
    output_file = f"{args.cache_dir}/output.jsonl"

    if os.path.exists(output_file):
        print(f"Before df {len(data)} ")
        uniques = []
        with open(output_file, "r") as f:
            for line in f:
                row = json.loads(line)
                uniques.append(
                    row["prompt"]
                    + " - "
                    + row["neg_prompt"]
                    + " - "
                    + row["start_frame"]
                    + " - "
                    + row["end_frame"]
                )
        print(f"unique size {len(uniques)}")
        data = data[
            ~data.apply(
                lambda x: (
                    x[args.prompt_col]
                    if args.prompt_col is not None and not pd.isna(x[args.prompt_col])
                    else ""
                )
                + " - "
                + (
                    x[args.neg_prompt_col]
                    if args.neg_prompt_col is not None
                    and not pd.isna(x[args.neg_prompt_col])
                    else ""
                )
                + " - "
                + (
                    x[args.start_frame_col]
                    if args.start_frame_col is not None
                    and not pd.isna(x[args.start_frame_col])
                    else ""
                )
                + " - "
                + (
                    x[args.end_frame_col]
                    if args.end_frame_col is not None
                    and not pd.isna(x[args.end_frame_col])
                    else ""
                )
                in uniques,
                axis=1,
            )
        ]
        print(f"Need to handle {len(data)} files")

    writer = open(output_file, "a+")

    assert (
        args.prompt_col in data.columns
    ), f"Column {args.prompt_col} not found in data."
    assert (
        args.start_frame_col in data.columns or args.end_frame_col in data.columns
    ), f"At least one image needed."

    pipe = prepare_pipeline(args)
    if pipe == None:
        print("Prepare pipeline error")
        return None

    cnt = 0
    for _, row in data.iterrows():
        _prompt = row[args.prompt_col] if not pd.isna(row[args.prompt_col]) else ""
        _neg_prompt = ""
        if args.neg_prompt_col != None:
            _neg_prompt = (
                row[args.neg_prompt_col]
                if not pd.isna(row[args.neg_prompt_col])
                else ""
            )
        _start_frame = ""
        if args.start_frame_col != None:
            _start_frame = (
                row[args.start_frame_col]
                if not pd.isna(row[args.start_frame_col])
                else ""
            )
        _camera = ""
        if (
            args.enable_cm_adaln
            and args.camera_col is not None
            and args.camera_col in data.columns
        ):
            _camera = row[args.camera_col] if not pd.isna(row[args.camera_col]) else ""
        video = generation(pipe, _start_frame, _prompt, _camera, args)
        if video != None:
            record = {
                "prompt": _prompt,
                "neg_prompt": _neg_prompt,
                "index_id": "",
                "start_frame": _start_frame,
                "end_frame": "",
                "url": video,
                "result": video,
            }
            writer.write(json.dumps(record) + "\n")
            cnt += 1
    print(f"Finish video gen for {cnt} videos")
    writer.close()
    output_file = f"{args.cache_dir}/output.jsonl"
    tsv_file = f"{args.cache_dir}/output.tsv"
    try:
        with (
            open(output_file, "r") as fin,
            open(tsv_file, "w", encoding="utf-8") as fout,
        ):
            # Read all json lines
            rows = [json.loads(line) for line in fin]
            if rows:
                # Write header
                header = list(rows[0].keys())
                # Write rows
                for row in rows:
                    fout.write(
                        "\t".join(str(row.get(col, "")) for col in header) + "\n"
                    )
        print(f"Converted {output_file} to {tsv_file}")
    except Exception as e:
        print(f"Error converting jsonl to tsv: {e}")


if __name__ == "__main__":
    args = _parse_args()
    if args.task == "i2v-14B":
        image2video(args)
    else:
        print(f"Unsupport task {args.task}")
