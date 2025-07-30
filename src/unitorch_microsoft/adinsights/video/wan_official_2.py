import argparse
import logging
import os
import sys
import warnings
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import re
import pandas as pd
import hashlib
import json
from io import BytesIO
import requests
import numpy as np

warnings.filterwarnings('ignore')

import random

import torch
import torch.distributed as dist
from PIL import Image

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.distributed.util import init_distributed_group
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import save_video, str2bool

from unitorch_microsoft.adinsights.video.wan_official import readimg, load_z3_model, get_model_list, load_model, check_state_dict
  
def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"

    cfg = WAN_CONFIGS[args.task]

    if args.sample_steps is None:
        args.sample_steps = cfg.sample_steps

    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift

    if args.sample_guide_scale is None:
        args.sample_guide_scale = cfg.sample_guide_scale

    if args.frame_num is None:
        args.frame_num = cfg.frame_num

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)
    # Size check
    assert args.size in SUPPORTED_SIZES[
        args.
        task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"
    
def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="i2v-A14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="How many frames of video are generated. The number should be 4n+1"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated video to.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the video from.")
    parser.add_argument(
        "--use_prompt_extend",
        action="store_true",
        default=False,
        help="Whether to use prompt extend.")
    parser.add_argument(
        "--prompt_extend_method",
        type=str,
        default="local_qwen",
        choices=["dashscope", "local_qwen"],
        help="The prompt extend method to use.")
    parser.add_argument(
        "--prompt_extend_model",
        type=str,
        default=None,
        help="The prompt extend model to use.")
    parser.add_argument(
        "--prompt_extend_target_lang",
        type=str,
        default="zh",
        choices=["zh", "en"],
        help="The target language of prompt extend.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=43,
        help="The seed to use for generating the video.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="The image to generate the video from.")
    parser.add_argument(
        "--sample_solver",
        type=str,
        default='unipc',
        choices=['unipc', 'dpm++'],
        help="The solver used to sample.")
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=None,
        help="Classifier free guidance scale.")
    parser.add_argument(
        "--convert_model_dtype",
        action="store_true",
        default=False,
        help="Whether to convert model paramerters dtype.")
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
        default='camera',
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

def generation(pipe, prompt_expander, start_frame, prompt, camera, args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    print(f"generation rank {rank}, world_size {world_size}, local_rank {local_rank}, device {device}")
    print(f"Process video gen for {start_frame}")
    image = readimg(start_frame, args.cache_dir, args.size)
    if image == None:
        return None
    print("finish read img")
    print(f"prompt {prompt}")
    if prompt_expander is not None:
        logging.info("Extending prompt ...")
        if rank == 0:
            prompt_output = prompt_expander(
                prompt,
                image=image,
                tar_lang=args.prompt_extend_target_lang,
                seed=args.base_seed)
            if prompt_output.status == False:
                logging.info(
                    f"Extending prompt failed: {prompt_output.message}")
                logging.info("Falling back to original prompt.")
            else:
                prompt = prompt_output.prompt
            input_prompt = [prompt]
        else:
            input_prompt = [None]
        if dist.is_initialized():
            dist.broadcast_object_list(input_prompt, src=0)
        prompt = input_prompt[0]
        logging.info(f"Extended prompt: {prompt}")
    if True:
        if "ti2v" in args.task:
            video = pipe.generate(
                prompt,
                image,
                size=SIZE_CONFIGS[args.size],
                max_area=MAX_AREA_CONFIGS[args.size],
                frame_num=args.frame_num,
                shift=args.sample_shift,
                sample_solver=args.sample_solver,
                sampling_steps=args.sample_steps,
                guide_scale=args.sample_guide_scale,
                seed=args.base_seed,
                offload_model=args.offload_model,
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
            

        if rank == 0:
            name = hashlib.md5(start_frame.encode()).hexdigest() + f"_.mp4"
            name = os.path.join(args.cache_dir, name)
            save_video(
                tensor=video[None],
                save_file=name,
                fps=16,
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )
        del video
        torch.cuda.synchronize()
        print(f"finish generation rank {rank}, world_size {world_size}, local_rank {local_rank}, device {device}")
        return name
    #except Exception as e:
    #    print(e)
    #    return None

def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)

def prepare_pipeline(args):
    try:
        print("Prepare I2V pipeline")
        rank = int(os.getenv("RANK", 0))
        world_size = int(os.getenv("WORLD_SIZE", 1))
        local_rank = int(os.getenv("LOCAL_RANK", 0))
        device = local_rank
        print(f"prepare PL rank {rank}, world_size {world_size}, local_rank {local_rank}, device {device}")

        _init_logging(rank)

        if args.offload_model is None:
            args.offload_model = False if world_size > 1 else True
            logging.info(
                f"offload_model is not specified, set to {args.offload_model}.")
        if world_size > 1:
            torch.cuda.set_device(local_rank)
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                rank=rank,
                world_size=world_size)
        else:
            assert not (
                args.t5_fsdp or args.dit_fsdp
            ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
            assert not (
                args.ulysses_size > 1
            ), f"sequence parallel are not supported in non-distributed environments."

        if args.ulysses_size > 1:
            assert args.ulysses_size == world_size, f"The number of ulysses_size should be equal to the world size."
            init_distributed_group()

        if args.use_prompt_extend:
            if args.prompt_extend_method == "dashscope":
                prompt_expander = DashScopePromptExpander(
                    model_name=args.prompt_extend_model,
                    task=args.task,
                    is_vl=args.image is not None)
            elif args.prompt_extend_method == "local_qwen":
                prompt_expander = QwenPromptExpander(
                    model_name=args.prompt_extend_model,
                    task=args.task,
                    is_vl=args.image is not None,
                    device=rank)
            else:
                raise NotImplementedError(
                    f"Unsupport prompt_extend_method: {args.prompt_extend_method}")
        else:
            prompt_expander = None
        
        cfg = WAN_CONFIGS[args.task]
        if args.ulysses_size > 1:
            assert cfg.num_heads % args.ulysses_size == 0, f"`{cfg.num_heads=}` cannot be divided evenly by `{args.ulysses_size=}`."

        logging.info(f"Generation job args: {args}")
        logging.info(f"Generation model config: {cfg}")

        if dist.is_initialized():
            base_seed = [args.base_seed] if rank == 0 else [None]
            dist.broadcast_object_list(base_seed, src=0)
            args.base_seed = base_seed[0]

        logging.info(f"Input prompt: {args.prompt}")

        if "ti2v" in args.task:
            print("Prepare WanTI2V pipeline")
            wan_pipe = wan.WanTI2V(
                config=cfg,
                checkpoint_dir=args.ckpt_dir,
                device_id=device,
                rank=rank,
                t5_fsdp=args.t5_fsdp,
                dit_fsdp=args.dit_fsdp,
                use_sp=(args.ulysses_size > 1),
                t5_cpu=args.t5_cpu,
                convert_model_dtype=args.convert_model_dtype,
            )
        else:
            print("Prepare WanI2V pipeline")
            wan_pipe = wan.WanI2V(
                config=cfg,
                checkpoint_dir=args.ckpt_dir,
                device_id=device,
                rank=rank,
                t5_fsdp=args.t5_fsdp,
                dit_fsdp=args.dit_fsdp,
                use_sp=(args.ulysses_size > 1),
                t5_cpu=args.t5_cpu,
                convert_model_dtype=args.convert_model_dtype,
            )
            
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

            check_state_dict(wan_pipe.model.state_dict(), state_dict)
            m, u = wan_pipe.model.load_state_dict(state_dict, strict=False)
            print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

        return wan_pipe, prompt_expander
    except Exception as e:
        print(f"Prepare I2V pipeline error {e}")
        return None, None


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

    pipe, prompt_expander = prepare_pipeline(args)
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
        if args.enable_cm_adaln and args.camera_col is not None and args.camera_col in data.columns:
            _camera = (
                row[args.camera_col]
                if not pd.isna(row[args.camera_col])
                else ""
            )
        video = generation(pipe, prompt_expander, _start_frame, _prompt, _camera, args)
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
        with open(output_file, "r") as fin, open(
            tsv_file, "w", encoding="utf-8"
        ) as fout:
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
    image2video(args)


