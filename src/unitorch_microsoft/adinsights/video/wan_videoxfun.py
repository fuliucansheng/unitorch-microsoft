import torch

# from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData
from PIL import Image
import fire
import os
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import re
import pandas as pd
import hashlib
import json
from io import BytesIO
import requests
import numpy as np
import argparse
import random
import sys
import time
from videox_fun.dist import set_multi_gpus_devices, shard_model
from videox_fun.models import (
    AutoencoderKLWan,
    AutoTokenizer,
    CLIPModel,
    WanT5EncoderModel,
    WanTransformer3DModel,
    Wan2_2Transformer3DModel,
)
from videox_fun.models.cache_utils import get_teacache_coefficients
from videox_fun.pipeline import WanI2VPipeline, Wan2_2I2VPipeline
from videox_fun.utils.fp8_optimization import (
    convert_model_weight_to_float8,
    replace_parameters_by_name,
    convert_weight_dtype_wrapper,
)
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from videox_fun.utils.utils import (
    filter_kwargs,
    get_image_to_video_latent,
    save_videos_grid,
)
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoTokenizer


def readimg(imagefile, cache_dir, max_area, return_bytes=True):
    try:
        if imagefile.startswith(("http://", "https://")):
            response = requests.get(imagefile)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(imagefile).convert("RGB")

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


def generation(pipe, generator, start_frame, prompt, negative_prompt, args):
    print(f"Process video gen for {start_frame}")
    max_area = args.sample_size[0] * args.sample_size[1]
    image = readimg(start_frame, args.cache_dir, max_area)
    width, height = image.size

    if image == None:
        return None
    print("finish read img")
    try:
        with torch.no_grad():
            input_video, input_video_mask, clip_image = get_image_to_video_latent(
                [image],
                None,
                video_length=args.video_length,
                sample_size=[height, width],
            )
            if args.version == 2.1:
                sample = pipe(
                    prompt,
                    num_frames=args.video_length,
                    negative_prompt="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
                    height=height,
                    width=width,
                    generator=generator,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                    video=input_video,
                    mask_video=input_video_mask,
                    clip_image=clip_image,
                    shift=args.shift,
                ).videos
            elif args.version == 2.2:
                sample = pipe(
                    prompt,
                    num_frames=args.video_length,
                    negative_prompt="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
                    height=height,
                    width=width,
                    generator=generator,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                    boundary=args.boundary,
                    video=input_video,
                    mask_video=input_video_mask,
                    shift=args.shift,
                ).videos

            name = hashlib.md5(start_frame.encode()).hexdigest() + f"_.mp4"
            name = os.path.join(args.cache_dir, name)
            save_videos_grid(sample, name, fps=args.fps)
            return name
    except Exception as e:
        print(e)
        return None


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )

    parser.add_argument(
        "--GPU_memory_mode",
        type=str,
        default="model_cpu_offload",
        help="GPU memory mode, e.g., 'sequential_cpu_offload'",
    )
    parser.add_argument(
        "--ulysses_degree", type=int, default=1, help="Ulysses parallelism degree"
    )
    parser.add_argument(
        "--ring_degree", type=int, default=1, help="Ring attention parallelism degree"
    )
    parser.add_argument(
        "--fsdp_dit",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.",
    )
    parser.add_argument(
        "--fsdp_text_encoder",
        action="store_true",
        default=True,
        help="Whether to use FSDP for text encoder.",
    )
    parser.add_argument(
        "--compile_dit",
        action="store_true",
        default=False,
        help="Enable torch.compile for DiT.",
    )
    parser.add_argument(
        "--enable_teacache", action="store_true", default=False, help="Enable TeaCache."
    )
    parser.add_argument(
        "--teacache_threshold", type=float, default=0.10, help="TeaCache threshold"
    )
    parser.add_argument(
        "--num_skip_start_steps",
        type=int,
        default=5,
        help="Number of steps to skip TeaCache at start",
    )
    parser.add_argument(
        "--teacache_offload",
        action="store_true",
        default=False,
        help="Offload TeaCache tensors to CPU",
    )
    parser.add_argument(
        "--z3_flag_disable",
        action="store_true",
        default=False,
        help="Disable Z3 flag",
    )
    parser.add_argument(
        "--cfg_skip_ratio", type=float, default=0, help="CFG skip ratio"
    )
    parser.add_argument(
        "--enable_riflex", action="store_true", default=False, help="Enable Riflex"
    )
    parser.add_argument(
        "--riflex_k",
        type=int,
        default=6,
        help="Index of intrinsic frequency for Riflex",
    )
    parser.add_argument(
        "--version",
        type=float,
        default=2.1,
        help="Version number of Wan model to use, e.g., 2.1 or 2.2",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="config/wan2.1/wan_civitai.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="models/Diffusion_Transformer/Wan2.1-I2V-14B-480P",
        help="Model name or path",
    )
    parser.add_argument(
        "--sampler_name",
        type=str,
        default="Flow_Unipc",
        choices=["Flow", "Flow_Unipc", "Flow_DPM++"],
        help="Sampler name",
    )
    parser.add_argument(
        "--shift", type=float, default=3, help="Noise schedule shift parameter"
    )
    parser.add_argument(
        "--transformer_path",
        type=str,
        default=None,
        help="Path to pretrained transformer checkpoint",
    )
    parser.add_argument(
        "--transformer_high_path",
        type=str,
        default=None,
        help="Path to pretrained transformer checkpoint",
    )
    parser.add_argument(
        "--vae_path", type=str, default=None, help="Path to pretrained VAE checkpoint"
    )
    parser.add_argument(
        "--lora_path", type=str, default=None, help="Path to LoRA checkpoint"
    )
    parser.add_argument(
        "--lora_high_path", type=str, default=None, help="Path to LoRA checkpoint"
    )
    parser.add_argument(
        "--sample_size",
        type=lambda s: [int(x) for x in s.split(",")],
        default="480,832",
        help="Sample size as 'height,width'",
    )
    parser.add_argument(
        "--video_length", type=int, default=81, help="Video length (number of frames)"
    )
    parser.add_argument("--fps", type=int, default=16, help="Frames per second")
    parser.add_argument(
        "--weight_dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16"],
        help="Weight dtype",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=6.0,
        help="Guidance scale for classifier-free guidance",
    )
    parser.add_argument("--seed", type=int, default=43, help="Random seed")
    parser.add_argument(
        "--num_inference_steps", type=int, default=40, help="Number of inference steps"
    )
    parser.add_argument("--lora_weight", type=float, default=0.55, help="LoRA weight")
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
        "--boundary",
        type=float,
        default=0.9,
        help="Boundary for the prompt",
    )
    args = parser.parse_args()

    return args


def load_z3_model(ckpt_dir):
    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

    state_dict = get_fp32_state_dict_from_zero_checkpoint(
        ckpt_dir,
        exclude_frozen_parameters=True,
    )
    for key in state_dict.keys():
        print(f"Key: {key}, Shape: {state_dict[key].shape}")
    """
    state_dict = {
        (k[6:] if k.startswith("model.") else k): v
        for k, v in state_dict.items()
    }
    """
    print(f"Loaded {len(state_dict)} parameters from {ckpt_dir}")

    return state_dict


def check_state_dict(old_state_dict, state_dict):
    import time

    load_keys = []
    non_load_keys = []
    for key, value in state_dict.items():
        if key in old_state_dict and old_state_dict[key].shape == state_dict[key].shape:
            print(f"Key {key} found in old state dict with matching shape")
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


def prepare_pipeline(args):
    try:
        print("Prepare I2V pipeline")
        device = set_multi_gpus_devices(args.ulysses_degree, args.ring_degree)
        config = OmegaConf.load(args.config_path)
        if "boundary" in config["transformer_additional_kwargs"]:
            args.boundary = config["transformer_additional_kwargs"]["boundary"]
        weight_dtype = (
            torch.bfloat16 if args.weight_dtype == "bfloat16" else torch.float16
        )
        if args.version == 2.1:
            WanModel = WanTransformer3DModel
        elif args.version == 2.2:
            WanModel = Wan2_2Transformer3DModel
        else:
            raise ValueError(f"Unsupported version {args.version}")

        if args.version == 2.1:
            transformer = WanModel.from_pretrained(
                os.path.join(
                    args.model_name,
                    config["transformer_additional_kwargs"].get(
                        "transformer_subpath", "transformer"
                    ),
                ),
                transformer_additional_kwargs=OmegaConf.to_container(
                    config["transformer_additional_kwargs"]
                ),
                low_cpu_mem_usage=True if not args.fsdp_dit else False,
                torch_dtype=weight_dtype,
            )
            transformer_2 = None

            print(
                f"Transformer model loaded Wan version {args.version} from {args.model_name} on device {device}"
            )
        elif args.version == 2.2:
            transformer = WanModel.from_pretrained(
                os.path.join(
                    args.model_name,
                    config["transformer_additional_kwargs"].get(
                        "transformer_low_noise_model_subpath", "transformer"
                    ),
                ),
                transformer_additional_kwargs=OmegaConf.to_container(
                    config["transformer_additional_kwargs"]
                ),
                low_cpu_mem_usage=True if not args.fsdp_dit else False,
                torch_dtype=weight_dtype,
            )
            print(
                f"Transformer model loaded Wan version {args.version} from {args.model_name} low_noise on device {device}"
            )

            transformer_2 = WanModel.from_pretrained(
                os.path.join(
                    args.model_name,
                    config["transformer_additional_kwargs"].get(
                        "transformer_high_noise_model_subpath", "transformer"
                    ),
                ),
                transformer_additional_kwargs=OmegaConf.to_container(
                    config["transformer_additional_kwargs"]
                ),
                low_cpu_mem_usage=True if not args.fsdp_dit else False,
                torch_dtype=weight_dtype,
            )
            print(
                f"Transformer model loaded Wan version {args.version} from {args.model_name} high_noise on device {device}"
            )

        if args.transformer_path is not None:
            state_dict = {}
            if not args.z3_flag_disable:
                state_dict = load_z3_model(args.transformer_path)
            else:
                if os.path.isdir(args.transformer_path):
                    model_files = get_model_list(args.transformer_path)
                    for model_file in model_files:
                        state_dict.update(load_model(model_file))
                    print(f"Loaded {len(state_dict)} parameters from {model_files}")
                else:
                    if os.path.exists(args.transformer_path):
                        state_dict = load_model(args.transformer_path)
                        print(
                            f"Loaded {len(state_dict)} parameters from {args.transformer_path}"
                        )
            state_dict = (
                state_dict["state_dict"] if "state_dict" in state_dict else state_dict
            )

            check_state_dict(transformer.state_dict(), state_dict)
            m, u = transformer.load_state_dict(state_dict, strict=False)
            print(
                f"load FT ckpt from {args.transformer_path} missing keys: {len(m)}, unexpected keys: {len(u)}"
            )

        if args.transformer_high_path is not None:
            print(f"From checkpoint: {args.transformer_high_path}")
            state_dict = {}
            if not args.z3_flag_disable:
                state_dict = load_z3_model(args.transformer_high_path)
            else:
                if os.path.isdir(args.transformer_high_path):
                    model_files = get_model_list(args.transformer_high_path)
                    for model_file in model_files:
                        state_dict.update(load_model(model_file))
                    print(f"Loaded {len(state_dict)} parameters from {model_files}")
                else:
                    if os.path.exists(args.transformer_high_path):
                        state_dict = load_model(args.transformer_high_path)
                        print(
                            f"Loaded {len(state_dict)} parameters from {args.transformer_high_path}"
                        )
            state_dict = (
                state_dict["state_dict"] if "state_dict" in state_dict else state_dict
            )

            check_state_dict(transformer_2.state_dict(), state_dict)
            m, u = transformer_2.load_state_dict(state_dict, strict=False)
            print(
                f"load FT ckpt from {args.transformer_high_path} missing keys: {len(m)}, unexpected keys: {len(u)}"
            )

        vae = AutoencoderKLWan.from_pretrained(
            os.path.join(
                args.model_name, config["vae_kwargs"].get("vae_subpath", "vae")
            ),
            additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
        ).to(weight_dtype)

        if args.vae_path is not None:
            print(f"From checkpoint: {args.vae_path}")
            if args.vae_path.endswith("safetensors"):
                from safetensors.torch import load_file, safe_open

                state_dict = load_file(args.vae_path)
            else:
                state_dict = torch.load(args.vae_path, map_location="cpu")
            state_dict = (
                state_dict["state_dict"] if "state_dict" in state_dict else state_dict
            )

            m, u = vae.load_state_dict(state_dict, strict=False)
            print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(
                args.model_name,
                config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer"),
            ),
        )

        text_encoder = WanT5EncoderModel.from_pretrained(
            os.path.join(
                args.model_name,
                config["text_encoder_kwargs"].get(
                    "text_encoder_subpath", "text_encoder"
                ),
            ),
            additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        )
        text_encoder = text_encoder.eval()

        if args.version == 2.1:
            clip_image_encoder = CLIPModel.from_pretrained(
                os.path.join(
                    args.model_name,
                    config["image_encoder_kwargs"].get(
                        "image_encoder_subpath", "image_encoder"
                    ),
                ),
            ).to(weight_dtype)
            clip_image_encoder = clip_image_encoder.eval()
        elif args.version == 2.2:
            clip_image_encoder = None

        Choosen_Scheduler = scheduler_dict = {
            "Flow": FlowMatchEulerDiscreteScheduler,
            "Flow_Unipc": FlowUniPCMultistepScheduler,
            "Flow_DPM++": FlowDPMSolverMultistepScheduler,
        }[args.sampler_name]
        if args.sampler_name == "Flow_Unipc" or args.sampler_name == "Flow_DPM++":
            config["scheduler_kwargs"]["shift"] = 1
        scheduler = Choosen_Scheduler(
            **filter_kwargs(
                Choosen_Scheduler, OmegaConf.to_container(config["scheduler_kwargs"])
            )
        )
        if args.version == 2.1:
            pipeline = WanI2VPipeline(
                transformer=transformer,
                vae=vae,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                scheduler=scheduler,
                clip_image_encoder=clip_image_encoder,
            )
        elif args.version == 2.2:
            pipeline = Wan2_2I2VPipeline(
                transformer=transformer,
                transformer_2=transformer_2,
                vae=vae,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                scheduler=scheduler,
            )

        if args.ulysses_degree > 1 or args.ring_degree > 1:
            from functools import partial

            transformer.enable_multi_gpus_inference()
            if transformer_2 is not None:
                transformer_2.enable_multi_gpus_inference()
            if args.fsdp_dit:
                shard_fn = partial(
                    shard_model, device_id=device, param_dtype=weight_dtype
                )
                pipeline.transformer = shard_fn(pipeline.transformer)
                if transformer_2 is not None:
                    pipeline.transformer_2 = shard_fn(pipeline.transformer_2)
                print("Add FSDP DIT")
            if args.fsdp_text_encoder:
                shard_fn = partial(
                    shard_model, device_id=device, param_dtype=weight_dtype
                )
                pipeline.text_encoder = shard_fn(pipeline.text_encoder)
                print("Add FSDP TEXT ENCODER")

        if args.compile_dit:
            for i in range(len(pipeline.transformer.blocks)):
                pipeline.transformer.blocks[i] = torch.compile(
                    pipeline.transformer.blocks[i]
                )
            if transformer_2 is not None:
                for i in range(len(pipeline.transformer_2.blocks)):
                    pipeline.transformer_2.blocks[i] = torch.compile(
                        pipeline.transformer_2.blocks[i]
                    )
            print("Add Compile")

        if args.GPU_memory_mode == "sequential_cpu_offload":
            replace_parameters_by_name(
                transformer,
                [
                    "modulation",
                ],
                device=device,
            )
            transformer.freqs = transformer.freqs.to(device=device)
            if transformer_2 is not None:
                replace_parameters_by_name(
                    transformer_2,
                    [
                        "modulation",
                    ],
                    device=device,
                )
                transformer_2.freqs = transformer_2.freqs.to(device=device)
            pipeline.enable_sequential_cpu_offload(device=device)
        elif args.GPU_memory_mode == "model_cpu_offload_and_qfloat8":
            convert_model_weight_to_float8(
                transformer,
                exclude_module_name=[
                    "modulation",
                ],
                device=device,
            )
            convert_weight_dtype_wrapper(transformer, weight_dtype)
            if transformer_2 is not None:
                convert_model_weight_to_float8(
                    transformer_2,
                    exclude_module_name=[
                        "modulation",
                    ],
                    device=device,
                )
                convert_weight_dtype_wrapper(transformer_2, weight_dtype)
            pipeline.enable_model_cpu_offload(device=device)
        elif args.GPU_memory_mode == "model_cpu_offload":
            pipeline.enable_model_cpu_offload(device=device)
        elif args.GPU_memory_mode == "model_full_load_and_qfloat8":
            convert_model_weight_to_float8(
                transformer,
                exclude_module_name=[
                    "modulation",
                ],
                device=device,
            )
            convert_weight_dtype_wrapper(transformer, weight_dtype)
            if transformer_2 is not None:
                convert_model_weight_to_float8(
                    transformer_2,
                    exclude_module_name=[
                        "modulation",
                    ],
                    device=device,
                )
                convert_weight_dtype_wrapper(transformer_2, weight_dtype)
            pipeline.to(device=device)
        else:
            pipeline.to(device=device)

        print(
            f"check param teacache {args.enable_teacache} {args.GPU_memory_mode} {args.cfg_skip_ratio} "
        )
        print(f"check arguments {args}")
        coefficients = (
            get_teacache_coefficients(args.model_name) if args.enable_teacache else None
        )
        if coefficients is not None:
            print(
                f"Enable TeaCache with threshold {args.teacache_threshold} and skip the first {args.num_skip_start_steps} steps."
            )
            pipeline.transformer.enable_teacache(
                coefficients,
                args.num_inference_steps,
                args.teacache_threshold,
                num_skip_start_steps=args.num_skip_start_steps,
                offload=args.teacache_offload,
            )
            if transformer_2 is not None:
                pipeline.transformer_2.share_teacache(transformer=pipeline.transformer)

        if args.cfg_skip_ratio is not None:
            print(f"Enable cfg_skip_ratio {args.cfg_skip_ratio}.")
            pipeline.transformer.enable_cfg_skip(
                args.cfg_skip_ratio, args.num_inference_steps
            )
            if not transformer_2 is None:
                pipeline.transformer_2.share_cfg_skip(transformer=pipeline.transformer)

        if args.lora_path is not None:
            print(f"Load LoRA from {args.lora_path}")
            pipeline = merge_lora(
                pipeline, args.lora_path, args.lora_weight, device=device
            )
        if args.lora_high_path is not None:
            print(f"Load LoRA from {args.lora_high_path}")
            pipeline = merge_lora(
                pipeline,
                args.lora_high_path,
                args.lora_weight,
                device=device,
                sub_transformer_name="transformer_2",
            )

        generator = torch.Generator(device=device).manual_seed(args.seed)
        print("Finish prepare I2V pipeline")
        return pipeline, generator
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

    pipe, generator = prepare_pipeline(args)
    if pipe == None:
        print("Prepare pipeline error")
        return None

    cnt = 0
    total_start = time.time()
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
        start = time.time()
        print(_start_frame)
        video = generation(pipe, generator, _start_frame, _prompt, _neg_prompt, args)
        print(f"Generation time: {time.time() - start} seconds for {_start_frame}")
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
    print(f"Total time: {time.time() - total_start} seconds for {cnt} videos")
    writer.close()
    # Convert output jsonl to tsv
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
    image2video(args)
