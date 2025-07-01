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
import argparse
import random
import sys
import torch
import numpy as np
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from transformers import CLIPVisionModel


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
            output = pipe(
                image=image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=81,
                generator=generator,
                guidance_scale=5.0,
            ).frames[0]

            name = hashlib.md5(start_frame.encode()).hexdigest() + f"_.mp4"
            name = os.path.join(args.cache_dir, name)
            export_to_video(output, name, fps=args.fps)
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
        "--vae_path", type=str, default=None, help="Path to pretrained VAE checkpoint"
    )
    parser.add_argument(
        "--lora_path", type=str, default=None, help="Path to LoRA checkpoint"
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


def prepare_pipeline(args):
    try:
        print("Prepare I2V pipeline")
        # model_id = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"
        image_encoder = CLIPVisionModel.from_pretrained(
            args.model_name, subfolder="image_encoder", torch_dtype=torch.float32
        )
        vae = AutoencoderKLWan.from_pretrained(
            args.model_name, subfolder="vae", torch_dtype=torch.float32
        )
        pipe = WanImageToVideoPipeline.from_pretrained(
            args.model_name,
            vae=vae,
            image_encoder=image_encoder,
            torch_dtype=torch.bfloat16,
        )
        device = torch.cuda.current_device()

        print(f"Transformer model loaded from {args.model_name}")
        print(f"device: {device}")

        if args.transformer_path is not None:
            print(f"From checkpoint: {args.transformer_path}")
            z3_flag = True
            if z3_flag:
                state_dict = load_z3_model(args.transformer_path)
            else:
                if args.transformer_path.endswith("safetensors"):
                    from safetensors.torch import load_file, safe_open

                    state_dict = load_file(args.transformer_path)
                else:
                    state_dict = torch.load(args.transformer_path, map_location="cpu")
            state_dict = (
                state_dict["state_dict"] if "state_dict" in state_dict else state_dict
            )

            # check_state_dict(transformer.state_dict(), state_dict)
            m, u = pipe.transformer.load_state_dict(state_dict, strict=False)
            print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

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

            m, u = pipe.vae.load_state_dict(state_dict, strict=False)
            print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

        if args.GPU_memory_mode == "sequential_cpu_offload":
            pipe.enable_sequential_cpu_offload(device=device)
        elif args.GPU_memory_mode == "model_cpu_offload":
            pipe.enable_model_cpu_offload(device=device)
        else:
            pipe.to(device=device)

        if args.lora_path is not None:
            print(f"Load LoRA from {args.lora_path}")
            # pipeline = merge_lora(pipeline, args.lora_path, args.lora_weight, device=device)

        generator = torch.Generator(device=device).manual_seed(args.seed)
        print("Finish prepare I2V pipeline")
        return pipe, generator
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
        video = generation(pipe, generator, _start_frame, _prompt, _neg_prompt, args)
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
