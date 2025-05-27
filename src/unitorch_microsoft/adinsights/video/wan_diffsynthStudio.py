import torch
from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData
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


def readimg(imagefile):
    try:
        if imagefile.startswith(("http://", "https://")):
            response = requests.get(imagefile)
            image = Image.open(BytesIO(response.content))
        else:
            image = Image.open(imagefile)
        max_area = 480 * 854
        aspect_ratio = image.height / image.width
        print(image.height, image.width, aspect_ratio)
        mod_value = 16
        height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        image = image.resize((width, height), resample=Image.LANCZOS)
        print(image.height, image.width)
        return image
    except Exception as e:
        print(e)
        return None


def load_model(ckpt_folder):
    # Load models
    model_manager = ModelManager(device="cpu")
    model_manager.load_models(
        [
            os.path.join(
                ckpt_folder, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
            )
        ],
        torch_dtype=torch.float32,  # Image Encoder is loaded with float32
    )
    model_manager.load_models(
        [
            [
                os.path.join(
                    ckpt_folder, "diffusion_pytorch_model-00001-of-00007.safetensors"
                ),
                os.path.join(
                    ckpt_folder, "diffusion_pytorch_model-00002-of-00007.safetensors"
                ),
                os.path.join(
                    ckpt_folder, "diffusion_pytorch_model-00003-of-00007.safetensors"
                ),
                os.path.join(
                    ckpt_folder, "diffusion_pytorch_model-00004-of-00007.safetensors"
                ),
                os.path.join(
                    ckpt_folder, "diffusion_pytorch_model-00005-of-00007.safetensors"
                ),
                os.path.join(
                    ckpt_folder, "diffusion_pytorch_model-00006-of-00007.safetensors"
                ),
                os.path.join(
                    ckpt_folder, "diffusion_pytorch_model-00007-of-00007.safetensors"
                ),
            ],
            os.path.join(ckpt_folder, "models_t5_umt5-xxl-enc-bf16.pth"),
            os.path.join(ckpt_folder, "Wan2.1_VAE.pth"),
        ],
        torch_dtype=torch.bfloat16,  # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization. #bfloat16
    )
    pipe = WanVideoPipeline.from_model_manager(
        model_manager, torch_dtype=torch.bfloat16, device="cuda"
    )
    pipe.enable_vram_management(
        num_persistent_param_in_dit=6 * 10**9
    )  # You can set `num_persistent_param_in_dit` to a small number to reduce VRAM required.
    return pipe


def generation(pipe, start_frame, prompt, neg_prompt, cache_dir, infer_step=10):
    print(f"Process video gen for {start_frame}")
    image = readimg(start_frame)
    if image == None:
        return None
    print("finish read img")
    try:
        video = pipe(
            prompt=prompt,
            negative_prompt=neg_prompt
            + "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
            input_image=image,
            height=image.height,
            width=image.width,
            num_inference_steps=infer_step,
            num_frames=161,
            seed=0,
            tiled=True,
        )
        name = (
            hashlib.md5(start_frame.encode()).hexdigest() + f"_step{infer_step}_10s.mp4"
        )
        print(name)
        name = os.path.join(cache_dir, name)
        print(name)
        save_video(video, name, fps=15, quality=5)
        return name
    except Exception as e:
        print(e)
        return None


def image2video(
    data_file: str,
    cache_dir: str,
    names: Union[str, List[str]],
    prompt_col: str,
    ckpt_dir: str,
    start_frame_col: Optional[str] = None,
    end_frame_col: Optional[str] = None,
    neg_prompt_col: Optional[str] = None,
):
    pipe = load_model(ckpt_dir)
    if isinstance(names, str) and names.strip() == "*":
        names = None
    if isinstance(names, str):
        names = re.split(r"[,;]", names)
        names = [n.strip() for n in names]

    data = pd.read_csv(data_file, names=names, sep="\t", quoting=3, header=None)
    os.makedirs(cache_dir, exist_ok=True)
    output_file = f"{cache_dir}/output.jsonl"
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
                    x[prompt_col]
                    if prompt_col is not None and not pd.isna(x[prompt_col])
                    else ""
                )
                + " - "
                + (
                    x[neg_prompt_col]
                    if neg_prompt_col is not None and not pd.isna(x[neg_prompt_col])
                    else ""
                )
                + " - "
                + (
                    x[start_frame_col]
                    if start_frame_col is not None and not pd.isna(x[start_frame_col])
                    else ""
                )
                + " - "
                + (
                    x[end_frame_col]
                    if end_frame_col is not None and not pd.isna(x[end_frame_col])
                    else ""
                )
                in uniques,
                axis=1,
            )
        ]
        print(f"Need to handle {len(data)} files")

    writer = open(output_file, "a+")

    assert prompt_col in data.columns, f"Column {prompt_col} not found in data."
    assert (
        start_frame_col in data.columns or end_frame_col in data.columns
    ), f"At least one image needed."

    cnt = 0
    for _, row in data.iterrows():
        _prompt = row[prompt_col] if not pd.isna(row[prompt_col]) else ""
        _neg_prompt = ""
        if neg_prompt_col != None:
            _neg_prompt = (
                row[neg_prompt_col] if not pd.isna(row[neg_prompt_col]) else ""
            )
        _start_frame = ""
        if start_frame_col != None:
            _start_frame = (
                row[start_frame_col] if not pd.isna(row[start_frame_col]) else ""
            )
        video = generation(
            pipe, _start_frame, _prompt, _neg_prompt, cache_dir, infer_step=10
        )
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


if __name__ == "__main__":
    fire.Fire()
