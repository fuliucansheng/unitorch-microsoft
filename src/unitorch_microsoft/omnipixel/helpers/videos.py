# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import re
import cv2
import fire
import json
import logging
import hashlib
import subprocess
import tempfile
import pandas as pd
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union


def save_image(folder, image):
    name = hashlib.md5(image.tobytes()).hexdigest() + ".jpg"
    image.save(f"{folder}/{name}")
    return f"{folder}/{name}"


def list_frames(video, folder):
    video = cv2.VideoCapture(video)
    frames = []
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frames.append(frame)

    video.release()
    os.makedirs(folder, exist_ok=True)
    for i, frame in enumerate(frames):
        frame.save(os.path.join(folder, f"frame_{i:04d}.jpg"))
    return


if __name__ == "__main__":
    fire.Fire(
        {
            "list_frames": list_frames,
        }
    )
