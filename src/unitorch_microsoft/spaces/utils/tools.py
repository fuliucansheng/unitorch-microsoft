# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import hashlib
import subprocess
from unitorch_microsoft import cached_path

hashed_link = lambda text, num=6: hashlib.sha1(text.encode()).hexdigest()[:num]

bg_colors = [
    "ut-ms-bg-color-red",
    "ut-ms-bg-color-green",
    "ut-ms-bg-color-blue",
    "ut-ms-bg-color-yellow",
    "ut-ms-bg-color-purple",
]


def random_bg_color(text):
    return bg_colors[hash(text) % len(bg_colors)]
