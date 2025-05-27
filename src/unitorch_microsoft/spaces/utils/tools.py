# Copyright (c) MICROSOFT.
# Licensed under the MIT License.
import hashlib
import socket
import subprocess
from unitorch_microsoft import cached_path


def get_random_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", 0))
        return s.getsockname()[1]


def get_host_name():
    return socket.gethostname()


def start_http_server():
    http_port = get_random_port()
    http_process = subprocess.Popen(
        [
            "unitorch-service",
            "start",
            cached_path("services/http_files.ini"),
            "--daemon_mode",
            "False",
            "--html_dir",
            "/",
            "--port",
            str(http_port),
        ],
    )
    http_url = f"http://{get_host_name()}:{http_port}/" + "{0}"
    return http_url


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
