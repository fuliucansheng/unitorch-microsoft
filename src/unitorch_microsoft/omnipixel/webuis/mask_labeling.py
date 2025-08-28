# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import io
import re
import socket
import requests
import tempfile
import hashlib
import subprocess
import pandas as pd
import gradio as gr
from PIL import Image, ImageOps
from collections import Counter, defaultdict
from torch.hub import download_url_to_file
from unitorch import get_temp_home
from unitorch.cli import CoreConfigureParser
from unitorch.cli import register_webui
from unitorch.cli.webuis import (
    create_element,
    create_accordion,
    create_row,
    create_column,
    create_flex_layout,
    create_group,
    create_tab,
    create_tabs,
    create_blocks,
)
from unitorch.cli.webuis import SimpleWebUI


def get_random_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", 0))
        return s.getsockname()[1]


def get_host_name():
    return socket.gethostname()


@register_webui("microsoft/omnipixel/webui/inpainting/mask")
class PicassoInpaintingMaskWebUI(SimpleWebUI):
    def __init__(
        self,
        config: CoreConfigureParser,
    ):
        self._config = config
        config.set_default_section("microsoft/omnipixel/webui/inpainting/mask")
        data_file = config.getoption("data_file", None)
        result_file = config.getoption("result_file", None)
        names = config.getoption("names", "*")
        if isinstance(names, str) and names == "*":
            names = None
        if isinstance(names, str):
            names = re.split(r"[,;]", names)
            names = [n.strip() for n in names]

        self.dataset = pd.read_csv(
            data_file,
            names=names,
            header="infer" if names is None else None,
            sep="\t",
            quoting=3,
        )
        self.dataset["Index"] = self.dataset.index.map(lambda x: f"No.{x}")
        self.result_file = result_file
        self.dataset["Comment"] = ""
        self.dataset["Label"] = ""

        if os.path.exists(result_file):
            self.dataset = pd.read_csv(result_file, sep="\t")
            self.dataset.fillna("", inplace=True)
        else:
            self.dataset.to_csv(result_file, sep="\t", index=False)

        self.guideline = config.getoption("guideline", None)
        self.image_col = config.getoption("image_col", None)
        self.image_folder = config.getoption("image_folder", "./images")

        self.min_image_width = config.getoption("min_image_width", "none")
        self.min_image_height = config.getoption("min_image_height", "none")
        self.max_image_width = config.getoption("max_image_width", "none")
        self.max_image_height = config.getoption("max_image_height", "100px")

        os.makedirs(self.image_folder, exist_ok=True)

        self.http_port = get_random_port()
        self.http_process = subprocess.Popen(
            [
                "unitorch-service",
                "start",
                "services/http_files.ini",
                "--daemon_mode",
                "False",
                "--html_dir",
                "/",
                "--port",
                str(self.http_port),
            ],
        )
        self.http_url = f"http://{get_host_name()}:{self.http_port}/" + "{0}"

        guideline_header = create_element(
            "markdown",
            label="# <div style='margin-top:10px'>Guideline</div>",
            interactive=False,
        )
        guideline = create_element(
            "markdown",
            label=f"{self.guideline}",
            interactive=False,
        )
        index = create_element(
            "text",
            label="Index",
            interactive=False,
        )
        progress = create_element(
            "text",
            label="Progress",
            interactive=False,
            scale=2,
        )

        image = create_element("image_editor", "Input Image")
        mask_image = create_element("image", "Input Image Mask")

        comment = create_element(
            "text",
            label="Comment",
        )
        submit = create_element(
            "button",
            label="Submit",
        )

        refresh = create_element(
            "button",
            label="Refresh",
        )
        results = create_element(
            "dataframe",
            label="Results",
            interactive=False,
        )

        tab1 = create_tab(
            create_row(index, progress),
            create_row(image, mask_image),
            create_row(comment, submit),
            name="Labeling",
        )
        tab2 = create_tab(
            create_row(progress, refresh),
            results,
            name="Results",
        )
        tabs = create_tabs(tab1, tab2)
        iface = create_blocks(guideline_header, guideline, tabs)

        iface.__enter__()
        image.change(fn=self.composite_images, inputs=[image], outputs=[mask_image])
        submit.click(
            self.label,
            inputs=[index, comment, mask_image],
            outputs=[
                index,
                progress,
                comment,
                image,
            ],
            trigger_mode="once",
        )

        refresh.click(
            self.show,
            inputs=[],
            outputs=[progress, results],
            trigger_mode="once",
        )
        iface.load(
            fn=self.sample,
            inputs=[],
            outputs=[
                index,
                progress,
                comment,
                image,
            ],
        )
        iface.load(
            fn=self.show,
            inputs=[],
            outputs=[progress, results],
        )

        iface.__exit__()

        super().__init__(config, iname="Human Mask Labeling Tools", iface=iface)

    def composite_images(self, images):
        if images is None:
            return None
        layers = images["layers"]
        if len(layers) == 0:
            return None
        image = layers[0]
        for i in range(1, len(layers)):
            image = Image.alpha_composite(image, layers[i])
        image = image.convert("L")
        image = image.point(lambda p: p < 5 and 255)
        image = ImageOps.invert(image)
        return image

    def sample(self):
        total = self.dataset.shape[0]
        labeled = self.dataset[self.dataset["Label"] != ""]
        non_labeled = self.dataset[self.dataset["Label"] == ""]
        progress = f"{len(labeled)} / {total}"

        if len(self.dataset[self.dataset["Label"] != ""]) == total:
            return None, progress, "", None
        if len(non_labeled) == 0:
            non_labeled = self.dataset[self.dataset["Label"] == ""]
        new_one = non_labeled.sample(1).iloc[0]
        new_one[self.image_col] = self.save_url(new_one[self.image_col])
        return new_one["Index"], progress, "", new_one[self.image_col]

    def show(self):
        total = self.dataset.shape[0]
        labeled = self.dataset[self.dataset["Label"] != ""]
        progress = f"{len(self.dataset[self.dataset['Label'] != ''])} / {total}"

        results = labeled.copy()
        url = lambda x: (
            x if x.startswith("http") else self.http_url.format(os.path.abspath(x))
        )
        for col in [self.image_col, "Label"]:
            results[col] = results[col].map(url)
            results[col] = results[col].map(
                lambda x: f'<img src="{x}" style="min-width: {self.min_image_width}; max-width: {self.max_image_width}; min-height: {self.min_image_height}; max-height: {self.max_image_height};">'
            )

        results = results[["Index", self.image_col, "Label", "Comment"]]
        return (progress, results)

    def save_url(self, url):
        if url.startswith("http"):
            name = os.path.join(
                self.image_folder, "url_" + hashlib.md5(url.encode()).hexdigest()
            )
            try:
                if not os.path.exists(name):
                    download_url_to_file(url, name)
                return name
            except Exception as e:
                print(e)
        return url

    def save_image(self, image: Image.Image):
        md5 = hashlib.md5()
        md5.update(image.tobytes())
        name = md5.hexdigest() + ".jpg"
        output_path = f"{self.image_folder}/{name}"
        image.save(output_path)
        return output_path

    def label(self, index, comment, mask_image):
        self.dataset.loc[self.dataset.Index == index, "Label"] = self.save_image(
            mask_image
        )
        self.dataset.loc[self.dataset.Index == index, "Comment"] = comment
        self.dataset.to_csv(self.result_file, sep="\t", index=False)
        return self.sample()
