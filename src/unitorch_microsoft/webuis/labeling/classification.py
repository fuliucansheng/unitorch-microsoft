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
from PIL import Image
from collections import Counter
from torch.hub import download_url_to_file
from unitorch import get_temp_home
from unitorch.cli import CoreConfigureParser
from unitorch.cli import register_webui
from unitorch.cli.webuis import (
    create_element,
    create_accordion,
    create_row,
    create_column,
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


def get_flex_layout(*eles, num_per_row=2):
    rows = [
        create_row(*eles[i : i + num_per_row]) for i in range(0, len(eles), num_per_row)
    ]
    return create_column(*rows)


@register_webui("microsoft/webui/labeling/classification")
class GenericClassificationLabelingWebUI(SimpleWebUI):
    def __init__(
        self,
        config: CoreConfigureParser,
        default_section: str = "microsoft/webui/labeling/classification",
    ):
        self._config = config
        config.set_default_section(default_section)
        data_file = config.getoption("data_file", None)
        result_file = config.getoption("result_file", None)
        names = config.getoption("names", "*")
        if isinstance(names, str) and names == "*":
            names = None
        if isinstance(names, str):
            names = re.split(r"[,;]", names)
            names = [n.strip() for n in names]

        temp_folder = config.getoption("temp_folder", get_temp_home())
        os.makedirs(temp_folder, exist_ok=True)
        self.temp_folder = temp_folder

        self.dataset = pd.read_csv(
            data_file,
            names=names,
            header="infer" if names is None else None,
            sep="\t",
            quoting=3,
        )
        self.dataset["Index"] = self.dataset.index.map(lambda x: f"No.{x}")
        self.result_file = result_file

        # start http server: unitorch-service start services/http_files.ini --daemon_mode False --html_dir /
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

        # show columns
        self.text_cols = config.getoption("text_cols", None)
        self.image_cols = config.getoption("image_cols", None)
        self.video_cols = config.getoption("video_cols", None)
        self.html_cols = config.getoption("html_cols", None)
        self.show_cols = config.getoption("show_cols", None)
        self.num_images_per_row = config.getoption("num_images_per_row", 4)
        self.num_videos_per_row = config.getoption("num_videos_per_row", 4)
        self.num_html_per_row = config.getoption("num_html_per_row", 4)

        self.text_cols = [] if self.text_cols is None else self.text_cols
        self.image_cols = [] if self.image_cols is None else self.image_cols
        self.video_cols = [] if self.video_cols is None else self.video_cols
        self.html_cols = [] if self.html_cols is None else self.html_cols

        if self.text_cols is not None:
            if isinstance(self.text_cols, str):
                self.text_cols = re.split(r"[,;]", self.text_cols)
                self.text_cols = [n.strip() for n in self.text_cols]
            assert all(
                [col in self.dataset.columns for col in self.text_cols]
            ), f"text_cols {self.text_cols} not found in dataset"

        if self.image_cols is not None:
            if isinstance(self.image_cols, str):
                self.image_cols = re.split(r"[,;]", self.image_cols)
                self.image_cols = [n.strip() for n in self.image_cols]
            assert all(
                [col in self.dataset.columns for col in self.image_cols]
            ), f"image_cols {self.image_cols} not found in dataset"

        if self.video_cols is not None:
            if isinstance(self.video_cols, str):
                self.video_cols = re.split(r"[,;]", self.video_cols)
                self.video_cols = [n.strip() for n in self.video_cols]
            assert all(
                [col in self.dataset.columns for col in self.video_cols]
            ), f"video_cols {self.video_cols} not found in dataset"

        if self.show_cols is not None:
            if isinstance(self.show_cols, str):
                self.show_cols = re.split(r"[,;]", self.show_cols)
                self.show_cols = [n.strip() for n in self.show_cols]
            assert all(
                [col in self.dataset.columns for col in self.show_cols]
            ), f"show_cols {self.show_cols} not found in dataset"
        else:
            self.show_cols = list(self.dataset.columns)

        self.num_text_cols = 0 if self.text_cols is None else len(self.text_cols)
        self.num_image_cols = 0 if self.image_cols is None else len(self.image_cols)
        self.num_video_cols = 0 if self.video_cols is None else len(self.video_cols)
        self.num_html_cols = 0 if self.html_cols is None else len(self.html_cols)
        self.guideline = config.getoption("guideline", None)
        self.choices = config.getoption("choices", None)
        self.checkbox = config.getoption("checkbox", False)
        self.default_choice = config.getoption("default_choice", "")
        self.dataset["User"] = ""
        self.dataset["Comment"] = ""
        self.dataset["Label"] = ""

        self.choices = [c.replace(",", " ").strip() for c in self.choices]
        self.default_choice = self.default_choice.replace(",", " ").strip()

        if os.path.exists(result_file):
            self.dataset = pd.read_csv(result_file, sep="\t")
            self.dataset.fillna("", inplace=True)
        else:
            self.dataset.to_csv(result_file, sep="\t", index=False)

        # create elements
        index = create_element(
            "text",
            label="Index",
            interactive=False,
        )
        user = create_element(
            "text",
            label="User",
            interactive=False,
            scale=2,
        )
        guideline = create_element(
            "markdown",
            label=f"# Guideline \n {self.guideline}",
            interactive=False,
        )
        texts = [
            create_element(
                "text",
                label=col,
                lines=2,
            )
            for col in self.text_cols
        ]
        images = [
            create_element(
                "image",
                label=col,
            )
            for col in self.image_cols
        ]
        videos = [
            create_element(
                "video",
                label=col,
            )
            for col in self.video_cols
        ]
        htmls = [
            create_element(
                "html",
                label=col,
            )
            for col in self.html_cols
        ]
        choices = (
            create_element(
                "radio",
                label="Label",
                values=self.choices,
            )
            if not self.checkbox
            else create_element(
                "checkboxgroup",
                label="Label",
                values=self.choices,
            )
        )
        comment = create_element(
            "text",
            label="Comment",
            lines=4,
        )
        random = create_element(
            "button",
            label="Random",
        )
        submit = create_element(
            "button",
            label="Submit",
        )
        progress = create_element(
            "text",
            label="Progress",
            interactive=False,
            scale=2,
        )

        adv_index = create_element(
            "text",
            label="Index",
            interactive=True,
            scale=2,
        )
        adv_load = create_element(
            "button",
            label="Load",
        )
        adv_reset = create_element(
            "button",
            label="Reset",
            variant="secondary",
        )
        adv_stats = create_element(
            "markdown",
            label="",
            interactive=False,
        )
        adv_preview = create_element(
            "markdown",
            label="## Preview",
            interactive=False,
        )

        # show results
        refresh = create_element(
            "button",
            label="Refresh",
        )
        download = create_element(
            "download_button",
            label="Download",
        )
        results = create_element(
            "dataframe",
            label="Results",
            interactive=False,
        )

        # create blocks
        text_layout = create_column(*texts)
        image_layout = get_flex_layout(*images, num_per_row=self.num_images_per_row)
        video_layout = get_flex_layout(*videos, num_per_row=self.num_videos_per_row)
        html_layout = get_flex_layout(*htmls, num_per_row=self.num_html_per_row)
        label_layout = create_row(
            comment, create_column(choices, create_row(random, submit))
        )

        layouts = []
        if self.num_text_cols > 0:
            layouts.append(text_layout)

        if self.num_image_cols > 0:
            layouts.append(image_layout)

        if self.num_video_cols > 0:
            layouts.append(video_layout)

        if self.num_html_cols > 0:
            layouts.append(html_layout)

        tab1 = create_tab(
            create_row(index, progress, user),
            *layouts,
            label_layout,
            name="Labeling",
        )
        tab2 = create_tab(
            create_row(progress, refresh, download), results, name="Results"
        )
        tab3 = create_tab(
            create_row(
                create_column(
                    create_row(adv_index, user),
                    create_row(adv_load, adv_reset),
                    scale=2,
                ),
                adv_stats,
            ),
            adv_preview,
            *layouts,
            name="Advanced",
        )
        tabs = create_tabs(tab1, tab2, tab3)
        iface = create_blocks(guideline, tabs)

        # create events
        iface.__enter__()
        submit.click(
            self.label,
            inputs=[index, user, choices, comment],
            outputs=[
                download,
                adv_stats,
                index,
                progress,
                choices,
                comment,
                *texts,
                *images,
                *videos,
                *htmls,
            ],
        )
        random.click(
            self.sample,
            inputs=[],
            outputs=[
                index,
                progress,
                choices,
                comment,
                *texts,
                *images,
                *videos,
                *htmls,
            ],
        )
        adv_load.click(
            self.load,
            inputs=[adv_index],
            outputs=[
                index,
                choices,
                comment,
                *texts,
                *images,
                *videos,
                *htmls,
            ],
        )
        adv_reset.click(
            self.reset,
            inputs=[adv_index],
            outputs=[adv_index, progress],
        )
        index.change(
            fn=lambda x: x,
            inputs=[index],
            outputs=[adv_index],
        )

        iface.load(
            fn=self.sample,
            inputs=[],
            outputs=[
                index,
                progress,
                choices,
                comment,
                *texts,
                *images,
                *videos,
                *htmls,
            ],
        )
        refresh.click(
            self.show,
            inputs=[],
            outputs=[progress, results],
        )
        iface.load(
            fn=lambda: tuple(self.show())
            + (os.path.abspath(self.result_file), self.stats()),
            inputs=[],
            outputs=[progress, results, download, adv_stats],
        )

        def get_user(request: gr.Request):
            if request:
                return request.username
            return None

        iface.load(fn=get_user, inputs=None, outputs=user)

        iface.__exit__()

        super().__init__(config, iname="Human Classification Labeling", iface=iface)

    def postprocess_texts(self, *texts, info=None):
        return texts

    def preprocess_images(self, *images, info=None):
        return images

    def postprocess_images(self, *images, info=None):
        return images

    def preprocess_videos(self, *videos, info=None):
        return videos

    def postprocess_videos(self, *videos, info=None):
        return videos

    def postprocess_htmls(self, *htmls, info=None):
        return htmls

    def process_show_cols(self, results, show_cols=None):
        return results

    def sample(self):
        total = self.dataset.shape[0]
        labeled = self.dataset[self.dataset["Label"] != ""].shape[0]
        progress = f"{labeled}/{total}"

        if labeled == total:
            return (None, progress, None, None) + tuple(
                [None]
                * (self.num_text_cols + self.num_image_cols + self.num_video_cols)
            )
        new_one = self.dataset[self.dataset["Label"] == ""].sample(1).iloc[0]
        new_index = new_one["Index"]
        new_texts = new_one[self.text_cols].tolist()
        new_texts = self.postprocess_texts(*new_texts, info=new_one)

        def save_url(url):
            if url.startswith("http"):
                name = os.path.join(
                    self.temp_folder, hashlib.md5(url.encode()).hexdigest()
                )
                try:
                    download_url_to_file(url, name)
                    return name
                except Exception as e:
                    print(e)
            return url

        new_images = new_one[self.image_cols].tolist()
        new_images = self.preprocess_images(*new_images, info=new_one)
        new_images = [save_url(p) for p in new_images]
        new_images = self.postprocess_images(*new_images, info=new_one)

        new_videos = new_one[self.video_cols].tolist()
        new_videos = self.preprocess_videos(*new_videos, info=new_one)
        new_videos = [save_url(p) for p in new_videos]
        new_videos = self.postprocess_videos(*new_videos, info=new_one)

        new_htmls = new_one[self.html_cols].tolist()
        new_htmls = self.postprocess_htmls(*new_htmls, info=new_one)

        return (
            (new_index, progress, None, None)
            + tuple(new_texts)
            + tuple(new_images)
            + tuple(new_videos)
            + tuple(new_htmls)
        )

    def show(self):
        total = self.dataset.shape[0]
        labeled = self.dataset[self.dataset["Label"] != ""].shape[0]
        progress = f"{labeled}/{total}"

        results = self.dataset[self.dataset["Label"] != ""].copy()
        results = self.process_show_cols(results, show_cols=self.show_cols)

        for col in set(self.image_cols):
            results[col] = results[col].map(
                lambda x: x
                if x.startswith("http")
                else self.http_url.format(os.path.abspath(x))
            )
            results[col] = results[col].map(lambda x: f'<img src="{x}" width="100%">')
        for col in set(self.video_cols):
            results[col] = results[col].map(
                lambda x: x
                if x.startswith("http")
                else self.http_url.format(os.path.abspath(x))
            )
            results[col] = results[col].map(
                lambda x: f'<video src="{x}" width="100%" preload="none" controls>'
            )

        results = results[
            ["Index"]
            + [
                col
                for col in self.show_cols
                if col not in ["Comment", "User", "Label", "Index"]
            ]
            + ["Label", "Comment"]
        ]
        return (progress, results)

    def reset(self, index):
        self.dataset.loc[self.dataset.Index == index, "User"] = ""
        self.dataset.loc[self.dataset.Index == index, "Label"] = ""
        self.dataset.loc[self.dataset.Index == index, "Comment"] = ""
        self.dataset.to_csv(self.result_file, sep="\t", index=False)
        total = self.dataset.shape[0]
        labeled = self.dataset[self.dataset["Label"] != ""].shape[0]
        progress = f"{labeled}/{total}"
        gr.Info(f"Reset {index} Success.")
        return None, progress

    def load(self, index):
        new_one = self.dataset[self.dataset["Index"] == index].iloc[0]
        new_texts = new_one[self.text_cols].tolist()
        new_texts = self.postprocess_texts(*new_texts, info=new_one)

        def save_url(url):
            if url.startswith("http"):
                name = os.path.join(
                    self.temp_folder, hashlib.md5(url.encode()).hexdigest()
                )
                try:
                    download_url_to_file(url, name)
                    return name
                except Exception as e:
                    print(e)
            return url

        new_images = new_one[self.image_cols].tolist()
        new_images = self.preprocess_images(*new_images, info=new_one)
        new_images = [save_url(p) for p in new_images]
        new_images = self.postprocess_images(*new_images, info=new_one)

        new_videos = new_one[self.video_cols].tolist()
        new_videos = self.preprocess_videos(*new_videos, info=new_one)
        new_videos = [save_url(p) for p in new_videos]
        new_videos = self.postprocess_videos(*new_videos, info=new_one)

        new_htmls = new_one[self.html_cols].tolist()
        new_htmls = self.postprocess_htmls(*new_htmls, info=new_one)

        return (
            (index, None, None)
            + tuple(new_texts)
            + tuple(new_images)
            + tuple(new_videos)
            + tuple(new_htmls)
        )

    def stats(self):
        labeled = self.dataset[self.dataset["Label"] != ""]
        choices = self.choices
        if self.default_choice not in choices and self.default_choice != "":
            choices = choices + [self.default_choice]

        counter = Counter()
        for labels in labeled["Label"].tolist():
            counter.update(labels.split(","))
        stats = pd.DataFrame(
            {"Label": choices, "Count": [counter[choice] for choice in choices]}
        )
        stats["Percentage"] = stats["Count"] / (len(labeled) if len(labeled) > 0 else 1)
        stats["Percentage"] = stats["Percentage"].map(lambda x: f"{x:.2%}")
        stats = pd.concat(
            [
                stats,
                pd.DataFrame(
                    {
                        "Label": ["Total"],
                        "Count": [f"{len(labeled)} / {len(self.dataset)}"],
                        "Percentage": [f"{len(labeled) / len(self.dataset):.2%}"],
                    }
                ),
            ]
        )
        stats = stats.to_markdown(index=False)
        return stats

    def label(
        self,
        index,
        user,
        choice,
        comment,
    ):
        if isinstance(choice, list) or isinstance(choice, tuple):
            choice = ",".join(choice) if len(choice) > 0 else self.default_choice
        elif choice is None:
            choice = self.default_choice
        self.dataset.loc[self.dataset.Index == index, "User"] = user
        self.dataset.loc[self.dataset.Index == index, "Label"] = choice
        self.dataset.loc[self.dataset.Index == index, "Comment"] = comment
        self.dataset.to_csv(self.result_file, sep="\t", index=False)
        return (os.path.abspath(self.result_file), self.stats()) + self.sample()
