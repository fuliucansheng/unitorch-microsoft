# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import io
import re
import requests
import tempfile
import hashlib
import pandas as pd
import gradio as gr
from PIL import Image
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


@register_webui("microsoft/webui/labeling/classification")
class GenericClassificationLabelingWebUI(SimpleWebUI):
    def __init__(self, config: CoreConfigureParser):
        self._config = config
        config.set_default_section("microsoft/webui/labeling/classification")
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
        self.dataset["__index__"] = range(len(self.dataset))
        self.dataset["__index__"] = self.dataset["__index__"].astype(str)
        self.result_file = result_file

        # show columns
        self.text_cols = config.getoption("text_cols", None)
        self.image_cols = config.getoption("image_cols", None)
        self.video_cols = config.getoption("video_cols", None)
        self.show_cols = config.getoption("show_cols", None)
        self.http_url = config.getoption("http_url", "{0}")

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
        self.guideline = config.getoption("guideline", None)
        self.choices = config.getoption("choices", None)
        self.dataset["__user__"] = ""
        self.dataset["__comment__"] = ""
        self.dataset["__label__"] = ""

        if os.path.exists(result_file):
            self.dataset = pd.read_csv(result_file, sep="\t")
            self.dataset["__index__"] = self.dataset["__index__"].astype(str)
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
        choices = create_element(
            "radio",
            label="Label",
            values=self.choices,
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

        # show results
        refresh = create_element(
            "button",
            label="Refresh",
        )
        download = create_element(
            "download_button",
            label="Download",
            link=os.path.abspath(self.result_file),
        )
        results = create_element(
            "dataframe",
            label="Results",
            interactive=False,
        )

        # create blocks
        text_layout = create_column(*texts)
        image_layout = create_row(*images)
        video_layout = create_row(*videos)
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

        tab1 = create_tab(
            create_row(index, progress, user),
            *layouts,
            label_layout,
            name="Labeling",
        )
        tab2 = create_tab(
            create_row(progress, refresh, download), results, name="Results"
        )
        tabs = create_tabs(tab1, tab2)
        iface = create_blocks(guideline, tabs)

        # create events
        iface.__enter__()
        submit.click(
            self.serve,
            inputs=[index, user, choices, comment],
            outputs=[index, progress, choices, comment, *texts, *images, *videos],
        )
        random.click(
            self.sample,
            inputs=[],
            outputs=[index, progress, choices, comment, *texts, *images, *videos],
        )
        iface.load(
            fn=self.sample,
            inputs=[],
            outputs=[index, progress, choices, comment, *texts, *images, *videos],
        )
        refresh.click(
            self.show,
            inputs=[],
            outputs=[progress, results],
        )
        iface.load(
            fn=self.show,
            inputs=[],
            outputs=[progress, results],
        )

        def get_user(request: gr.Request):
            if request:
                return request.username
            return None

        iface.load(fn=get_user, inputs=None, outputs=user)

        iface.__exit__()

        super().__init__(config, iname="Human Classification Labeling", iface=iface)

    def sample(self):
        total = self.dataset.shape[0]
        labeled = self.dataset[self.dataset["__label__"] != ""].shape[0]
        progress = f"{labeled}/{total}"

        if labeled == total:
            return (None, progress, None, None) + tuple(
                [None]
                * (self.num_text_cols + self.num_image_cols + self.num_video_cols)
            )
        new_one = self.dataset[self.dataset["__label__"] == ""].sample(1).iloc[0]
        new_index = new_one["__index__"]
        new_texts = new_one[self.text_cols].tolist()
        new_images = new_one[self.image_cols].tolist()

        def save_url(url):
            if url.startswith("http"):
                try:
                    response = requests.get(url, stream=True)
                    name = os.path.join(
                        self.temp_folder, hashlib.md5(url.encode()).hexdigest()
                    )
                    if response.status_code == 200:
                        with open(name, "wb") as f:
                            for chunk in response.iter_content(8192):
                                f.write(chunk)
                        return name
                except Exception as e:
                    print(e)
            return url

        new_images = [save_url(p) for p in new_images]
        new_videos = new_one[self.video_cols].tolist()

        return (
            (new_index, progress, None, None)
            + tuple(new_texts)
            + tuple(new_images)
            + tuple(new_videos)
        )

    def show(self):
        total = self.dataset.shape[0]
        labeled = self.dataset[self.dataset["__label__"] != ""].shape[0]
        progress = f"{labeled}/{total}"

        results = self.dataset[self.dataset["__label__"] != ""].copy()
        for col in self.image_cols:
            results[col] = results[col].map(
                lambda x: x if x.startswith("http") else self.http_url.format(x)
            )
            results[col] = results[col].map(lambda x: f'<img src="{x}" width="100%">')
        for col in self.video_cols:
            results[col] = results[col].map(
                lambda x: x if x.startswith("http") else self.http_url.format(x)
            )
            results[col] = results[col].map(
                lambda x: f'<video src="{x}" width="100%" preload="none" controls>'
            )
        results.rename(
            columns={
                "__index__": "Index",
                "__user__": "User",
                "__label__": "Label",
                "__comment__": "Comment",
            },
            inplace=True,
        )
        results = results[
            ["Index"]
            + [
                col
                for col in results.columns
                if col != "Comment"
                and col != "User"
                and col != "Label"
                and col != "Index"
            ]
            + ["Label", "Comment"]
        ]
        return (progress, results)

    def serve(
        self,
        index,
        user,
        choice,
        comment,
    ):
        self.dataset.loc[self.dataset.__index__ == index, "__user__"] = user
        self.dataset.loc[self.dataset.__index__ == index, "__label__"] = choice
        self.dataset.loc[self.dataset.__index__ == index, "__comment__"] = comment
        self.dataset.to_csv(self.result_file, sep="\t", index=False)
        return self.sample()
