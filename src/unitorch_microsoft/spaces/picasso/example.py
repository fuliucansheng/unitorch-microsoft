# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import re
import subprocess
import hashlib
import pandas as pd
import gradio as gr
from torch.hub import download_url_to_file
from unitorch import get_temp_home
from unitorch.utils import read_file
from unitorch.models import GenericOutputs
from unitorch.cli import CoreConfigureParser
from unitorch.cli.webuis import SimpleWebUI
from unitorch_microsoft import cached_path
from unitorch_microsoft.spaces import (
    create_element,
    create_row,
    create_column,
    create_tab,
    create_tabs,
    create_flex_layout,
    create_blocks,
    create_toper_menus,
    create_footer,
    create_dashboard_card,
    create_card,
)

config_path = cached_path("spaces/config.ini")
config = CoreConfigureParser(config_path)


class ExampleWebUI(SimpleWebUI):
    def __init__(
        self,
        config: CoreConfigureParser,
        default_section: str = "microsoft/spaces/picasso/example",
        default_title: str = "Example Title",
        default_description: str = "Example Description",
        http_url: str = None,
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
        if http_url is not None:
            self.http_url = http_url
        else:
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
        self.text_cols = config.getoption("text_cols", [])
        self.image_cols = config.getoption("image_cols", [])
        self.video_cols = config.getoption("video_cols", [])
        self.url_cols = config.getoption("url_cols", [])
        self.html_cols = config.getoption("html_cols", [])
        self.show_cols = config.getoption("show_cols", None)
        self.group_col = config.getoption("group_col", None)
        self.num_images_per_row = config.getoption("num_images_per_row", 4)
        self.num_videos_per_row = config.getoption("num_videos_per_row", 4)
        self.num_mix_images_videos_per_row = config.getoption(
            "num_mix_images_videos_per_row", 4
        )
        self.num_html_per_row = config.getoption("num_html_per_row", 4)

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
        self.html_styles = config.getoption("html_styles", {})
        self.dataset["User"] = ""
        self.dataset["Comment"] = ""
        self.dataset["Label"] = ""

        self.choices = [c.replace(",", " ").strip() for c in self.choices]
        self.default_choice = self.default_choice.replace(",", " ").strip()
        if self.default_choice not in self.choices and self.default_choice != "":
            self.choices = self.choices + [self.default_choice]

        if os.path.exists(result_file):
            self.dataset = pd.read_csv(result_file, sep="\t")
            self.dataset.fillna("", inplace=True)
        else:
            self.dataset.to_csv(result_file, sep="\t", index=False)

        # create elements
        toper_menus = create_toper_menus()
        footer = create_footer()
        index = create_element(
            "text",
            label="Index",
            interactive=False,
        )
        group_values = [" - "]
        if self.group_col is not None:
            group_values += self.dataset[self.group_col].unique().tolist()
        group = create_element(
            "dropdown",
            label="Group",
            default=" - ",
            values=group_values,
        )
        user = create_element(
            "text",
            label="User",
            interactive=False,
            scale=2,
        )
        guideline_header = create_element(
            "markdown",
            label=f"# <div style='margin-top:10px'>🪄 {default_title}</div>",
            interactive=False,
        )
        guideline = create_element(
            "markdown",
            label=f"{default_description}",
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
        choices = create_element(
            "radio" if not self.checkbox else "checkboxgroup",
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
        res_label = create_element(
            "dropdown",
            label="Label",
            default=" - ",
            values=[" - "] + self.choices,
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
        adv_stats_header = create_element(
            "markdown",
            label="## Analytics",
            interactive=False,
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
        if (
            self.num_image_cols + self.num_video_cols
            > self.num_mix_images_videos_per_row
        ):
            image_layout = create_flex_layout(
                *images, num_per_row=self.num_images_per_row
            )
            video_layout = create_flex_layout(
                *videos, num_per_row=self.num_videos_per_row
            )
            mix_image_video_layout = None
        else:
            mix_image_video_layout = create_flex_layout(
                *images, *videos, num_per_row=self.num_mix_images_videos_per_row
            )
            image_layout, video_layout = None, None
        html_layout = create_flex_layout(*htmls, num_per_row=self.num_html_per_row)
        label_layout = create_row(
            comment, create_column(choices, create_row(random, submit))
        )

        layouts = []
        if self.num_text_cols > 0:
            layouts.append(text_layout)

        if self.num_image_cols > 0 and image_layout is not None:
            layouts.append(image_layout)

        if self.num_video_cols > 0 and video_layout is not None:
            layouts.append(video_layout)

        if mix_image_video_layout is not None:
            layouts.append(mix_image_video_layout)

        if self.num_html_cols > 0:
            layouts.append(html_layout)

        tab1 = create_tab(
            create_row(index, group, progress, user),
            *layouts,
            label_layout,
            name="Labeling",
        )
        tab2 = create_tab(
            create_row(progress, group, res_label, refresh, download),
            results,
            name="Results",
        )
        tab3 = create_tab(
            create_row(
                adv_index,
                adv_load,
                adv_reset,
            ),
            adv_stats_header,
            adv_stats,
            adv_preview,
            *layouts,
            name="Advanced",
        )
        tabs = create_tabs(
            tab1,
            tab2,
            tab3,
            elem_classes="ut-ms-min-75-height",
        )
        iface = create_blocks(toper_menus, guideline_header, guideline, tabs, footer)

        # create events
        iface.__enter__()
        submit.click(
            self.label,
            inputs=[index, group, user, choices, comment],
            outputs=[
                download,
                adv_stats,
                index,
                progress,
                group,
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
            inputs=[group],
            outputs=[
                index,
                progress,
                group,
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
            inputs=[group, res_label],
            outputs=[progress, results],
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
            inputs=[group],
            outputs=[
                index,
                progress,
                group,
                choices,
                comment,
                *texts,
                *images,
                *videos,
                *htmls,
            ],
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

        super().__init__(config, iname=default_title, iface=iface)

    def process_sample(self, sample):
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

        for col in self.image_cols + self.video_cols:
            sample[col] = save_url(sample[col])

        url = (
            lambda x: x
            if x.startswith("http")
            else self.http_url.format(os.path.abspath(x))
        )
        for col in self.url_cols:
            sample[col] = url(sample[col])

        for col in self.html_cols:
            sample[col] = self.html_styles.get(sample[col], "{0}").format(sample[col])
        return sample

    def process_results(self, results):
        url = (
            lambda x: x
            if x.startswith("http")
            else self.http_url.format(os.path.abspath(x))
        )
        for col in set(self.image_cols):
            results[col] = results[col].map(url)
            results[col] = results[col].map(
                lambda x: f'<img src="{x}" style="max-height: 100px;">'
            )
        for col in set(self.video_cols):
            results[col] = results[col].map(
                lambda x: x
                if x.startswith("http")
                else self.http_url.format(os.path.abspath(x))
            )
            results[col] = results[col].map(
                lambda x: f'<video src="{x}" style="max-height: 100px;" preload="none" controls>'
            )

        for col in self.url_cols:
            results[col] = results[col].map(url)

        for col in set(self.html_cols):
            results[col] = results[col].map(
                lambda x: self.html_styles.get(x, "{0}").format(x)
            )
        return results

    def sample(self, group=None):
        total = self.dataset.shape[0]
        if group is not None and group not in ["", " - "]:
            labeled = self.dataset[
                (self.dataset[self.group_col] == group) & (self.dataset["Label"] != "")
            ]
            non_labeled = self.dataset[
                (self.dataset[self.group_col] == group) & (self.dataset["Label"] == "")
            ]
        else:
            labeled = self.dataset[self.dataset["Label"] != ""]
            non_labeled = self.dataset[self.dataset["Label"] == ""]
        progress = f"{len(labeled)} / {total}"

        if len(self.dataset[self.dataset["Label"] != ""]) == total:
            return (None, progress, group, None, None) + tuple(
                [None]
                * (
                    self.num_text_cols
                    + self.num_image_cols
                    + self.num_video_cols
                    + self.num_html_cols
                )
            )
        if len(non_labeled) == 0:
            non_labeled = self.dataset[self.dataset["Label"] == ""]
        new_one = non_labeled.sample(1).iloc[0]
        new_one = self.process_sample(new_one)
        new_index = new_one["Index"]
        new_texts = new_one[self.text_cols].tolist()
        new_images = new_one[self.image_cols].tolist()
        new_videos = new_one[self.video_cols].tolist()
        new_htmls = new_one[self.html_cols].tolist()

        return (
            (new_index, progress, group, None, None)
            + tuple(new_texts)
            + tuple(new_images)
            + tuple(new_videos)
            + tuple(new_htmls)
        )

    def show(self, group=None, choice=None):
        total = self.dataset.shape[0]
        if choice is not None and choice not in ["", " - "]:
            labeled = self.dataset[self.dataset["Label"].map(lambda x: choice in x)]
        else:
            labeled = self.dataset[self.dataset["Label"] != ""]

        if group is not None and group not in [" - "]:
            labeled = labeled[labeled[self.group_col] == group]

        progress = f"{len(self.dataset[self.dataset['Label'] != ''])} / {total}"

        results = labeled.copy()
        if len(results) > 0:
            results = self.process_results(results)

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
        progress = f"{labeled} / {total}"
        gr.Info(f"Reset {index} Success.")
        return None, progress

    def load(self, index):
        new_one = self.dataset[self.dataset["Index"] == index].iloc[0]
        new_one = self.process_sample(new_one)
        new_texts = new_one[self.text_cols].tolist()
        new_images = new_one[self.image_cols].tolist()
        new_videos = new_one[self.video_cols].tolist()
        new_htmls = new_one[self.html_cols].tolist()

        return (
            (index, None, None)
            + tuple(new_texts)
            + tuple(new_images)
            + tuple(new_videos)
            + tuple(new_htmls)
        )

    def stats(self):
        dataset = self.dataset.copy()
        dataset["Num"] = 1
        labeled = dataset[dataset["Label"] != ""]
        choices = self.choices

        labeled["Label"] = labeled["Label"].map(
            lambda x: x.split(",") if x != "" else []
        )
        labeled_exploded = labeled.explode("Label")

        percent = lambda x, y: f"{x / (y if y > 0 else 1):.2%}"

        if self.group_col is not None:
            group = labeled.groupby(self.group_col)["Num"].count().to_dict()

            stats = labeled_exploded.groupby([self.group_col, "Label"])["Num"].count()
            stats = [
                {
                    "Group": g,
                    "Label": c,
                    "Count": f"{stats.get((g, c), 0)} / {group[g]}",
                    "Percentage": percent(stats.get((g, c), 0), group[g]),
                }
                for g in group.keys()
                for c in choices
            ]
            stats = pd.DataFrame(stats)
            stats = stats.append(
                {
                    "Group": "-",
                    "Label": "Total",
                    "Count": f"{len(labeled)} / {len(dataset)}",
                    "Percentage": percent(len(labeled), len(dataset)),
                },
                ignore_index=True,
            )

        else:
            stats = labeled_exploded.groupby("Label")["Num"].count()
            stats = [
                {
                    "Label": c,
                    "Count": stats.get(c, 0),
                    "Percentage": percent(stats.get(c, 0), len(labeled)),
                }
                for c in choices
            ]
            stats = pd.DataFrame(stats)
            stats = stats.append(
                {
                    "Label": "Total",
                    "Count": len(labeled),
                    "Percentage": percent(len(labeled), len(dataset)),
                },
                ignore_index=True,
            )

        stats = stats.to_markdown(index=False)
        return stats

    def label(
        self,
        index,
        group,
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
        return (os.path.abspath(self.result_file), self.stats()) + self.sample(group)
