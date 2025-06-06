# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import pandas as pd
import gradio as gr
from unitorch.utils import read_file
from unitorch.models import GenericOutputs
from unitorch.cli import CoreConfigureParser
from unitorch.cli.webuis import SimpleWebUI
from unitorch_microsoft import cached_path
from unitorch_microsoft.spaces import (
    create_element,
    create_row,
    create_column,
    create_flex_layout,
    create_blocks,
    create_toper_menus,
    create_footer,
    create_dashboard_card,
    create_card,
    create_dashboard_cards_group,
    create_cards_group,
)


class DashboardWebUI(SimpleWebUI):
    _title = "Picasso Dashboard"
    _description = "This is a demo for Picasso Dashboard, which provides a comprehensive view of the Picasso image generative models performance and progress. It serves as a centralized hub for tracking key metrics related to the models powering Picasso, ensuring stakeholders and developers can monitor model health, accuracy, and efficiency."

    def __init__(
        self,
        config: CoreConfigureParser,
    ):
        self._config = config
        config.set_default_section("microsoft/spaces/picasso/dashboard")

        self.monitored_dataset = self.get_latest_data()

        # create elements
        toper_menus = create_toper_menus()
        dashboard_header = create_element(
            "markdown",
            "# <div style='margin:30px; min-height: 3em; text-align:center; font-weight: 600; font-size: 1.2em; color: darkslategray; display: flex; justify-content: center'>🌐 {self._title} </div>",
        )

        start_date = (pd.Timestamp.now() - pd.DateOffset(months=1)).strftime("%Y-%m-%d")
        end_date = (pd.Timestamp.now() + pd.DateOffset(days=1)).strftime("%Y-%m-%d")
        start = gr.DateTime(
            f"{start_date}", label="Start Date", include_time=False, type="string"
        )
        end = gr.DateTime(
            f"{end_date}", label="End Date", include_time=False, type="string"
        )
        apply = gr.Button("Apply", scale=1, variant="primary")
        refresh = gr.Button("Refresh", scale=1)

        def create_monitor(x, y, g, h):
            header = gr.HTML(f"<h3>{h}</h3>")
            plot = gr.ScatterPlot(
                x=x,
                y=y,
                color=g,
                y_aggregate="mean",
                x_label=x,
                y_label=y,
            )
            table = gr.Markdown()
            return GenericOutputs(
                plot=plot,
                table=table,
                layout=create_column(header, plot, table),
            )

        monitors = [
            create_monitor("Date", "GoodRate", "Group", "Segmentation"),
            create_monitor("Date", "GoodRate", "Group", "ControlNet"),
            create_monitor("Date", "BadRecall", "Group", "RAI Text | BadRecall"),
            create_monitor("Date", "GoodMisJudge", "Group", "RAI Text | GoodMisJudge"),
            create_monitor("Date", "BadRecall", "Group", "RAI Image | BadRecall"),
            create_monitor("Date", "GoodMisJudge", "Group", "RAI Image | GoodMisJudge"),
        ]

        footer = create_footer()

        # create layout
        iface = create_blocks(
            toper_menus,
            create_row(dashboard_header),
            create_row(start, end, apply, refresh),
            create_flex_layout(
                *[monitor.layout for monitor in monitors],
                num_per_row=2,
                do_padding=True,
            ),
            footer,
        )

        iface._title = self._title
        iface._description = self._description

        # create events
        iface.__enter__()
        apply.click(
            fn=self.filter_data,
            inputs=[start, end],
            outputs=[monitor.plot for monitor in monitors],
            trigger_mode="once",
        )

        refresh.click(
            self.refresh_data,
            outputs=[monitor.plot for monitor in monitors]
            + [monitor.table for monitor in monitors],
            trigger_mode="once",
        )
        iface.load(
            fn=self.refresh_data,
            outputs=[monitor.plot for monitor in monitors]
            + [monitor.table for monitor in monitors],
        )
        iface.__exit__()

        super().__init__(config, iname=self._title, iface=iface)

    def get_latest_data(self):
        seg_data = pd.DataFrame(
            {
                "Tasks": ["Segmentation"] * 3,
                "Date": ["2024-09-27"] * 3,
                "Group": ["White", "Simple", "Complex"],
                "ImageCount": [108, 165, 96],
                "GoodRate": [0.815, 0.886, 0.75],
            }
        )

        controlnet_data = pd.DataFrame(
            {
                "Tasks": ["ControlNet"] * 2,
                "Date": ["2024-09-27"] * 2,
                "Group": ["ProductInfo", "ProductInfo Rewrite"],
                "GoodRate": [0.82, 0.9318],
            }
        )

        rai_text_data = pd.DataFrame(
            {
                "Tasks": ["RAI Text"] * 3,
                "Date": ["2024-09-27"] * 3,
                "Group": ["Base AACS", "Our AACS", "Our AACS+GPT"],
                "BadRecall": [0.20, 0.477, 0.986],
                "GoodMisJudge": [0.002, 0.057, 0.222],
            }
        )

        rai_image_data = pd.DataFrame(
            {
                "Tasks": ["RAI Image"] * 2,
                "Date": ["2024-09-27"] * 2,
                "Group": ["Base AACS", "Our AACS"],
                "BadRecall": [0.734, 0.906],
                "GoodMisJudge": [0.008, 0.041],
            }
        )

        return pd.concat([seg_data, controlnet_data, rai_text_data, rai_image_data])

    def filter_data(self, start, end):
        dataset = self.monitored_dataset
        dataset = dataset[dataset["Date"].between(start, end)]
        segmentation = dataset[dataset["Tasks"] == "Segmentation"]
        controlnet = dataset[dataset["Tasks"] == "ControlNet"]
        rai_text = dataset[dataset["Tasks"] == "RAI Text"]
        rai_image = dataset[dataset["Tasks"] == "RAI Image"]

        return (
            segmentation,
            controlnet,
            rai_text,
            rai_text,
            rai_image,
            rai_image,
        )

    def refresh_data(self):
        self.monitored_dataset = self.get_latest_data()

        dataset = self.monitored_dataset
        segmentation = dataset[dataset["Tasks"] == "Segmentation"]
        controlnet = dataset[dataset["Tasks"] == "ControlNet"]
        rai_text = dataset[dataset["Tasks"] == "RAI Text"]
        rai_image = dataset[dataset["Tasks"] == "RAI Image"]

        segmentation_table = (
            segmentation.sort_values("Group", ascending=True)
            .sort_values("Date", ascending=False)
            .head(3)
        )
        controlnet_table = (
            controlnet.sort_values("Group", ascending=True)
            .sort_values("Date", ascending=False)
            .head(3)
        )
        rai_text_table = (
            rai_text.sort_values("Group", ascending=True)
            .sort_values("Date", ascending=False)
            .head(3)
        )
        rai_image_table = (
            rai_image.sort_values("Group", ascending=True)
            .sort_values("Date", ascending=False)
            .head(3)
        )

        return (
            segmentation,
            controlnet,
            rai_text,
            rai_text,
            rai_image,
            rai_image,
            segmentation_table[["Group", "Date", "GoodRate"]].to_markdown(index=False),
            controlnet_table[["Group", "Date", "GoodRate"]].to_markdown(index=False),
            rai_text_table[["Group", "Date", "BadRecall"]].to_markdown(index=False),
            rai_text_table[["Group", "Date", "GoodMisJudge"]].to_markdown(index=False),
            rai_image_table[["Group", "Date", "BadRecall"]].to_markdown(index=False),
            rai_image_table[["Group", "Date", "GoodMisJudge"]].to_markdown(index=False),
        )
