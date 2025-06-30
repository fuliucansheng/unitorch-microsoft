# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import io
import json
import re
import random
import logging
import requests
import numpy as np
import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from PIL import Image
from typing import Optional
from unitorch.cli import register_webui, CoreConfigureParser
from unitorch.cli.webuis import SimpleWebUI
from unitorch.cli.webuis import (
    matched_pretrained_names,
    create_element,
    create_accordion,
    create_row,
    create_column,
    create_group,
    create_tab,
    create_tabs,
    create_blocks,
    create_pretrain_layout,
    create_lora_layout,
)


@register_webui("microsoft/webui/tools/calibration")
class CalibrationWebUI(SimpleWebUI):
    def __init__(self, config: CoreConfigureParser):
        config.set_default_section("microsoft/webui/tools/calibration")
        data_file = create_element("file", "Input Data File")
        n_bins = create_element(
            "slider", "Number of Bins", min_value=0, max_value=1000, step=1, default=50
        )
        pred_index = create_element("number", "Prediction Index", default=0)
        label_index = create_element("number", "Label Index", default=1)
        generate = create_element("button", "Generate")
        output_image = create_element("image", "Output Image")

        left = create_column(
            data_file, n_bins, create_row(pred_index, label_index), generate
        )
        right = create_column(output_image)
        iface = create_blocks(create_row(left, right))

        iface.__enter__()

        generate.click(
            fn=self.generate_calibration_plot,
            inputs=[data_file, n_bins, pred_index, label_index],
            outputs=[output_image],
            trigger_mode="once",
        )

        iface.__exit__()

        super().__init__(config, "Calibration", iface)

    def generate_calibration_plot(
        self,
        data_file: str,
        n_bins: int = 10,
        pred_index: int = 0,
        label_index: int = 1,
    ):
        df = pd.read_csv(data_file, sep="\t", header=None)
        y_prob = df[int(pred_index)].values
        y_true = df[int(label_index)].values

        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy="uniform"
        )

        plt.figure(figsize=(6, 6))
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
        plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
        plt.title("Calibration Plot")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        return Image.open(buf)
