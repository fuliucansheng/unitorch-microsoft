# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import re
import torch
import pandas as pd
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from sklearn.metrics import roc_auc_score
from unitorch.models import GenericOutputs
from unitorch.models.clip import (
    ClipForPretrain as _ClipForPretrain,
    ClipProcessor,
)
from unitorch.utils import pop_value, nested_dict_value, read_file, read_json_file

from unitorch.cli import (
    hf_endpoint_url,
    cached_path,
    add_default_section_for_init,
    add_default_section_for_function,
)
from unitorch.cli import CoreConfigureParser, GenericScript
from unitorch.cli import register_script
from unitorch.cli.models.image_utils import ImageProcessor
from unitorch.cli.pipelines.stable.interrogator import ClipInterrogatorPipeline


@register_script("microsoft/picasso/script/diffusion/interrogator/clip")
class ClipInterrogatorScript(GenericScript):
    def __init__(self, config: CoreConfigureParser):
        self.config = config

    def launch(self, **kwargs):
        config = self.config

        pipe = ClipInterrogatorPipeline.from_core_configure(config)

        config.set_default_section(
            "microsoft/picasso/script/diffusion/interrogator/clip"
        )

        data_file = config.getoption("data_file", None)
        names = config.getoption("names", None)
        if isinstance(names, str) and names.strip() == "*":
            names = None
        if isinstance(names, str):
            names = re.split(r"[,;]", names)
            names = [n.strip() for n in names]

        image_col = config.getoption("image_col", None)

        data = pd.read_csv(
            data_file,
            names=names,
            sep="\t",
            quoting=3,
            header=None,
        )

        assert image_col in data.columns, f"Column {image_col} not found in data."

        output_file = config.getoption("output_file", None)

        processor = ImageProcessor.from_core_configure(config)

        results = []
        for image in data[image_col]:
            image = processor._read(image)
            caption = pipe.blip_pipe(image, num_beams=5, max_gen_seq_length=32)
            image_embeds = pipe.get_image_embeds(image)
            result = pipe.get_best_prompt(image_embeds, caption)
            results.append(result)

        data["result"] = results

        data.to_csv(output_file, sep="\t", header=None, index=None, quoting=3)
