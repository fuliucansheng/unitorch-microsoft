# Copyright (c) MICROSOFT.
# Licensed under the MIT License.
import io
import re
import fire
import requests
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

from unitorch.cli import cached_path
from unitorch.cli import CoreConfigureParser
from unitorch.cli.models.image_utils import ImageProcessor
from unitorch.cli.pipelines.stable.interrogator import ClipInterrogatorPipeline


def launch(
    data_file: str,
    output_file: str,
    names: Union[str, List[str]],
    image_col: str,
    pretrained_name: Optional[str] = "clip-vit-large-patch14",
    http_url: Optional[str] = "http://0.0.0.0:11230/?file={0}",
):
    pipe = ClipInterrogatorPipeline.from_core_configure(
        config=CoreConfigureParser(),
        pretrained_name=pretrained_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    if isinstance(names, str) and names.strip() == "*":
        names = None
    if isinstance(names, str):
        names = re.split(r"[,;]", names)
        names = [n.strip() for n in names]

    data = pd.read_csv(
        data_file,
        names=names,
        sep="\t",
        quoting=3,
        header=None,
    )

    assert image_col in data.columns, f"Column {image_col} not found in data."

    results = []
    for image in data[image_col]:
        if http_url is not None:
            url = http_url.format(image)
            doc = requests.get(url, timeout=600)
            image = Image.open(io.BytesIO(doc.content)).convert("RGB")
        else:
            image = Image.open(image).convert("RGB")
        caption = pipe.blip_pipe(image, num_beams=5, max_gen_seq_length=32)
        image_embeds = pipe.get_image_embeds(image)
        result = pipe.get_best_prompt(image_embeds, caption)
        results.append(result)

    data["result"] = results

    data.to_csv(output_file, sep="\t", header=None, index=None, quoting=3)


if __name__ == "__main__":
    fire.Fire(launch)
