# Copyright (c) MICROSOFT.
# Licensed under the MIT License.
import io
import re
import fire
import requests
import numpy as np
import pandas as pd
import torch
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from paddleocr import PaddleOCR


def ocr(
    data_file: str,
    output_file: str,
    names: Union[str, List[str]],
    image_col: str,
    http_url: str = "http://0.0.0.0:11230/?file={0}",
):
    use_gpu = True if torch.cuda.is_available() else False
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

    ocr = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=use_gpu, show_log=False)

    images = data[image_col].unique()
    results = pd.DataFrame({image_col: images})
    roi_results = []
    for image in images:
        if http_url is not None:
            url = http_url.format(image)
            doc = requests.get(url, timeout=600)
            image = Image.open(io.BytesIO(doc.content)).convert("RGB")
        else:
            image = Image.open(image).convert("RGB")

        bboxes = ""
        try:
            outputs = ocr.ocr(np.array(image), det=True, cls=False)
            if outputs is None or len(outputs) < 1:
                continue
            elif outputs[0] is None or len(outputs[0]) < 1:
                continue
            outputs = outputs[0]
            _results = []
            for result in outputs:
                xs, ys = zip(*result[0])
                x1, y1, x2, y2 = min(*xs), min(*ys), max(*xs), max(*ys)
                _results.append(f"{int(x1)},{int(y1)},{int(x2)},{int(y2)}")
            bboxes = ";".join(_results)
        except Exception as e:
            print("Error OCR: %s" % e)
        finally:
            roi_results.append(bboxes)
    results["roi"] = roi_results
    results = pd.merge(
        data,
        results,
        on=image_col,
        how="inner",
    )
    results.to_csv(output_file, sep="\t", index=False, header=None, quoting=3)


if __name__ == "__main__":
    fire.Fire(ocr)
