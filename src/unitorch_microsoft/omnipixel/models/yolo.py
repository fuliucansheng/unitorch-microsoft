import torch
import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO
import requests,re
from ultralytics import YOLO
from typing import List, Tuple, Union
import fire
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union


def load_image(image: str, http_url: str) -> Union[np.ndarray, None]:
    try:
        if http_url is not None:
            url = http_url.format(image)
            doc = requests.get(url, timeout=600)
            image = Image.open(BytesIO(doc.content)).convert("RGB")
        else:
            image = Image.open(image).convert("RGB")
        return np.array(image)
    except Exception as e:
        print(f"[Warning] Cannot load image {image}: {e}")
        return None


def process_batch(model, batch_images: List[np.ndarray]) -> List[Tuple[int, float, str]]:
    results = model.predict(batch_images, classes=0, verbose=False)
    outputs = []

    for i, result in enumerate(results):
        img_np = batch_images[i]
        h, w = img_np.shape[:2]

        if hasattr(result, 'boxes') and result.boxes.xyxy is not None and len(result.boxes.xyxy) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            human_count = len(boxes)

            max_area = 0
            max_box = None
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    max_box = box

            max_ratio = max_area / (h * w) if max_area > 0 else 0
            max_box_str = ",".join(map(str, map(int, max_box)))
        else:
            human_count = 0
            max_ratio = 0.0
            max_box_str = "0,0,0,0"

        outputs.append((human_count, round(max_ratio, 6), max_box_str))

    return outputs


def infer_yolov12_batch(
    data_file: str,
    output_file: str,
    names: Union[str, List[str]],
    image_col: str = "image_path",
    model_path: str = "yolo12n.pt",
    batch_size: int = 128,
    http_url: str = "http://0.0.0.0:11230/?file={0}",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(names, str) and names.strip() == "*":
        names = None
    if isinstance(names, str):
        names = re.split(r"[,;]", names)
        names = [n.strip() for n in names]

    df = pd.read_csv(
        data_file,
        names=names,
        sep="\t",
        quoting=3,
        header=None,
    )
    assert image_col in df.columns, f"{image_col} not found in input file"

    model = YOLO(model_path)
    model.eval()

    human_counts, max_ratios, max_boxes = [], [], []

    buffer_images = []
    buffer_indices = []

    for idx, row in df.iterrows():
        img_path = row[image_col]
        img_np = load_image(img_path, http_url=http_url)

        if img_np is not None:
            buffer_images.append(img_np)
            buffer_indices.append(idx)
        else:
            human_counts.append(0)
            max_ratios.append(0.0)
            max_boxes.append("0,0,0,0")

        if len(buffer_images) == batch_size or idx == len(df) - 1:
            if buffer_images:
                batch_results = process_batch(model, buffer_images)
                for res in batch_results:
                    h_cnt, h_ratio, h_box = res
                    human_counts.append(h_cnt)
                    max_ratios.append(h_ratio)
                    max_boxes.append(h_box)
                buffer_images.clear()
                buffer_indices.clear()

            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(df)}")

    df["human_count"] = human_counts
    df["max_area_ratio"] = max_ratios
    df["max_box"] = max_boxes

    df.to_csv(output_file, sep="\t", index=False, quoting=3, header=False)
    print(f"[Done] Saved results to: {output_file}")

if __name__ == "__main__":
    fire.Fire(infer_yolov12_batch)
