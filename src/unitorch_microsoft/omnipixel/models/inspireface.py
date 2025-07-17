import os
import cv2
import fire
import torch
import pandas as pd
import numpy as np
import inspireface as isf
from typing import List, Tuple, Union
import base64
import re
import requests
from PIL import Image
from io import BytesIO

def load_image(image: str, http_url: str) -> Union[np.ndarray, None]:
    try:
        if http_url:
            url = http_url.format(image)
            doc = requests.get(url, timeout=600)
            image = Image.open(BytesIO(doc.content)).convert("RGB")
        else:
            image = Image.open(image).convert("RGB")
        return np.array(image)[:, :, ::-1]  # convert RGB to BGR for cv2
    except Exception as e:
        print(f"[Warning] Cannot load image {image}: {e}")
        return None


def process_batch(session, batch_images: List[np.ndarray]) -> List[Tuple[str, str, str, str, float, str, str, str]]:
    results = []

    race_tags = ["Black", "Asian", "Latino/Hispanic", "Middle Eastern", "White"]
    gender_tags = ["Female", "Male"]
    age_bracket_tags = [
        "0-2 years old", "3-9 years old", "10-19 years old", "20-29 years old",
        "30-39 years old", "40-49 years old", "50-59 years old", "60-69 years old",
        "more than 70 years old"
    ]

    pipeline_opt = (isf.HF_ENABLE_QUALITY | isf.HF_ENABLE_MASK_DETECT |
                    isf.HF_ENABLE_LIVENESS | isf.HF_ENABLE_FACE_ATTRIBUTE)

    for img in batch_images:
        h, w = img.shape[:2]
        faces = session.face_detection(img)

        if not faces:
            results.append(("[]", "0,0,0,0", 0.0, "", "", ""))
            continue

        extensions = session.face_pipeline(img, faces, pipeline_opt)
        if not extensions or len(extensions) != len(faces):
            results.append(("[]", "0,0,0,0", 0.0, "", "", ""))
            continue

        # 合并 (face, ext)，计算面积
        face_data = []
        for face, ext in zip(faces, extensions):
            x1, y1, x2, y2 = face.location
            area = (x2 - x1) * (y2 - y1)
            face_data.append(((x1, y1, x2, y2), area, ext))

        box_strs = ["{},{},{},{}".format(*fd[0]) for fd in face_data]

        # 找最大框对应项
        max_fd = max(face_data, key=lambda x: x[1])
        max_box, max_area, max_ext = max_fd
        max_area_ratio = max_area / (h * w)

        gender = gender_tags[max_ext.gender]
        race = race_tags[max_ext.race]
        age_bracket = age_bracket_tags[max_ext.age_bracket]

        results.append((
            "|".join(box_strs),
            "{},{},{},{}".format(*max_box),
            round(max_area_ratio, 6),
            gender,
            race,
            age_bracket
        ))

    return results


def infer_inspireface_batch(
    data_file: str,
    output_file: str,
    names: Union[str, List[str]],
    image_col: str = "image_path",
    batch_size: int = 64,
    http_url: str = None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(names, str) and names.strip() == "*":
        names = None
    if isinstance(names, str):
        names = re.split(r"[,;]", names)
        names = [n.strip() for n in names]

    df = pd.read_csv(data_file, names=names, sep="\t", quoting=3, header=None)
    assert image_col in df.columns, f"{image_col} not found in input file"

    # 初始化 InspireFace
    opt = (isf.HF_ENABLE_FACE_RECOGNITION | isf.HF_ENABLE_QUALITY | 
       isf.HF_ENABLE_MASK_DETECT | isf.HF_ENABLE_LIVENESS | 
       isf.HF_ENABLE_INTERACTION | isf.HF_ENABLE_FACE_ATTRIBUTE | 
       isf.HF_ENABLE_FACE_EMOTION)
    session = isf.InspireFaceSession(opt, isf.HF_DETECT_MODE_ALWAYS_DETECT)

    results = []
    buffer_images = []
    buffer_indices = []

    for idx, row in df.iterrows():
        img_path = row[image_col]
        img_np = load_image(img_path, http_url=http_url)

        if img_np is not None:
            buffer_images.append(img_np)
            buffer_indices.append(idx)
        else:
            results.append(("[]", "[]", "0,0,0,0", "", 0.0))

        if len(buffer_images) == batch_size or idx == len(df) - 1:
            batch_results = process_batch(session, buffer_images)
            results.extend(batch_results)
            buffer_images.clear()
            buffer_indices.clear()

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(df)}")


    df["face_boxes"] = [r[0] for r in results]
    df["max_face_box"] = [r[1] for r in results]
    df["max_area_ratio"] = [r[2] for r in results]
    df["max_gender"] = [r[3] for r in results]
    df["max_race"] = [r[4] for r in results]
    df["max_age_bracket"] = [r[5] for r in results]



    df.to_csv(output_file, sep="\t", index=False, quoting=3, header=False)
    print(f"[Done] Saved results to: {output_file}")

if __name__ == "__main__":
    fire.Fire(infer_inspireface_batch)
