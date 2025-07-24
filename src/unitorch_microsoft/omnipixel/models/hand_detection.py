import cv2
import mediapipe as mp
import pandas as pd
import fire
import re
import requests
from typing import List, Union
from PIL import Image
from io import BytesIO
import numpy as np

def load_image(image: str, http_url: str) -> Union[np.ndarray, None]:
    try:
        if http_url:
            url = http_url.format(image)
            doc = requests.get(url, timeout=30)
            image = Image.open(BytesIO(doc.content)).convert("RGB")
        else:
            image = Image.open(image).convert("RGB")
        return np.array(image)[:, :, ::-1]  # RGB to BGR for OpenCV
    except Exception as e:
        print(f"[Warning] Cannot load image {image}: {e}")
        return None

def process_batch(images: List[np.ndarray], detector) -> List[int]:
    counts = []

    for img in images:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = detector.process(img_rgb)

        if results.multi_hand_landmarks:
            counts.append(len(results.multi_hand_landmarks))
        else:
            counts.append(0)

    return counts

def infer_hand_count_batch(
    data_file: str,
    output_file: str,
    names: Union[str, List[str]],
    image_col: str = "image_path",
    batch_size: int = 64,
    http_url: str = "http://0.0.0.0:11230/?file={0}",
):
    if isinstance(names, str) and names.strip() == "*":
        names = None
    if isinstance(names, str):
        names = re.split(r"[,;]", names)
        names = [n.strip() for n in names]

    df = pd.read_csv(data_file, sep="\t", quoting=3, names=names, header=None)
    assert image_col in df.columns, f"{image_col} not found in input"

    mp_hands = mp.solutions.hands
    detector = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=10,
        min_detection_confidence=0.1
    )

    hand_counts = []
    buffer_images = []
    buffer_indices = []

    for idx, row in df.iterrows():
        img_path = row[image_col]
        img = load_image(img_path, http_url)

        if img is not None:
            buffer_images.append(img)
            buffer_indices.append(idx)
        else:
            hand_counts.append(0)

        if len(buffer_images) == batch_size or idx == len(df) - 1:
            if buffer_images:
                batch_counts = process_batch(buffer_images, detector)
                hand_counts.extend(batch_counts)
                buffer_images.clear()
                buffer_indices.clear()

            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(df)}")

    df["hand_count"] = hand_counts
    df.to_csv(output_file, sep="\t", index=False, quoting=3, header=False)
    print(f"[✓] Done. Results saved to: {output_file}")

if __name__ == "__main__":
    fire.Fire(infer_hand_count_batch)
