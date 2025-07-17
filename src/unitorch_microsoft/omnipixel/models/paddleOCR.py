import cv2
import pandas as pd
import fire
import re
import requests
import json
from typing import List, Union
from PIL import Image
from io import BytesIO
import numpy as np
from paddleocr import PaddleOCR


def load_image(image: str, http_url: str) -> Union[np.ndarray, None]:
    try:
        if http_url:
            url = http_url.format(image)
            doc = requests.get(url, timeout=30)
            image = Image.open(BytesIO(doc.content)).convert("RGB")
        else:
            image = Image.open(image).convert("RGB")
        return np.array(image)
    except Exception as e:
        print(f"[Warning] Cannot load image {image}: {e}")
        return None

def process_ocr_batch(images: List[np.ndarray], ocr_model, paths: List[str], min_score: float = 0.8) -> List[dict]:
    results = []
    for img, path in zip(images, paths):
        try:
            ocr_result = ocr_model.predict(img)
            entries = []
            for res in ocr_result:
                texts = res['rec_texts']        # 识别出的文字
                scores = res['rec_scores']      # 识别置信度
                rec_polys = res['rec_polys']         # 文本区域的多边形坐标

                for text, score, polys in zip(texts, scores, rec_polys):
                    if score >= min_score:
                        entries.append({"text": text, "score": round(score, 4), "box": polys.tolist() })
                
            results.append({"image": path, "ocr_results": entries})
        except Exception as e:
            print(f"[Warning] OCR failed on image {path}: {e}")
            results.append({"image": path, "ocr_results": []})
    return results

def write_results(results: List[dict], output_file: str, output_format: str):
    with open(output_file, "w", encoding="utf-8") as f:
        if output_format == "json":
            json.dump(results, f, ensure_ascii=False, indent=2)

        elif output_format == "tsv_json":
            f.write("image_path\tocr_results\n")
            for res in results:
                f.write(f"{res['image']}\t{json.dumps(res['ocr_results'], ensure_ascii=False)}\n")

        elif output_format == "tsv_long":
            f.write("image_path\ttext\tscore\tbox\n")
            for res in results:
                for r in res["ocr_results"]:
                    f.write(f"{res['image']}\t{r['text']}\t{r['score']}\t{r['box']}\n")
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

def infer_ocr_text_batch(
    data_file: str,
    output_file: str,
    names: Union[str, List[str]],
    image_col: str = "image_path",
    batch_size: int = 32,
    http_url: str = None,
    output_format: str = "tsv_long",  # "tsv_long", "tsv_json", or "json"
    min_score: float = 0.8,
):
    if isinstance(names, str) and names.strip() == "*":
        names = None
    if isinstance(names, str):
        names = re.split(r"[,;]", names)
        names = [n.strip() for n in names]

    df = pd.read_csv(data_file, sep="\t", quoting=3, names=names, header=None)
    assert image_col in df.columns, f"{image_col} not found in input"

    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False
    )

    buffer_images = []
    buffer_paths = []
    all_results = []

    for idx, row in df.iterrows():
        img_path = row[image_col]
        img = load_image(img_path, http_url)

        if img is not None:
            buffer_images.append(img)
            buffer_paths.append(img_path)

        if len(buffer_images) == batch_size or idx == len(df) - 1:
            if buffer_images:
                batch_results = process_ocr_batch(buffer_images, ocr, buffer_paths, min_score)
                all_results.extend(batch_results)
                buffer_images.clear()
                buffer_paths.clear()

            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(df)}")

    write_results(all_results, output_file, output_format)
    print(f"[✓] Done. Results saved to: {output_file}")

if __name__ == "__main__":
    fire.Fire(infer_ocr_text_batch)
