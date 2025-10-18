# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import re
import json
import fire
import logging
import pandas as pd
import hashlib
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torch.hub import download_url_to_file
from unitorch_microsoft.externals.papyrus import get_gpt4_response


def save_image_from_url(folder, url):
    name = hashlib.md5(url.encode()).hexdigest() + ".jpg"
    path = f"{folder}/{name}"
    try:
        download_url_to_file(url, path, progress=False)
        return path
    except:
        return None


def covert(
    data_file: str,
    cache_dir: str,
    names: Union[str, List[str]] = None,
):
    if isinstance(names, str) and names.strip() == "*":
        names = None
    if isinstance(names, str):
        names = re.split(r"[,;]", names)
        names = [n.strip() for n in names]

    data = pd.read_csv(
        data_file,
        names=names,
        sep=",",
        quoting=3,
        header=None if names is not None else "infer",
    )

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    data.to_csv(
        os.path.join(cache_dir, "converted_data.tsv"),
        index=False,
        sep="\t",
        quoting=3,
    )


def preprocess(
    data_file: str,
    cache_dir: str,
    image_folder: Optional[str] = None,
    names: Union[str, List[str]] = None,
    images: Optional[Union[str, List[str]]] = None,
    keep_columns: Optional[Union[str, List[str]]] = None,
    url_prefix: Optional[str] = None,
):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

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
        header=None if names is not None else "infer",
    )
    if image_folder is not None and not os.path.exists(image_folder):
        os.makedirs(image_folder)

    all_images = []
    if isinstance(images, str):
        images = re.split(r"[,;]", images)
        images = [i.strip() for i in images]
    for image in images or []:
        if image not in data.columns:
            raise ValueError(f"Image column '{image}' not found in the data file.")
        all_images += data[image].tolist()
    all_images = set(all_images)
    all_images = {
        im: save_image_from_url(image_folder, im) for im in all_images if im is not None
    }
    all_images = {k: v for k, v in all_images.items() if v is not None}

    for image in images or []:
        data[f"{image}_local"] = data[image].apply(lambda x: all_images.get(x, None))
        if url_prefix is not None:
            data[f"{image}_url"] = data[f"{image}_local"].apply(
                lambda x: f"{url_prefix}/{os.path.basename(x)}" if x else None
            )
        data = data[data[f"{image}_local"].notna()]
    print(
        f"Preprocessed data with {len(data)} rows and {len(data.columns)} columns. Header: {data.columns.tolist()}"
    )
    data.to_csv(
        os.path.join(cache_dir, "processed_data.tsv"),
        index=False,
        sep="\t",
        quoting=3,
    )

    json.dump(
        all_images,
        open(os.path.join(cache_dir, "images.json"), "w"),
        indent=2,
    )

    if keep_columns is not None:
        if isinstance(keep_columns, str):
            keep_columns = re.split(r"[,;]", keep_columns)
            keep_columns = [c.strip() for c in keep_columns]
        more_columns = []
        for col in keep_columns:
            if f"{col}_local" in data.columns:
                more_columns.append(f"{col}_local")
            if f"{col}_url" in data.columns:
                more_columns.append(f"{col}_url")
        keep_columns = keep_columns + more_columns
        data = data[keep_columns]
        print(f"Filtered data to {len(data.columns)} columns: {data.columns.tolist()}")
        data.to_csv(
            os.path.join(cache_dir, "filtered_data.tsv"),
            index=False,
            sep="\t",
            quoting=3,
        )


def gpt(
    data_file: str,
    cache_dir: str,
    title_col: str,
    image1_col: str,
    image2_col: str,
    names: Union[str, List[str]] = None,
    model: str = "gpt-41-2025-04-14-Eval",
):
    prompt = """
    In this image, the product title is '{adTitle}'

    The first image is the cropped version, and the second image is the original.  Please decide whether the first image is a good crop or bad crop based on the following criteria:
    
    - Product Cut: 
    Product Bad crop: The product in the product title is cut off or incomplete. As long as the product gets cut off from the original image, even the product is recognizable, it still counts as bad crop. Unless it meets the following exceptions 
    Product good crop: If this product is incomplete in both the original and displayed image, it doesn't count as 'cut off', consider the image good product crop.
    Product Good crop: An exception is when the product has a repeating pattern. For example, a net or a pile of rocks. 
    Product Good crop: If the product doesn't appear in the image, or the product is a virtual concept(like service, application). Always consider as product good crop
    Product Good crop: if the product is in the background of the image, it should not be considered as factors to decide good or bad crop.
    
    - Face Cut:
    Face bad crop: Eyes, nose, mouth, ears, eyebrows are in original image, but get fully or partially cut off in cropped images. Consider all the faces in the image including the blurry faces. Unless it meet the following exceptions:
    Face good cut: If the human face is far in the background,
    Face good cut: Only the 5 features listed in 'Face bad crop' are crucial, as long these features are not cut off from original image, consider the image good crop. If this features are not in neither the original or displayed image, it doesn't count as 'cut off', consider the image good face crop. Even if the other facial features like hair, chin, forehead or hands/arms are cut off in the cropped image,  still consider the image face good crop. 
    
    - Text Cut: 
    Text bad crop: Text is cut off in the middle, resulting in incomplete words or sentences. 
    Text good crop: As long as the characters are fully visible, it is acceptable if the text becomes blurry or if the border of the text is cut off. However, if the text is unreadable or too blurry, this criterion does not apply and consider it as good text cut.
    
    If one of the category fall into bad crop, consider the image as bad crop(Y). If all the criterion are good or do not apply, consider the image good crop (N)
    
    Please decide whether the first image is a blurry or clear image based on following criterion. Do not zoom in or zoom out the image. Evaluate based on the provided image size, which is the size the end users will see on their monitors.
    - Product Recognition clear: Users can tell what the product is when see the image. If the product is not on the original image, the users should at least be able to tell the topic of the image. Otherwise, the display image is blurry
    - Detail preservation clear: Text, patterns, and objects should have smooth, well-defined edges without noticeable jaggedness, blurry or ghosting. Otherwise, the display image is blurry
    - Text readability: Following info count as Key information: Product name; Brand Name/Text in Logo; Key Selling Point: Highlights the ad's core message and shapes first impressions (e.g., long-lasting hydration) ; Price, Discount and Promotional Information: Impacts purchasing decisions; Call-to-Action(CTA): Encourages user actions (e.g., “Buy Now”, “Learn More”). If key information text is overlaid separately on the image, the largest text must be clear and readable. For example, if both the product name and key selling point appear on the image, and the product name is large and clear, while the key selling point is smaller and slightly blurry, the image should still be considered clear. If the key information text is printed on the product and the product is clear, the image is still considered clear even if the text is blurry or unreadable. If the text is too small or blurry to determine its importance, use other criteria to decide if the image is blurry.
    Violating any of the three criterion: Product recognition, detail preservation and text readability, makes the image blurry.
    
    N for good crop, Y for bad crop. Organize the answer with following structure:  <ans1>Blurry or Clear</ans1><reasoning1></reasoning1> <ans2>N or Y</ans2><reasoning2></reasoning2>
    """
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

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
        header=None if names is not None else "infer",
    )
    if title_col not in data.columns:
        raise ValueError(f"Title column '{title_col}' not found in the data file.")
    if image1_col not in data.columns:
        raise ValueError(f"Image1 column '{image1_col}' not found in the data file.")
    if image2_col not in data.columns:
        raise ValueError(f"Image2 column '{image2_col}' not found in the data file.")

    data = data[[title_col, image1_col, image2_col]].drop_duplicates()
    output_file = os.path.join(cache_dir, "gpt_results.jsonl")
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            results = [json.loads(line) for line in f]
        results = pd.DataFrame(results)
        existed = (
            set(
                zip(
                    results["title"],
                    results["image1"],
                    results["image2"],
                )
            )
            if results.shape[0] > 0
            else set()
        )
        data = data[
            ~data.apply(
                lambda row: (row[title_col], row[image1_col], row[image2_col])
                in existed,
                axis=1,
            )
        ]

    with open(output_file, "a") as f:
        for _, row in data.iterrows():
            title = row[title_col]
            image1 = Image.open(row[image1_col])
            image2 = Image.open(row[image2_col])
            prompt_text = prompt.format(adTitle=title)
            result = get_gpt4_response(
                prompt=prompt_text,
                images=[image1, image2],
                model=model,
                max_tokens=4096,
            )
            if result is None:
                print(
                    f"Failed to get response for title: {title}, image1: {image1}, image2: {image2}"
                )
                continue
            record = {
                "title": title,
                "image1": row[image1_col],
                "image2": row[image2_col],
                "result": result,
            }
            f.write(json.dumps(record) + "\n")
            f.flush()


def model(
    data_file: str,
    cache_dir: str,
    image_col: str,
    names: Union[str, List[str]] = None,
):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

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
        header=None if names is not None else "infer",
    )
    if image_col not in data.columns:
        raise ValueError(f"Image column '{image_col}' not found in the data file.")

    data = data[[image_col]].drop_duplicates()
    data.to_csv(
        os.path.join(cache_dir, "processed_model_images.tsv"),
        index=False,
        sep="\t",
        header=None,
        quoting=3,
    )
    cmd = f"unitorch-infer picasso/configs/bletchley.bad_crop.ini --test_file {cache_dir}/processed_model_images.tsv --core/task/supervised@output_path {cache_dir}/model_results.tsv --core/process/image@http_url None"
    os.system(cmd)


def postprocess(
    data_file: str,
    cache_dir: str,
    label_col: str,
    title_col: str = None,
    image1_col: str = None,
    image2_col: str = None,
    image_col: str = None,
    names: Union[str, List[str]] = None,
    threshold: float = 0.3,
):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

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
        header=None if names is not None else "infer",
    )

    def check(y_true, y_pred):
        ds = pd.DataFrame(
            {
                "y_true": y_true,
                "y_pred": y_pred,
            }
        )
        acc = (ds["y_true"] == ds["y_pred"]).mean()
        yes = ds[ds["y_true"] == 1]
        no = ds[ds["y_true"] == 0]
        y_acc = (yes["y_pred"] == 1).mean()
        n_acc = (no["y_pred"] == 0).mean()
        b_acc = (y_acc + n_acc) / 2.0
        p_yes = ds[ds["y_pred"] == 1]
        pre = (p_yes["y_true"] == 1).mean()
        rec = (yes["y_pred"] == 1).mean()
        logging.info(
            f"Accuracy: {acc:.4f}, Yes Accuracy: {y_acc:.4f}, No Accuracy: {n_acc:.4f}, Balanced Accuracy: {b_acc:.4f}, Precision: {pre:.4f}, Recall: {rec:.4f}"
        )

    if os.path.exists(os.path.join(cache_dir, "gpt_results.jsonl")):
        gpt_result = pd.read_csv(
            os.path.join(cache_dir, "gpt_results.jsonl"),
            sep="\t",
            names=["jsonl"],
            quoting=3,
            header=None,
        )
        gpt_result["jsonl"] = gpt_result["jsonl"].apply(lambda x: json.loads(x))
        result_dict = {
            f"{row['title']}_{row['image1']}_{row['image2']}": row["result"]
            for _, row in gpt_result["jsonl"].iteritems()
        }
        data["gpt_result"] = data.apply(
            lambda row: result_dict.get(
                f"{row[title_col]}_{row[image1_col]}_{row[image2_col]}", None
            ),
            axis=1,
        )
        data = data[data["gpt_result"].notna()]
        print(data.shape)
        p1 = re.compile(r"<ans1>(.*?)</ans1>")
        p2 = re.compile(r"<ans2>(.*?)</ans2>")
        data["gpt_result_ans1"] = data["gpt_result"].apply(
            lambda x: p1.search(x).group(1) if p1.search(x) else None
        )
        data["gpt_result_ans2"] = data["gpt_result"].apply(
            lambda x: p2.search(x).group(1) if p2.search(x) else None
        )
        data = data[data["gpt_result_ans2"].notna()]
        check(
            data[label_col].apply(lambda x: 1 if x == "Yes" else 0),
            data["gpt_result_ans2"].apply(
                lambda x: 1 if x.lower().strip() == "y" else 0
            ),
        )

    if os.path.exists(os.path.join(cache_dir, "model_results.tsv")):
        model_result = pd.read_csv(
            os.path.join(cache_dir, "model_results.tsv"),
            sep="\t",
            names=["image", "score"],
            quoting=3,
            header=None,
        )
        model_result["score"] = model_result["score"].astype(float)
        result_dict = {row["image"]: row["score"] for _, row in model_result.iterrows()}
        data["model_score"] = data[image_col].apply(lambda x: result_dict.get(x, None))
        check(
            data[label_col].apply(lambda x: 1 if x == "Yes" else 0),
            data["model_score"].apply(lambda x: 1 if x >= threshold else 0),
        )


def process_all(
    data_file: str,
    cache_dir: str,
    title_col: str = None,
    image1_col: str = None,
    image2_col: str = None,
    image_col: str = None,
    label_col: str = None,
    names: Union[str, List[str]] = None,
    image_folder: str = None,
    images: Optional[Union[str, List[str]]] = None,
    keep_columns: Optional[Union[str, List[str]]] = None,
    url_prefix: Optional[str] = None,
    gpt_model: str = "gpt-41-2025-04-14-Eval",
    threshold: float = 0.3,
):
    if not os.path.exists(os.path.join(cache_dir, "converted_data.tsv")):
        covert(data_file, cache_dir, names)

    if not os.path.exists(os.path.join(cache_dir, "processed_data.tsv")):
        preprocess(
            os.path.join(cache_dir, "converted_data.tsv"),
            cache_dir,
            image_folder=image_folder,
            images=images,
            keep_columns=keep_columns,
            url_prefix=url_prefix,
        )

    # gpt(
    #     os.path.join(cache_dir, "processed_data.tsv"),
    #     cache_dir,
    #     names,
    #     title_col,
    #     image1_col,
    #     image2_col,
    #     model=gpt_model,
    # )
    model(
        os.path.join(cache_dir, "processed_data.tsv"),
        cache_dir,
        image_col,
    )
    # postprocess(
    #     os.path.join(cache_dir, "processed_data.tsv"),
    #     cache_dir,
    #     label_col,
    #     title_col,
    #     image1_col,
    #     image2_col,
    #     image_col,
    #     threshold=threshold,
    # )


if __name__ == "__main__":
    fire.Fire()
