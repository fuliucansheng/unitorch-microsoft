# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import io
import base64
import zipfile
import logging
import hashlib
import numpy as np
import pandas as pd
from PIL import Image


def get_image_hash(image):
    m = hashlib.md5()
    m.update(str(image).encode("utf-8"))
    return m.hexdigest()


positive_zip = zipfile.ZipFile("./positiveCat.zip")
negative_zip = zipfile.ZipFile("./negative.zip")

output_folder = "./sensitive_dataset"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
output_zipfile = zipfile.ZipFile(
    os.path.join(output_folder, "./zip_images.zip"),
    mode="w",
)
output_train_file = os.path.join(output_folder, "./train.tsv")
output_test_file = os.path.join(output_folder, "./test.tsv")

dataset = []

for name in positive_zip.namelist():
    if not name.endswith(".png"):
        continue
    image_bytes = positive_zip.read(name)
    image_hash = get_image_hash(image_bytes)
    image_class = name.split("/")[-2]
    output_zipfile.writestr(image_hash, image_bytes)
    dataset.append([image_hash, image_class])

for name in negative_zip.namelist():
    if not name.endswith(".png"):
        continue
    image_bytes = negative_zip.read(name)
    image_hash = get_image_hash(image_bytes)
    image_class = "Other"
    output_zipfile.writestr(image_hash, image_bytes)
    dataset.append([image_hash, image_class])

dataset = pd.DataFrame(dataset, columns=["name", "label"])

train = dataset.sample(frac=0.9)
test = dataset[~dataset.isin(train)]
test = test[~test.name.isna()]

train.to_csv(output_train_file, index=False, header=None, sep="\t", quoting=3)
test.to_csv(output_test_file, index=False, header=None, sep="\t", quoting=3)
