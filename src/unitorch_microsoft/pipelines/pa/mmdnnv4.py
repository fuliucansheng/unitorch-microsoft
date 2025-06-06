# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import zipfile
import io
import time
import os
import json
import random
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import torchvision
import pandas as pd
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torch import autocast
from transformers.activations import quick_gelu
from unitorch.models import GenericModel
from unitorch.cli import (
    cached_path,
    hf_endpoint_url,
    add_default_section_for_init,
    add_default_section_for_function,
)
from unitorch.cli import CoreConfigureParser, GenericScript
from unitorch.cli import register_script
from unitorch_microsoft.models.bletchley.modeling_v1 import (
    get_bletchley_image_config,
    BletchleyImageEncoder,
)
from PIL import Image
from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize, Compose
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import asyncio


class MMDNNv4OfferModel(GenericModel):
    def __init__(self, pretrained_weight_path):
        super().__init__()
        projection_dim = 288
        hidden_dim = 32
        output_hidden_dim = 64

        num_seller = 15020
        num_brand = 1000001

        image_config = get_bletchley_image_config("0.3B", gradient_checkpointing=False)
        self.image_embed_dim = image_config.hidden_size

        self.image_encoder = BletchleyImageEncoder(
            image_config,
            add_projection_layer=False,
        )
        self.image_projection = nn.Linear(
            self.image_embed_dim,
            projection_dim,
            bias=False,
        )
        self.image_layer_norm = nn.LayerNorm(projection_dim)

        self.seller_embedding = nn.Embedding(num_seller, hidden_dim)
        self.seller_layer_norm = nn.LayerNorm(hidden_dim)
        self.brand_embedding = nn.Embedding(num_brand, hidden_dim)
        self.brand_layer_norm = nn.LayerNorm(hidden_dim)

        self.final_visual_projection = nn.Linear(
            projection_dim + hidden_dim * 2,
            output_hidden_dim,
        )

        self.from_pretrained(pretrained_weight_path)

    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(
        self,
        images=None,
        seller_ids=None,
        brand_ids=None,
    ):
        image_outputs = self.image_encoder(
            images=images,
        )
        image_embeds = image_outputs[:, 0]
        image_embeds = self.image_projection(image_embeds)
        image_embeds = self.image_layer_norm(quick_gelu(image_embeds))

        outputs = []
        bsz = image_embeds.size(0)
        for i in range(bsz):
            seller_embeds = self.seller_embedding(seller_ids[i])
            brand_embeds = self.brand_embedding(brand_ids[i])
            seller_embeds = self.seller_layer_norm(seller_embeds)
            brand_embeds = self.brand_layer_norm(brand_embeds)

            sz = seller_embeds.size(0)

            new_embeds = torch.cat(
                [
                    image_embeds[i].unsqueeze(0).repeat(sz, 1),
                    seller_embeds,
                    brand_embeds,
                ],
                dim=-1,
            )
            new_embeds = self.final_visual_projection(quick_gelu(new_embeds))
            new_embeds = new_embeds / new_embeds.norm(dim=-1, keepdim=True)
            outputs.append(new_embeds)

        assert len(outputs) == bsz
        return outputs


@register_script("microsoft/pipeline/pa/mmdnnv4/offer")
class MMDNNv4ForChunkInferencePipeline(GenericScript):
    def __init__(self, config: CoreConfigureParser):
        self.config = config

    def launch(self, **kwargs):
        config = self.config
        config.set_default_section("microsoft/pipeline/pa/mmdnnv4/offer")

        input_file = config.getoption("input_file", None)
        names = config.getoption("names", "*")
        zip_folder = config.getoption("zip_folder", "*")
        device = config.getoption("device", "0")
        pretrained_weight_path = config.getoption("pretrained_weight_path", None)
        output_path = config.getoption("output_path", None)
        max_batch_size = config.getoption("max_batch_size", 256)

        image_transform = Compose(
            [
                Resize([224, 224]),
                CenterCrop([224, 224]),
                ToTensor(),
                Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        model = MMDNNv4OfferModel(pretrained_weight_path)
        device = "cpu" if device == "cpu" else int(device)
        model.to(device)
        model.eval()

        def load_zip_to_memory(zip_path):
            file_contents = {}

            with open(zip_path, "rb") as f:
                zip_buffer = io.BytesIO(f.read())

            with zipfile.ZipFile(zip_buffer, "r") as zip_file:
                for file_name in zip_file.namelist():
                    with zip_file.open(file_name) as file:
                        file_contents[file_name] = file.read()
            return file_contents

        def process_image(key, zip_file_contents):
            image_bytes = zip_file_contents.get(key)
            if image_bytes is not None:
                try:
                    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    return image_transform(image)
                except Exception as e:
                    print(f"处理图片 {key} 时出错: {str(e)}")
                    return image_transform(
                        Image.new("RGB", [224, 224], (255, 255, 255))
                    )

        async def async_process_image(key, zip_file_contents, executor):
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                executor, process_image, key, zip_file_contents
            )

        async def process_batch(images, zip_file_contents, executor):
            tasks = [
                async_process_image(key, zip_file_contents, executor)
                for key in images.to_list()
            ]
            return await asyncio.gather(*tasks)

        def process_num(nums, device):
            num_list = nums.str.split(",").apply(lambda x: list(map(int, x))).tolist()
            return [torch.tensor(sublist, device=device) for sublist in num_list]

        def postprocess(rowids, embeddings):
            embedding_arrays = [t.cpu().detach().numpy() for t in embeddings]
            expanded_rowids = (
                rowids.str.split(",", expand=True).stack().reset_index(drop=True)
            )

            flat_embeddings = np.vstack(
                [
                    arr.squeeze(0) if arr.shape[0] == 1 else arr
                    for arr in embedding_arrays
                ]
            )
            embedding_strings = [",".join(map(str, emb)) for emb in flat_embeddings]

            result_dict = dict(zip(expanded_rowids, embedding_strings))
            return result_dict

        def save_dict_to_tsv(data_dict, path):
            pd.DataFrame.from_dict(data_dict, orient="index").reset_index().to_csv(
                path, sep="\t", index=False, header=None
            )

        start_time = time.time()
        df = pd.read_csv(input_file, sep="\t", quoting=3, names=names)

        with ThreadPoolExecutor(max_workers=16) as executor:
            for zip_file in os.listdir(zip_folder):
                if zip_file.endswith(
                    "zip"
                ) and f"{zip_file[:-4]}_output.tsv" not in os.listdir(output_path):
                    zip_start_time = time.time()
                    zip_path = os.path.join(zip_folder, zip_file)
                    zip_file_contents = load_zip_to_memory(zip_path)
                    keys = zip_file_contents.keys()
                    result_df = df[df["image"].isin(keys)]

                    results = {}
                    for i in range(0, len(result_df["image"]), max_batch_size):
                        images = result_df["image"][i : i + max_batch_size]
                        sellers = result_df["seller"][i : i + max_batch_size]
                        brands = result_df["brand"][i : i + max_batch_size]
                        rowids = result_df["rowid"][i : i + max_batch_size]

                        sellers = process_num(sellers, device)
                        brands = process_num(brands, device)

                        images = asyncio.run(
                            process_batch(images, zip_file_contents, executor)
                        )
                        images = torch.stack(images, dim=0).to(device)

                        embeds = model(
                            images=images, seller_ids=sellers, brand_ids=brands
                        )

                        result_tmp_dict = postprocess(rowids, embeds)
                        results = {**result_tmp_dict, **results}

                        del images, sellers, brands, embeds, result_tmp_dict
                        torch.cuda.empty_cache()

                    save_dict_to_tsv(
                        results, f"{output_path}/{zip_file[:-4]}_output.tsv"
                    )
                    del zip_file_contents, results
                    zip_time = time.time() - zip_start_time
                    print(
                        f"running time for one zip: {zip_time:.4f}s {len(result_df['image'])/zip_time} samples/s"
                    )

            elapsed_time = time.time() - start_time
            print(
                f"total running time: {elapsed_time:.4f}s {len(df['image'])/elapsed_time} samples/s"
            )
