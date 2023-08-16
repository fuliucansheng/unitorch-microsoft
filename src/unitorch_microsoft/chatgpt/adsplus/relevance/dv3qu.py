# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import io
import json
import base64
import zipfile
import logging
import hashlib
import numpy as np
import pandas as pd
from unitorch.cli import CoreConfigureParser, GenericScript
from unitorch.cli import register_script

prompt = """
We are currently optimizing a commercial search engine advertising system. One of our focus is on relevance tasks, which involve understanding the relationship between a given query and the ad text that users see alongside the search results. We also aim to measure the relevance between the query and the ad landing page, which is the web page that users are directed to after clicking on the ad. Your task is to identify the underlying intents of a user who typed a specific query. These intents will then be used for the aforementioned tasks.\n\n#Instruction\nPlease provide the following information, enclosed within the designated tags, based on the given query:\n1. What product, service, information, etc. is the user looking for?\n2. What's mentioned brand, retailer name, etc. in the query?\n3. What category does the product belong to?\n4. What are the other products within the same category?\n5. Can you suggest similar products, services, information, accessories, etc. that might be useful to the user?\n6. Can you suggest any related products, services, information, accessories, etc. that might be useful to the user?\n7. Can you suggest any other products, brands, or information that the user might be interested in, based on the query they searched?\nGiven Query {query}, provide above items between tags <AdQueryIntentMG> and </AdQueryIntentMG>:
"""


@register_script("microsoft/script/chatgpt/adsplus/relevance/dv3qu")
class DV3QUScript(GenericScript):
    def __init__(self, config: CoreConfigureParser):
        self.config = config

    def run(self, **kwargs):
        config = self.config
        config.set_default_section("microsoft/script/chatgpt/adsplus/relevance/dv3qu")
        data_file = config.getoption("data_file", None)
        assert data_file is not None

        names = config.getoption("names", None)
        chunksize = config.getoption("chunksize", 1000)
        freq = config.getoption("freq", 20)
        query_col = config.getoption("query_col", "query")
        row_col = config.getoption("row_col", "rowid")

        if names is None:
            data = pd.read_csv(
                data_file,
                names=names,
                sep="\t",
                chunksize=chunksize,
                quoting=3,
                header="infer",
                iterator=True,
                low_memory=True,
            )
        else:
            data = pd.read_csv(
                data_file,
                names=names,
                sep="\t",
                chunksize=chunksize,
                quoting=3,
                header=None,
                iterator=True,
                low_memory=True,
            )

        def processing(row):
            res = {
                "prompt": prompt.format(query=row[query_col]),
                "max_tokens": 200,
                "temperature": 0,
                "top_p": 1,
                "_batch_request_metadata": {
                    "ConversationId": str(row[row_col]),
                },
            }
            return json.dumps(res)

        output_file = config.getoption("output_file", "./output.txt")
        for i, _data in enumerate(data):
            assert query_col in _data.columns and row_col in _data.columns
            _data.fillna("", inplace=True)
            _data["jsonl"] = _data.apply(processing, axis=1)

            _data["jsonl"].to_csv(
                output_file,
                sep="\t",
                index=False,
                header=None,
                mode="a",
                quoting=3,
            )
            logging.info(f"partition {i} processed finish.")
