# Copyright (c) MICROSOFT.
# Licensed under the MIT License.
import os
import json
import re
import requests
import logging
import time
import pandas as pd
from unitorch.cli import register_script
from unitorch.cli import GenericScript
from unitorch.cli.core import CoreConfigureParser


@register_script("microsoft/msan/pa_title/chatgpt")
class AzureChatGPTScript(GenericScript):
    def __init__(self, config: CoreConfigureParser):
        self.config = config
        self.config.set_default_section("microsoft/msan/pa_title/chatgpt")
        self.prompt = """<|im_start|>system
    You are an AI assistant that helps people find information.
    <|im_end|>
    <|im_start|>user
    I want to promote my product in a news feed. Please generate compelling and engaging titles with a maximum of 48 characters. Make sure it's accurate to the product. Consider titles from different product highlighting perspectives and output the five best titles for choice. Don't show other options or reason. Use the following format:
        #1. <title 1> 
        #2. <title 2>
        #3. <title 3>
        #4. <title 4>
        #5. <title 5>
    Use following product:
        `
        Product Title: {Title}
        Product Url: {ProductUrl}
        Product Brand:  {Brand}
        `
    <|im_end|>
    <|im_start|>assistant
        """
        self.api_key = self.config.getoption("key", None)
        self.api_endpoint = self.config.getoption("endpoint", None)
        self.api_deploy_name = self.config.getoption("deploy_name", None)
        self.max_tokens = self.config.getoption("max_tokens", 500)
        self.max_chars = self.config.getoption("max_chars", 64)
        self.temperature = self.config.getoption("temperature", 1.0)
        self.top_p = self.config.getoption("top_p", 1.0)
        self.retry_times = self.config.getoption("retry_times", 3)
        self.payload = {
            "prompt": self.prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "n": 1,
            "stream": False,
            "stop": "<|im_end|>",
        }
        self.api_url = (
            self.api_endpoint
            + "/openai/deployments/"
            + self.api_deploy_name
            + "/completions?api-version=2023-05-15"
        )

    def _request(self, title, url, brand):
        payload = self.payload.copy()
        payload["prompt"] = self.prompt.format(Title=title, ProductUrl=url, Brand=brand)
        result = ""
        for _ in range(self.retry_times):
            try:
                response = requests.post(
                    self.api_url,
                    headers={
                        "api-key": self.api_key,
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=30,
                )
                response.raise_for_status()
                response = json.loads(response.text)
                text = response["choices"][0]["text"].strip()
                text = text.replace("#N#", "\n")
                title1 = re.findall("1\..*\n", text)
                title2 = re.findall("2\..*\n", text)
                title3 = re.findall("3\..*\n", text)
                title4 = re.findall("4\..*\n", text)
                title5 = re.findall("5\..*\n", text)
                titles = title1 + title2 + title3 + title4 + title5
                titles = sorted(titles, key=len)
                for title in titles:
                    title = re.sub("[1,2,3,4,5]\.", "", title).strip(" \n")
                    if len(title) <= self.max_chars:
                        result = title
            except Exception as e:
                print(e)
                continue

            if result != "":
                break
            else:
                time.sleep(2)

        return result

    def launch(self):
        self.config.set_default_section("microsoft/msan/pa_title/chatgpt")
        data_file = self.config.getoption("data_file", None)
        names = self.config.getoption("names", None)
        chunksize = self.config.getoption("chunksize", 5)
        freq = self.config.getoption("freq", 1)
        title = self.config.getoption("title", "title")
        url = self.config.getoption("url", "url")
        brand = self.config.getoption("brand", "brand")
        if data_file is None:
            raise ValueError("Please specify the data file path.")

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

        output_file = self.config.getoption("output_file", "./output.tsv")
        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
        logging.info(f"Start processing data file {data_file}.")
        for i, _data in enumerate(data):
            _data.fillna("", inplace=True)
            _data["result"] = _data.apply(
                lambda x: self._request(x[title], x[url], x[brand]),
                axis=1,
            )
            _data.to_csv(
                output_file,
                sep="\t",
                index=False,
                header=None,
                mode="a",
                quoting=3,
            )
            if (i + 1) % freq == 0:
                logging.info(f"Processed {i + 1} chunks finish.")
        logging.info(f"Finish processing data file {data_file}.")
