# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import re
import requests
import json
import time
import logging
import tempfile
import asyncio
import aiohttp
import urllib
import urllib.parse
import pandas as pd
from multiprocessing import Process, Queue
from unitorch.cli import CoreConfigureParser, GenericScript
from unitorch.cli import register_script
from unitorch_microsoft import cached_path

try:
    from lxml import etree
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.options import Options
except ImportError:
    raise ImportError(
        "lxml and selenium are required for this script. "
        "Please install them using `pip install lxml selenium`."
    )


@register_script("microsoft/script/china/slab/crawler")
class SLABCrawler(GenericScript):
    def __init__(self, config: CoreConfigureParser):
        self.config = config

    def launch(self, **kwargs):
        config = self.config
        config.set_default_section("microsoft/script/china/slab/crawler")
        data_file = config.getoption("data_file", None)
        use_google_chrome = config.getoption("use_google_chrome", True)
        names = config.getoption("names", None)
        url_col = config.getoption("url_col", None)
        num_processes = config.getoption("num_processes", 5)
        output_file = config.getoption("output_file", "./output.jsonl")
        result_file = config.getoption("result_file", "./output.tsv")

        assert data_file is not None and os.path.exists(data_file)

        if isinstance(names, str) and names.strip() == "*":
            names = None

        if isinstance(names, str):
            names = re.split(r"[,;]", names)
            names = [n.strip() for n in names]

        dataset = pd.read_csv(
            data_file,
            names=names,
            header="infer" if names is None else None,
            sep="\t",
            quoting=3,
        )

        assert url_col is not None and url_col in dataset.columns

        results = pd.DataFrame()
        results["url"] = dataset[url_col].drop_duplicates()

        logging.info(f"start rendering {len(results)} pages")

        def render_htmls(part, Q):
            chrome_options = Options()
            chrome_options.add_argument("--disable-extensions")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--no-sandbox")  # linux only
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--headless=new")
            chrome_options.add_argument("start-maximized")
            if use_google_chrome:
                # wget -q https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
                # dpkg -i ./google-chrome-stable_current_amd64.deb
                # apt --fix-broken install
                chrome_options.binary_location = "/usr/bin/google-chrome"
            driver = webdriver.Chrome(options=chrome_options)
            driver.set_page_load_timeout(90)

            for _, row in part.iterrows():
                url = row["url"]
                for _ in range(3):
                    try:
                        driver.get(url)
                        break
                    except:
                        driver.quit()
                        time.sleep(10)
                        driver = webdriver.Chrome(options=chrome_options)
                        driver.set_page_load_timeout(90)
                        # if "ERR_CONNECTION_REFUSED" in driver.page_source:
                        #     logging.info(f"ERR_CONNECTION_REFUSED {url}")
                        #     driver.quit()
                        #     time.sleep(10)
                        #     driver = webdriver.Chrome(options=chrome_options)
                        #     driver.set_page_load_timeout(90)

                try:
                    lptitle = driver.title
                    links = driver.find_elements(By.TAG_NAME, "a")
                    suburls = [link.get_attribute("href") for link in links]
                    suburls = [url for url in suburls if url is not None]
                    Q.put({"url": url, "lptitle": lptitle, "suburls": suburls})
                except:
                    logging.info(f"ERR {url}")
                time.sleep(1)

            try:
                driver.quit()
            except:
                pass
            Q.put("Done")

        def write_file(fpath, Q, cnt):
            f = open(fpath, "w")
            done = 0
            while True:
                item = Q.get()
                if item == "Done":
                    done += 1
                    if done == cnt:
                        break
                else:
                    f.write(json.dumps(item) + "\n")

        # partition the data into num_processes parts
        data_parts = []
        for i in range(num_processes):
            data_parts.append(results.iloc[i::num_processes])

        # start the rendering processes
        processes = []
        queue = Queue()
        for i in range(num_processes):
            p = Process(
                target=render_htmls,
                args=(
                    data_parts[i],
                    queue,
                ),
            )
            processes.append(p)

        jsonl_file = tempfile.NamedTemporaryFile(suffix=".jsonl").name
        io_process = Process(target=write_file, args=(jsonl_file, queue, num_processes))
        processes.append(io_process)

        for p in processes:
            p.start()

        # wait for all processes to finish
        for p in processes:
            p.join()

        # merge the results
        results = pd.read_csv(jsonl_file, names=["result"], sep="\t")
        results["obj"] = results.result.map(json.loads)
        results["url"] = results.obj.map(lambda x: x["url"])
        results["lptitle"] = results.obj.map(lambda x: x["lptitle"])
        results["suburls"] = results.obj.map(lambda x: x["suburls"])
        results.drop(columns=["result", "obj"], inplace=True)

        results.to_json(output_file, orient="records", lines=True)

        logging.info(
            f"finish rendering {len(results)} pages, {len(results[results.lptitle != ''])} success. "
        )
        results.to_csv(result_file, sep="\t", index=False, header=False)
