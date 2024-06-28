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

async def crawl(urls, api_token=None):
    semaphore = asyncio.Semaphore(100)

    tasks = []
    results = []

    async def fetch(url, session):
        api_url = "https://api.crawlbase.com"
        params = {
            "token": api_token,
            "url": urllib.parse.unquote(url),
        }
        async with semaphore:
            async with session.get(api_url, params=params) as response:
                if response.status != 200:
                    return "<html></html>"
                text = await response.text()
                return text
            
    async def send(urls, session):
        task = []
        for url in urls:
            _task = asyncio.create_task(fetch(url, session))
            _task.add_done_callback(lambda x: results.append(x.result()))
            task.append(_task)
        await asyncio.gather(*task)

    async with aiohttp.ClientSession() as sess:
        for i in range(0, len(urls), 20):
            chunk = urls[i:i+20]
            _task = asyncio.create_task(send(chunk, sess))
            tasks.append(_task)
            await asyncio.sleep(1)
        await asyncio.gather(*tasks)
    
    return pd.DataFrame({"url": urls, "html": results})


@register_script("microsoft/script/china/alibaba/crawler/crawlbase/1688/crawling")
class Alibaba1688Crawler(GenericScript):
    def __init__(self, config: CoreConfigureParser):
        self.config = config

    def launch(self, **kwargs):
        config = self.config
        config.set_default_section(
            "microsoft/script/china/alibaba/crawler/crawlbase/1688/crawling"
        )
        api_token = config.getoption("api_token", None)
        data_file = config.getoption("data_file", None)
        names = config.getoption("names", "*")
        url_col = config.getoption("url_col", None)
        input_escapechar = config.getoption("input_escapechar", None)
        output_file = config.getoption("output_file", "./output.jsonl")

        assert (
            api_token is not None
            and data_file is not None
            and os.path.exists(data_file)
        )

        if isinstance(names, str):
            names = re.split(r"[,;]", names)
            names = [n.strip() for n in names]

        dataset = pd.read_csv(
            data_file,
            names=names,
            sep="\t",
            escapechar=input_escapechar,
        )
        assert url_col is not None and url_col in dataset.columns

        urls = dataset[url_col].drop_duplicates().tolist()

        logging.info(f"start crawling {len(urls)} urls")
        results = asyncio.run(crawl(urls, api_token))
        results["title"] = results.html.map(
            lambda x: (etree.HTML(x).xpath("//title/text()") + [""])[0]
        )

        logging.info(
            f"finish crawling {len(urls)} urls, {len(results[results.title != ''])} success. "
        )
        results.to_json(output_file, orient="records", lines=True)


@register_script("microsoft/script/china/alibaba/crawler/crawlbase/1688/rendering")
class Alibaba1688Render(GenericScript):
    def __init__(self, config: CoreConfigureParser):
        self.config = config

    def launch(self, **kwargs):
        config = self.config
        config.set_default_section(
            "microsoft/script/china/alibaba/crawler/crawlbase/1688/rendering"
        )
        data_file = config.getoption("data_file", None)
        output_file = config.getoption("output_file", "./output.jsonl")
        result_file = config.getoption("output_file", "./output.tsv")

        assert data_file is not None and os.path.exists(data_file)

        dataset = pd.read_csv(
            data_file,
            names=["result"],
            sep="\t",
        )

        dataset["obj"] = dataset.result.map(json.loads)

        num_processes = config.getoption("num_processes", 5)
        html_dir = config.getoption("html_dir", "./html")
        web_port = config.getoption("web_port", 8000)
        os.makedirs(html_dir, exist_ok=True)

        results = pd.DataFrame()
        results["url"] = dataset.obj.map(lambda x: x["url"])
        results["html"] = dataset.obj.map(lambda x: x["html"])

        logging.info(f"start rendering {len(results)} pages")

        def render_htmls(part, Q):
            chrome_options = Options()
            chrome_options.add_argument("--disable-extensions")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--no-sandbox")  # linux only
            chrome_options.add_argument("--headless=new")
            chrome_options.add_argument("start-maximized")
            driver = webdriver.Chrome(options=chrome_options)
            driver.set_page_load_timeout(90)

            html_page = os.path.join(html_dir, f"{os.getpid()}.html")
            html_url = f"http://127.0.0.1:{web_port}/{os.path.basename(html_page)}"
            for _, row in part.iterrows():
                url, html = row["url"], row["html"]
                with open(html_page, "w") as f:
                    f.write(html)
                for _ in range(3):
                    try:
                        driver.get(html_url)
                        break
                    except:
                        if "ERR_CONNECTION_REFUSED" in driver.page_source:
                            logging.info(f"ERR_CONNECTION_REFUSED {html_url}")
                            driver.quit()
                            time.sleep(10)
                            driver = webdriver.Chrome(options=chrome_options)
                            driver.set_page_load_timeout(90)

                product_htmls = driver.find_elements(
                    By.XPATH, "//div[contains(@class, 'offer_item')]"
                )
                items = []

                def first(objs, attr):
                    for obj in objs:
                        try:
                            if obj.get_attribute(attr) is not None:
                                return obj.get_attribute(attr)
                        except:
                            pass
                    return ""

                for product_html in product_htmls:
                    item = {}
                    item["content"] = product_html.text
                    item_urls = product_html.find_elements(By.XPATH, ".//div//a")
                    item["url"] = first(item_urls, "href")
                    item_imgs = product_html.find_elements(By.XPATH, ".//div//img")
                    item["images"] = first(item_imgs, "src")
                    items.append(item)

                products = json.dumps(items)
                Q.put({"url": url, "products": products})

                if products == "[]":
                    logging.info(f"error rendering {url}, {html_page}")

            driver.quit()
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
        products = pd.read_csv(jsonl_file, names=["result"], sep="\t")
        products["obj"] = products.result.map(json.loads)
        products["url"] = products.obj.map(lambda x: x["url"])
        products["products"] = products.obj.map(lambda x: x["products"])
        products.drop(columns=["result", "obj"], inplace=True)
        results = pd.merge(results, products, on="url", how="left")

        results.to_json(output_file, orient="records", lines=True)

        logging.info(
            f"finish rendering {len(results)} pages, {len(results[results.products != '[]'])} success. "
        )

        # get title & products
        # https://p4psearch.1688.com/hamlet.html?scene=7&keywords=%32%34k%E9%87%91&m_exp=&a_o=730121404983&cosite=bing_feeds
        results["keywords"] = results.url.map(
            lambda x: urllib.parse.parse_qs(urllib.parse.urlparse(x).query).get(
                "keywords", [""]
            )[0]
        )
        results["title"] = results.html.map(
            lambda x: (etree.HTML(x).xpath("//title/text()") + [""])[0]
        )
        results = results[["url", "keywords", "title", "products"]]
        results.to_csv(result_file, sep="\t", index=False, header=False)
