# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import io
import time
import base64
import requests
from PIL import Image
from typing import Optional, List
from azure.identity import AzureCliCredential
from unitorch_microsoft.scripts.tools.report_items import reported_item

"""
pip3 install azure-identity
az login
"""

# https://eng.ms/docs/microsoft-ai/webxt/bing-fundamentals/dlis/dlis/papyrus/serviceusage/serviceusage
# https://eng.ms/docs/microsoft-ai/webxt/bing-fundamentals/dlis/dlis/papyrus/modelmigration/models
# You can only call /images/generations api through papyrus large endpoint.
papyrus_endpoint1 = "https://westus2large.papyrus.binginternal.com/images/generations"
papyrus_endpoint2 = "https://westus2large.papyrus.binginternal.com/images/edits"
papyrus_endpoint3 = "https://westus2large.papyrus.binginternal.com/chat/completions"
verify_scope = "api://5fe538a8-15d5-4a84-961e-be66cd036687/.default"

credential = AzureCliCredential()


def timed_cache(ttl_seconds=300):
    def decorator(func):
        cache = {}

        def wrapper(*args, **kwargs):
            key = (args, frozenset(kwargs.items()))
            now = time.time()

            if key in cache:
                result, timestamp = cache[key]
                if now - timestamp < ttl_seconds:
                    return result

            result = func(*args, **kwargs)
            cache[key] = (result, now)
            return result

        return wrapper

    return decorator


@timed_cache(ttl_seconds=300)
def get_access_token():
    access_token = credential.get_token(verify_scope).token
    return access_token


from unitorch_microsoft.externals.papyrus.text import (
    get_response as get_gpt4_response,
    get_tools_response as get_gpt4_tools_response,
    get_chat_response as get_gpt4_chat_response,
    get_gpt5_response,
    get_gpt5_tools_response,
    get_gpt5_chat_response,
)
from unitorch_microsoft.externals.papyrus.image import (
    get_image_response as get_gpt_image_response,
)
