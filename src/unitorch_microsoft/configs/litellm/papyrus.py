# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import time
import litellm
from typing import List, Dict
from litellm.integrations.custom_logger import CustomLogger
from azure.identity import AzureCliCredential

credential = AzureCliCredential()
verify_scope = "api://5fe538a8-15d5-4a84-961e-be66cd036687/.default"

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

def fetch_token():
    return credential.get_token(verify_scope).token

@timed_cache(ttl_seconds=300)
def get_cached_token():
    return fetch_token()

class PapyrusHandler(CustomLogger):
    def __init__(self):
        pass

    async def async_pre_call_deployment_hook(self, kwargs, call_type):
        model = kwargs.get("model", "")
        papyrus_model = kwargs.get("proxy_server_request", {}).get("body", {}).get("model", "")
        
        if papyrus_model.startswith("papyrus-"):
            access_token = get_cached_token()
            papyrus_model = "-".join(papyrus_model.split("-")[1:])  # remove "papyrus-" prefix
            
            kwargs["api_key"] = access_token
            kwargs["headers"] = {
                "papyrus-model-name": papyrus_model,
                "papyrus-quota-id": "",
                "papyrus-timeout-ms": "120000",
            }
            model 
        
        return kwargs 

handler = PapyrusHandler()
