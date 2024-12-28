# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import logging
import importlib
import importlib_resources
import importlib.metadata as importlib_metadata
import torch
import ast
import hashlib
import configparser
import unitorch
import unitorch.cli
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.utils import replace
from transformers.utils import is_remote_url
from unitorch.cli import cached_path as _cached_path

VERSION = "0.0.0.1"

logger = logging.getLogger()


UNITORCH_MS_ENDPOINT = os.environ.get(
    "UNITORCH_MS_ENDPOINT", "https://unitorchazureblob.blob.core.windows.net"
)


def ms_endpoint_url(url):
    if is_remote_url(url):
        return url
    if url.startswith("/"):
        url = url[1:]
    return f"{UNITORCH_MS_ENDPOINT}/{url}"


# faiss
_faiss_available = importlib.util.find_spec("faiss") is not None
if _faiss_available:
    logging.debug(f"Successfully imported faiss")


def is_faiss_available():
    return _faiss_available


# openai
_openai_available = importlib.util.find_spec("openai") is not None
try:
    _openai_version = importlib_metadata.version("openai")
    logging.debug(f"Successfully imported openai version {_openai_version}")
except importlib_metadata.PackageNotFoundError:
    _openai_available = False


def is_openai_available():
    return _openai_available


@replace(unitorch.cli.cached_path)
def cached_path(
    url_or_filename,
    cache_dir=None,
    force_download=False,
    proxies=None,
    resume_download=False,
    user_agent: Union[Dict, str, None] = None,
    extract_compressed_file=False,
    force_extract=False,
    use_auth_token: Union[bool, str, None] = None,
    local_files_only=False,
) -> Optional[str]:
    if not is_remote_url(url_or_filename):
        pkg_filename = os.path.join(
            importlib_resources.files("unitorch_microsoft"), url_or_filename
        )
        if os.path.exists(pkg_filename):
            return pkg_filename

    return _cached_path(
        url_or_filename,
        cache_dir=cache_dir,
        force_download=force_download,
        proxies=proxies,
        resume_download=resume_download,
        user_agent=user_agent,
        extract_compressed_file=extract_compressed_file,
        force_extract=force_extract,
        use_auth_token=use_auth_token,
        local_files_only=local_files_only,
    )


import unitorch_microsoft.scores
import unitorch_microsoft.models
import unitorch_microsoft.modules
import unitorch_microsoft.scripts
import unitorch_microsoft.services

UNITORCH_DEBUG = os.environ.get("UNITORCH_DEBUG", "INFO").upper()

if UNITORCH_DEBUG == "ALL":
    import unitorch_microsoft.adinsights
    import unitorch_microsoft.adsplus
    import unitorch_microsoft.aether
    import unitorch_microsoft.chatgpt
    import unitorch_microsoft.deepgen
    import unitorch_microsoft.models.bletchley
    import unitorch_microsoft.models.bloom
    import unitorch_microsoft.models.culr
    import unitorch_microsoft.models.llama
    import unitorch_microsoft.models.mmdnn
    import unitorch_microsoft.models.sam
    import unitorch_microsoft.models.tnlr
    import unitorch_microsoft.models.tribert
    import unitorch_microsoft.models.tulg
    import unitorch_microsoft.models.tulr
    import unitorch_microsoft.msan
    import unitorch_microsoft.pa
    import unitorch_microsoft.pa.intl
    import unitorch_microsoft.pa.l2
    import unitorch_microsoft.utils
    import unitorch_microsoft.vpr
    import unitorch_microsoft.webuis
