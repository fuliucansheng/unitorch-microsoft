# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import logging
import pkg_resources
import importlib
import importlib.metadata as importlib_metadata
import torch
import unitorch
import unitorch.cli
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.utils import replace
from transformers.utils import is_remote_url
from unitorch.cli import cached_path as _cached_path
import unitorch_microsoft.score
import unitorch_microsoft.models
import unitorch_microsoft.scripts
import unitorch_microsoft.services

VERSION = "0.0.0.1"

logger = logging.getLogger()

# auto_gptq
_auto_gptq_available = importlib.util.find_spec("auto_gptq") is not None
try:
    _auto_gptq_version = importlib_metadata.version("auto_gptq")
    logging.debug(f"Successfully imported auto_gptq version {_auto_gptq_version}")
except importlib_metadata.PackageNotFoundError:
    _auto_gptq_available = False


def is_auto_gptq_available():
    return _auto_gptq_available


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
        pkg_filename = pkg_resources.resource_filename(
            "unitorch_microsoft", url_or_filename
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


if logger.getEffectiveLevel() <= logging.DEBUG:
    import unitorch_microsoft.models.bletchley
    import unitorch_microsoft.models.culr
    import unitorch_microsoft.models.florence
    import unitorch_microsoft.models.mmdnn
    import unitorch_microsoft.models.tnlr
    import unitorch_microsoft.models.tulg
    import unitorch_microsoft.models.tulr
    import unitorch_microsoft.adsplus.relevance
    import unitorch_microsoft.pa
    import unitorch_microsoft.vpr
