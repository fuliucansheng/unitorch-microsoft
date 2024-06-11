# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import logging
import pkg_resources
import importlib
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
import unitorch_microsoft.scores
import unitorch_microsoft.models
import unitorch_microsoft.modules
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


@replace(unitorch.cli.CoreConfigureParser)
class CoreConfigureParser(configparser.ConfigParser):
    def __init__(
        self,
        fpath: Optional[str] = "./config.ini",
        params: Optional[Tuple[str]] = [],
    ):
        super().__init__(interpolation=configparser.ExtendedInterpolation())
        self.fpath = fpath
        self.read(fpath)
        for param in params:
            assert len(param) == 3
            k0, k1, v = param
            if not self.has_section(k0):
                self.add_section(k0)
            self.set(k0, k1, str(v))

        self._freeze_section = None

        self._default_section = self.getdefault("core/config", "default_section", None)
        self._freeze_section = self.getdefault("core/config", "freeze_section", None)

    def _getdefault(self, section, option, value=None):
        if not self.has_section(section):
            return value
        if self.has_option(section, option):
            return self.get(section, option)
        return value

    def _ast_replacement(self, node):
        value = node.id
        if value in ("True", "False", "None"):
            return node
        return ast.Str(value)

    def _ast_literal_eval(self, value):
        root = ast.parse(value, mode="eval")
        if isinstance(root.body, ast.BinOp):
            raise ValueError(value)

        for node in ast.walk(root):
            for field, child in ast.iter_fields(node):
                if isinstance(child, list):
                    for index, subchild in enumerate(child):
                        if isinstance(subchild, ast.Name):
                            child[index] = self._ast_replacement(subchild)
                elif isinstance(child, ast.Name):
                    replacement = self._ast_replacement(child)
                    node.__setattr__(field, replacement)
        return ast.literal_eval(root)

    def get(
        self,
        section,
        option,
        raw=False,
        vars=None,
        fallback=configparser._UNSET,
    ):
        value = super().get(
            section,
            option,
            raw=raw,
            vars=vars,
            fallback=fallback,
        )
        if raw:
            return value
        try:
            return self._ast_literal_eval(value)
        except (SyntaxError, ValueError):
            return value

    def set_freeze_section(self, section):
        self._freeze_section = section

    def set_default_section(self, section):
        self._default_section = section

    def getdefault(self, section, option, value=None):
        value = self._getdefault(section, option, value)

        if self._freeze_section:
            value = self._getdefault(self._freeze_section, option, value)

        return value

    def getoption(self, option, value=None):
        return self.getdefault(self._default_section, option, value)

    def print(self):
        print("#" * 30, "Config Info".center(20, " "), "#" * 30)
        for sec, item in self.items():
            for k, v in item.items():
                print(
                    sec.rjust(10, " "),
                    ":".center(5, " "),
                    k.ljust(30, " "),
                    ":".center(5, " "),
                    v.ljust(30, " "),
                )
        print("#" * 30, "Config Info End".center(20, " "), "#" * 30)

    def save(self, save_path="./config.ini"):
        self.write(open(save_path, "w"))
        return save_path

    def hexsha(self, length=None):
        string = sorted(
            [f"{k}_{kk}_{vv}" for k, v in self.items() for kk, vv in v.items()]
        )
        string = "|".join(string)
        hexsha = hashlib.sha1(string.encode()).hexdigest()
        if length is not None:
            hexsha = hexsha[:length]
        return hexsha

    def __copy__(self):
        setting = [(sec, k, v) for sec in self.sections() for k, v in self[sec].items()]
        return type(self)(self.fpath, setting)

    def __deepcopy__(self, memo):
        setting = [(sec, k, v) for sec in self.sections() for k, v in self[sec].items()]
        return type(self)(self.fpath, setting)


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


import unitorch_microsoft.models

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
    import unitorch_microsoft.models.florence
    import unitorch_microsoft.models.llama
    import unitorch_microsoft.models.minigpt4
    import unitorch_microsoft.models.mmdnn
    import unitorch_microsoft.models.tnlr
    import unitorch_microsoft.models.tribert
    import unitorch_microsoft.models.tulg
    import unitorch_microsoft.models.tulr
    import unitorch_microsoft.msan
    import unitorch_microsoft.pa
    import unitorch_microsoft.pa.intl
    import unitorch_microsoft.pa.l2
    import unitorch_microsoft.tools
    import unitorch_microsoft.utils
    import unitorch_microsoft.vpr
    import unitorch_microsoft.webuis
