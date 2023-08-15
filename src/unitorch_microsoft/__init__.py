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


@replace(unitorch.models.peft.PeftCheckpointMixin)
class PeftCheckpointMixinV2(unitorch.models.peft.PeftCheckpointMixin):
    checkpoint_name = "pytorch_model.bin"

    modules_to_save_checkpoints = ["lora"]

    def save_checkpoint(
        self,
        ckpt_dir: str,
        weight_name: str = None,
        **kwargs,
    ):
        """
        Save the model's current state as a checkpoint.

        Args:
            ckpt_dir (str): Directory path to save the checkpoint.
            weight_name (str): Name of the weight file.

        Returns:
            None
        """
        if weight_name is None:
            weight_name = self.checkpoint_name
        state_dict = self.state_dict()
        state_dict = {
            k: v
            for k, v in state_dict.items()
            if any(m in k for m in self.modules_to_save_checkpoints)
        }
        weight_path = os.path.join(ckpt_dir, weight_name)
        torch.save(state_dict, weight_path)
        logging.info(f"{type(self).__name__} model save checkpoint to {weight_path}")

    def from_checkpoint(
        self,
        ckpt_dir: str,
        weight_name: Optional[str] = None,
        **kwargs,
    ):
        """
        Load model weights from a checkpoint.

        Args:
            ckpt_dir (str): Directory path of the checkpoint.
            weight_name (str): Name of the weight file.

        Returns:
            None
        """
        if weight_name is None:
            weight_name = self.checkpoint_name
        weight_path = os.path.join(ckpt_dir, weight_name)
        if not os.path.exists(weight_path):
            return
        _state_dict = self.state_dict()
        _state_dict = {
            k: v
            for k, v in _state_dict.items()
            if any(m in k for m in self.modules_to_save_checkpoints)
        }
        state_dict = torch.load(weight_path, map_location="cpu")
        assert all(
            k in state_dict.keys() and state_dict[k].shape == v.shape
            for k, v in _state_dict.items()
        )
        self.load_state_dict(state_dict, strict=False)
        logging.info(
            f"{type(self).__name__} model load weight from checkpoint {weight_path}"
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
