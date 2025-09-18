# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

from unitorch_microsoft.spaces.demos.tools.gpt_caption import GPT4WebUI
from unitorch_microsoft.spaces.demos.tools.gpt5v_caption import GPT5VisonWebUI
from unitorch_microsoft.spaces.demos.tools.bg_remove import RemoveBGWebUI
from unitorch_microsoft.spaces.demos.tools.gpt_img_create import GPTCreateImgWebUI
from unitorch_microsoft.spaces.demos.tools.img_caption import CaptionImgWebUI
from unitorch_microsoft.spaces.demos.tools.joycaption2 import JoyCaption2WebUI
from unitorch_microsoft.spaces.demos.tools.nano_banana import NanoBananaWebUI

tools_pages = [
    GPT4WebUI,
    GPT5VisonWebUI,
    GPTCreateImgWebUI,
    NanoBananaWebUI,
    RemoveBGWebUI,
    CaptionImgWebUI,
    JoyCaption2WebUI,
]
