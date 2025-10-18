# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

from unitorch_microsoft.spaces.demos.tools.bg_remove import RemoveBGWebUI
from unitorch_microsoft.spaces.demos.tools.img_caption import CaptionImgWebUI
from unitorch_microsoft.spaces.demos.tools.joycaption2 import JoyCaption2WebUI
from unitorch_microsoft.spaces.demos.tools.opencv import OpenCVZoomInWebUI

tools_pages = [
    RemoveBGWebUI,
    CaptionImgWebUI,
    JoyCaption2WebUI,
    OpenCVZoomInWebUI,
]
