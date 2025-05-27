# Copyright (c) MICROSOFT.
# Licensed under the MIT License.
import os
import re
import sys
import platform
import logging
from itertools import chain
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup()
