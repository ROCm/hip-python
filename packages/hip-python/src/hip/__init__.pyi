# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# HAND-MAINTAINED, unlike the generated stubs next to the extension modules.
# `tests/hip-python` fails when this file and `__init__.py` drift apart.

from typing import Any

from rocm.bindings import hip as hip
from rocm.bindings import hiprtc as hiprtc
from rocm.version import HIP_VERSION as HIP_VERSION
from rocm.version import HIP_VERSION_NAME as HIP_VERSION_NAME
from rocm.version import HIP_VERSION_TUPLE as HIP_VERSION_TUPLE
from rocm.version import ROCM_VERSION as ROCM_VERSION
from rocm.version import ROCM_VERSION_NAME as ROCM_VERSION_NAME
from rocm.version import ROCM_VERSION_TUPLE as ROCM_VERSION_TUPLE
from rocm.version import hip_version_name as hip_version_name
from rocm.version import hip_version_tuple as hip_version_tuple
from rocm.version import rocm_version_name as rocm_version_name
from rocm.version import rocm_version_tuple as rocm_version_tuple

__author__: str

# Read from the installed distribution metadata at import.
VERSION: str
__version__: str

# `hiprtc.ext` is attached to the `rocm.bindings.hiprtc` module object at
# import time and cannot be declared from here.

# The optional bindings are bound only where their wheel is installed, which
# a stub cannot express: declaring them would type-check code that fails at
# runtime. Stub-only -- the runtime module has no `__getattr__`.
def __getattr__(name: str) -> Any: ...
