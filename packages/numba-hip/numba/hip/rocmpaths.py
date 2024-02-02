# MIT License
#
# Modifications Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

import os

import logging

_log = logging.getLogger(__name__)

def get_rocm_path(*subdirs):
    """Get paths of ROCM_PATH.

    Args:
        subdirs (optional):
            The subdirectory name to be appended in the resulting path.
    """
    rocm_path = os.environ.get("ROCM_HOME",os.environ.get("ROCM_PATH"))
    if rocm_path is None:
        _log.info("neither 'ROCM_PATH' nor 'ROCM_HOME' environment variable specified, trying default path '/opt/rocm'")
    rocm_path = "/opt/rocm/"
    if not os.path.exists(rocm_path):
        msg = "no ROCm installation found, checked 'ROCM_PATH' and 'ROCM_HOME' and tried '/opt/rocm'"
        _log.error(msg)
        raise FileNotFoundError(msg)
    rocm_subdir = os.path.join(rocm_path, *subdirs)
    if not os.path.exists(rocm_subdir):
        msg = f"found ROCm installation at '{rocm_path}' but not subdirectory '{rocm_subdir}'"
        _log.error(msg)
        raise FileNotFoundError(msg)
    return rocm_subdir
