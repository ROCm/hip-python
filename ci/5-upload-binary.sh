#!/usr/bin/env bash
# MIT License
# 
# Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
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

# env var BINARY_REPO - A repo such as 'pypi' or 'testpypi'.
# env var BINARY_REPO_TOKEN - The token to use.

if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
   echo "ERROR: script must not be sourced";
   return 1
fi

set -e
set -o xtrace

if [ -z ${BINARY_REPO+x} ]; then
  echo "ERROR: environment variable 'BINARY_REPO' not set."
  exit 1
fi

if [ -z ${BINARY_REPO_TOKEN+x} ]; then
  echo "ERROR: environment variable 'BINARY_REPO_TOKEN' not set."
  exit 1
fi

python3 -m twine upload --repository ${BINARY_REPO} _upload/*.whl -u__token__ -p${BINARY_REPO_TOKEN}
