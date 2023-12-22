#!/usr/bin/bash
# MIT License
#
# Copyright (c) 2023 Advanced Micro Devices, Inc.
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

# env var ROCM_VER - The ROCm version to consider.
# env var BASE_BRANCH - The branch to base this version on.

if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
   echo "ERROR: script must not be sourced";
   return
fi

set -e
set -o xtrace

if [ -z ${ROCM_VER+x} ]; then
   echo "ERROR: environment variable 'ROCM_VER' not set."
  return 1
fi

if [ -z ${BASE_BRANCH+x} ]; then
   echo "ERROR: environment variable 'BASE_BRANCH' not set."
  return 1
fi

sudo apt update
sudo apt install -y git

NEW_BRANCH=develop/rocm-rel-${ROCM_VER}

RELEASE_REPO_DIR=${RELEASE_REPO_DIR:-hip-python-release-repo}
git clone https://github.com/ROCmSoftwarePlatform/hip-python.git ${RELEASE_REPO_DIR}
cd ${RELEASE_REPO_DIR}
git checkout ${BASE_BRANCH}
git checkout ${NEW_BRANCH} || git branch ${NEW_BRANCH}
git checkout ${NEW_BRANCH}

# checkout all necessary tools
bash init.sh
