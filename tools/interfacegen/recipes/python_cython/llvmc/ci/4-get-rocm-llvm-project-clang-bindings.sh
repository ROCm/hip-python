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

#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
   echo "ERROR: script must not be sourced";
   return
fi

set -e
set -o xtrace

if [ -z ${ROCM_VER+x} ]; then
  echo "ERROR: environment variable 'ROCM_VER' not set."
  exit 1
fi

ROCM_VER_SHORT=$(echo ${ROCM_VER} | sed "s,\([0-9]\+\.[0-9]\+\)\.0,\1,g")

RELEASE_REPO_DIR=${RELEASE_REPO_DIR:-rocm-llvm-python-release-repo}
RELEASE_REPO_DIR=$(realpath ${RELEASE_REPO_DIR})

sudo apt install wget

cd ${RELEASE_REPO_DIR}/rocm-llvm-python/rocm/clang/

# 1) copy clang bindings into ROCm LLVM Python 'rocm.llvm.clang' package
echo "copy clang bindings into ROCm LLVM Python 'rocm.llvm.clang' package"
for f in "__init__.py" "cindex.py" "enumerations.py"; do
  rm -f ${f}
  wget https://raw.githubusercontent.com/ROCm/llvm-project/rocm-${ROCM_VER_SHORT}.x/clang/bindings/python/clang/${f}
done
sed -s -i "s,clang\.enumerations,rocm.clang.enumerations," ${RELEASE_REPO_DIR}/rocm-llvm-python/rocm/clang/cindex.py

# 2) copy LLVM LICENSE.TXT int into ROCm LLVM Python 'rocm.llvm.clang' package
echo "copy LLVM LICENSE.TXT int into ROCm LLVM Python 'rocm.llvm.clang' package"

rm -f LICENSE.TXT
wget https://raw.githubusercontent.com/ROCm/llvm-project/rocm-${ROCM_VER_SHORT}.x/LICENSE.TXT
