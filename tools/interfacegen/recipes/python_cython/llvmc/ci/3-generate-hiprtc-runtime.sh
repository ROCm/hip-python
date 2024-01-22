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

ROCM_VER_X=$(echo "${ROCM_VER}.x" | sed "s,\([0-9]\+\.[0-9]\+\)\.[0-9]\+,\1,g")

DEPSDIR=${DEPSDIR:-"__deps"}
DEPSDIR=$(realpath ${DEPSDIR})

RELEASE_REPO_DIR=${RELEASE_REPO_DIR:-rocm-llvm-python-release-repo}
RELEASE_REPO_DIR=$(realpath ${RELEASE_REPO_DIR})

mkdir -p ${DEPSDIR}
cd ${DEPSDIR}
echo "get ROCm/clr"
git clone https://github.com/ROCm/clr.git -b rocm-${ROCM_VER_X} ||\
        git clone https://github.com/ROCm/clr.git -b develop # < ROCm 5.6 does not have a separate branch/tag
echo "get ROCm/HIP"
git clone https://github.com/ROCm/HIP.git -b rocm-${ROCM_VER_X}

# run cmake
echo "run cmake"
declare -x ROCM_PATH=/opt/rocm
declare -x HIP_DIR=${DEPSDIR}/HIP
declare -x HIPCC_BIN_DIR=/opt/rocm/bin
declare -x OPENCL_DIR=${DEPSDIR}/clr/opencl
declare -x ROCCLR_DIR=${DEPSDIR}/clr/rocclr
declare -x HIP_PLATFORM=amd

# create and soure a venv
python3 -m venv _venv
source _venv/bin/activate
pip install cmake cppheaderparser

cd ${DEPSDIR}/clr/hipamd/
mkdir -p build
cd build

cmake -DHIP_COMMON_DIR="${HIP_DIR}" \
        -DHIPCC_BIN_DIR="${HIPCC_DIR}/bin" \
        -DAMD_OPENCL_PATH=${OPENCL_DIR} \
        -DROCCLR_PATH=${ROCCLR_DIR} \
        -DCMAKE_PREFIX_PATH="${ROCM_PATH}/" \
        -DCMAKE_INSTALL_PREFIX=install .. --fresh --debug-output		

# generate hiprtc runtime header
echo "generate hiprtc runtime header"
make hiprtc-builtins VERBOSE=1

# copy header file into release repo
echo "copy header file into ROCm LLVM Python 'rocm.llvm.amd_comgr' package"
cp ${DEPSDIR}/clr/hipamd/build/src/hiprtc/hip_rtc_gen/hipRTC ${RELEASE_REPO_DIR}/rocm-llvm-python/rocm/amd_comgr/hiprtc_runtime.h

# decativate venv and remove folder
deactivate
rm -rf _venv
