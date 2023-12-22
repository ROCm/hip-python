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

if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
   echo "ERROR: script must not be sourced";
   return
fi

set -e
set -o xtrace

echo ${ROCM_VER+x}
if [ -z ${ROCM_VER+x} ]; then
  echo "ERROR: environment variable 'ROCM_VER' not set."
  exit 1
fi

# preinstall tzdata without install recommendations
export DEBIAN_FRONTEND=noninteractive
sudo apt-get install -y --no-install-recommends tzdata

# install latest rocm installation tool
sudo apt update
wget -np -r -nH --cut-dirs=4 -A "amdgpu-install*deb" https://repo.radeon.com/amdgpu-install/latest/ubuntu/focal/
sudo apt install -y --no-install-recommends ./amdgpu-install_*.deb
rm ./amdgpu-install_*.deb

ROCM_VER_SHORT=$(echo $ROCM_VER | sed "s,\([0-9]\+\.[0-9]\+\)\.0,\1,g")

echo "deb [arch=amd64 signed-by=/etc/apt/trusted.gpg.d/rocm-keyring.gpg] https://repo.radeon.com/rocm/apt/${ROCM_VER_SHORT} focal main" | sudo tee /etc/apt/sources.list.d/rocm.list
echo -e 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600' | sudo tee /etc/apt/preferences.d/rocm-pin-600
sudo apt update
 
# install ROCm
sudo amdgpu-install -y --usecase=rocm --rocmrelease=${ROCM_VER} --no-dkms
