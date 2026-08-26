#!/usr/bin/env bash
# MIT License
#
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

# Container OS helpers.
#
# Trimmed copy of aiss-3p-dev-pipelines'
# public/python/jdp/toolbox/linuxcontainers/nodes/libos.sh, reduced to the
# manylinux interpreter selection plus the probes CI reports. Upstream's
# prepare_os/prepare_ubuntu_os/prepare_rhel_os are deliberately absent: they
# install a package set (java, dkms, qemu-kvm, a second python, cmake3) that
# would disturb the interpreter and toolchain of a manylinux image. The
# retained functions are byte-identical to upstream so a re-sync stays a
# readable diff.
#
# Sourced, not executed:
#
#   . ci/internal/libos.sh

# If this is an AlmaLinux based manylinux installation.
# Prints "1" in case of success.
# Example usage: `if [ $(is_manylinux) ]; then <do-something>; fi`
function is_almalinux_manylinux() {
  os_id=$(. /etc/os-release; echo $ID)
  [[ ${AUDITWHEEL_PLAT:-} == "manylinux"* ]] && [[ "${os_id}" == "almalinux" ]] && echo "1"
}

function get_python_version() {
  python3 -V | cut -d " " -f2
}

function get_glibc_version() {
  # note: example output of `ldd --version`:
  # ```text
  # ldd (GNU libc) 2.28
  # ldd (Ubuntu GLIBC 2.35-0ubuntu3) 2.35
  # [more lines...]
  # ```
  ldd --version | head -n1 | grep -o "ldd (.\+) [0-9]\+\(\.[0-9]\+\)\{1\}" | rev | cut -d " " -f1 | rev
}

function get_gcc_version() {
  # note: example output of `gcc --version`:
  # ```text
  # gcc (GCC) 11.5.0
  # gcc (GCC) 11.2.1 20220127 (Red Hat 11.2.1-9)
  # [more lines...]
  # ```
  gcc --version | head -n1 | grep -o "gcc (.\+) [0-9]\+\(\.[0-9]\+\)\{2\}" | rev | cut -d " " -f1 | rev
}

# Makes the manylinux CPython of the given version (default:
# MANYLINUX_DEFAULT_PYTHON_VERSION) the container's python3/pip3/pip.
function set_almalinux_manylinux_python3() {
  local default_python_version=${1:-${MANYLINUX_DEFAULT_PYTHON_VERSION:-$(get_python_version)}}
  local maj=$(echo "$default_python_version" | cut  -d "." -f 1)
  local min=$(echo "$default_python_version" | cut  -d "." -f 2)
  local python_bin_dir="/opt/python/cp${maj}${min}-cp${maj}${min}/bin"
  local python3_path="${python_bin_dir}/python${maj}.${min}"
  local pip3_path="${python_bin_dir}/pip${maj}.${min}"

  alternatives --install /usr/bin/python3 python3 ${python3_path} 0 \
               --slave /usr/bin/pip3 pip3 ${pip3_path} \
               --slave /usr/bin/pip pip ${pip3_path}
  alternatives --set python3 ${python3_path}

  python3 --version
  pip --version
  pip3 --version
  python3 -m pip --version

  python3 -m pip install --upgrade pip
  python3 -m pip install virtualenv
}

# Prints the bin directory of the manylinux CPython of the given version
# (default: MANYLINUX_DEFAULT_PYTHON_VERSION). Console scripts of packages
# installed with `pip` land there, so it must be on PATH.
function get_almalinux_manylinux_python3_bin_dir() {
  local default_python_version=${1:-${MANYLINUX_DEFAULT_PYTHON_VERSION:-$(get_python_version)}}
  local maj=$(echo "$default_python_version" | cut -d "." -f 1)
  local min=$(echo "$default_python_version" | cut -d "." -f 2)
  printf "/opt/python/cp${maj}${min}-cp${maj}${min}/bin"
}

# Local addition, no upstream counterpart: the same path, but checked.
#
# Prints the bin directory of the manylinux CPython of the given version and
# fails when the image does not carry that interpreter, naming the ones it
# does. The wheel matrix asks for several versions in a row, and which of them
# a given ROCm-flavoured manylinux base ships is not obvious from the outside.
function get_manylinux_python_bin() {
  local version=${1:?get_manylinux_python_bin needs a CPython version, e.g. 3.12}
  local bin_dir
  bin_dir=$(get_almalinux_manylinux_python3_bin_dir "${version}")

  if [[ ! -x "${bin_dir}/python${version}" ]]; then
    local present
    present=$(ls -d /opt/python/cp*-cp*/ 2>/dev/null | xargs -r -n1 basename | tr '\n' ' ')
    echo "ERROR: no CPython ${version} in this image; ${bin_dir}/python${version} is missing." >&2
    echo "       Interpreters present: ${present:-none, this is not a manylinux image}" >&2
    return 1
  fi

  printf '%s' "${bin_dir}"
}
