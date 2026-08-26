#!/usr/bin/env bash
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

set -eu

# Build the hip-python wheel matrix inside a manylinux container.
#
# One build-wheels.sh pass per matrix entry, all of them writing into the
# same BUILD_ARTIFACTS_DIR, so the stage ends with a single wheel pool that
# covers the whole supported interpreter range:
#
#   3.10, no SABI     -> cp310-cp310 wheels
#   3.11, no SABI     -> cp311-cp311 wheels
#   3.12, SABI 3.12   -> cp312-abi3 wheels, which install on 3.12 and on
#                        every later CPython, so the tail of the range
#                        needs no build of its own
#
# 3.10 and 3.11 are built version-specific because the stable ABI only
# carries the buffer protocol from 3.11 on and an abi3 floor below that
# cannot express these bindings; see USE_SABI in build-wheels.sh.
#
# Manylinux-specific by construction: the interpreters come from
# /opt/python/cp<xy>-cp<xy>, which is a manylinux layout. Another distro
# wants a sibling driver rather than a fallback hidden in here, the same way
# build-wheels.ps1 sits next to build-wheels.sh.
#
# Required env:
#   SRC_DIR              parent of the checkout, as for build-wheels.sh
#   BUILD_DIR            scratch dir; every entry gets a subdirectory of it
#   BUILD_ARTIFACTS_DIR  shared wheel output dir for the whole matrix
#
# Optional env:
#   HIP_PYTHON_WHEEL_MATRIX   default "3.10:no 3.11:no 3.12:3.12".
#                             Whitespace-separated <python version>:<USE_SABI>
#                             entries
#
# Everything else build-wheels.sh reads (ROCM_PATH, ROCM_VERSION, MAX_JOBS,
# LIGHT_MODE, SKIP_CODEGEN, ...) is passed through untouched. BUILD_DIR and
# USE_SABI are the exceptions: this script sets them per entry.

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=./libos.sh
. "${script_dir}/libos.sh"

: "${SRC_DIR:?SRC_DIR must be set}"
: "${BUILD_DIR:?BUILD_DIR must be set}"
: "${BUILD_ARTIFACTS_DIR:?BUILD_ARTIFACTS_DIR must be set}"

# build-wheels.sh reads these from the environment, and a caller that assigned
# them without exporting would otherwise hand it nothing.
export SRC_DIR BUILD_ARTIFACTS_DIR

wheel_matrix=${HIP_PYTHON_WHEEL_MATRIX:-"3.10:no 3.11:no 3.12:3.12"}

mkdir -p "${BUILD_ARTIFACTS_DIR}"

### parse the matrix, resolve every interpreter up front
#
# A matrix entry naming a version this image does not carry should fail in
# seconds rather than after the first entry has spent an hour compiling LLVM.

python_versions=()
sabi_floors=()
python_bins=()

for entry in ${wheel_matrix}; do
  python_version=${entry%%:*}
  sabi=${entry##*:}
  if [[ "${python_version}" == "${entry}" || -z "${python_version}" || -z "${sabi}" ]]; then
    echo "ERROR: malformed HIP_PYTHON_WHEEL_MATRIX entry \"${entry}\"." >&2
    echo "       Expected <python version>:<USE_SABI>, e.g. \"3.10:no\" or \"3.12:3.12\"." >&2
    exit 1
  fi

  python_versions+=("${python_version}")
  sabi_floors+=("${sabi}")
  python_bins+=("$(get_manylinux_python_bin "${python_version}")")
done

if [[ ${#python_versions[@]} -eq 0 ]]; then
  echo "ERROR: HIP_PYTHON_WHEEL_MATRIX is empty; nothing to build." >&2
  exit 1
fi

### build

base_path=${PATH}

for i in "${!python_versions[@]}"; do
  python_version=${python_versions[${i}]}
  sabi=${sabi_floors[${i}]}
  python_bin=${python_bins[${i}]}
  tag=cp${python_version//./}

  echo "=============================================================="
  echo "hip-python wheels for CPython ${python_version} (USE_SABI=${sabi})"
  echo "=============================================================="

  # Its own BUILD_DIR per entry: build-wheels.sh copies the checkout there and
  # configures CMake inside it, and a cache configured against one interpreter
  # is not reusable for the next.
  PATH="${python_bin}:${base_path}" \
  BUILD_DIR="${BUILD_DIR}/${tag}" \
  USE_SABI="${sabi}" \
    "${script_dir}/build-wheels.sh"
done

### verify the matrix produced the tags it promised
#
# The build is long and its failure modes are quiet — a mistagged wheel set
# only surfaces as an unsatisfiable install in the test stage, a stage and a
# container later. Check here instead, where the build log is still at hand.

shopt -s nullglob

missing=()

echo "=============================================================="
echo "Wheels in ${BUILD_ARTIFACTS_DIR}"
echo "=============================================================="

for i in "${!python_versions[@]}"; do
  python_version=${python_versions[${i}]}
  sabi=${sabi_floors[${i}]}

  if [[ "${sabi}" == "no" ]]; then
    tag=cp${python_version//./}
    expected_tag="${tag}-${tag}"
  else
    expected_tag="cp${sabi//./}-abi3"
  fi

  found=("${BUILD_ARTIFACTS_DIR}"/*-"${expected_tag}"-*.whl)
  printf '  %-14s %d compiled wheel(s)\n' "${expected_tag}" "${#found[@]}"
  if [[ ${#found[@]} -eq 0 ]]; then
    missing+=("${expected_tag}, from CPython ${python_version} with USE_SABI=${sabi}")
  fi
done

# The two interpreter-agnostic distributions, built once by whichever entry ran
# last and needed by every tested interpreter. Only hip-python is pure enough
# to land on py3-none-any: numba-hip declares itself non-pure in its setup.py
# so that numba/hip installs into the same scheme as numba on a lib/lib64-split
# system, which keeps py3-none but pins the platform. Nothing repairs that
# wheel afterwards, so the platform it is given here is AUDITWHEEL_PLAT or the
# linux_x86_64 no index accepts, and the glob asks for the former.
distributions=(hip_python numba_hip)
distribution_tags=("py3-none-any" "py3-none-manylinux*")

for i in "${!distributions[@]}"; do
  distribution=${distributions[${i}]}
  expected_tag=${distribution_tags[${i}]}

  found=("${BUILD_ARTIFACTS_DIR}/${distribution}"-*-${expected_tag}.whl)
  printf '  %-14s %d wheel(s)\n' "${distribution}" "${#found[@]}"
  if [[ ${#found[@]} -eq 0 ]]; then
    missing+=("${distribution} (${expected_tag})")
  fi
done

if [[ ${#missing[@]} -gt 0 ]]; then
  echo "ERROR: the wheel matrix is incomplete. Nothing produced:" >&2
  printf '  - %s\n' "${missing[@]}" >&2
  echo "Wheels that are present:" >&2
  ls -1 "${BUILD_ARTIFACTS_DIR}" >&2 || true
  exit 1
fi

echo "Wheel matrix complete."
