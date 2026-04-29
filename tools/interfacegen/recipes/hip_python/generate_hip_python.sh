#!/usr/bin/env bash
# MIT License
#
# Copyright (c) 2024-2026 Advanced Micro Devices, Inc.
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
#
# Unified hip-python code generator (replaces the per-recipe scripts
# generate_hip_python_pkgs.sh, generate_llvmc_pkg.sh, generate_amd_comgr_pkg.sh).
#
# Writes only Cython sources, namespace markers, and CMake module-list/
# version include files into <output_dir>/python/<package>/. Python-packaging
# files (handcoded in the hip-python repo) are NEVER written.
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
  echo "ERROR: script must not be sourced"
  return 1
fi

set -e

HELP_MSG="
Usage: ./$(basename "$0") output_dir [OPTIONS]

Required:
  output_dir            Root of the hip-python repo. Generator writes into
                        <output_dir>/python/<package>/ ...
  --rocm-version        ROCm version (e.g. 7.13.0). Can also be set via the
                        ROCM_VER environment variable.

Options:
  --rocm-path           Path to a ROCm installation. Default: \$ROCM_PATH or /opt/rocm.
  --recipes             Comma-separated subset of {hip,llvm,comgr}. Default: all.
  --hip-libs LIST       HIP library subset. Default: '*'.
  --llvm-libs LIST      LLVM library subset. Default: '*'.
  --comgr-libs LIST     COMGR library subset. Default: '*'.
  --no-rt-linking       Disable runtime linking (link directly against shared libs).
  --pre-clean           Remove the venv subdir '_venv' before all other tasks.
  --post-clean          Remove the venv subdir '_venv' after all other tasks.
  -n, --no-venv         Do not create/use a virtual environment.
  -h, --help            Show this help message.
"

case ${1:-} in
  -h|--help) echo "${HELP_MSG}"; exit 0 ;;
esac

OUTPUT_DIR=${1:-}
if [ -z "${OUTPUT_DIR}" ]; then
  echo "ERROR: no output dir specified."
  exit 1
fi
shift

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --pre-clean)     PRE_CLEAN=1; shift ;;
    --post-clean)    POST_CLEAN=1; shift ;;
    -n|--no-venv)    NO_VENV=1; shift ;;
    --rocm-path)     ROCM_PATH=$2; shift; shift ;;
    --rocm-version)  ROCM_VER=$2; shift; shift ;;
    --recipes)       EXTRA_ARGS+=(--recipes "$2"); shift; shift ;;
    --hip-libs)      EXTRA_ARGS+=(--hip-libs "$2"); shift; shift ;;
    --llvm-libs)     EXTRA_ARGS+=(--llvm-libs "$2"); shift; shift ;;
    --comgr-libs)    EXTRA_ARGS+=(--comgr-libs "$2"); shift; shift ;;
    --no-rt-linking) EXTRA_ARGS+=(--no-rt-linking); shift ;;
    -h|--help)       echo "${HELP_MSG}"; exit 0 ;;
    *)               echo "ERROR: unknown option '$1'"; exit 1 ;;
  esac
done

if [ -z "${ROCM_VER:-}" ]; then
  echo "ERROR: no ROCm version specified (--rocm-version or ROCM_VER)."
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

[ -z "${PRE_CLEAN+x}" ] || rm -rf "${SCRIPT_DIR}/_venv"

if [ -z "${NO_VENV+x}" ]; then
  [ ! -d "${SCRIPT_DIR}/_venv" ] && python3 -m venv "${SCRIPT_DIR}/_venv"
  PYTHON="${SCRIPT_DIR}/_venv/bin/python3"
else
  PYTHON="python3"
fi

"${PYTHON}" -m pip install -r "${SCRIPT_DIR}/requirements.txt"

declare -x HIP_PLATFORM=${HIP_PLATFORM:-amd}
declare -x ROCM_PATH=${ROCM_PATH:-/opt/rocm}
HIP_PYTHON_CLANG_RES_DIR=$("${ROCM_PATH}/llvm/bin/clang" -print-resource-dir)

# Run the unified codegen as a module so relative imports inside
# recipes/hip_python/{hip,llvm,comgr}/ resolve.
PYTHONPATH="${SCRIPT_DIR}/..:${PYTHONPATH:-}" \
  "${PYTHON}" -m hip_python.codegen \
    "${OUTPUT_DIR}" \
    --rocm-version "${ROCM_VER}" \
    --rocm-path "${ROCM_PATH}" \
    --clang-resource-dir "${HIP_PYTHON_CLANG_RES_DIR}" \
    "${EXTRA_ARGS[@]}"

[ -z "${POST_CLEAN+x}" ] || rm -rf "${SCRIPT_DIR}/_venv"
