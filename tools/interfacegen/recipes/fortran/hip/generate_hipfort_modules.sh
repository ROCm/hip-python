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
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
   echo "ERROR: script must not be sourced";
   return 1
fi

set -e

HELP_MSG="
Usage: ./$(basename $0) output_dir [OPTIONS]

Required:
  output_dir        The output directory to which the files should be written to. Must contain 'hip-python' and 'hip-python-as-cuda' subfolders.
  --rocm-version    The ROCm version, e.g. '5.6.0'. Can also be specified via the 'ROCM_VER' environment variable.

Options:
  --rocm-path       Path to a ROCm installation, defaults to variable 'ROCM_PATH' if set or '/opt/rocm'.
  --libs            HIP Python libraries to generate as comma separated list without whitespaces, defaults to variable 'HIP_PYTHON_LIBS' if set or '*'.
                    Add a prefix '^' to NOT generate code for the comma-separated list of libraries that follows but all other libraries.
  --pre-clean       Remove the virtual Python environment subfolder '_venv' --- if it exists --- before all other tasks.
  --post-clean      Remove the virtual Python environment subfolder '_venv' --- if it exists --- after all other tasks.
  -n, --no-venv     Do not create and use a virtual Python environment.
  -h, --help        Show this help message.
"

case $1 in
  -h|--help)
    echo "${HELP_MSG}"
    exit 0
    ;;
esac

OUTPUT_DIR=$1
if [ -z ${OUTPUT_DIR} ]; then
  echo "ERROR: no output dir specified."
  exit 1
fi
shift

while [[ $# -gt 0 ]]; do
  case $1 in
    --pre-clean)
      PRE_CLEAN=1
      shift
      ;;
    --post-clean)
      POST_CLEAN=1
      shift
      ;;
    -n|--no-venv)
      NO_VENV=1
      shift
      ;;
    --libs)
      HIP_PYTHON_LIBS=$2
      shift; shift
      ;;
    --rocm-path)
      ROCM_PATH=$2
      shift; shift
      ;;
    --rocm-version)
      ROCM_VER=$2
      shift; shift
      ;;
    -h|--help)
      echo "${HELP_MSG}"
      exit 0
      ;;
    -*|--*)
      echo "ERROR: unknown option '$1'"
      exit 1
      ;;
    *)
      echo "ERROR: unknown option '$1'"
      exit 1
      ;;
  esac
done

if [ -z ${ROCM_VER} ]; then
  echo "ERROR: no ROCm version specified."
  exit 1
fi

[ -z ${PRE_CLEAN+x} ] || rm -rf venv

alias PYTHON="python3"
if [ -z ${NO_ENV+x} ]; then
  [ ! -d "venv" ] && python3 -m venv _venv
  alias PYTHON="_venv/bin/python3"
fi
shopt -s expand_aliases

PYTHON -m pip install -r requirements.txt

declare -x HIP_PLATFORM=${HIP_PLATFORM:-amd}
declare -x HIP_PYTHON_LIBS=${HIP_PYTHON_LIBS:-*}
declare -x ROCM_PATH=${ROCM_PATH:-/opt/rocm}
declare -x CLANG_RES_DIR=$(${ROCM_PATH}/llvm/bin/clang -print-resource-dir)
echo $ROCM_PATH
echo $CLANG_RES_DIR
PYTHON codegen_hipfort.py ${OUTPUT_DIR} --rocm-version ${ROCM_VER}

[ -z ${POST_CLEAN+x} ] || rm -rf venv
