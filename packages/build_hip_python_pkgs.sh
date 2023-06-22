#!/usr/bin/env bash
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
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
   echo "ERROR: script must not be sourced";
   return 1
fi

HELP_MSG="
Usage: ./$(basename $0) [OPTIONS]

Options:
  --rocm-path        Path to a ROCm installation, defaults to variable 'ROCM_PATH' if set or '/opt/rocm'.
  --no-hip           Do not build package 'hip-python'.
  --no-cuda          Do not build package 'hip-python-as-cuda'.
  --no-docs          Do not build the docs of package 'hip-python'.
  --no-api-docs      Temporarily move the 'hip-python/docs/python_api' subfolder so that sphinx does not see it.
  --no-clean-docs    Do not generate docs from scratch, i.e. don't run sphinx with -E switch.
  -j,--num-jobs      Number of build jobs to use (currently only applied for building docs). Defaults to 1.
  --pre-clean        Remove the virtual Python environment subfolder '_venv' --- if it exists --- before all other tasks.
  --post-clean       Remove the virtual Python environment subfolder '_venv' --- if it exists --- after all other tasks.
  -n, --no-venv      Do not create and use a virtual Python environment.
  -h, --help         Show this help message.
"

NUM_JOBS=1
while [[ $# -gt 0 ]]; do
  case $1 in
    -b|--pre-clean)
      PRE_CLEAN=1
      shift
      ;;
    -a|--post-clean)
      POST_CLEAN=1
      shift
      ;;
    -n|--no-venv)
      NO_VENV=1
      shift
      ;;
    -h|--help)
      echo "${HELP_MSG}"
      exit 0
      ;;
    --rocm-path)
      ROCM_PATH=$2
      shift; shift
      ;;
    --no-hip)
      NO_HIP=1
      shift
      ;;
    --no-cuda)
      NO_CUDA=1
      shift
      ;;
    --no-docs)
      NO_DOCS=1
      shift
      ;;
    --no-clean-docs)
      NO_CLEAN_DOCS=1
      shift
      ;;
    --no-api-docs)
      NO_API_DOCS=1
      shift
      ;;
    -j|--num-jobs)
      NUM_JOBS=$2
      shift; shift
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
    *)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

#echo "PRE_CLEAN=$PRE_CLEAN"
#echo "POST_CLEAN=$POST_CLEAN"
#echo "NO_VENV=$NO_VENV"
#echo "NO_HIP=$NO_HIP"
#echo "NO_CUDA=$NO_CUDA"
#echo "NO_DOCS=$NO_DOCS"

SAVED_ROCM_PATH=${ROCM_PATH}
SAVED_HIP_PLATFORM=${HIP_PLATFORM}

export ROCM_PATH=${ROCM_PATH:-"/opt/rocm"} # adjust accordingly
export HIP_PLATFORM=${HIP_PLATFORM:-"amd"}

# note: [ -z {var+x} ] evaluates to true if `var` is unset!

[ -z ${PRE_CLEAN+x} ] || rm -rf venv

alias PYTHON="python3"
if [ -z ${NO_ENV+x} ]; then
  [ ! -d "venv" ] && python3 -m venv _venv
  alias PYTHON="_venv/bin/python3"
fi
shopt -s expand_aliases

if [ -z ${NO_HIP+x} ]; then
  # build hip-python
  echo "building package hip-python"
  PKG="hip-python"
  mkdir -p ${PKG}/dist/
  mkdir -p ${PKG}/dist/archive
  mv ${PKG}/dist/*.whl ${PKG}/dist/archive/
  mv ${PKG}/dist/*.tar.gz ${PKG}/dist/archive/
  PYTHON -m pip install -r ${PKG}/requirements.txt
  PYTHON -m build ${PKG} -n
fi
  
if [ -z ${NO_CUDA+x} ]; then
  # build hip-python-as-cuda
  echo "building package hip-python-as-cuda"
  PKG="hip-python-as-cuda"
  mkdir -p ${PKG}/dist/
  mkdir -p ${PKG}/dist/archive
  mv ${PKG}/dist/*.whl ${PKG}/dist/archive/
  mv ${PKG}/dist/*.tar.gz ${PKG}/dist/archive/
  PYTHON -m pip install --force-reinstall hip-python/dist/hip*whl
  PYTHON -m pip install -r ${PKG}/requirements.txt
  PYTHON -m build ${PKG} -n
fi

if [ -z ${NO_DOCS+x} ]; then
  echo "building docs for package hip-python"
  # build docs
  PYTHON -m pip install --force-reinstall hip-python/dist/hip*whl \
                                                    hip-python-as-cuda/dist/hip*whl
  PYTHON -m pip install -r hip-python/docs/requirements.txt
  DOCS_DIR="hip-python/docs"
  
  if [ ! -z ${NO_API_DOCS+x} ]; then
     mv "$DOCS_DIR/python_api" "./_python_api"
  fi

  if [ -z ${NO_CLEAN_DOCS+x} ]; then
    PYTHON -m sphinx -j ${NUM_JOBS} -T -E -b html -d _build/doctrees -D language=en ${DOCS_DIR} ${DOCS_DIR}/_build/html
  else
    echo "reuse saved sphinx environment" 
    PYTHON -m sphinx -j ${NUM_JOBS} -T -b html -d _build/doctrees -D language=en ${DOCS_DIR} ${DOCS_DIR}/_build/html
  fi
  
  if [ ! -z ${NO_API_DOCS+x} ]; then
     mv "./_python_api" "$DOCS_DIR/python_api"
  fi
fi

export ROCM_PATH="${SAVED_ROCM_PATH}"
export HIP_PLATFORM="${SAVED_HIP_PLATFORM}"

[ -z ${POST_CLEAN+x} ] || rm -rf venv
