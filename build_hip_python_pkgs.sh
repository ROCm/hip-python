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

set -e
set -o xtrace

HELP_MSG="
Usage: ./build_hip_python_pkgs.sh [OPTIONS]

Options:   
  -c,--checkout        The 'release/rocm-rel-X.Y.Z' branch to checkout out the package source files from. If this option is not used,
                       the user is assumed to checkout the files by himself beforehand.
  --rocm-path          Path to a ROCm installation, defaults to variable 'ROCM_PATH' if set or '/opt/rocm'.
  --libs               HIP Python libraries to build as comma separated list without whitespaces, defaults to variable 'HIP_PYTHON_LIBS' if set or '*'.
                       Add a prefix '^' to NOT build the comma-separated list of libraries that follows but all other libraries.
  --cuda-libs          HIP Python CUDA interop libraries to build as comma separated list without whitespaces, defaults to variable 'HIP_PYTHON_CUDA_LIBS' if set or '*'.
                       Add a prefix '^' to NOT build the comma-separated list of libraries that follows but all other libraries.
  --hip                Build package 'hip-python'.
  --cuda               Build package 'hip-python-as-cuda'.
  --docs               Build the docs.
  --no-api-docs        Temporarily move the 'docs/python_api' subfolder so that sphinx does not see it.
  --no-clean-docs      Do not generate docs from scratch, i.e. don't run sphinx with -E switch.
  --docs-use-testpypi  Get the HIP Python packages for building the docs from Test PyPI.
  --docs-use-pypi      Get the HIP Python packages for building the docs from PyPI.
  --no-archive         Do not put previously created packages into the archive folder.
  --run-tests          Run the tests.
  -j,--num-jobs        Number of build jobs to use. Defaults to 1.
  --pre-clean          Remove the virtual Python environment subfolder '_venv' --- if it exists --- before all other tasks.
  --post-clean         Remove the virtual Python environment subfolder '_venv' --- if it exists --- after all other tasks.
  -n, --no-venv        Do not create and use a virtual Python environment.
  -h, --help           Show this help message.
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
    --libs)
      HIP_PYTHON_LIBS=$2
      shift; shift
      ;;
    --cuda-libs)
      HIP_PYTHON_CUDA_LIBS=$2
      shift; shift
      ;;
    -h|--help)
      echo "${HELP_MSG}"
      exit 0
      ;;
    -c|--checkout)
      BRANCH=$2
      shift; shift
      ;;
    --rocm-path)
      ROCM_PATH=$2
      shift; shift
      ;;
    --hip)
      HIP=1
      shift
      ;;
    --cuda)
      CUDA=1
      shift
      ;;
    --docs)
      DOCS=1
      shift
      ;;
    --run-tests)
      RUN_TESTS=1
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
    --docs-use-pypi)
      DOCS_USE_PYPI=1
      shift
      ;;
    --docs-use-testpypi)
      DOCS_USE_TESTPYPI=1
      shift
      ;;
    --no-archive)
      NO_ARCHIVE_OLD_PACKAGES=1
      shift
      ;;
    -j|--num-jobs)
      NUM_JOBS=$2
      shift; shift
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

declare -x ROCM_PATH=${ROCM_PATH:-/opt/rocm}
declare -x HIP_PLATFORM=${HIP_PLATFORM:-amd}
declare -x HIP_PYTHON_LIBS=${HIP_PYTHON_LIBS:-*}
declare -x HIP_PYTHON_CUDA_LIBs=${HIP_PYTHON_CUDA_LIBs:-*}

# note: [ -z {var+x} ] evaluates to true if 'var' is unset!

[ -z ${PRE_CLEAN+x} ] || rm -rf _venv

if [ ! -z ${BRANCH+x} ]; then
  rm -rf hip-python/
  rm -rf hip-python-as-cuda/
  git checkout ${BRANCH} -- hip-python
  git checkout ${BRANCH} -- hip-python-as-cuda
fi

alias PYTHON="python3"
PYTHON_PATH="python3"
if [ -z ${NO_VENV+x} ]; then
  [ ! -d "_venv" ] && python3 -m venv _venv
  alias PYTHON="$(pwd)/_venv/bin/python3"
  PYTHON_PATH="$(pwd)/_venv/bin/python3"
fi
shopt -s expand_aliases

if [ ! -z ${HIP+x} ]; then
  # build hip-python
  echo "building package hip-python"
  PKG="hip-python"
  mkdir -p ${PKG}/dist/
  mkdir -p ${PKG}/dist/archive
  if [ -z ${NO_ARCHIVE_OLD_PACKAGES+x} ]; then
    mv ${PKG}/dist/*.whl ${PKG}/dist/archive/    2> /dev/null || true
    mv ${PKG}/dist/*.tar.gz ${PKG}/dist/archive/ 2> /dev/null || true
  fi
  PYTHON -m pip install -r ${PKG}/requirements.txt
  PYTHON _render_update_version.py
  # PYTHON -m build ${PKG} -n
  cd ${PKG}
  PYTHON setup.py build_ext -j ${NUM_JOBS} bdist_wheel
  cd ..
fi
  
if [ ! -z ${CUDA+x} ]; then
  # build hip-python-as-cuda
  echo "building package hip-python-as-cuda"
  PKG="hip-python-as-cuda"
  mkdir -p ${PKG}/dist/
  mkdir -p ${PKG}/dist/archive
  if [ -z ${NO_ARCHIVE_OLD_PACKAGES+x} ]; then
    mv ${PKG}/dist/*.whl ${PKG}/dist/archive/    2> /dev/null || true
    mv ${PKG}/dist/*.tar.gz ${PKG}/dist/archive/ 2> /dev/null || true
  fi
  for f in $(find hip-python/dist -maxdepth 1 -name "*.whl"); do
    PYTHON -m pip install --force-reinstall $f || true
  done
  PYTHON -m pip install -r ${PKG}/requirements.txt
  PYTHON _render_update_version.py
  # PYTHON -m build ${PKG} -n
  cd ${PKG}
  PYTHON setup.py build_ext -j ${NUM_JOBS} bdist_wheel
  cd ..
fi

if [ ! -z ${DOCS+x} ]; then
  echo "building docs for package hip-python"
  # build docs
  if [ ! -z ${DOCS_USE_PYPI+x} ]; then
    echo "docs: obtaining hip-python and hip-python-as-cuda from PyPI"
    PYTHON -m pip install --force-reinstall hip-python hip-python-as-cuda
  elif [ ! -z ${DOCS_USE_TESTPYPI+x} ]; then
    echo "docs: obtaining hip-python and hip-python-as-cuda from Test PyPI"
    PYTHON -m pip install -i https://test.pypi.org/simple --force-reinstall hip-python hip-python-as-cuda
  else
    for f in $(find hip-python/dist -maxdepth 1 -name "*.whl"); do
      PYTHON -m pip install --force-reinstall $f || true
    done
    for f in $(find hip-python-as-cuda/dist -maxdepth 1 -name "*.whl"); do
      PYTHON -m pip install --force-reinstall $f || true
    done
  fi
  DOCS_DIR="docs"
  PYTHON -m pip install -r ${DOCS_DIR}/requirements.txt
  
  if [ ! -z ${NO_API_DOCS+x} ]; then
     mv "${DOCS_DIR}/python_api" "./_python_api"
  fi

  if [ -z ${NO_CLEAN_DOCS+x} ]; then
    PYTHON -m sphinx -j ${NUM_JOBS} -T -E -b html -d _build/doctrees -D language=en ${DOCS_DIR} ${DOCS_DIR}/_build/html
  else
    echo "reuse saved sphinx environment" 
    PYTHON -m sphinx -j ${NUM_JOBS} -T -b html -d _build/doctrees -D language=en ${DOCS_DIR} ${DOCS_DIR}/_build/html
  fi
  
  if [ ! -z ${NO_API_DOCS+x} ]; then
     mv "./_python_api" "${DOCS_DIR}/python_api"
  fi
fi

if [ ! -z ${RUN_TESTS+x} ]; then
  for f in $(find hip-python/dist -maxdepth 1 -name "*.whl"); do
    PYTHON -m pip install --force-reinstall $f || true
  done
  for f in $(find hip-python-as-cuda/dist -maxdepth 1 -name "*.whl"); do
    PYTHON -m pip install --force-reinstall $f || true
  done
  declare -x HIP_PYTHON_cudaError_t_HALLUCINATE=1
  PYTHON -m pytest -v examples
fi

[ -z ${POST_CLEAN+x} ] || rm -rf _venv
