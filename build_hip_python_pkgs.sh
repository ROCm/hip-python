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
Usage: ./build_hip_python_pkgs.sh [OPTIONS]

Options:
  --rocm-path            Path to a ROCm installation, defaults to variable 'ROCM_PATH' if set or '/opt/rocm'.
  --libs                 HIP Python libraries to build as comma separated list without whitespaces, defaults to variable 'HIP_PYTHON_LIBS' if set or '*'.
                         Add a prefix '^' to NOT build the comma-separated list of libraries that follows but all other libraries.
  --cuda-libs            HIP Python CUDA interop libraries to build as comma separated list without whitespaces, defaults to variable 'HIP_PYTHON_CUDA_LIBS' if set or '*'.
                         Add a prefix '^' to NOT build the comma-separated list of libraries that follows but all other libraries.
  --no-hip               Do not build package 'hip-python'.
  --no-cuda              Do not build package 'hip-python-as-cuda'.
  --no-docs              Do not build the docs of package 'hip-python'.
  --no-api-docs          Temporarily move the 'hip-python/docs/python_api' subfolder so that sphinx does not see it.
  --no-clean-docs        Do not generate docs from scratch, i.e. don't run sphinx with -E switch.
  --docs-use-testpypi    Get the HIP Python packages for building the docs from Test PyPI.
  --docs-use-pypi        Get the HIP Python packages for building the docs from PyPI.
  --run-tests            Run the tests.
  -j,--num-jobs          Number of build jobs to use (currently only applied for building docs). Defaults to 1.
  --pre-clean            Remove the virtual Python environment subfolder '_venv' --- if it exists --- before all other tasks.
  --post-clean           Remove the virtual Python environment subfolder '_venv' --- if it exists --- after all other tasks.
  -n, --no-venv          Do not create and use a virtual Python environment.
  -h, --help             Show this help message.
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

# note: [ -z {var+x} ] evaluates to true if `var` is unset!

[ -z ${PRE_CLEAN+x} ] || rm -rf _venv

alias PYTHON="python3"
PYTHON_PATH="python3"
if [ -z ${NO_VENV+x} ]; then
  [ ! -d "_venv" ] && python3 -m venv _venv
  alias PYTHON="$(pwd)/_venv/bin/python3"
  PYTHON_PATH="$(pwd)/_venv/bin/python3"
fi
shopt -s expand_aliases

if [ -z ${NO_HIP+x} ]; then
  # build hip-python
  echo "building package hip-python"
  PKG="hip-python"
  mkdir -p ${PKG}/dist/
  mkdir -p ${PKG}/dist/archive
  mv ${PKG}/dist/*.whl ${PKG}/dist/archive/    2> /dev/null
  mv ${PKG}/dist/*.tar.gz ${PKG}/dist/archive/ 2> /dev/null
  PYTHON -m pip install -r ${PKG}/requirements.txt
  PYTHON _render_update_version.py
  PYTHON -m build ${PKG} -n
fi
  
if [ -z ${NO_CUDA+x} ]; then
  # build hip-python-as-cuda
  echo "building package hip-python-as-cuda"
  PKG="hip-python-as-cuda"
  mkdir -p ${PKG}/dist/
  mkdir -p ${PKG}/dist/archive
  mv ${PKG}/dist/*.whl ${PKG}/dist/archive/    2> /dev/null
  mv ${PKG}/dist/*.tar.gz ${PKG}/dist/archive/ 2> /dev/null
  PYTHON -m pip install --force-reinstall hip-python/dist/hip*whl
  PYTHON -m pip install -r ${PKG}/requirements.txt
  PYTHON _render_update_version.py
  PYTHON -m build ${PKG} -n
fi

if [ -z ${NO_DOCS+x} ]; then
  echo "building docs for package hip-python"
  # build docs
  if [ ! -z ${DOCS_USE_PYPI+x} ]; then
    echo "docs: obtaining hip-python and hip-python-as-cuda from PyPI"
    PYTHON -m pip install --force-reinstall hip-python hip-python-as-cuda
  elif [ ! -z ${DOCS_USE_TESTPYPI+x} ]; then
    echo "docs: obtaining hip-python and hip-python-as-cuda from Test PyPI"
    PYTHON -m pip install -i https://test.pypi.org/simple --force-reinstall hip-python hip-python-as-cuda
  else
    PYTHON -m pip install --force-reinstall hip-python/dist/hip*whl \
                                                    hip-python-as-cuda/dist/hip*whl
    PYTHON -m pip install -r hip-python/docs/requirements.txt
  fi
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

if [ ! -z ${RUN_TESTS+x} ]; then
  PYTHON -m pip install --force-reinstall hip-python/dist/hip*whl \
                                          hip-python-as-cuda/dist/hip*whl
  cd hip-python/examples/0_*
  PYTHON -m pip install -r requirements.txt
  PYTHON hip_deviceattributes.py
  PYTHON hip_deviceproperties.py
  PYTHON hip_python_device_array.py
  PYTHON hip_stream.py
  PYTHON hipblas_with_numpy.py
  PYTHON hipfft.py
  #PYTHON hiprand_monte_carlo_pi.py
  PYTHON hiprtc_launch_kernel_args.py
  PYTHON hiprtc_launch_kernel_no_args.py
  PYTHON rccl_comminitall_bcast.py
  # 10x ok

  cd ../1_*
  PYTHON -m pip install -r requirements.txt
  PYTHON cuda_stream.py
  PYTHON=${PYTHON_PATH} make clean run
  # 2x ok
fi

[ -z ${POST_CLEAN+x} ] || rm -rf _venv
