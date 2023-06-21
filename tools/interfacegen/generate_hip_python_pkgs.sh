#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
   echo "ERROR: script must not be sourced";
   return 1
fi

HELP_MSG="
Usage: ./$(basename $0) [OPTIONS]

Options:
  --rocm-path       Path to a ROCm installation, defaults to variable 'ROCM_PATH' if set or '/opt/rocm'.
  -l, --libs        HIP Python modules to generate as comma separated list without whitespaces, defaults to variable 'HIP_PYTHON_LIBS' if set or '*'.
  --pre-clean       Remove the virtual Python environment subfolder '_venv' --- if it exists --- before all other tasks.
  --post-clean      Remove the virtual Python environment subfolder '_venv' --- if it exists --- after all other tasks.
  -n, --no-venv     Do not create and use a virtual Python environment.
  -h, --help        Show this help message.
"

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
    -l|--libs)
      HIP_PYTHON_LIBS=$2
      shift; shift
      ;;
    --rocm-path)
      ROCM_PATH=$2
      shift; shift
      ;;
    -h|--help)
      echo "${HELP_MSG}"
      exit 0
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

[ -z ${PRE_CLEAN+x} ] || rm -rf venv

alias PYTHON="python3"
if [ -z ${NO_ENV+x} ]; then
  [ ! -d "venv" ] && python3 -m venv _venv
  alias PYTHON="_venv/bin/python3"
fi
shopt -s expand_aliases

PYTHON -m pip install -r requirements.txt

HIP_PLATFORM=${HIP_PLATFORM:-amd} \
HIP_PYTHON_LIBS=${HIP_PYTHON_LIBS:-*} \
ROCM_PATH=${ROCM_PATH:-/opt/rocm} \
HIP_PYTHON_CLANG_RES_DIR=$(${ROCM_PATH}/llvm/bin/clang -print-resource-dir) \
PYTHON codegen_hip_python.py

[ -z ${POST_CLEAN+x} ] || rm -rf venv
