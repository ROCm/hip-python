#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
   echo "ERROR: script must not be sourced";
   return 1
fi

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
  [ ! -d "venv" ] && python3 -m venv venv
  alias PYTHON="venv/bin/python3"
fi
shopt -s expand_aliases

PYTHON -m pip install -r requirements.txt

HIP_PLATFORM=amd \
HIP_PYTHON_LIBS=* \
ROCM_PATH=/opt/rocm \
HIP_PYTHON_CLANG_RES_DIR=$(${ROCM_PATH}/llvm/bin/clang -print-resource-dir) \
PYTHON codegen_hip_python.py

[ -z ${POST_CLEAN+x} ] || rm -rf venv
