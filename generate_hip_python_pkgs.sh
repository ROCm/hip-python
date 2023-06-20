#!/usr/bin/env bash

[ ! -d "venv" ] && python3 -m venv venv

  venv/bin/python3 -m pip install -r requirements.txt

  HIP_PLATFORM=amd \
  HIP_PYTHON_LIBS=* \
  ROCM_PATH=/opt/rocm \
  HIP_PYTHON_CLANG_RES_DIR=$(${ROCM_PATH}/llvm/bin/clang -print-resource-dir) \
  venv/bin/python3 codegen_hip_python.py
