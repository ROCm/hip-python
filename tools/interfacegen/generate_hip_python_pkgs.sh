#!/usr/bin/env bash
HIP_PLATFORM=amd \
HIP_PYTHON_LIBS=* \
ROCM_PATH=/opt/rocm \
HIP_PYTHON_CLANG_RES_DIR=$(${ROCM_PATH}/llvm/bin/clang -print-resource-dir) \
python3 codegen_hip_python.py
