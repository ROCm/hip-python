#!/usr/bin/env bash

SAVED_ROCM_PATH=${ROCM_PATH}
SAVED_HIP_PLATFORM=${HIP_PLATFORM}
SAVED_PYTHONPATH="${PYTHONPATH}"

export ROCM_PATH=/opt/rocm # adjust accordingly
export HIP_PLATFORM="amd"
export PYTHONPATH="${PYTHONPATH}:$(pwd)/hip-python/:$(pwd)/hip-python-as-cuda" 

# put local path into PYTHONPATH,
# as `build` will copy only the setup script and package files into a temporary dir

python3 -m build hip-python $@
python3 -m build hip-python-as-cuda $@

export ROCM_PATH="${SAVED_ROCM_PATH}"
export HIP_PLATFORM="${SAVED_HIP_PLATFORM}"
export PYTHONPATH="${SAVED_PYTHONPATH}"
