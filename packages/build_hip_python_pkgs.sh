#!/usr/bin/env bash
SAVED_ROCM_PATH=${ROCM_PATH}
SAVED_HIP_PLATFORM=${HIP_PLATFORM}

export ROCM_PATH=${ROCM_PATH:-"/opt/rocm"} # adjust accordingly
export HIP_PLATFORM=${HIP_PLATFORM:-"amd"}

rm -rf venv
python3 -m venv venv

  # install hip-python
  mkdir -p hip-python/dist/
  find hip-python/dist/ -name "*.whl" -o -name "*.tar.gz" -delete
  venv/bin/python3 -m pip install -r hip-python/requirements.txt
  python3 -m build hip-python -n
  
  # install hip-python-as-cuda
  mkdir -p hip-python-as-cuda/dist/
  find hip-python-as-cuda/dist/ -name "*.whl" -o -name "*.tar.gz" -delete
  venv/bin/python3 -m pip install --force-reinstall hip-python/dist/hip*whl
  venv/bin/python3 -m pip install -r hip-python-as-cuda/requirements.txt
  venv/bin/python3 -m build hip-python-as-cuda -n

rm -rf venv

export ROCM_PATH="${SAVED_ROCM_PATH}"
export HIP_PLATFORM="${SAVED_HIP_PLATFORM}"
