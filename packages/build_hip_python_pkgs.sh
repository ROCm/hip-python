#!/usr/bin/env bash
SAVED_ROCM_PATH=${ROCM_PATH}
SAVED_HIP_PLATFORM=${HIP_PLATFORM}

export ROCM_PATH=${ROCM_PATH:-"/opt/rocm"} # adjust accordingly
export HIP_PLATFORM=${HIP_PLATFORM:-"amd"}

rm -rf venv
python3 -m venv venv

#  # build hip-python
#  PKG="hip-python"
#  mkdir -p ${PKG}/dist/
#  mkdir -p ${PKG}/dist/archive
#  mv ${PKG}/dist/*.whl ${PKG}/dist/archive/
#  mv ${PKG}/dist/*.tar.gz ${PKG}/dist/archive/
#  venv/bin/python3 -m pip install -r ${PKG}/requirements.txt
#  python3 -m build ${PKG} -n
#  
#  # build hip-python-as-cuda
#  PKG="hip-python-as-cuda"
#  mkdir -p ${PKG}/dist/
#  mkdir -p ${PKG}/dist/archive
#  mv ${PKG}/dist/*.whl ${PKG}/dist/archive/
#  mv ${PKG}/dist/*.tar.gz ${PKG}/dist/archive/
#  venv/bin/python3 -m pip install --upgrade hip-python/dist/hip*whl
#  venv/bin/python3 -m pip install -r ${PKG}/requirements.txt
#  venv/bin/python3 -m build ${PKG} -n

  # build docs
  venv/bin/python3 -m pip install --upgrade hip-python/dist/hip*whl
  venv/bin/python3 -m pip install --upgrade hip-python-as-cuda/dist/hip*whl
  venv/bin/python3 -m pip install -r hip-python/docs/requirements.txt
  ( cd hip-python/docs && source build.sh )


rm -rf venv

export ROCM_PATH="${SAVED_ROCM_PATH}"
export HIP_PLATFORM="${SAVED_HIP_PLATFORM}"
