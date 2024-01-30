#!/usr/bin/env bash

# Applies the patch to a conda environment
# Modify and adjust to your own environment.

ENV_ROOT=${CONDA_PREFIX}
PYVER=$(python --version | grep -o "\b3\.[0-9]\+\b")

PATCH_DIR=numba/hip
NUMBA_DIR=${CONDA_PREFIX}/lib/python${PYVER}/site-packages/numba/

if [ ! -d ${NUMBA_DIR} ]; then
  echo "ERROR: Numba package directory does not exist: '${NUMBA_DIR}'" >&2
  exit 1
fi
cp -v -f -R ${PATCH_DIR} ${NUMBA_DIR}
echo ""
echo "Never import numba from this repository's root directory!"
echo ""
