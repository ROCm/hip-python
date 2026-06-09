#!/usr/bin/env bash
set -xeu

# Test numba-hip against the wheels produced by the build stage.
#
# Installs the hip-python wheels (rocm_bindings_*, hip_python_interop,
# hip_python) and the numba-hip wheel from BUILD_ARTIFACTS_DIR into a fresh
# venv, then runs the test suite.
#
# The tests live OUTSIDE the importable package (in numba_hip/tests, not under
# the src/ source root) and use only absolute imports, so they exercise the
# *installed* numba.hip. Running them from a directory with no `numba/` parent
# guarantees the source tree cannot shadow the installed package.
#
# Required env:
#   SRC_DIR              parent dir that contains numba_hip/
#   BUILD_ARTIFACTS_DIR  dir holding the built wheels

project_dir=numba_hip

### resolved paths

src_dir=${SRC_DIR}/${project_dir}
test_venv=${TEST_VENV:-$(mktemp -d)}

### venv with test deps

python3 -m venv ${test_venv}
. ${test_venv}/bin/activate
pip install --upgrade pip pytest cffi

### install the built wheels
#
# The merged hip-python tree ships rocm_bindings_*, hip_python, and
# hip_python_interop wheels; numba-hip depends on the compiler bindings
# (formerly the rocm-llvm-python wheel, now rocm_bindings_compiler).
shopt -s nullglob
hip_python_wheels=(
  ${BUILD_ARTIFACTS_DIR}/rocm_bindings_*.whl
  ${BUILD_ARTIFACTS_DIR}/hip_python_interop*.whl
  ${BUILD_ARTIFACTS_DIR}/hip_python-*.whl
)
shopt -u nullglob
pip install "${hip_python_wheels[@]}"
pip install $(find ${BUILD_ARTIFACTS_DIR}/ -name "numba_hip*.whl")

### run tests
#
# Enable fallback to AMDSMI for GPU UUID retrieval for architectures like
# gfx1151 (Strix Halo).
export NUMBA_HIP_FALLBACK_TO_AMDSMI_FOR_UUID=1

pytest -v -s ${src_dir}/tests

deactivate
rm -rf ${test_venv}
