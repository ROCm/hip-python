#!/usr/bin/env bash
set -xeu

# Run the unified hip-python test suites against the wheels produced by
# the build stage (sibling pipeline stage):
#
#   1. The hip-python example suite (the merged hip-python +
#      rocm-llvm-python examples), run against the rocm_bindings_*,
#      hip_python_interop, and hip_python wheels.
#
#   2. The numba-hip test suite (tests/numba-hip), run against those same
#      wheels plus the numba_hip wheel. numba-hip's tests and CI were
#      folded in here when its standalone packages/numba-hip/ci/ scripts
#      were retired; the unified `all_wheels` target now also builds the
#      numba-hip wheel, so no separate numba-hip build/test stage remains.
#
# Each suite runs in its own fresh venv so their dependency closures
# (examples vs numba) cannot interfere.
#
# Required env:
#   SRC_DIR              parent dir that contains hip_python/
#   BUILD_ARTIFACTS_DIR  dir holding the built wheels

project_dir=hip_python
src_dir=${SRC_DIR}/${project_dir}

### -------------------------------------------------------------------
### Suite 1 — hip-python examples
### -------------------------------------------------------------------

examples_build_dir=$(mktemp -d)
examples_venv=${TEST_VENV:-$(mktemp -d)}

python3 -m venv ${examples_venv}
. ${examples_venv}/bin/activate
pip install --upgrade pip pytest

cp -av ${src_dir}/examples ${examples_build_dir}/examples
pip install -r ${examples_build_dir}/examples/requirements.txt

# Install every hip-python wheel produced by the build stage. The merged
# tree emits rocm_bindings_*, hip_python, and hip_python_interop wheels.
shopt -s nullglob
wheels=(
  ${BUILD_ARTIFACTS_DIR}/rocm_bindings_*.whl
  ${BUILD_ARTIFACTS_DIR}/hip_python_interop*.whl
  ${BUILD_ARTIFACTS_DIR}/hip_python-*.whl
)
shopt -u nullglob
if [ ${#wheels[@]} -eq 0 ]; then
  echo "ERROR: no hip-python wheels found under ${BUILD_ARTIFACTS_DIR}" >&2
  exit 1
fi
pip install "${wheels[@]}"

export HIP_PYTHON_cudaError_t_HALLUCINATE=1
pytest -v ${examples_build_dir}/examples

# Mocked unit tests for the hip-python-interop pynvml/NVML shim. These live
# OUTSIDE the importable package (tests/hip-python-interop, not under src/) and
# stub rocm.bindings.amdsmi, so they exercise the *installed* hip_python_interop
# wheel without requiring a GPU.
pytest -v ${src_dir}/tests/hip-python-interop

deactivate
rm -rf ${examples_venv} ${examples_build_dir}

### -------------------------------------------------------------------
### Suite 2 — numba-hip
### -------------------------------------------------------------------
#
# The tests live OUTSIDE the importable package (tests/numba-hip, not
# under src/) and use only absolute imports, so they exercise the
# *installed* numba.hip. Running them from a directory with no `numba/`
# parent guarantees the source tree cannot shadow the installed package.

numba_venv=$(mktemp -d)

python3 -m venv ${numba_venv}
. ${numba_venv}/bin/activate
pip install --upgrade pip pytest cffi

# numba-hip depends on the compiler bindings (formerly the
# rocm-llvm-python wheel, now rocm_bindings_compiler).
shopt -s nullglob
numba_deps=(
  ${BUILD_ARTIFACTS_DIR}/rocm_bindings_*.whl
  ${BUILD_ARTIFACTS_DIR}/hip_python_interop*.whl
  ${BUILD_ARTIFACTS_DIR}/hip_python-*.whl
)
shopt -u nullglob
pip install "${numba_deps[@]}"
pip install $(find ${BUILD_ARTIFACTS_DIR}/ -name "numba_hip*.whl")

# Enable fallback to AMDSMI for GPU UUID retrieval for architectures like
# gfx1151 (Strix Halo).
export NUMBA_HIP_FALLBACK_TO_AMDSMI_FOR_UUID=1

pytest -v -s ${src_dir}/tests/numba-hip

deactivate
rm -rf ${numba_venv}
