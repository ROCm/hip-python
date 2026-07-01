#!/usr/bin/env bash
set -xeu

# Run the unified hip-python test suites against the wheels produced by
# the build stage (sibling pipeline stage):
#
#   1. The hip-python example suite (the merged hip-python +
#      rocm-llvm-python examples), run against the rocm_bindings_*,
#      hip_python_interop, and hip_python wheels.
#
#   2. The hip-python-interop pynvml/NVML shim unit tests.
#
#   3. The numba-hip test suite (tests/numba-hip), run against those same
#      wheels plus the numba_hip wheel. numba-hip's tests and CI were
#      folded in here when its standalone packages/numba-hip/ci/ scripts
#      were retired; the unified `all_wheels` target now also builds the
#      numba-hip wheel, so no separate numba-hip build/test stage remains.
#
# All suites share a single venv: their dependency closures only overlap
# on numpy, which the examples leave unpinned, so numba's constraint (via
# numba_hip -> numba<0.64) wins cleanly during resolution.
#
# Required env:
#   SRC_DIR              parent dir that contains hip_python/
#   BUILD_ARTIFACTS_DIR  dir holding the built wheels

project_dir=hip_python
src_dir=${SRC_DIR}/${project_dir}

### -------------------------------------------------------------------
### Shared venv + dependency install
### -------------------------------------------------------------------

examples_build_dir=$(mktemp -d)
test_venv=${TEST_VENV:-$(mktemp -d)}

python3 -m venv ${test_venv}
. ${test_venv}/bin/activate
pip install --upgrade pip pytest cffi

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

# numba-hip depends on the compiler bindings (formerly the
# rocm-llvm-python wheel, now rocm_bindings_compiler) which are already
# installed above.
pip install $(find ${BUILD_ARTIFACTS_DIR}/ -name "numba_hip*.whl")

# Suite env vars. Each only affects its own suite and is harmless to the
# others, so export both up front:
#   - HIP_PYTHON_cudaError_t_HALLUCINATE: hip-python examples.
#   - NUMBA_HIP_FALLBACK_TO_AMDSMI_FOR_UUID: fall back to AMDSMI for GPU
#     UUID retrieval on architectures like gfx1151 (Strix Halo).
export HIP_PYTHON_cudaError_t_HALLUCINATE=1
export NUMBA_HIP_FALLBACK_TO_AMDSMI_FOR_UUID=1

### -------------------------------------------------------------------
### Suite 1 — hip-python examples
### -------------------------------------------------------------------

pytest -v ${examples_build_dir}/examples

### -------------------------------------------------------------------
### Suite 2 — hip-python-interop shim unit tests
### -------------------------------------------------------------------
#
# Mocked unit tests for the hip-python-interop pynvml/NVML shim. These live
# OUTSIDE the importable package (tests/hip-python-interop, not under src/) and
# stub rocm.bindings.amdsmi, so they exercise the *installed* hip_python_interop
# wheel without requiring a GPU.

pytest -v ${src_dir}/tests/hip-python-interop

### -------------------------------------------------------------------
### Suite 3 — rocm-bindings unit tests (core + compiler)
### -------------------------------------------------------------------
#
# GPU-free unit tests for the rocm-bindings-core and rocm-bindings-compiler
# wheels (path resolution, CStr lifetime pinning, comgr enum lookups, the
# libclang loader fallback). Like the suites above they live OUTSIDE the
# importable packages (tests/, not under src/) and exercise the *installed*
# wheels, so they only run once the bindings have been materialized/built.

pytest -v ${src_dir}/tests/rocm-bindings-core
pytest -v ${src_dir}/tests/rocm-bindings-compiler

### -------------------------------------------------------------------
### Suite 4 — numba-hip
### -------------------------------------------------------------------
#
# The tests live OUTSIDE the importable package (tests/numba-hip, not
# under src/) and use only absolute imports, so they exercise the
# *installed* numba.hip. Run them from a directory with no `numba/`
# parent (the examples build dir) so the source tree cannot shadow the
# installed package.

cd ${examples_build_dir}
pytest -v -s ${src_dir}/tests/numba-hip

deactivate
rm -rf ${test_venv} ${examples_build_dir}
