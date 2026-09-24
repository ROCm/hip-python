#!/usr/bin/env bash
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
#   3. The rocm-bindings-core and rocm-bindings-compiler unit tests.
#
#   4. The type-information suite (tests/stubs): the hand-maintained
#      `cuda.bindings.cufile` stub against the installed module, and
#      pyright over annotated sample scripts.
#
#   5. The hip compat package suite (tests/hip-python): what
#      `from hip import hip, hiprtc` resolves to, and its stub.
#
#   6. The numba-hip test suite (tests/numba-hip), run against those same
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
#
# Optional env:
#   TEST_VENV               venv to reuse; a temp dir otherwise
#   HIP_PYTHON_PROJECT_DIR  default hip_python. Name of the checkout directory
#                           under SRC_DIR, for CI systems that clone the
#                           repository under its GitHub name

project_dir=${HIP_PYTHON_PROJECT_DIR:-hip_python}
src_dir=${SRC_DIR}/${project_dir}

### -------------------------------------------------------------------
### Shared venv + dependency install
### -------------------------------------------------------------------

examples_build_dir=$(mktemp -d)
test_venv=${TEST_VENV:-$(mktemp -d)}

python3 -m venv ${test_venv}
. ${test_venv}/bin/activate
# pyright drives suite 4's sample scripts. Its wheel is a launcher that
# fetches a node runtime on first use; where that is unavailable the suite
# skips rather than fails.
pip install --upgrade pip pytest cffi pyright

cp -av ${src_dir}/examples ${examples_build_dir}/examples
pip install -r ${examples_build_dir}/examples/requirements.txt

# Install the hip-python wheel set the build stage produced: the merged tree
# emits rocm_bindings_*, hip_python_interop, hip_python and numba_hip.
#
# Named by distribution and version rather than by file, because one
# artifacts directory can hold several tag families side by side — the
# manylinux matrix leaves cp310, cp311 and cp312-abi3 wheels in the same
# pool. Handing pip the filenames would install whichever the glob happened
# to expand to; handing it names lets it pick the wheel whose tag fits the
# interpreter this venv was made from, and report an interpreter with no
# compatible wheel in its own words.
#
# Versions are pinned to what is on disk so a same-named release on the index
# cannot win, while leaving the index reachable, which numpy, numba and the
# rest of the dependency closure still need. numba-hip's dependency on the
# compiler bindings (the former rocm-llvm-python wheel, now
# rocm_bindings_compiler) resolves within this one call.
shopt -s nullglob
wheels=(
  ${BUILD_ARTIFACTS_DIR}/rocm_bindings_*.whl
  ${BUILD_ARTIFACTS_DIR}/hip_python_interop*.whl
  ${BUILD_ARTIFACTS_DIR}/hip_python-*.whl
  ${BUILD_ARTIFACTS_DIR}/numba_hip*.whl
)
shopt -u nullglob
if [ ${#wheels[@]} -eq 0 ]; then
  echo "ERROR: no hip-python wheels found under ${BUILD_ARTIFACTS_DIR}" >&2
  exit 1
fi

# {distribution}-{version}-{python tag}-{abi tag}-{platform tag}.whl, and a
# version may not itself contain a dash, so the first two fields are enough.
# Keyed by distribution, which collapses the tag variants of one package into
# the single requirement they share.
declare -A requirements=()
for wheel in "${wheels[@]}"; do
  wheel_file=$(basename "${wheel}")
  distribution=${wheel_file%%-*}
  wheel_file=${wheel_file#*-}
  version=${wheel_file%%-*}
  requirements[${distribution}]="${distribution}==${version}"
done

pip install --find-links "${BUILD_ARTIFACTS_DIR}" "${requirements[@]}"

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
#
# -rs makes pytest print the reason for every SKIPPED test in the summary
# (e.g. "hipblaslt runtime library not available"). The examples suite
# skips whole example tests when their backing ROCm runtime library is
# absent, so surfacing those reasons keeps CI logs self-explanatory.

pytest -v -rs ${examples_build_dir}/examples

### -------------------------------------------------------------------
### Suite 2 — hip-python-interop shim unit tests
### -------------------------------------------------------------------
#
# Mocked unit tests for the hip-python-interop pynvml/NVML shim. These live
# OUTSIDE the importable package (tests/hip-python-interop, not under src/) and
# stub rocm.bindings.amdsmi, so they exercise the *installed* hip_python_interop
# wheel without requiring a GPU.

pytest -v -rs ${src_dir}/tests/hip-python-interop

### -------------------------------------------------------------------
### Suite 3 — rocm-bindings unit tests (core + compiler)
### -------------------------------------------------------------------
#
# GPU-free unit tests for the rocm-bindings-core and rocm-bindings-compiler
# wheels (path resolution, CStr lifetime pinning, comgr enum lookups, the
# libclang loader fallback). Like the suites above they live OUTSIDE the
# importable packages (tests/, not under src/) and exercise the *installed*
# wheels, so they only run once the bindings have been materialized/built.

pytest -v -rs ${src_dir}/tests/rocm-bindings-core
pytest -v -rs ${src_dir}/tests/rocm-bindings-compiler

### -------------------------------------------------------------------
### Suite 4 — type information
### -------------------------------------------------------------------
#
# `cuda.bindings.cufile` is the one handcoded module whose `.pyi` is
# written by hand instead of generated, so a `.pyx` change can ship
# without the matching stub edit. This suite checks the stub against the
# *installed* extension, which is the artifact users consume.
#
# It then runs pyright over annotated sample scripts, which is what the
# PEP 561 markers and the shipped stubs are for. Without pyright the
# sample cases skip.

pytest -v -rs ${src_dir}/tests/stubs

### -------------------------------------------------------------------
### Suite 5 — hip backward-compatibility package
### -------------------------------------------------------------------
#
# The hip_python wheel re-exports rocm.bindings under the old
# `from hip import hip, hiprtc` spelling. The tests live OUTSIDE the
# importable package (tests/hip-python, not under src/) and cover the shim,
# its hand-maintained stub and the usage pattern applications drive it with.
# The device-touching cases skip themselves where no HIP device is present.

pytest -v -rs ${src_dir}/tests/hip-python

### -------------------------------------------------------------------
### Suite 6 — numba-hip
### -------------------------------------------------------------------
#
# The tests live OUTSIDE the importable package (tests/numba-hip, not
# under src/) and use only absolute imports, so they exercise the
# *installed* numba.hip. Run them from a directory with no `numba/`
# parent (the examples build dir) so the source tree cannot shadow the
# installed package.

cd ${examples_build_dir}
pytest -v -s -rs ${src_dir}/tests/numba-hip

deactivate
rm -rf ${test_venv} ${examples_build_dir}
