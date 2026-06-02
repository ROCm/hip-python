#!/usr/bin/env bash
set -xeu

# Run the unified hip-python example suite against the wheels produced
# by build-hip-python-wheels (sibling pipeline stage). The merged
# hip-python tree ships one examples/ directory covering both the
# former hip-python and rocm-llvm-python example sets; one test
# script is enough.
#
# Required env:
#   SRC_DIR              parent dir that contains hip_python/
#   BUILD_ARTIFACTS_DIR  dir holding the built wheels

project_dir=hip_python

### resolved paths

src_dir=${SRC_DIR}/${project_dir}
build_dir=$(mktemp -d)
test_venv=${TEST_VENV:-$(mktemp -d)}

### venv with test deps

python3 -m venv ${test_venv}
. ${test_venv}/bin/activate
pip install --upgrade pip pytest

cp -av ${src_dir}/examples ${build_dir}/examples

pip install -r ${build_dir}/examples/requirements.txt

### install the built wheels
#
# Install every wheel produced by the build stage. The merged tree
# emits rocm_bindings_*, hip_python, and hip_python_interop wheels.
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

### run tests

export HIP_PYTHON_cudaError_t_HALLUCINATE=1
pytest -v ${build_dir}/examples

deactivate
rm -rf ${test_venv} ${build_dir}
