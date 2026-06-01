#!/usr/bin/env bash
set -xeu

# Build the numba-hip wheel.
#
# numba-hip is a (currently) pure-Python overlay that installs as `numba.hip`.
# It is built with setuptools via `pip wheel`. A `setup.py` in the project root
# forces a non-pure (platlib) wheel so it co-locates with `numba` on
# lib/lib64-split systems (see numba_hip/setup.py); auditwheel then relabels the
# platform tag to manylinux.
#
# Required env:
#   SRC_DIR              parent dir that contains numba_hip/
#   BUILD_ARTIFACTS_DIR  where the repaired wheel lands
#
# Optional env:
#   BUILD_DIR            scratch dir for the working copy (default: mktemp -d)

project_dir=numba_hip

### resolved paths

src_dir=${SRC_DIR}/${project_dir}
build_dir=${BUILD_DIR:-$(mktemp -d)}
dist_dir=$(mktemp -d)
wheels_venv=$(mktemp -d)

# note: --no-deps & --no-build-isolation prevent pip from installing
#       dependencies in pyproject.toml
WHEEL_ARGS=(
  -v
  --no-deps
  --no-build-isolation
  --disable-pip-version-check
)

### venv with build deps

python3 -m venv ${wheels_venv}
. ${wheels_venv}/bin/activate
pip install --upgrade pip
pip install auditwheel patchelf  # auditwheel requirements
pip install wheel setuptools     # numba-hip build requirements

### prepare working copy + build the wheel

mkdir -p ${build_dir}
cp -av ${src_dir}/. ${build_dir}/

python3 -m pip wheel ${build_dir} -w "${dist_dir}" "${WHEEL_ARGS[@]}"

### relabel the platform tag to manylinux and publish
#
# setup.py builds a platlib (non-pure) wheel so numba-hip co-locates with numba
# (see numba_hip/setup.py), but its content is pure Python with no compiled
# extensions. `auditwheel repair` therefore cannot process it (it errors on a
# platform-tagged wheel that contains no ELF binary), so we instead relabel the
# bare `linux_x86_64` platform tag to portable manylinux with `wheel tags`.
#
# NOTE: once numba-hip gains real compiled extensions, replace this step with
# `auditwheel repair ... -w "${BUILD_ARTIFACTS_DIR}"` so the bundled shared
# libraries are vendored and the manylinux tag reflects their actual deps.
mkdir -p "${BUILD_ARTIFACTS_DIR}"
(
  cd "${dist_dir}"
  wheel_file=$(ls *.whl)
  retagged=$(python3 -m wheel tags \
    --platform-tag manylinux_2_17_x86_64.manylinux2014_x86_64 \
    --remove "${wheel_file}")
  cp "${retagged}" "${BUILD_ARTIFACTS_DIR}/"
)

deactivate
rm -rf ${wheels_venv} ${dist_dir}
