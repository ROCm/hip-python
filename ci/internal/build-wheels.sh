#!/usr/bin/env bash
set -xeu

# Build hip-python wheels.
#
# Two modes:
#
#   - Full mode (default): builds all six packages — core, hip,
#     libraries, systems, compiler, hip-python-interop.
#
#   - Light mode (LIGHT_MODE=true): builds only core, hip, and
#     compiler. Skips libraries / systems / interop. Use this for
#     a quick CI smoke build that exercises the bulk of the
#     codegen and the heaviest cython compile passes (hip + LLVM)
#     without paying for the math/vendor library wheels.
#
# Steps:
#
#   1. (optional) Run the hip-python codegen against /opt/rocm
#      (+ optional rocm-systems / rocm-libraries source trees) to
#      (re)populate packages/<wheel>/src/rocm/bindings/*.{pxd,pyx,pyi}
#      and cmake/generated_modules.cmake. SKIPPED BY DEFAULT —
#      pass SKIP_CODEGEN=false to enable. CI normally consumes the
#      pre-generated content already committed to the source tree.
#
#   2. Configure + build the enabled wheels with scikit-build-core
#      via the unified `packages/CMakeLists.txt` aggregate target.
#
# Required env:
#   SRC_DIR              parent of hip_python/ AND interfacegen/
#   BUILD_DIR            scratch dir for the working copy
#   BUILD_ARTIFACTS_DIR  where wheels land (HIP_PYTHON_WHEEL_OUTPUT_DIR)
#
# Optional env:
#   ROCM_PATH                 default /opt/rocm
#   ROCM_VERSION              default 7.13.0
#   ROCM_SYSTEMS_DIR          ${SRC_DIR}/rocm-systems if it exists
#   ROCM_LIBRARIES_DIR        ${SRC_DIR}/rocm-libraries if it exists
#   ROCM_LLVM_PROJECT_DIR     ${SRC_DIR}/llvm-project if it exists
#   INTERFACEGEN_DIR          ${SRC_DIR}/interfacegen
#   HIP_PYTHON_CODEGEN_DIR    ${INTERFACEGEN_DIR}/recipes/hip-python
#   MAX_JOBS                  default 16
#   SCCACHE_ENABLE            default false
#   SKIP_CODEGEN              default true (skip step 1; consume
#                             already-generated content). Set to
#                             "false" to re-run codegen.
#   LIGHT_MODE                default false. If "true", build only
#                             core + hip + compiler (skips libraries,
#                             systems, interop).

project_dir=hip_python

### resolved paths

src_dir=${SRC_DIR}/${project_dir}
build_dir=${BUILD_DIR}/${project_dir}
wheels_venv=$(mktemp -d)

rocm_path=${ROCM_PATH:-/opt/rocm}
rocm_version=${ROCM_VERSION:-7.13.0}
interfacegen_dir=${INTERFACEGEN_DIR:-${SRC_DIR}/interfacegen}
hip_python_codegen_dir=${HIP_PYTHON_CODEGEN_DIR:-${interfacegen_dir}/recipes/hip-python}
max_jobs=${MAX_JOBS:-16}

# Default optional source-tree paths to ${SRC_DIR}/<repo> if they exist.
rocm_systems_dir=${ROCM_SYSTEMS_DIR:-}
if [[ -z "${rocm_systems_dir}" && -d "${SRC_DIR}/rocm-systems" ]]; then
  rocm_systems_dir=${SRC_DIR}/rocm-systems
fi
rocm_libraries_dir=${ROCM_LIBRARIES_DIR:-}
if [[ -z "${rocm_libraries_dir}" && -d "${SRC_DIR}/rocm-libraries" ]]; then
  rocm_libraries_dir=${SRC_DIR}/rocm-libraries
fi
rocm_llvm_project_dir=${ROCM_LLVM_PROJECT_DIR:-}
if [[ -z "${rocm_llvm_project_dir}" && -d "${SRC_DIR}/llvm-project" ]]; then
  rocm_llvm_project_dir=${SRC_DIR}/llvm-project
fi

# Light mode trims the package set to core + hip + compiler.
if [[ "${LIGHT_MODE:-false}" == "true" ]]; then
  build_libraries=OFF
  build_systems=OFF
  build_interop=OFF
else
  build_libraries=ON
  build_systems=ON
  build_interop=ON
fi

### prepare working copy

rm -rf ${build_dir}
mkdir -p ${BUILD_DIR}
cp -av ${src_dir} ${build_dir}

### venv with build deps

python3 -m venv ${wheels_venv}
. ${wheels_venv}/bin/activate
# The outer cmake invocation below forces ``-G "Unix Makefiles"`` so the
# huge generated ``.c`` files (hipblas.c is 100+ MB) compile reliably —
# ninja's jobserver-pipe inheritance from the outer make breaks down on
# them and gcc intermittently fails to write the dependency file
# mid-compile. The per-wheel scikit-build-core subprocess runs in its
# own forked context (the inherited jobserver pipe is closed across the
# fork) so ninja can be used safely there, and scikit-build-core
# requires ninja unless ``ninja.make-fallback = true`` is set in the
# package pyproject. Install it explicitly so the per-wheel build step
# doesn't fail with ``Missing dependencies: ninja>=1.5``.
# pyproject-metadata is used by cmake/hip_python_assemble_wheel.py to
# generate each wheel's core metadata from its pyproject.toml (the same
# PEP 621 -> METADATA path scikit-build-core uses). The unified
# all_wheels flow assembles the six compiled wheels from the already
# compiled build output instead of recompiling via `python -m build`;
# see share/design/BUILDING.md.
pip install --upgrade pip auditwheel patchelf \
    build "scikit-build-core>=0.11.2" "cmake>=3.26" "ninja>=1.5" \
    "cython>=3.1.0" setuptools "pyproject-metadata>=0.9"

if [ -d "/opt/rh/gcc-toolset-$(g++ -dumpversion)" ]; then
  toolchain="/opt/rh/gcc-toolset-$(g++ -dumpversion)/root/usr"
  export CCC_OVERRIDE_OPTIONS="+--gcc-toolchain=${toolchain}"
fi

### step 1 — codegen (skipped by default)

if [[ "${SKIP_CODEGEN:-true}" != "true" ]]; then
  # Install the codegen tool. interfacegen is referenced as a path
  # dependency from hip-python-codegen's dev-requirements.txt; the
  # relative `../../../` path resolves only from the recipes dir,
  # so install both explicitly with absolute paths to keep this
  # script invocation-location-independent.
  pip install --upgrade "${interfacegen_dir}"
  pip install --upgrade "${hip_python_codegen_dir}"

  # libclang Python bindings often lag the system libclang.so. The
  # interfacegen typehandler tolerates unknown TypeKind ids by
  # mapping them to UNEXPOSED, but we still want a reasonably new
  # binding so common newer types resolve naturally.
  pip install --upgrade "libclang>=18,<19"

  codegen_args=(
    "${build_dir}"
    --rocm-version "${rocm_version}"
    --rocm-path "${rocm_path}"
    --license-path "${build_dir}/LICENSE"
  )
  [[ -n "${rocm_systems_dir}"      ]] && codegen_args+=(--rocm-systems-dir   "${rocm_systems_dir}")
  [[ -n "${rocm_libraries_dir}"    ]] && codegen_args+=(--rocm-libraries-dir "${rocm_libraries_dir}")
  [[ -n "${rocm_llvm_project_dir}" ]] && codegen_args+=(--rocm-llvm-project-dir "${rocm_llvm_project_dir}")

  hip-python-generate "${codegen_args[@]}"
fi

### step 2 — configure + build the enabled wheels

cd ${build_dir}

cmake_args=(
  # Force Unix Makefiles. Ninja (cmake's default when ninja is on PATH)
  # is faster but unreliable here: scikit-build-core invokes us from
  # within its own jobserver, ninja can't initialize the inherited
  # jobserver pipe, and on the largest generated .c files (hipblas.c
  # is 100+ MB) gcc intermittently fails to write the dependency file
  # mid-compile. Make has no jobserver-inheritance quirk and serializes
  # per-recipe deterministically — slightly slower, much more stable.
  # Override with `CMAKE_GENERATOR=Ninja` env var if you really want it.
  -G "${CMAKE_GENERATOR:-Unix Makefiles}"
  -S packages
  -B packages/build
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_VERBOSE_MAKEFILE=ON
  -DROCM_PATH=${rocm_path}
  -DHIP_PLATFORM=amd
  -DHIP_PYTHON_BUILD_CORE=ON
  -DHIP_PYTHON_BUILD_HIP=ON
  -DHIP_PYTHON_BUILD_LIBRARIES=${build_libraries}
  -DHIP_PYTHON_BUILD_SYSTEMS=${build_systems}
  -DHIP_PYTHON_BUILD_COMPILER=ON
  -DHIP_PYTHON_BUILD_INTEROP=${build_interop}
  -DHIP_PYTHON_AUDITWHEEL_REPAIR=ON
  -DHIP_PYTHON_WHEEL_OUTPUT_DIR=${BUILD_ARTIFACTS_DIR}
)

# Add compiler launcher if sccache is enabled
if [[ "${SCCACHE_ENABLE:-false}" == "true" ]]; then
  cmake_args+=(
    -DCMAKE_C_COMPILER_LAUNCHER=sccache
    -DCMAKE_CXX_COMPILER_LAUNCHER=sccache
  )
fi

cmake "${cmake_args[@]}"

cmake --build packages/build --target all_wheels -j${max_jobs}

deactivate
rm -rf ${wheels_venv}
