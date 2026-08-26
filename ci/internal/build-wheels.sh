#!/usr/bin/env bash
set -xeu

# Build hip-python wheels.
#
# Two modes:
#
#   - Full mode (default): builds all six compiled packages — core, hip,
#     libraries, systems, compiler, hip-python-interop — plus the two
#     pure-Python ones, hip-python and numba-hip.
#
#   - Light mode (LIGHT_MODE=true): builds only core, hip, and
#     compiler. Skips libraries / systems / interop. Use this for
#     a quick CI smoke build that exercises the bulk of the
#     codegen and the heaviest cython compile passes (hip + LLVM)
#     without paying for the math/vendor library wheels.
#
# Steps:
#
#   1. (optional) Install the in-tree codegen tool so the
#      `hip-python-generate` console script is on PATH. Codegen itself
#      now runs at CMake CONFIGURE time (step 2) via
#      -DHIP_PYTHON_RUN_CODEGEN=ON, which (re)populates
#      packages/<wheel>/src/rocm/bindings/*.{pxd,pyx,pyi} and
#      cmake/generated_modules.cmake before the build graph is created.
#      SKIPPED BY DEFAULT — pass SKIP_CODEGEN=false to enable. CI
#      normally consumes the pre-generated content committed to the tree.
#
#   2. Configure + build the enabled wheels with scikit-build-core
#      via the unified `packages/CMakeLists.txt` aggregate target. When
#      codegen is enabled the configure call BLOCKS while it runs
#      (several minutes up to ~30 min). `all_wheels` also builds
#      numba-hip.
#
# Required env:
#   SRC_DIR              parent of hip_python/ (interfacegen + the codegen
#                        tool now live inside hip_python/tools/)
#   BUILD_DIR            scratch dir for the working copy
#   BUILD_ARTIFACTS_DIR  where wheels land (HIP_PYTHON_WHEEL_OUTPUT_DIR)
#
# Optional env:
#   ROCM_PATH                 default /opt/rocm
#   ROCM_VERSION              default 7.13.0
#   ROCM_SYSTEMS_DIR          ${SRC_DIR}/rocm-systems if it exists
#   ROCM_LIBRARIES_DIR        ${SRC_DIR}/rocm-libraries if it exists
#   ROCM_LLVM_PROJECT_DIR     ${SRC_DIR}/llvm-project if it exists
#   INTERFACEGEN_DIR          ${src_dir}/tools/interfacegen
#   HIP_PYTHON_CODEGEN_DIR    ${src_dir}/tools/hip-python-generate
#   MAX_JOBS                  default 16
#   SCCACHE_ENABLE            default false
#   SKIP_CODEGEN              default true (skip step 1; consume
#                             already-generated content). Set to
#                             "false" to re-run codegen.
#   LIGHT_MODE                default false. If "true", build only
#                             core + hip + compiler (skips libraries,
#                             systems, interop).
#   USE_SABI                  default "no". When set to a CPython version
#                             (e.g. "3.11"), build limited-API (abi3) wheels
#                             against the CPython stable ABI using that value
#                             as the abi3 floor (forwarded as
#                             -DHIP_PYTHON_ABI3_FLOOR). Independent of the
#                             active build interpreter, but the floor must be
#                             <= the active Python and >= 3.11 (the bindings
#                             use the buffer protocol, which is only in the
#                             stable ABI since CPython 3.11). "no" disables it.

project_dir=hip_python

### resolved paths

src_dir=${SRC_DIR}/${project_dir}
build_dir=${BUILD_DIR}/${project_dir}
wheels_venv=$(mktemp -d)

rocm_path=${ROCM_PATH:-/opt/rocm}
rocm_version=${ROCM_VERSION:-7.13.0}
interfacegen_dir=${INTERFACEGEN_DIR:-${src_dir}/tools/interfacegen}
hip_python_codegen_dir=${HIP_PYTHON_CODEGEN_DIR:-${src_dir}/tools/hip-python-generate}
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

# Stable-ABI (abi3) floor. USE_SABI carries the abi3 floor version (e.g.
# "3.11") or "no" to disable. When a version is given, forward it to CMake as
# HIP_PYTHON_ABI3_FLOOR so the compiled extensions are built against the
# CPython stable ABI and the wheels are tagged cp<floor>-abi3.
abi3_floor=""
if [[ "${USE_SABI:-no}" != "no" ]]; then
  if [[ ! "${USE_SABI}" =~ ^3\.[0-9]{2}$ ]]; then
    echo "ERROR: USE_SABI must be \"no\" or a CPython floor version matching 3.[0-9][0-9] (e.g. \"3.11\"); got \"${USE_SABI}\"." >&2
    exit 1
  fi
  abi3_floor=${USE_SABI}
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

### step 1 — install the codegen tool (skipped by default)
#
# Codegen now runs at CMake configure time (step 2) via
# -DHIP_PYTHON_RUN_CODEGEN=ON. Here we only need to make the
# `hip-python-generate` console script importable/on PATH so the
# configure-time find_program(hip-python-generate) succeeds.

run_codegen=false
if [[ "${SKIP_CODEGEN:-true}" != "true" ]]; then
  run_codegen=true

  # Install the codegen tool. interfacegen is referenced as a path
  # dependency from hip-python-codegen's dev-requirements.txt; the
  # relative path resolves only from the recipe dir, so install both
  # explicitly with absolute paths to keep this script
  # invocation-location-independent.
  pip install --upgrade "${interfacegen_dir}"
  pip install --upgrade "${hip_python_codegen_dir}"

  # libclang Python bindings often lag the system libclang.so. The
  # interfacegen typehandler tolerates unknown TypeKind ids by
  # mapping them to UNEXPOSED, but we still want a reasonably new
  # binding so common newer types resolve naturally.
  pip install --upgrade "libclang>=18,<19"
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
  # Build a self-contained libLLVM.so from the LLVM static archives
  # (--whole-archive) so static-only symbols such as
  # LLVMInitializeAllTargetInfos are exported; the system shared
  # libLLVM.so omits them and numba.hip fails at dlsym otherwise.
  -DHIP_PYTHON_FORCE_BUILD_LIBLLVM=ON
  -DHIP_PYTHON_AUDITWHEEL_REPAIR=ON
  -DHIP_PYTHON_WHEEL_OUTPUT_DIR=${BUILD_ARTIFACTS_DIR}
)

# Forward the abi3 floor when stable-ABI builds are requested.
if [[ -n "${abi3_floor}" ]]; then
  cmake_args+=(-DHIP_PYTHON_ABI3_FLOOR=${abi3_floor})
fi

# Add compiler launcher if sccache is enabled
if [[ "${SCCACHE_ENABLE:-false}" == "true" ]]; then
  cmake_args+=(
    -DCMAKE_C_COMPILER_LAUNCHER=sccache
    -DCMAKE_CXX_COMPILER_LAUNCHER=sccache
  )
fi

# Configure-time codegen: when enabled the `cmake` call below BLOCKS
# while hip-python-generate runs (several minutes up to ~30 min) and
# regenerates the sources + cmake/generated_modules.cmake before any
# target is created, so the subsequent single `cmake --build` compiles
# the freshly generated set in one pass.
if [[ "${run_codegen}" == "true" ]]; then
  cmake_args+=(
    -DHIP_PYTHON_RUN_CODEGEN=ON
    -DHIP_PYTHON_ROCM_PATH=${rocm_path}
    -DHIP_PYTHON_ROCM_VERSION=${rocm_version}
  )
  [[ -n "${rocm_systems_dir}"      ]] && cmake_args+=(-DHIP_PYTHON_ROCM_SYSTEMS_DIR=${rocm_systems_dir})
  [[ -n "${rocm_libraries_dir}"    ]] && cmake_args+=(-DHIP_PYTHON_ROCM_LIBRARIES_DIR=${rocm_libraries_dir})
  [[ -n "${rocm_llvm_project_dir}" ]] && cmake_args+=(-DHIP_PYTHON_ROCM_LLVM_PROJECT_DIR=${rocm_llvm_project_dir})
fi

cmake "${cmake_args[@]}"

cmake --build packages/build --target all_wheels -j${max_jobs}

deactivate
rm -rf ${wheels_venv}
