#!/usr/bin/env bash
set -xeu

# Build + install hipFILE (AMD Infinity Storage GPU-direct I/O) from the
# fetched rocm-systems tree into ${ROCM_PATH} so the hip-python *systems*
# wheel's `find_package(hipfile)` succeeds and the `rocm.bindings.hipfile`
# / `rocm.bindings.cyhipfile` Cython modules get built.
#
# This is the FIRST build stage of the rocm_python pipeline: it must run
# BEFORE ci/internal/build-wheels.sh, because that script's CMake configure
# calls `find_package(hipfile QUIET)` (packages/rocm-bindings-systems/
# CMakeLists.txt) and only builds the hipfile bindings when hipFILE is
# already installed under ${ROCM_PATH}.
#
# Steps:
#   1. Install build dependencies (libmount-devel is REQUIRED to compile
#      libhipfile.so; boost-devel only matters if tests/examples are on).
#   2. Configure + build + install hipFILE into ${ROCM_PATH} with the AMD
#      HIP toolchain, tests/examples OFF (keeps the CI build lean and Boost
#      out of the critical path).
#   3. Verify the install artifacts + the exported CMake package, and run
#      ais-check non-fatally (an AIS-incapable builder must not fail here).
#
# Required env:
#   SRC_DIR   parent of hip_python/ and the fetched rocm-systems tree.
#
# Optional env:
#   ROCM_PATH            default /opt/rocm (also the install prefix).
#   ROCM_VERSION         default 7.13.0 (informational; logged only).
#   ROCM_SYSTEMS_DIR     default ${SRC_DIR}/rocm_systems (falls back to
#                        ${SRC_DIR}/rocm-systems).
#   HIPFILE_SRC_DIR      default ${ROCM_SYSTEMS_DIR}/projects/hipfile.
#   HIPFILE_BUILD_DIR    default ${HIPFILE_SRC_DIR}/build.
#   HIPFILE_GPU_TARGETS  default "gfx90a;gfx942;gfx1100" (CMAKE_HIP_ARCHITECTURES).
#   MAX_JOBS             default 16.

### resolved paths

rocm_path=${ROCM_PATH:-/opt/rocm}
rocm_version=${ROCM_VERSION:-7.13.0}
max_jobs=${MAX_JOBS:-16}

rocm_systems_dir=${ROCM_SYSTEMS_DIR:-}
if [[ -z "${rocm_systems_dir}" ]]; then
  if [[ -d "${SRC_DIR}/rocm_systems" ]]; then
    rocm_systems_dir=${SRC_DIR}/rocm_systems
  elif [[ -d "${SRC_DIR}/rocm-systems" ]]; then
    rocm_systems_dir=${SRC_DIR}/rocm-systems
  else
    echo "ERROR: could not locate the rocm-systems tree; set ROCM_SYSTEMS_DIR." >&2
    exit 1
  fi
fi

hipfile_src_dir=${HIPFILE_SRC_DIR:-${rocm_systems_dir}/projects/hipfile}
hipfile_build_dir=${HIPFILE_BUILD_DIR:-${hipfile_src_dir}/build}
hipfile_gpu_targets=${HIPFILE_GPU_TARGETS:-gfx90a;gfx942;gfx1100}

echo "[build-hipfile] ROCm ${rocm_version} | src=${hipfile_src_dir} | prefix=${rocm_path}"

### step 1 — build dependencies
#
# libmount-devel: REQUIRED — src/amd_detail/mountinfo.cpp includes
#   <libmount/libmount.h>.
# boost-devel: only needed when tests/examples are enabled (program_options);
#   installed defensively so flipping BUILD_TESTING/AIS_INSTALL_EXAMPLES back
#   on doesn't break. On Debian/Ubuntu builders the equivalents are
#   `libmount-dev` and `libboost-program-options-dev`.
if command -v dnf >/dev/null 2>&1; then
  dnf install -y libmount-devel boost-devel || \
    echo "[build-hipfile] WARN: dnf dependency install failed (already present / not root?); continuing"
elif command -v apt-get >/dev/null 2>&1; then
  apt-get update && apt-get install -y libmount-dev libboost-program-options-dev || \
    echo "[build-hipfile] WARN: apt dependency install failed; continuing"
fi

### toolchain

# ROCm's amdclang++ cannot find libstdc++/pthread on the manylinux_2_28
# (AlmaLinux 8) builders unless pointed at the gcc-toolset. Mirrors
# ci/internal/build-wheels.sh.
if [ -d "/opt/rh/gcc-toolset-$(g++ -dumpversion)" ]; then
  toolchain="/opt/rh/gcc-toolset-$(g++ -dumpversion)/root/usr"
  export CCC_OVERRIDE_OPTIONS="+--gcc-toolchain=${toolchain}"
fi

# EL8/manylinux_2_28 ships a glibc (2.28) whose headers predate two symbols
# hipFILE uses (kernel is new enough, only the userspace headers lag):
#   * SYS_pidfd_open      — x86_64 syscall 434 (src/amd_detail/sys.cpp)
#   * F_SEAL_FUTURE_WRITE — fcntl seal 0x0010  (src/amd_detail/stats.cpp)
# Define them if the toolchain doesn't. `-pthread` is required so the
# std::thread users (MT stress tests + library) link. On newer distros whose
# headers already define these, hipFILE has no -Werror on macro redefinition
# so the compile is unaffected.
hipfile_compat_flags="-pthread"
case "$(uname -m)" in
  x86_64)
    hipfile_compat_flags="-DSYS_pidfd_open=434 -DF_SEAL_FUTURE_WRITE=0x0010 ${hipfile_compat_flags}"
    ;;
esac

### step 2 — configure + build + install

rm -rf "${hipfile_build_dir}"
cmake -S "${hipfile_src_dir}" -B "${hipfile_build_dir}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_PLATFORM=amd \
  -DCMAKE_CXX_COMPILER=amdclang++ \
  -DCMAKE_HIP_ARCHITECTURES="${hipfile_gpu_targets}" \
  -DCMAKE_CXX_FLAGS="${hipfile_compat_flags}" \
  -DCMAKE_HIP_FLAGS="${hipfile_compat_flags}" \
  -DCMAKE_EXE_LINKER_FLAGS="-pthread" \
  -DROCM_PATH="${rocm_path}" \
  -DCMAKE_INSTALL_PREFIX="${rocm_path}" \
  -DBUILD_TESTING=OFF \
  -DAIS_INSTALL_EXAMPLES=OFF

cmake --build "${hipfile_build_dir}" -j"${max_jobs}"
cmake --install "${hipfile_build_dir}"

### step 3 — verify

test -f "${rocm_path}/lib/libhipfile.so"            || { echo "ERROR: libhipfile.so missing" >&2; exit 1; }
test -f "${rocm_path}/include/hipfile.h"             || { echo "ERROR: hipfile.h missing" >&2; exit 1; }
test -f "${rocm_path}/lib/cmake/hipfile/hipfile-config.cmake" || { echo "ERROR: hipfile CMake package missing (find_package would fail)" >&2; exit 1; }
echo "[build-hipfile] installed libhipfile.so + hipfile.h + CMake package OK"

# ais-check reports P2PDMA/HIP/amdgpu support; non-fatal (build stage may run
# on a host without an AIS-capable NVMe / P2PDMA-enabled kernel).
"${rocm_path}/bin/ais-check" || echo "[build-hipfile] ais-check reported unsupported capabilities (non-fatal)"
