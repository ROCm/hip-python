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

# Prepare a fresh container for a hip-python build, test or codegen run.
#
# Turns a stock manylinux image into the environment the other ci/internal
# scripts expect: the requested CPython as python3, ROCm installed under
# ${ROCM_PATH}, and the ROCm/HIP environment exported. It is the counterpart of
# the `manylinux` and `rocm` node init phases in aiss-3p-dev-pipelines, and it
# uses the same two libraries (ci/internal/libos.sh, ci/internal/librocm.sh) so
# a GitHub Actions run and a Jenkins run install ROCm the same way.
#
# Run once per job, after the checkout and before any other ci/internal script:
#
#   ROCM_SPECIFIER=therock:7.14.0 bash ci/internal/prepare-container.sh
#
# Under GitHub Actions the resolved environment is appended to ${GITHUB_ENV}
# and ${GITHUB_PATH}, so subsequent steps see it. Everywhere else, source the
# emitted file:
#
#   . ${ROCM_ENV_FILE:-/tmp/rocm-env.sh}
#
# Optional env:
#   ROCM_SPECIFIER    default therock:7.14.0. See ci/internal/librocm.sh for the
#                     accepted forms; 'therock:X.Y.Z' installs the TheRock
#                     tarball, 'X.Y.Z' the repo.radeon.com packages,
#                     'preinstalled' keeps whatever the image ships.
#   AMDGPU_TARGETS    default gfx90a (the MI210 of the internal runner). A
#                     single target; picks the TheRock artifact group.
#   ROCM_PATH         default /opt/rocm.
#   ROCM_PKGS         packages for the non-TheRock path; empty means the 'rocm'
#                     meta package.
#   MANYLINUX_DEFAULT_PYTHON_VERSION
#                     default 3.12. Which /opt/python interpreter becomes
#                     python3. Ignored off manylinux.
#   ROCM_ENV_FILE     where the sourceable exports are written; default
#                     ${RUNNER_TEMP:-/tmp}/rocm-env.sh.

### resolved configuration

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

. ${script_dir}/libos.sh
. ${script_dir}/librocm.sh

export ROCM_SPECIFIER=${ROCM_SPECIFIER:-therock:7.14.0}
export AMDGPU_TARGETS=${AMDGPU_TARGETS:-gfx90a}
export ROCM_PATH=${ROCM_PATH:-/opt/rocm}
export ROCM_PKGS=${ROCM_PKGS:-}

manylinux_python_version=${MANYLINUX_DEFAULT_PYTHON_VERSION:-3.12}
rocm_env_file=${ROCM_ENV_FILE:-${RUNNER_TEMP:-/tmp}/rocm-env.sh}

os_id=$(. /etc/os-release; echo ${ID})
os_id_like=$(. /etc/os-release; echo ${ID_LIKE:-})

### step 1 - the few tools the image lacks

# wget: install_therock_from_tarball downloads with it. ssh: the codegen
# workflow pushes over it. git: manylinux ships one, other images may not.
missing=()
command -v wget >/dev/null 2>&1 || missing+=(wget)
command -v git >/dev/null 2>&1 || missing+=(git)
if ! command -v ssh >/dev/null 2>&1; then
  if [[ "${os_id}" == "ubuntu" || "${os_id}" == "debian" ]]; then
    missing+=(openssh-client)
  else
    missing+=(openssh-clients)
  fi
fi

if [[ ${#missing[@]} -gt 0 ]]; then
  if command -v dnf >/dev/null 2>&1; then
    dnf install -y "${missing[@]}"
  elif command -v apt-get >/dev/null 2>&1; then
    apt-get update
    apt-get install -y --no-install-recommends "${missing[@]}"
  else
    echo "ERROR: cannot install ${missing[*]}: neither dnf nor apt-get found" >&2
    exit 1
  fi
fi

### step 2 - make the requested interpreter python3

# The manylinux images ship every CPython under /opt/python and no default
# python3. build-wheels.sh, test.sh and generate-bindings.sh all call `python3`
# and `python3 -m venv`, so one of them has to be selected here.
python_bin_dir=
if [ $(is_almalinux_manylinux) ]; then
  set_almalinux_manylinux_python3 "${manylinux_python_version}"
  python_bin_dir=$(get_almalinux_manylinux_python3_bin_dir "${manylinux_python_version}")
  # Console scripts of pip-installed packages land here, which is how
  # generate-bindings.sh finds `hip-python-generate` after installing it.
  export PATH="${python_bin_dir}:${PATH}"
else
  echo "[info] not a manylinux image; using the python3 the image provides"
fi
python3 --version

### step 3 - install ROCm

install_rocm # via ${script_dir}/librocm.sh

# The generator derives its clang resource dir from ${ROCM_PATH}/llvm. TheRock
# tarballs bundle LLVM (${ROCM_PATH}/llvm -> lib/llvm), the packages split it
# into a separate development package -- the same distinction the
# rocm_python_codegen node makes in its prepare phase.
if [[ "${ROCM_SPECIFIER}" != "therock:"* ]]; then
  if ! "${ROCM_PATH}/llvm/bin/clang" -print-resource-dir >/dev/null 2>&1; then
    if command -v dnf >/dev/null 2>&1; then
      dnf install -y rocm-llvm-devel libzstd-devel
    elif command -v apt-get >/dev/null 2>&1; then
      apt-get install -y --no-install-recommends rocm-llvm-dev libzstd-dev
    fi
  fi
fi

### step 4 - export the resolved environment

# Read back from the installation rather than trusting the specifier: a
# wildcard specifier has no version in it at all, and a tarball can carry a
# different patch level than the one that was asked for.
rocm_version=$(get_rocm_version)
rocm_version_short=$(get_rocm_version_short)
rocm_version_major=$(echo ${rocm_version} | cut -d "." -f 1)
rocm_version_minor=$(echo ${rocm_version} | cut -d "." -f 2)
rocm_version_patch=$(echo ${rocm_version} | cut -d "." -f 3)

requested_version=${ROCM_SPECIFIER#therock:}
if [[ "${requested_version}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ && "${requested_version}" != "${rocm_version}" ]]; then
  message="ROCM_SPECIFIER '${ROCM_SPECIFIER}' names ROCm ${requested_version} but ${ROCM_PATH} reports ${rocm_version}"
  if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
    echo "::warning::${message}"
  else
    echo "[warn] ${message}"
  fi
fi

# The variable set of the rocm_base node's init-exports.env, so that a build
# driven from here and one driven from Jenkins see the same environment.
{
  echo "export ROCM_PATH=${ROCM_PATH}"
  echo "export ROCM_HOME=${ROCM_PATH}"
  echo "export ROCM_SPECIFIER=${ROCM_SPECIFIER}"
  echo "export ROCM_VERSION=${rocm_version}"
  echo "export ROCM_VERSION_SHORT=${rocm_version_short}"
  echo "export ROCM_VERSION_MAJOR=${rocm_version_major}"
  echo "export ROCM_VERSION_MINOR=${rocm_version_minor}"
  echo "export ROCM_VERSION_PATCH=${rocm_version_patch}"
  echo "export AMDGPU_TARGETS=${AMDGPU_TARGETS}"
  echo "export HIP_PLATFORM=amd"
  echo "export HIP_HIPCC_EXECUTABLE=${ROCM_PATH}/bin/hipcc"
  echo "export LD_LIBRARY_PATH=${ROCM_PATH}/lib:\${LD_LIBRARY_PATH:-}"
  echo "export CMAKE_PREFIX_PATH=\${CMAKE_PREFIX_PATH:+\${CMAKE_PREFIX_PATH}:}${ROCM_PATH}/lib/cmake"
  echo "export PATH=${python_bin_dir:+${python_bin_dir}:}${ROCM_PATH}/bin:${ROCM_PATH}/lib/llvm/bin:\${PATH}"
  if [[ "${ROCM_SPECIFIER}" == "therock:"* ]]; then
    # NOTE: HIP_DEVICE_LIB_PATH does not apply to a package-based ROCm install.
    echo "export THEROCK_BIN_DIR=${ROCM_PATH}/bin/"
    echo "export HIP_PATH=${ROCM_PATH}"
    echo "export HIP_DEVICE_LIB_PATH=${ROCM_PATH}/lib/llvm/amdgcn/bitcode/"
  fi
} >"${rocm_env_file}"

cat "${rocm_env_file}"

if [[ -n "${GITHUB_ENV:-}" ]]; then
  # GITHUB_ENV takes bare assignments and expands nothing, so the values that
  # extend an existing variable are resolved here.
  {
    echo "ROCM_PATH=${ROCM_PATH}"
    echo "ROCM_HOME=${ROCM_PATH}"
    echo "ROCM_SPECIFIER=${ROCM_SPECIFIER}"
    echo "ROCM_VERSION=${rocm_version}"
    echo "ROCM_VERSION_SHORT=${rocm_version_short}"
    echo "ROCM_VERSION_MAJOR=${rocm_version_major}"
    echo "ROCM_VERSION_MINOR=${rocm_version_minor}"
    echo "ROCM_VERSION_PATCH=${rocm_version_patch}"
    echo "AMDGPU_TARGETS=${AMDGPU_TARGETS}"
    echo "HIP_PLATFORM=amd"
    echo "HIP_HIPCC_EXECUTABLE=${ROCM_PATH}/bin/hipcc"
    echo "LD_LIBRARY_PATH=${ROCM_PATH}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    echo "CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH:+${CMAKE_PREFIX_PATH}:}${ROCM_PATH}/lib/cmake"
    if [[ "${ROCM_SPECIFIER}" == "therock:"* ]]; then
      echo "THEROCK_BIN_DIR=${ROCM_PATH}/bin/"
      echo "HIP_PATH=${ROCM_PATH}"
      echo "HIP_DEVICE_LIB_PATH=${ROCM_PATH}/lib/llvm/amdgcn/bitcode/"
    fi
  } >>"${GITHUB_ENV}"
fi

if [[ -n "${GITHUB_PATH:-}" ]]; then
  # One entry per line, each prepended to PATH by the runner.
  {
    echo "${ROCM_PATH}/lib/llvm/bin"
    echo "${ROCM_PATH}/bin"
    if [[ -n "${python_bin_dir}" ]]; then
      echo "${python_bin_dir}"
    fi
  } >>"${GITHUB_PATH}"
fi

if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
  {
    echo "- ROCm: \`${rocm_version}\` from \`${ROCM_SPECIFIER}\` (${AMDGPU_TARGETS})"
    echo "- python: \`$(python3 --version 2>&1)\`, glibc \`$(get_glibc_version)\`, gcc \`$(get_gcc_version)\`"
  } >>"${GITHUB_STEP_SUMMARY}"
fi
