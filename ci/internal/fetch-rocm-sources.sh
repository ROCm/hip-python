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

# Fetch the ROCm component sources the code generation reads.
#
# hip-python's generators consume three repositories next to the hip_python
# tree: rocm-systems (HIP/rccl/roctracer/hipfile headers, and the clr tree the
# hiprtc runtime header is built from), rocm-libraries (the math and vendor
# library headers) and llvm-project (the comgr header and the clang Python
# bindings). The `extraGitRepos` entries of the rocm_python and
# rocm_python_codegen nodes in aiss-3p-dev-pipelines do the same thing.
#
# Run after ci/internal/prepare-container.sh, whose ROCM_VERSION this uses:
#
#   SRC_DIR=$PWD bash ci/internal/fetch-rocm-sources.sh
#
# The clones land at ${SRC_DIR}/{rocm_systems,rocm_libraries,rocm_llvm_project},
# the paths generate-bindings.sh defaults to.
#
# Required env:
#   SRC_DIR         parent directory holding hip_python/; the clones become its
#                   siblings.
#
# Optional env:
#   ROCM_SPECIFIER  default therock:7.14.0. Only the 'therock:' prefix matters
#                   here: it selects which ref naming scheme to derive.
#   ROCM_VERSION    the X.Y.Z the ref is derived from. Defaults to the version
#                   of the installed ROCm, which is what makes a wildcard
#                   specifier like 'therock:*' (or 'therock:?', the spelling
#                   that survives a GitHub comment) resolve to a concrete ref.
#   ROCM_SOURCES_REF
#                   one ref for all three repositories, overriding the derived
#                   one. For a version whose release refs do not exist yet,
#                   'develop' is the usual value; llvm-project, which calls
#                   that branch 'amd-staging', is mapped for you.
#   ROCM_SOURCES_FULL
#                   default false. 'true' checks the repositories out whole
#                   instead of only the paths the generators read.
#   ROCM_SOURCES_GIT_BASE_URL
#                   default https://github.com/ROCm.

### resolved configuration

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

rocm_specifier=${ROCM_SPECIFIER:-therock:7.14.0}
sources_full=${ROCM_SOURCES_FULL:-false}
base_url=${ROCM_SOURCES_GIT_BASE_URL:-https://github.com/ROCm}

rocm_version=${ROCM_VERSION:-}
if [[ -z "${rocm_version}" ]]; then
  # Sourced only for this fallback, so the script also works when the caller
  # already knows the version and ROCm is not on PATH yet.
  . ${script_dir}/librocm.sh
  rocm_version=$(get_rocm_version)
fi

rocm_version_major=$(echo ${rocm_version} | cut -d "." -f 1)
rocm_version_minor=$(echo ${rocm_version} | cut -d "." -f 2)

### the ref, derived from the ROCm specifier

# A TheRock build and a packaged release are cut from differently named refs:
# the TheRock ones carry no patch level ('therock:7.14.0' -> 'therock-7.14'),
# the release ones are the maintenance branch of the patch series
# ('7.2.1' -> 'rocm-7.2.x').
#
# Deliberately a single ref and not the rocm-X.Y.Z -> rocm-X.Y.x ->
# therock-X.Y -> amd-staging fallback chain of the Jenkins nodes: generating
# release bindings from whatever ref happens to exist is worse than failing and
# being told to pass ROCM_SOURCES_REF.
if [[ -n "${ROCM_SOURCES_REF:-}" ]]; then
  ref=${ROCM_SOURCES_REF}
elif [[ "${rocm_specifier}" == "therock:"* ]]; then
  ref=therock-${rocm_version_major}.${rocm_version_minor}
else
  ref=rocm-${rocm_version_major}.${rocm_version_minor}.x
fi

### the paths each generator actually reads

# Justified by tools/hip-python-generate/src/hip_python_codegen/
# binding_generator.py: get_systems_header()/get_libraries_header()/
# get_llvm_header() map every bound header to one of these directories, and
# build_generator_include_paths() adds the same set as -I paths. clr and hip
# are needed whole rather than for headers: generate-bindings.sh copies clr and
# builds its hiprtc-builtins target against HIP_COMMON_DIR=projects/hip.
#
# A header that is missing from a sparse checkout is not fatal -- the generator
# resolves ${ROCM_PATH}/include first anyway and only falls back to the source
# trees -- but it does change what is generated, so keep this list in step with
# binding_generator.py.
function sparse_paths_for() {
  case "${1}" in
    rocm-systems)
      printf "%s" "projects/clr projects/hip projects/hipfile projects/roctracer projects/rccl projects/amdsmi"
      ;;
    rocm-libraries)
      # Down to the include directories rather than the projects: the ten
      # include trees are ~8 MB together, while the projects that contain them
      # are 6.7 GB, almost all of it hipblaslt's Tensile logic and hipsparselt.
      printf "%s" "projects/hipblas/library/include projects/hipblaslt/library/include projects/hipsolver/library/include projects/hiprand/library/include projects/rocrand/library/include projects/hipsparse/library/include projects/hipfft/library/include projects/hiptensor/library/include projects/hipsparselt/library/include projects/hipdnn/backend/include"
      ;;
    llvm-project)
      # LICENSE.TXT sits at the root, which a cone-mode checkout always
      # includes; generate-bindings.sh copies it next to the clang bindings.
      printf "%s" "amd/comgr clang/bindings/python llvm/include"
      ;;
    *)
      echo "ERROR: no sparse path set for '${1}'" >&2
      return 1
      ;;
  esac
}

### headers a cmake configure would have written

# hipTensor's public header includes two headers that only exist after a cmake
# configure: an export header from generate_export_header() and a version
# header from its .in template. No ROCm install covers the gap -- the SDK the
# wheel images carry ships no hipTensor at all, so this checkout is the only
# source the recipe has, and without these the library fails to parse and the
# whole generation reports a failure.
function materialize_hiptensor_headers() {
  local project=${SRC_DIR}/rocm_libraries/projects/hiptensor
  local internal=${project}/library/include/hiptensor/internal
  [[ -d "${internal}" ]] || return 0

  if [[ ! -f "${internal}/hiptensor-export.h" ]]; then
    # What generate_export_header() writes for a shared library built with
    # default visibility.
    cat >"${internal}/hiptensor-export.h" <<'EOF'
#ifndef HIPTENSOR_EXPORT_H
#define HIPTENSOR_EXPORT_H

#define HIPTENSOR_EXPORT __attribute__((visibility("default")))
#define HIPTENSOR_NO_EXPORT __attribute__((visibility("hidden")))
#define HIPTENSOR_DEPRECATED __attribute__((__deprecated__))
#define HIPTENSOR_DEPRECATED_EXPORT HIPTENSOR_EXPORT HIPTENSOR_DEPRECATED
#define HIPTENSOR_DEPRECATED_NO_EXPORT HIPTENSOR_NO_EXPORT HIPTENSOR_DEPRECATED

#endif
EOF
  fi

  if [[ -f "${internal}/hiptensor-version.h" || ! -f "${internal}/hiptensor-version.h.in" ]]; then
    return 0
  fi

  local version
  version=$(sed -n 's/^[[:space:]]*set[[:space:]]*([[:space:]]*VERSION_STRING[[:space:]]*"\([0-9][0-9.]*\)".*/\1/p' \
                   "${project}/CMakeLists.txt" | head -1)
  # HIPTENSOR_{MAJOR,MINOR,PATCH}_VERSION reach the bindings as constants, so a
  # placeholder version would be shipped as fact.
  if [[ -z "${version}" ]]; then
    echo "ERROR: no VERSION_STRING in ${project}/CMakeLists.txt, so the hipTensor version macros cannot be filled in" >&2
    exit 1
  fi
  sed -e "s/@hiptensor_VERSION_MAJOR@/$(echo "${version}" | cut -d. -f1)/" \
      -e "s/@hiptensor_VERSION_MINOR@/$(echo "${version}" | cut -d. -f2)/" \
      -e "s/@hiptensor_VERSION_PATCH@/$(echo "${version}" | cut -d. -f3)/" \
      "${internal}/hiptensor-version.h.in" >"${internal}/hiptensor-version.h"
  echo "[info] hiptensor: wrote the cmake-generated headers, version ${version}"
}

### fetch

for entry in rocm_systems:rocm-systems \
             rocm_libraries:rocm-libraries \
             rocm_llvm_project:llvm-project; do
  dir=${entry%%:*}
  name=${entry#*:}
  target=${SRC_DIR}/${dir}
  url=${base_url}/${name}.git

  # llvm-project calls the branch the other two call 'develop' 'amd-staging',
  # and carries a 'develop' TAG months behind it that `git clone -b` would take
  # without a word. Mapped here rather than in every caller.
  repo_ref=${ref}
  if [[ "${name}" == "llvm-project" && "${repo_ref}" == "develop" ]]; then
    repo_ref=amd-staging
  fi

  rm -rf "${target}"

  if [[ "${sources_full}" == "true" ]]; then
    clone_rc=0
    git clone --depth=1 --single-branch -b "${repo_ref}" "${url}" "${target}" || clone_rc=$?
  else
    # Blobless and checkout-less first: with --filter=blob:none the file
    # contents are only fetched for the paths the sparse checkout selects,
    # which is what keeps these three very large repositories cheap.
    clone_rc=0
    git clone --depth=1 --single-branch --filter=blob:none --no-checkout \
        -b "${repo_ref}" "${url}" "${target}" || clone_rc=$?
    if [[ ${clone_rc} -eq 0 ]]; then
      git -C "${target}" sparse-checkout set --cone $(sparse_paths_for "${name}")
      git -C "${target}" checkout
    fi
  fi

  if [[ ${clone_rc} -ne 0 ]]; then
    message="cannot fetch ${name} at '${repo_ref}' (derived from ROCM_SPECIFIER '${rocm_specifier}' and ROCm ${rocm_version}); pass ROCM_SOURCES_REF to name the ref explicitly"
    if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
      echo "::error::${message}"
    fi
    echo "ERROR: ${message}" >&2
    exit 1
  fi

  sha=$(git -C "${target}" rev-parse HEAD)
  # The committer date too: a ref can resolve to a tag that stopped moving, and
  # the generator would read it without a word of complaint.
  committed=$(git -C "${target}" log -1 --format=%cI 2>/dev/null || true)
  echo "[info] ${name}: ${repo_ref} (${sha}, ${committed:-date unknown}) -> ${target}"
  if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    echo "- ${name}: \`${repo_ref}\` (\`${sha}\`, ${committed:-date unknown})" >>"${GITHUB_STEP_SUMMARY}"
  fi
  du -sh "${target}" || true
done

materialize_hiptensor_headers
