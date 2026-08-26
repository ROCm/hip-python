#!/usr/bin/env bash
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
#                   one. For a version whose release branches do not exist yet,
#                   'amd-staging' is the usual value.
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

### fetch

for entry in rocm_systems:rocm-systems \
             rocm_libraries:rocm-libraries \
             rocm_llvm_project:llvm-project; do
  dir=${entry%%:*}
  name=${entry#*:}
  target=${SRC_DIR}/${dir}
  url=${base_url}/${name}.git

  rm -rf "${target}"

  if [[ "${sources_full}" == "true" ]]; then
    clone_rc=0
    git clone --depth=1 --single-branch -b "${ref}" "${url}" "${target}" || clone_rc=$?
  else
    # Blobless and checkout-less first: with --filter=blob:none the file
    # contents are only fetched for the paths the sparse checkout selects,
    # which is what keeps these three very large repositories cheap.
    clone_rc=0
    git clone --depth=1 --single-branch --filter=blob:none --no-checkout \
        -b "${ref}" "${url}" "${target}" || clone_rc=$?
    if [[ ${clone_rc} -eq 0 ]]; then
      git -C "${target}" sparse-checkout set --cone $(sparse_paths_for "${name}")
      git -C "${target}" checkout
    fi
  fi

  if [[ ${clone_rc} -ne 0 ]]; then
    message="cannot fetch ${name} at '${ref}' (derived from ROCM_SPECIFIER '${rocm_specifier}' and ROCm ${rocm_version}); pass ROCM_SOURCES_REF to name the ref explicitly"
    if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
      echo "::error::${message}"
    fi
    echo "ERROR: ${message}" >&2
    exit 1
  fi

  sha=$(git -C "${target}" rev-parse HEAD)
  echo "[info] ${name}: ${ref} (${sha}) -> ${target}"
  if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    echo "- ${name}: \`${ref}\` (\`${sha}\`)" >>"${GITHUB_STEP_SUMMARY}"
  fi
  du -sh "${target}" || true
done
