#!/usr/bin/env bash
set -xeu

# Stage the generated hip-python tree, commit it, and optionally push.
#
# Run AFTER ci/internal/generate-bindings.sh has populated
# ${BUILD_DIR}/hip_python with the generated outputs. This authors the
# release-only VERSION.in (via ci/internal/prepare-release.sh), stages the
# generator outputs (Cython bindings + VERSION.in + the docs-side outputs
# under docs_src/), commits as the configured author, and — when
# HIP_PYTHON_CODEGEN_PUSH=true — force-pushes to the codegen branch.
#
# Required env:
#   BUILD_DIR                    scratch dir; the working copy is at
#                                ${BUILD_DIR}/hip_python
#   ROCM_VERSION                 ROCm version (commit message + VERSION.in)
#   HIP_PYTHON_CODEGEN_GIT_AUTHOR
#                                git author/committer, "Name <email>" format
#
# Optional env:
#   HIP_PYTHON_CODEGEN_PUSH      "true" to push the commit. default false
#   HIP_PYTHON_CODEGEN_BRANCH    target branch for the push (required when
#                                HIP_PYTHON_CODEGEN_PUSH=true)
#   HIP_PYTHON_CODEGEN_REMOTE    remote to push to. default origin. Set this
#                                when the destination is not the repository the
#                                working copy was cloned from — publishing the
#                                same generated tree to another organization,
#                                say — and add the remote before calling this.

build_dir=${BUILD_DIR}/hip_python

cd ${build_dir}
  author_name="${HIP_PYTHON_CODEGEN_GIT_AUTHOR%% <*}"
  author_email="${HIP_PYTHON_CODEGEN_GIT_AUTHOR##*<}"
  author_email="${author_email%>}"

  # Author the release-only VERSION.in template embedding the ROCm version.
  # The script writes "<rocm>.@HIP_PYTHON_VERSION@"; @HIP_PYTHON_VERSION@
  # stays literal for CMake's configure_file to substitute the hardcoded
  # HIP_PYTHON_VERSION at build time.
  bash ci/internal/prepare-release.sh "${ROCM_VERSION}"

  # git add captures the rendered rocm/version.py and the other generator
  # outputs (none are git-ignored; they are simply absent on the codegen
  # base branch). This includes the docs-side outputs under docs_src/
  # (_toc.yml.in + the python_api/*.rst cy* pages) that ci/docs/build.sh
  # consumes without re-running codegen.
  git add packages VERSION.in docs_src/sphinx/_toc.yml.in docs_src/python_api

  git -c user.name="${author_name}" -c user.email="${author_email}" \
      commit --author="${HIP_PYTHON_CODEGEN_GIT_AUTHOR}" \
      -m "[chore] generate bindings for ROCm ${ROCM_VERSION}"

  if [[ "${HIP_PYTHON_CODEGEN_PUSH:-false}" == "true" ]]; then
    remote=${HIP_PYTHON_CODEGEN_REMOTE:-origin}
    git fetch ${remote}
    git push -u ${remote} HEAD:${HIP_PYTHON_CODEGEN_BRANCH} -f
  fi
