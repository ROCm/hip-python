#!/usr/bin/env bash
set -xeu

# Author the release-only VERSION.in template for a hip-python release branch.
#
# Run this on the branch that will become a release branch (typically from the
# codegen publish step), AFTER codegen has rendered the generated bindings
# (including rocm/version.py). It writes:
#
#   VERSION.in = "<rocm_version>.@HIP_PYTHON_VERSION@"
#
# @HIP_PYTHON_VERSION@ is left literal for CMake's configure_file to substitute
# the hardcoded HIP_PYTHON_VERSION at build time.
#
# The rendered rocm/version.py and the other generator outputs are committed on
# the release branch by the publish step's `git add packages` (they are not
# git-ignored — they are simply absent on the codegen base branch). This script
# only adds the VERSION.in template, the point where the ROCm version becomes
# the wheel-version prefix on the release branch. On the codegen base branch
# there is no VERSION.in, so VERSION == HIP_PYTHON_VERSION verbatim.
#
# Usage: prepare-release.sh <rocm_version> [repo_dir]

rocm_version=${1:?usage: prepare-release.sh <rocm_version> [repo_dir]}
repo_dir=${2:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"}

printf '%s.@HIP_PYTHON_VERSION@\n' "${rocm_version}" >"${repo_dir}/VERSION.in"
