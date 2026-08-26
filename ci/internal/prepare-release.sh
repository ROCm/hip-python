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
