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

# ci/docs/regenerate-stubs.sh — regenerate handcoded-Cython .pyi stubs.
#
# Usage:  ci/docs/regenerate-stubs.sh [<target>]
# Default target: all_stubs
# Available targets: all_stubs | <pkg>_stubs (core, hip, ...)
#                  | <cython-target>_stub
#
# Wraps the developer-only HIP_PYTHON_ENABLE_STUBGEN cmake flow. The
# stubs land in the source tree next to their .pyx and should be
# committed.
#
# Requires:
#   * mypy installed in the active Python env (provides `stubgen`),
#   * a working hip_python_initialize() — i.e., /opt/rocm present
#     (the stubgen target depends on the compiled .so of each
#     handcoded cython module, so the wheel build is set up).
#
# Why this isn't part of ci/docs/build.sh: stubgen needs the
# compiled extensions + ROCm + mypy. The docs build does not.
# Keeping them separate so each script has a clean precondition
# story (the docs build runs on Read the Docs; stubgen does not).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD_DIR="${REPO_ROOT}/build/stubs"
TARGET="${1:-all_stubs}"

cmake -B "${BUILD_DIR}" -S "${REPO_ROOT}/packages" \
  -DHIP_PYTHON_ENABLE_STUBGEN=ON

cmake --build "${BUILD_DIR}" --target "${TARGET}"

echo
echo "Regenerated stubs (see git status for the .pyi diff):"
git -C "${REPO_ROOT}" status --short -- '*.pyi' || true
