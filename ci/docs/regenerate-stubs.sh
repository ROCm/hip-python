#!/usr/bin/env bash
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
