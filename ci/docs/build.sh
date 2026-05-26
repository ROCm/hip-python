#!/usr/bin/env bash
# ci/docs/build.sh — render the hip-python Sphinx documentation.
#
# Usage:  ci/docs/build.sh [<output-html-dir>]
# Default output: <repo>/docs (matches packages/CMakeLists.txt)
#
# Thin wrapper around the cmake docs target — same pipeline cmake
# runs natively, just invokable from environments where typing the
# full cmake command is inconvenient (CI, Read the Docs).
#
# Preconditions (assumed already in place on the checked-out
# branch — the script does no codegen):
#   * packages/rocm-bindings-hip/cmake/generated_versions.cmake
#   * docs_src/sphinx/_toc.yml.in
#   * packages/*/src/rocm/bindings/*.{pyx,pyi}
#
# Requires: python3, cmake, sphinx (+ docs_src/sphinx/requirements.txt).
# cmake is pip-installable, so this works on RTD without system cmake.

set -euo pipefail

# Parallelism: standard cmake env var, default 8. cmake --build
# picks it up automatically AND the patched docs target forwards
# the value to sphinx -j.
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-8}"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD_DIR="${REPO_ROOT}/build/docs"
OUTPUT_DIR="${1:-${REPO_ROOT}/docs}"

cmake -B "${BUILD_DIR}" -S "${REPO_ROOT}/packages" \
  -DHIP_PYTHON_BUILD_DOCS_ONLY=ON \
  -DHIP_PYTHON_DOCS_OUTPUT_DIR="${OUTPUT_DIR}"

cmake --build "${BUILD_DIR}" --target docs
