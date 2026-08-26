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

set -eu

# Run the hip-python test suites once per supported interpreter, against the
# wheel pool the manylinux build matrix produced.
#
# The pool holds three tag families (cp310, cp311, cp312-abi3) and each
# interpreter is served by exactly one of them. Nothing here decides which:
# test.sh asks pip for the distributions by name and pip matches the tags, so
# 3.12 and everything after it land on the abi3 wheels for free. That is the
# point of the abi3 set — 3.13 and 3.14 are tested without being built for.
#
# An interpreter the image does not carry is reported as MISSING and fails the
# run at the end rather than at the point of discovery, so one run still says
# what every other interpreter did. The same reason the loop does not stop at
# the first failing suite: a wheel set broken on one version only is worth
# seeing as such.
#
# Manylinux-specific by construction, like its build-side sibling: the
# interpreters come from /opt/python/cp<xy>-cp<xy>.
#
# Required env:
#   SRC_DIR              parent dir that contains the checkout
#   BUILD_ARTIFACTS_DIR  dir holding the wheels the build matrix produced
#
# Optional env:
#   HIP_PYTHON_TEST_PYTHONS   default "3.10 3.11 3.12 3.13 3.14".
#                             Whitespace-separated CPython versions to test
#   HIP_PYTHON_PROJECT_DIR    passed through to test.sh

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=./libos.sh
. "${script_dir}/libos.sh"

: "${SRC_DIR:?SRC_DIR must be set}"
: "${BUILD_ARTIFACTS_DIR:?BUILD_ARTIFACTS_DIR must be set}"

# test.sh reads these from the environment, and a caller that assigned them
# without exporting would otherwise hand it nothing.
export SRC_DIR BUILD_ARTIFACTS_DIR

test_pythons=${HIP_PYTHON_TEST_PYTHONS:-"3.10 3.11 3.12 3.13 3.14"}

# test.sh removes its own venv when it succeeds; this cleans up after the runs
# that do not get that far.
venv_root=$(mktemp -d)
trap 'rm -rf "${venv_root}"' EXIT

base_path=${PATH}

tested_versions=()
statuses=()

for python_version in ${test_pythons}; do
  echo "=============================================================="
  echo "hip-python test suites on CPython ${python_version}"
  echo "=============================================================="

  tested_versions+=("${python_version}")

  if ! python_bin=$(get_manylinux_python_bin "${python_version}"); then
    statuses+=("MISSING")
    continue
  fi

  if PATH="${python_bin}:${base_path}" \
     TEST_VENV="${venv_root}/cp${python_version//./}" \
       "${script_dir}/test.sh"; then
    statuses+=("PASS")
  else
    statuses+=("FAIL")
  fi
done

if [[ ${#tested_versions[@]} -eq 0 ]]; then
  echo "ERROR: HIP_PYTHON_TEST_PYTHONS is empty; nothing was tested." >&2
  exit 1
fi

echo "=============================================================="
echo "Summary"
echo "=============================================================="

failed=0
for i in "${!tested_versions[@]}"; do
  printf '  CPython %-5s %s\n' "${tested_versions[${i}]}" "${statuses[${i}]}"
  if [[ "${statuses[${i}]}" != "PASS" ]]; then
    failed=$((failed + 1))
  fi
done

if [[ ${failed} -gt 0 ]]; then
  echo "${failed} of ${#tested_versions[@]} interpreters did not pass." >&2
  exit 1
fi

echo "All ${#tested_versions[@]} interpreters passed."
