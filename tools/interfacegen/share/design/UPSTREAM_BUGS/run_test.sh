#!/bin/bash
# Cross-version Cython codegen probe for the `*const *` initializer
# bug. For each Cython version: cythonize ./test_const_bug.pyx → C,
# then grep the generated C for the assignment to `__pyx_v_x` in
# each function. Reports ASSIGNED (initializer emitted) or
# **NO ASSIGNMENT** (bug bites) per function.
#
# Setup (do once before running):
#
#   mkdir -p /tmp/cython_versions && cd /tmp/cython_versions
#   for v in 3.0.12 3.1.0 3.1.8 3.2.4; do
#     python3 -m pip install --target=cy${v} "cython==${v}"
#   done
#   cp <this-dir>/test_const_bug.pyx /tmp/cython_versions/
#
# Then run from /tmp/cython_versions:
#
#   PYTHON=python3 CYTHON_VERSIONS="3.0.12 3.1.0 3.1.8 3.2.4" bash run_test.sh
#
# Override PYTHON / CYTHON_VERSIONS to suit your environment.

set -u

: "${PYTHON:=python3}"
: "${CYTHON_VERSIONS:=3.0.12 3.1.0 3.1.8 3.2.4}"
: "${CYTHON_TARGET_BASE:=$(pwd)}"

for v in ${CYTHON_VERSIONS}; do
    echo "=========================================================="
    echo "Cython ${v}"
    echo "=========================================================="
    out_c=test_${v}.c
    log=test_${v}.warnings
    PYTHONPATH=${CYTHON_TARGET_BASE}/cy${v} \
        ${PYTHON} -m cython -3 -o ${out_c} test_const_bug.pyx 2>&1 | tee ${log}
    echo
    echo "--- assignments to __pyx_v_x in C output ---"
    for fn in buggy_const_T_const_pp buggy_T_const_pp ok_const_T_pp ok_T_pp \
              workaround_split_const_T_const_pp ok_const_T_p; do
        # Find the function block, then look for `__pyx_v_x =` inside it.
        block=$(awk -v fn="${fn}" '
            /^static PyObject .__pyx_pf_/ && index($0, fn) { in_fn=1 }
            in_fn { print }
            in_fn && /^}/ { in_fn=0 }
        ' ${out_c})
        n=$(echo "${block}" | grep -cE '__pyx_v_x[[:space:]]*=[[:space:]]*[^;[:space:]]')
        if [ "${n}" -ge 1 ]; then
            printf '  %-44s ASSIGNED (%d times)\n' "${fn}" "${n}"
        else
            printf '  %-44s **NO ASSIGNMENT**\n' "${fn}"
        fi
    done
    echo
    echo "--- warnings ---"
    grep -E "warning|referenced before assignment" ${log} || echo "  (none)"
    echo
done
