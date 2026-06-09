#!/bin/bash
# Runtime verification of the `*const *` initializer bug. Compiles
# each Cython version's generated C to a .so, calls each function,
# and prints whether the returned pointer is the wrapper's heap
# address (OK) or 0x0 (NULL — bug bites at runtime).
#
# Setup is the same as for run_test.sh — install each Cython
# version into /tmp/cython_versions/cy<X.Y.Z>/ via
# `pip install --target` and copy test_const_bug.pyx into the same
# directory. Then:
#
#   PYTHON=python3 CC=gcc CYTHON_VERSIONS="3.0.12 3.1.0 3.1.8 3.2.4" \
#       bash runtime_verify.sh
#
# Override PYTHON / CC / CYTHON_VERSIONS to suit your environment.

set -u

: "${PYTHON:=python3}"
: "${CC:=gcc}"
: "${CYTHON_VERSIONS:=3.0.12 3.1.0 3.1.8 3.2.4}"
: "${CYTHON_TARGET_BASE:=$(pwd)}"

PYINC=$(${PYTHON} -c "import sysconfig; print(sysconfig.get_path('include'))")
PY_EXT_SUFFIX=$(${PYTHON} -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

for v in ${CYTHON_VERSIONS}; do
    echo "=========================================================="
    echo "Cython ${v} runtime verification"
    echo "=========================================================="
    out_c=test_${v}.c
    PYTHONPATH=${CYTHON_TARGET_BASE}/cy${v} \
        ${PYTHON} -m cython -3 -o ${out_c} test_const_bug.pyx > /dev/null 2>&1 \
        || { echo "  cython failed (likely a hard error rejected by this version)"; continue; }
    mkdir -p ${CYTHON_TARGET_BASE}/test_const_${v}
    out_so=${CYTHON_TARGET_BASE}/test_const_${v}/test_const_bug${PY_EXT_SUFFIX}
    ${CC} -fPIC -O2 -I${PYINC} -shared -o ${out_so} ${out_c} 2>&1 | tail -3
    PYTHONPATH=${CYTHON_TARGET_BASE}/test_const_${v} ${PYTHON} -c "
import test_const_bug as m
w = m.W()
# All variants should return the same address — the wrapper's
# malloc'd buffer. NULL means the bug bit at runtime.
for name in ['buggy_const_T_const_pp', 'buggy_T_const_pp',
             'ok_const_T_pp', 'ok_T_pp',
             'workaround_split_const_T_const_pp', 'ok_const_T_p']:
    fn = getattr(m, name)
    p = fn(w)
    ok = 'OK    ' if p != 0 else 'NULL!!'
    print(f'  {ok}  {name:<40} -> 0x{p:016x}')
" 2>&1
done
