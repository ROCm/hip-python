# Cython codegen regression: `cdef T x = <T>expr` silently drops the
# initializer when T contains the inner `*const *` pattern (Cython
# 3.0.x; fixed in 3.1.0). Each function body is one statement; the
# check is whether the C output contains an assignment to `x` and
# whether the runtime call returns the wrapper's non-NULL pointer.

from libc.stdlib cimport malloc, free


cdef class W:
    cdef void* _ptr
    def __cinit__(self):
        self._ptr = malloc(8)
        (<long*>self._ptr)[0] = 0xdeadbeef
    def __dealloc__(self):
        if self._ptr:
            free(self._ptr)
    cdef void* getPtr(self):
        return self._ptr


# CASE 1 (BUG): const T *const * — the dangerous shape.
def buggy_const_T_const_pp(W w):
    cdef const char *const * x = <const char *const *>w.getPtr()
    return <long>x


# CASE 2 (BUG): T *const * — same inner-const-on-pointer shape, no
# leftmost const.
def buggy_T_const_pp(W w):
    cdef void *const * x = <void *const *>w.getPtr()
    return <long>x


# CASE 3 (OK): const T ** — only outer const-on-pointee, no inner const.
def ok_const_T_pp(W w):
    cdef const char ** x = <const char **>w.getPtr()
    return <long>x


# CASE 4 (OK): T ** — no const at all.
def ok_T_pp(W w):
    cdef void ** x = <void **>w.getPtr()
    return <long>x


# CASE 5 (WORKAROUND): split form — bare cdef + separate assignment.
# The bug is in the cdef-with-initializer position; a stand-alone
# assignment statement is unaffected.
def workaround_split_const_T_const_pp(W w):
    cdef const char *const * x
    x = <const char *const *>w.getPtr()
    return <long>x


# CASE 6 (OK): const T * (single-pointer) — sanity check, no inner const.
def ok_const_T_p(W w):
    cdef const char * x = <const char *>w.getPtr()
    return <long>x
