# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.

import os

import clang.cindex


def _configure_libclang():
    """Set the libclang shared library before any tree.* import touches it.

    Default behavior: do **nothing** — let `clang.cindex` auto-load the
    `libclang.so` bundled inside the pip-installed `libclang` package
    (under `clang/native/libclang.so`). This pins the codegen to a
    well-tested libclang version that ships with the Python bindings.

    The previous implementation probed `/opt/rocm/lib/llvm/lib/libclang.so`
    and other system locations, which led to AST-shape mismatches when
    the system libclang was much newer than the pip wheel (the
    well-known anonymous-typedef enum spelling shift in libclang 17+
    showed up here as the `cyhipblas.pxd` `cdef enum hipblasStatus_t:`
    regression).

    Honors `INTERFACEGEN_LIBCLANG` if set, for advanced use only.
    """
    if clang.cindex.Config.loaded:
        return
    env = os.environ.get("INTERFACEGEN_LIBCLANG")
    if env and os.path.exists(env):
        clang.cindex.Config.set_library_file(env)
        return
    # Fall through: clang.cindex.Config auto-detects the libclang.so
    # bundled in the pip `libclang` package via Config.library_path.


_configure_libclang()
