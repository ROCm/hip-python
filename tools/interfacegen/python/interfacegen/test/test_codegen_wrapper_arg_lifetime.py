# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
"""Regression tests for the lifetime of an argument's wrapper temporary.

When a parameter arrives as a Python list, the generated wrapper allocates
the C array through a ``ListOf*`` adapter and passes the adapter's pointer
to the C function. The pointer only stays valid while the adapter is alive,
and Cython drops an intermediate object as soon as the object itself is no
longer needed -- which is the moment ``getPtr()`` returns, one line *before*
the C call in the generated C. So the adapter has to be bound to a local.

Both emitters must do that. The with-gil emitter used to inline the whole
``ListOfPointer.fromPyobj(params).getPtr()`` chain into the call on the
theory that holding the GIL kept the temporary alive; it does not, and the
LLVM bindings (the only ones the with-gil emitter produces) shipped a
use-after-free because of it. ``LLVMFunctionType`` received a freed
``void *[2]``, glibc had already overwritten both slots with tcache
bookkeeping, and the first ``LLVMGetParam`` on the resulting function type
dereferenced junk and took the interpreter down with SIGSEGV.
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))

from _codegen_helpers import make_generator, write_module  # noqa: E402

# A list of pointers in, like LLVMFunctionType's ParamTypes.
_HEADER = """
typedef struct OpaqueType_st * TypeRef;

TypeRef function_type(TypeRef ret_type, TypeRef *param_types,
                      unsigned int param_count, int is_var_arg);
"""

# An IntEnum in, like LLVMSetLinkage's Linkage — the plain-hoist path,
# where the hoisted expression is a Python attribute lookup rather than
# a wrapper temporary.
_ENUM_HEADER = """
typedef enum { LINKAGE_EXTERNAL = 0, LINKAGE_INTERNAL = 1 } Linkage;

void set_linkage(Linkage linkage);
"""


def _emit(
    tmp_path, *, module_name: str, nogil: bool, header: str = _HEADER
) -> str:
    gen = make_generator(
        header,
        module_name=module_name,
        modifiers_lazy_loader=" noexcept nogil" if nogil else "",
    )
    return write_module(gen, tmp_path)[f"{module_name}.pyx"]


def _def_block(pyx: str, name: str = "function_type") -> str:
    """Return the ``def <name>`` block without its docstring."""
    lines = pyx.splitlines()
    start = next(
        (i for i, line in enumerate(lines) if line.startswith(f"def {name}(")),
        None,
    )
    assert start is not None, f"def {name}(...) missing from:\n{pyx}"
    end = next(
        (
            i
            for i in range(start + 1, len(lines))
            if lines[i].strip() and not lines[i].startswith((" ", "\t"))
        ),
        len(lines),
    )
    body = "\n".join(lines[start:end])
    # Drop the docstring, which quotes the C signature and the wrapper
    # type names the assertions below look for.
    return re.sub(r'r?""".*?"""', "", body, flags=re.DOTALL)


def _assert_wrapper_outlives_call(body: str) -> None:
    # Which adapter the recipe picks for the array is not the point here;
    # that it is bound to a local before the call is.
    bound = re.search(
        r"cdef\s+(\S+)\s+(\S+_obj)\s*=\s*\1\.fromPyobj\(param_types\)", body
    )
    assert bound, (
        "the array adapter must be bound to a local so it outlives the C "
        f"call, got:\n{body}"
    )
    obj_name = bound.group(2)
    assert (
        f"{obj_name}.getPtr()" in body
    ), f"the array pointer must be taken from {obj_name}, got:\n{body}"
    # The freed-pointer form: adapter constructed and dropped inside the
    # call expression.
    assert not re.search(r"fromPyobj\(param_types\)\.getPtr\(\)", body), (
        "the adapter is still constructed inline in the call, so its array "
        f"is freed before the callee reads it:\n{body}"
    )


def test_with_gil_emitter_binds_adapter_to_local(tmp_path):
    _assert_wrapper_outlives_call(
        _def_block(_emit(tmp_path, module_name="mod_gil", nogil=False))
    )


def test_nogil_emitter_binds_adapter_to_local(tmp_path):
    _assert_wrapper_outlives_call(
        _def_block(_emit(tmp_path, module_name="mod_nogil", nogil=True))
    )


def _assert_enum_arg_hoisted(body: str, module_name: str) -> None:
    hoisted = re.search(
        rf"cdef\s+cy{module_name}\.Linkage\s+(\S+)\s*=\s*linkage\.value", body
    )
    assert (
        hoisted
    ), f"the IntEnum arg must be bound to a typed local, got:\n{body}"
    assert (
        f"set_linkage({hoisted.group(1)})" in body
    ), f"the cy* call must reference the hoisted local, got:\n{body}"


def test_with_gil_emitter_hoists_enum_arg(tmp_path):
    """The plain-hoist path is hoisted in the with-gil emitter too.

    Nothing here can dangle -- ``linkage.value`` yields an ``int``, not
    a borrowed pointer -- but the codegen has a single rendering for
    every Python-derived argument, so the shape is the same as for the
    array adapter above.
    """
    _assert_enum_arg_hoisted(
        _def_block(
            _emit(
                tmp_path,
                module_name="mod_gil_enum",
                nogil=False,
                header=_ENUM_HEADER,
            ),
            name="set_linkage",
        ),
        "mod_gil_enum",
    )


def test_nogil_emitter_hoists_enum_arg(tmp_path):
    _assert_enum_arg_hoisted(
        _def_block(
            _emit(
                tmp_path,
                module_name="mod_nogil_enum",
                nogil=True,
                header=_ENUM_HEADER,
            ),
            name="set_linkage",
        ),
        "mod_nogil_enum",
    )
