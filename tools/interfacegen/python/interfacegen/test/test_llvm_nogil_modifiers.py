# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
"""Tests for the per-header nogil opt-in of the LLVM-C recipe.

`llvm_c.nogil_node_init` marks the lazy-loader shims of one header as
`nogil`-callable, which is what moves the C call into a `with nogil:`
block. libLLVM is optional at runtime, so the shims must keep raising
when a symbol cannot be resolved -- these tests pin the exception
sentinel picked for each return shape, since a wrong one either fails to
compile or silently swallows the load error.
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))

from _codegen_helpers import make_generator, write_module  # noqa: E402
from interfacegen.support.recipes.rocm import llvm_c  # noqa: E402

_HEADER = """
typedef struct OpaqueModule_st * ModuleRef;
typedef int Bool32;
typedef _Bool BoolC;
typedef enum { STATUS_OK = 0, STATUS_INVALID = 3 } Status;
typedef enum { KIND_A = 0, KIND_B = 1 } Kind;
typedef enum { SIGNED_ERROR = -1, SIGNED_OK = 0 } SignedKind;
typedef struct { void * buffer; unsigned long size; } ObjectBuffer;
typedef void (* Handler)(int code, void * ctx);

ModuleRef parse_module(const char * path);
const char * get_error_message(void);
Bool32 verify_module(ModuleRef m);
BoolC is_object_file(const char * path);
unsigned int api_version(void);
double to_float(ModuleRef m);
void dispose_module(ModuleRef m);
ObjectBuffer get_object(unsigned int index);
Status get_status(ModuleRef m);
Kind get_kind(ModuleRef m);
SignedKind get_signed_kind(ModuleRef m);
void set_handler(Handler handler, void * ctx);
"""

# (function, expected modifiers on the shim declaration)
_EXPECTED_MODIFIERS = (
    ("parse_module", " except? NULL nogil"),
    ("get_error_message", " except? NULL nogil"),
    ("verify_module", " except? -1 nogil"),
    ("is_object_file", " except? -1 nogil"),
    ("api_version", " except? -1 nogil"),
    ("to_float", " noexcept nogil"),
    ("dispose_module", " noexcept nogil"),
    ("get_object", " noexcept nogil"),
    # -1 is out of band for every llvm-c enum, and none of them defines a
    # constant that means "error", so the sentinel is a cast rather than
    # an enumerator.
    ("get_status", " except? <Status>-1 nogil"),
    ("get_kind", " except? <Kind>-1 nogil"),
    # ... unless the enum itself uses -1, which leaves no sentinel.
    ("get_signed_kind", " noexcept nogil"),
    # A callback parameter is the one route by which LLVM could re-enter
    # the caller's code mid-call, so these keep the module default.
    ("set_handler", ""),
)


def _emit(tmp_path, *, nogil: bool):
    gen = make_generator(
        _HEADER,
        module_name="mod",
        runtime_linking=True,
        dll="libLLVM.so",
        node_init=llvm_c.nogil_node_init("input.h") if nogil else None,
    )
    return write_module(gen, tmp_path)


def _decl(pxd: str, name: str) -> str:
    match = re.search(rf"^cdef .*\b{name}\(.*$", pxd, flags=re.MULTILINE)
    assert match, f"declaration of {name} missing from:\n{pxd}"
    return match.group(0)


def _def_block(pyx: str, name: str) -> str:
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
    return "\n".join(lines[start:end])


def test_nogil_headers_are_the_heavyweight_ones():
    for relpath in (
        "llvm-c/Linker.h",
        "llvm-c/TargetMachine.h",
        "llvm-c/Transforms/PassBuilder.h",
        "llvm-c/lto.h",
    ):
        assert llvm_c.is_nogil_header(relpath), relpath
    for relpath in (
        "llvm-c/Core.h",
        "llvm-c/Orc.h",
        "llvm-c/DebugInfo.h",
        "llvm-c/Object.h",
    ):
        assert not llvm_c.is_nogil_header(relpath), relpath


def test_shim_modifiers_follow_the_return_type(tmp_path):
    pxd = _emit(tmp_path, nogil=True)["cymod.pxd"]
    for name, modifiers in _EXPECTED_MODIFIERS:
        decl = _decl(pxd, name)
        assert decl.endswith(modifiers), (
            f"{name}: expected the declaration to end in {modifiers!r}, "
            f"got {decl!r}"
        )


def test_shim_returns_its_sentinel_when_the_symbol_is_missing(tmp_path):
    cy_pyx = _emit(tmp_path, nogil=True)["cymod.pyx"]
    for name, sentinel in (
        ("parse_module", "return NULL"),
        ("verify_module", "return -1"),
        ("get_status", "return <Status>-1"),
        ("get_kind", "return <Kind>-1"),
    ):
        body = cy_pyx.split(f'__init_symbol(&_{name}__funptr,"{name}")')[1]
        first_line = body.splitlines()[1].strip()
        assert first_line == sentinel, (
            f"{name}: expected {sentinel!r} on symbol-load failure, "
            f"got {first_line!r}"
        )
    # Shapes without a sentinel are `noexcept`: Cython aborts the shim at
    # the raising `__init_symbol` rather than falling through to a call
    # via the NULL function pointer, so `pass` is safe here.
    body = cy_pyx.split(
        '__init_symbol(&_dispose_module__funptr,"dispose_module")'
    )[1]
    assert body.splitlines()[1].strip() == "pass"


def test_wrappers_release_the_gil(tmp_path):
    pyx = _emit(tmp_path, nogil=True)["mod.pyx"]
    for name, _ in _EXPECTED_MODIFIERS:
        block = _def_block(pyx, name)
        if name == "set_handler":
            assert "with nogil:" not in block, (
                "callback-taking functions must keep the GIL, got:\n" + block
            )
        else:
            assert (
                "with nogil:" in block
            ), f"{name} must release the GIL, got:\n{block}"


def test_headers_without_the_hook_are_untouched(tmp_path):
    out = _emit(tmp_path, nogil=False)
    for name, _ in _EXPECTED_MODIFIERS:
        # The `__<module>_has_symbol` probe is `noexcept nogil` in every
        # module, so only the function declarations are of interest.
        assert _decl(out["cymod.pxd"], name).endswith(")"), name
    assert "with nogil:" not in out["mod.pyx"]
