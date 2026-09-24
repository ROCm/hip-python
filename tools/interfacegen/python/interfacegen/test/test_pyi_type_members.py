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

"""The .pyi must declare what the .pyx binds: enum constants, record
properties, and the hoisted anonymous records under the names the module
exports them as.

A stub that declares the class but none of its members is worse than no
stub at all -- `hipError_t.hipSuccess` then reads as an error, and the
one way out is the checker-specific suppression comment this suite
exists to make unnecessary.
"""

import re

from interfacegen.test._codegen_helpers import make_generator, write_module

HEADER = r"""
typedef enum myStatus_t { MY_OK = 0, MY_FAIL = 1 } myStatus_t;

enum { MY_LOOSE_FLAG = 4 };

typedef struct myRecord_t { int count; float ratio; } myRecord_t;

typedef union myUnion_t { int i; float f; } myUnion_t;

typedef struct myFirst_t { struct { int a; } nested; } myFirst_t;

typedef struct mySecond_t { struct { int b; } nested; } mySecond_t;

struct myOpaque_t;
"""


def _emitted(tmp_path):
    return write_module(make_generator(HEADER), tmp_path)


def _class_body(stub_text: str, name: str) -> str:
    """The lines of one class stub, up to the next top-level statement."""
    match = re.search(rf"^class {name}\b.*?(?=^\S)", stub_text, re.M | re.S)
    assert match, f"no class {name} in the stub"
    return match.group(0)


def test_enum_carries_its_constants(tmp_path):
    stub = _emitted(tmp_path)["mod.pyi"]
    assert "class myStatus_t(enum.IntEnum):" in stub
    body = _class_body(stub, "myStatus_t")
    assert "    MY_OK: int" in body
    assert "    MY_FAIL: int" in body
    # enum.IntEnum is what makes the constants members; the import has to
    # come with them.
    assert "import enum" in stub


def test_enum_has_no_placeholder_init(tmp_path):
    """A checker derives the constructor from `enum.IntEnum`."""
    body = _class_body(_emitted(tmp_path)["mod.pyi"], "myStatus_t")
    assert "__init__" not in body


def test_an_anonymous_enum_stubs_as_module_level_constants(tmp_path):
    stub = _emitted(tmp_path)["mod.pyi"]
    assert "MY_LOOSE_FLAG: int" in stub
    assert "'MY_LOOSE_FLAG'," in stub
    # libclang spells the type `enum (unnamed at input.h:4:1)`; there is
    # no class for it in the .pyx either.
    assert "class enum" not in stub


def test_record_declares_the_properties_the_pyx_renders(tmp_path):
    emitted = _emitted(tmp_path)
    body = _class_body(emitted["mod.pyi"], "myRecord_t")
    assert "    count: Any" in body
    assert "    ratio: Any" in body
    # The stub and PROPERTIES() are driven off one generator, so what the
    # class returns at runtime is what the stub declares.
    properties = re.search(
        r"def PROPERTIES\(\):\s*\n\s*return \[([^\]]*)\]", emitted["mod.pyx"]
    )
    assert properties
    assert properties.group(1) == '"count","ratio"'


def test_record_declares_its_fixed_method_set(tmp_path):
    body = _class_body(_emitted(tmp_path)["mod.pyi"], "myRecord_t")
    for method in (
        "PROPERTIES",
        "fromObj",
        "allocate",
        "c_sizeof",
        "as_c_void_p",
        "__contains__",
        "__getitem__",
    ):
        assert f"def {method}(" in body


def test_a_union_gets_no_item_access(tmp_path):
    """`__contains__`/`__getitem__` are emitted for structs only."""
    body = _class_body(_emitted(tmp_path)["mod.pyi"], "myUnion_t")
    assert "    i: Any" in body
    assert "__getitem__" not in body


def test_an_incomplete_record_stays_opaque(tmp_path):
    body = _class_body(_emitted(tmp_path)["mod.pyi"], "myOpaque_t")
    assert "def __init__(self, *args, **kwargs): ..." in body
    assert "PROPERTIES" not in body


def test_hoisted_records_keep_their_parents_apart(tmp_path):
    """Both nested structs are `struct_0` locally; the module has one name
    for each, and the stub must use it or one silently obscures the other.
    """
    stub = _emitted(tmp_path)["mod.pyi"]
    assert "class myFirst_t_struct_0:" in stub
    assert "class mySecond_t_struct_0:" in stub
    assert "class struct_0:" not in stub
    assert "    a: Any" in _class_body(stub, "myFirst_t_struct_0")
    assert "    b: Any" in _class_body(stub, "mySecond_t_struct_0")
