# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
"""Regression test for `_persist_shim_header` and the
`_apply_header_workarounds` round-trip.

Two upstream-broken ROCm headers (hipblaslt.h, hipsparselt.h)
need their includes patched. The codegen patches them in-memory
for libclang's parse via `unsaved_files`. But the wheel-build's
gcc compile of the Cython-generated `.c` would re-`#include` the
unpatched on-disk header without a corresponding patch on disk.

`_persist_shim_header` writes the patched content to a known
location under the wheel's source tree
(`<output_dir>/packages/<pkg>/shim_includes/<header_relpath>`),
which the package CMakeLists prepends to the include search path
so gcc resolves against the shim copy first.

These tests pin both sides of the round-trip:
  - workaround application produces well-formed patched content
    for both hipblaslt and hipsparselt.
  - the persist helper writes that content to the expected
    on-disk path, creating intermediate directories as needed.
"""

import os
import textwrap

import pytest

from hip_python_codegen.binding_generator import (
    _apply_header_workarounds,
    _neutralize_rocrand_c_fallback,
    _persist_shim_header,
    SHIM_INCLUDES_SUBDIR,
)


# The C-mode fallback block from ROCm 7.14 rocrand.h that collides with
# HIP's uint4 under a C compile (see
# share/design/UPSTREAM_BUGS/rocrand_duplicate_uint4_definition.md).
_ROCRAND_FALLBACK_SRC = textwrap.dedent(
    """\
    #define ROCRAND_DEFAULT_MAX_BLOCK_SIZE 256

    #if defined(__cplusplus)
        #include <hip/hip_fp16.h>
        #include <hip/hip_runtime.h>
        #include <hip/hip_vector_types.h>
    #else
        #include <stddef.h>
        #include <stdint.h>
    typedef unsigned short __half;
    struct ihipStream_t;
    typedef struct ihipStream_t* hipStream_t;
    typedef struct
    {
        uint32_t x, y, z, w;
    } uint4;
    #endif

    typedef __half half;
    """
)


def test_apply_workarounds_strips_hipblaslt_cxx_includes(tmp_path):
    """hipblaslt.h: <memory>/<regex>/<vector>/<hip/hip_bfloat16.h>
    AND `"hipblaslt-types.h"` (transitively pulls in the C++-only
    hipblaslt extension types — none of which are referenced by
    hipblaslt.h's own public API) are all stripped."""
    src = textwrap.dedent(
        """\
        #include <memory>
        #include <regex>
        #include <vector>
        #include <hip/hip_bfloat16.h>
        #include "hipblaslt-types.h"
        """
    )
    h = tmp_path / "fake_hipblaslt.h"
    h.write_text(src)
    _path, content = _apply_header_workarounds(
        "hipblaslt/hipblaslt.h", str(h), src
    )
    # All five offending includes are commented out (not removed —
    # easier to spot in diffs). The provenance marker is present
    # on every replacement so future readers can find this code.
    for stripped in (
        "#include <memory>",
        "#include <regex>",
        "#include <vector>",
        "#include <hip/hip_bfloat16.h>",
        '#include "hipblaslt-types.h"',
    ):
        assert f"// {stripped}" in content
        assert "stripped by hip-python codegen" in content


def test_apply_workarounds_strips_hipsparselt_unused_includes(tmp_path):
    """hipsparselt.h: <hip/hip_bfloat16.h> and <hip/hip_fp8.h> are
    stripped. No POD substitute needed because the types aren't
    referenced anywhere in the public API surface."""
    src = textwrap.dedent(
        """\
        #if defined(__HIP_PLATFORM_AMD__)
        #include <hip/hip_bfloat16.h>
        #include <hip/hip_fp16.h>
        #include <hip/hip_fp8.h>
        #endif
        """
    )
    h = tmp_path / "fake_hipsparselt.h"
    h.write_text(src)
    _path, content = _apply_header_workarounds(
        "hipsparselt/hipsparselt.h", str(h), src
    )
    assert "// #include <hip/hip_bfloat16.h>" in content
    assert "// #include <hip/hip_fp8.h>" in content
    # hip_fp16 is NOT stripped (the type IS used elsewhere).
    assert "// #include <hip/hip_fp16.h>" not in content
    assert "#include <hip/hip_fp16.h>" in content
    # No POD typedef injection for hipsparselt.
    assert "typedef struct hip_bfloat16" not in content


def test_apply_workarounds_pass_through_unaffected_header(tmp_path):
    """A header that doesn't match any registered patch returns the
    input content unmodified (None on entry → None out)."""
    h = tmp_path / "fake_hipblas.h"
    h.write_text("#include <stdint.h>\n")
    _path, content = _apply_header_workarounds(
        "hipblas/hipblas.h", str(h), None
    )
    # No content load forced, no patches applied.
    assert content is None


def test_apply_workarounds_neutralizes_rocrand_c_fallback(tmp_path):
    """rocrand.h: the `#if defined(__cplusplus) ... #else <fallback> #endif`
    block is replaced with the unconditional HIP includes, so the C-mode
    fallback that redefines uint4/__half/hipStream_t is gone."""
    h = tmp_path / "fake_rocrand.h"
    h.write_text(_ROCRAND_FALLBACK_SRC)
    _path, content = _apply_header_workarounds(
        "rocrand/rocrand.h", str(h), None
    )
    # The colliding fallback typedefs are gone.
    assert "} uint4;" not in content
    assert "typedef struct ihipStream_t* hipStream_t;" not in content
    assert "typedef unsigned short __half;" not in content
    # The guard and its #else are gone (block fully replaced).
    assert "#if defined(__cplusplus)" not in content
    assert "#else" not in content
    # The HIP includes are now unconditional.
    assert "#include <hip/hip_fp16.h>" in content
    assert "#include <hip/hip_runtime.h>" in content
    assert "#include <hip/hip_vector_types.h>" in content
    # Provenance marker + upstream-bug reference are present.
    assert "hip-python codegen" in content
    assert "rocrand_duplicate_uint4_definition.md" in content
    # Surrounding content is preserved.
    assert "#define ROCRAND_DEFAULT_MAX_BLOCK_SIZE 256" in content
    assert "typedef __half half;" in content


def test_neutralize_rocrand_noop_and_warns_when_block_absent(caplog):
    """When the expected fallback block is absent (upstream reshuffle),
    the content is returned unchanged and a warning is logged so the
    regression surfaces at codegen time."""
    src = "#include <hip/hip_runtime.h>\ntypedef int rocrand_generator;\n"
    with caplog.at_level("WARNING"):
        out = _neutralize_rocrand_c_fallback(src)
    assert out == src
    assert any("rocrand.h" in r.message for r in caplog.records)


def test_persist_shim_header_creates_dirs_and_writes(tmp_path):
    """`_persist_shim_header` writes to
    `<output_dir>/packages/<pkg>/shim_includes/<header_relpath>` and
    creates any missing intermediate directories."""
    output_dir = str(tmp_path)
    content = "// patched content\n#include <stdint.h>\n"
    _persist_shim_header(
        "hipblaslt/hipblaslt.h",
        content,
        output_dir,
        package="rocm-bindings-libraries",
    )
    expected = os.path.join(
        output_dir, "packages", "rocm-bindings-libraries",
        SHIM_INCLUDES_SUBDIR, "hipblaslt", "hipblaslt.h",
    )
    assert os.path.isfile(expected)
    with open(expected) as f:
        assert f.read() == content


def test_persist_shim_header_noop_on_none_content(tmp_path):
    """When `content is None` (i.e. no patch fired) the helper is a
    no-op — nothing is written."""
    output_dir = str(tmp_path)
    _persist_shim_header("hipblaslt/hipblaslt.h", None, output_dir)
    # The shim_includes directory should not even exist.
    assert not os.path.exists(
        os.path.join(output_dir, "packages")
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
