# MIT License
#
# Copyright (c) 2023-2026 Advanced Micro Devices, Inc.
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

"""Per-module generators for the **rocm-bindings-libraries** wheel.

Math / FFT / random / sparse libraries: hipBLAS, hipSOLVER, hipRAND,
hipFFT, hipSPARSE. Each `generate_*` function returns a configured
`CythonModuleGenerator` for the named module.

Generator functions accept explicit `include_dir` and `header_relpath` so the
Cython `extern from` statements use the exact paths intended (no derivation
via os.path.dirname). Optional `header_content` enables in-memory rendering
of .h.in templates without requiring CMake configure.
"""

import textwrap

from interfacegen.cython import CythonModuleGenerator
from interfacegen.support.recipes import rocm as controls


def _make_header_arg(header_relpath: str, header_content: str = None):
    """Build the header argument for CythonModuleGenerator."""
    if header_content is not None:
        return (header_relpath, header_content)
    return header_relpath


def generate_hipblas(
    *,
    include_dir: str,
    header_relpath: str = "hipblas/hipblas.h",
    header_content: str = None,
    runtime_linking: bool,
    generator_args: list,
    default_ptr_handler,
):
    generator = CythonModuleGenerator(
        "rocm.bindings.hipblas",
        include_dir,
        _make_header_arg(header_relpath, header_content),
        runtime_linking=runtime_linking,
        util_pkg="rocm.bindings.util",
        dll="libhipblas.so",
        node_filter=controls.hipblas.node_filter,
        ptr_parm_intent=controls.hipblas.ptr_parm_intent,
        ptr_rank=controls.hipblas.ptr_rank,
        ptr_complicated_type_handler=default_ptr_handler,
        raw_comment_cleaner=controls.hipblas.raw_comment_cleaner,
        cflags=generator_args,
    )
    generator.c_interface_decl_prolog += textwrap.dedent(
        """\
    from rocm.bindings.cyhip cimport *
    """
    )
    generator.python_interface_decl_prolog += textwrap.dedent(
        """\
    from rocm.bindings.hip cimport *
    """
    )
    generator.python_interface_impl_prolog += textwrap.dedent(
        """\
    from rocm.bindings.hip import _hipDataType__Base
    """
    )
    return generator


def generate_hipsolver(
    *,
    include_dir: str,
    header_relpath: str = "hipsolver/hipsolver.h",
    header_content: str = None,
    runtime_linking: bool,
    generator_args: list,
    default_ptr_handler,
):
    generator = CythonModuleGenerator(
        "rocm.bindings.hipsolver",
        include_dir,
        _make_header_arg(header_relpath, header_content),
        runtime_linking=runtime_linking,
        util_pkg="rocm.bindings.util",
        dll="libhipsolver.so",
        node_filter=controls.hipsolver.node_filter,
        ptr_parm_intent=controls.hipsolver.ptr_parm_intent,
        ptr_rank=controls.hipsolver.ptr_rank,
        ptr_complicated_type_handler=default_ptr_handler,
        raw_comment_cleaner=controls.hipsolver.raw_comment_cleaner,
        cflags=generator_args,
    )
    generator.c_interface_decl_prolog += textwrap.dedent(
        """\
    from rocm.bindings.cyhip cimport *
    """
    )
    generator.python_interface_decl_prolog += textwrap.dedent(
        """\
    from rocm.bindings.hip cimport *
    """
    )
    generator.python_interface_impl_prolog += textwrap.dedent(
        """\
    from rocm.bindings.hip import _hipDataType__Base
    """
    )
    return generator


def generate_hiprand(
    *,
    include_dir: str,
    header_relpath: str = "hiprand/hiprand.h",
    header_content: str = None,
    runtime_linking: bool,
    generator_args: list,
    default_ptr_handler,
):
    generator = CythonModuleGenerator(
        "rocm.bindings.hiprand",
        include_dir,
        _make_header_arg(header_relpath, header_content),
        runtime_linking=runtime_linking,
        util_pkg="rocm.bindings.util",
        dll="libhiprand.so",
        node_filter=controls.hiprand.node_filter,
        macro_type=controls.hiprand.macro_type,
        ptr_parm_intent=controls.hiprand.ptr_parm_intent,
        ptr_rank=controls.hiprand.ptr_rank,
        ptr_complicated_type_handler=default_ptr_handler,
        cflags=generator_args,
    )
    generator.c_interface_decl_prolog += textwrap.dedent(
        """\
    from rocm.bindings.cyhip cimport hipStream_t
    """
    )
    generator.python_interface_decl_prolog += textwrap.dedent(
        """\
    from rocm.bindings.hip cimport ihipStream_t
    """
    )
    return generator


def generate_hipfft(
    *,
    include_dir: str,
    header_relpath: str = "hipfft/hipfft.h",
    header_content: str = None,
    runtime_linking: bool,
    generator_args: list,
    default_ptr_handler,
):
    generator = CythonModuleGenerator(
        "rocm.bindings.hipfft",
        include_dir,
        _make_header_arg(header_relpath, header_content),
        runtime_linking=runtime_linking,
        util_pkg="rocm.bindings.util",
        dll="libhipfft.so",
        node_filter=controls.hipfft.node_filter,
        macro_type=controls.hipfft.macro_type,
        ptr_parm_intent=controls.hipfft.ptr_parm_intent,
        ptr_rank=controls.hipfft.ptr_rank,
        ptr_complicated_type_handler=default_ptr_handler,
        cflags=generator_args,
    )
    generator.c_interface_decl_prolog += textwrap.dedent(
        """\
    from rocm.bindings.cyhip cimport hipStream_t, float2, double2
    """
    )
    generator.python_interface_decl_prolog += textwrap.dedent(
        """\
    from rocm.bindings.hip cimport ihipStream_t, float2, double2
    """
    )
    return generator


def generate_hipsparse(
    *,
    include_dir: str,
    header_relpath: str = "hipsparse/hipsparse.h",
    header_content: str = None,
    runtime_linking: bool,
    generator_args: list,
    default_ptr_handler,
):
    generator = CythonModuleGenerator(
        "rocm.bindings.hipsparse",
        include_dir,
        _make_header_arg(header_relpath, header_content),
        runtime_linking=runtime_linking,
        util_pkg="rocm.bindings.util",
        dll="libhipsparse.so",
        node_filter=controls.hipsparse.node_filter,
        macro_type=controls.hipsparse.macro_type,
        ptr_parm_intent=controls.hipsparse.ptr_parm_intent,
        ptr_rank=controls.hipsparse.ptr_rank,
        raw_comment_cleaner=controls.hipsparse.raw_comment_cleaner,
        ptr_complicated_type_handler=default_ptr_handler,
        cflags=generator_args,
    )
    generator.c_interface_decl_prolog += textwrap.dedent(
        """\
    from rocm.bindings.cyhip cimport *
    """
    )
    generator.python_interface_decl_prolog += textwrap.dedent(
        """\
    from rocm.bindings.hip cimport ihipStream_t, float2, double2 # C import structs/union types
    """
    )
    generator.python_interface_impl_prolog += textwrap.dedent(
        """\
    from rocm.bindings.hip import hipError_t, _hipDataType__Base # PY import enums
    """
    )
    return generator


def generate_hipblaslt(
    *,
    include_dir: str,
    header_relpath: str = "hipblaslt/hipblaslt.h",
    header_content: str = None,
    runtime_linking: bool,
    generator_args: list,
    default_ptr_handler,
):
    generator = CythonModuleGenerator(
        "rocm.bindings.hipblaslt",
        include_dir,
        _make_header_arg(header_relpath, header_content),
        runtime_linking=runtime_linking,
        util_pkg="rocm.bindings.util",
        dll="libhipblaslt.so",
        node_filter=controls.hipblaslt.node_filter,
        ptr_parm_intent=controls.hipblaslt.ptr_parm_intent,
        ptr_rank=controls.hipblaslt.ptr_rank,
        ptr_complicated_type_handler=default_ptr_handler,
        raw_comment_cleaner=controls.hipblaslt.raw_comment_cleaner,
        cflags=generator_args,
    )
    generator.c_interface_decl_prolog += textwrap.dedent(
        """\
    from rocm.bindings.cyhip cimport *
    from rocm.bindings.cyhipblas cimport *
    """
    )
    generator.python_interface_decl_prolog += textwrap.dedent(
        """\
    from rocm.bindings.hip cimport *
    from rocm.bindings.hipblas cimport *
    """
    )
    generator.python_interface_impl_prolog += textwrap.dedent(
        """\
    from rocm.bindings.hip import _hipDataType__Base
    """
    )
    return generator


def generate_hiptensor(
    *,
    include_dir: str,
    header_relpath: str = "hiptensor/hiptensor.h",
    header_content: str = None,
    runtime_linking: bool,
    generator_args: list,
    default_ptr_handler,
):
    generator = CythonModuleGenerator(
        "rocm.bindings.hiptensor",
        include_dir,
        _make_header_arg(header_relpath, header_content),
        runtime_linking=runtime_linking,
        util_pkg="rocm.bindings.util",
        dll="libhiptensor.so",
        node_filter=controls.hiptensor.node_filter,
        ptr_parm_intent=controls.hiptensor.ptr_parm_intent,
        ptr_rank=controls.hiptensor.ptr_rank,
        ptr_complicated_type_handler=default_ptr_handler,
        cflags=generator_args,
    )
    generator.c_interface_decl_prolog += textwrap.dedent(
        """\
    from rocm.bindings.cyhip cimport *
    """
    )
    generator.python_interface_decl_prolog += textwrap.dedent(
        """\
    from rocm.bindings.hip cimport *
    """
    )
    return generator


def generate_hipdnn(
    *,
    include_dir: str,
    header_relpath: str = "hipdnn_backend.h",
    header_content: str = None,
    runtime_linking: bool,
    generator_args: list,
    default_ptr_handler,
):
    generator = CythonModuleGenerator(
        "rocm.bindings.hipdnn",
        include_dir,
        _make_header_arg(header_relpath, header_content),
        runtime_linking=runtime_linking,
        util_pkg="rocm.bindings.util",
        dll="libhipdnn.so",
        node_filter=controls.hipdnn.node_filter,
        ptr_parm_intent=controls.hipdnn.ptr_parm_intent,
        ptr_rank=controls.hipdnn.ptr_rank,
        ptr_complicated_type_handler=default_ptr_handler,
        cflags=generator_args,
    )
    generator.c_interface_decl_prolog += textwrap.dedent(
        """\
    from rocm.bindings.cyhip cimport *
    """
    )
    generator.python_interface_decl_prolog += textwrap.dedent(
        """\
    from rocm.bindings.hip cimport *
    """
    )
    return generator


def generate_hipsparselt(
    *,
    include_dir: str,
    header_relpath: str = "hipsparselt/hipsparselt.h",
    header_content: str = None,
    runtime_linking: bool,
    generator_args: list,
    default_ptr_handler,
):
    generator = CythonModuleGenerator(
        "rocm.bindings.hipsparselt",
        include_dir,
        _make_header_arg(header_relpath, header_content),
        runtime_linking=runtime_linking,
        util_pkg="rocm.bindings.util",
        dll="libhipsparselt.so",
        node_filter=controls.hipsparselt.node_filter,
        macro_type=controls.hipsparselt.macro_type,
        ptr_parm_intent=controls.hipsparselt.ptr_parm_intent,
        ptr_rank=controls.hipsparselt.ptr_rank,
        raw_comment_cleaner=controls.hipsparselt.raw_comment_cleaner,
        ptr_complicated_type_handler=default_ptr_handler,
        cflags=generator_args,
    )
    generator.c_interface_decl_prolog += textwrap.dedent(
        """\
    from rocm.bindings.cyhip cimport *
    from rocm.bindings.cyhipsparse cimport *
    """
    )
    generator.python_interface_decl_prolog += textwrap.dedent(
        """\
    from rocm.bindings.hip cimport ihipStream_t, float2, double2
    from rocm.bindings.hipsparse cimport *
    """
    )
    generator.python_interface_impl_prolog += textwrap.dedent(
        """\
    from rocm.bindings.hip import hipError_t, _hipDataType__Base
    """
    )
    return generator
