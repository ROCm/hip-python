# MIT License
#
# Modifications Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

__author__ = "Advanced Micro Devices, Inc."

"""AMD COMGR Utilities.

Utitilies that make use of the `rocm.amd_comgr` interfaces
that are shipped with the ROCm LLVM Python project.
"""

from rocm.amd_comgr import amd_comgr as comgr

from . import llvmutils

llvm_amdgpu_kernel_visibility = "protected"
llvm_amdgpu_kernel_calling_convention = "amdgpu_kernel"
llvm_amdgpu_device_fun_visibility = "hidden"
llvm_amdgpu_kernel_address_significance = "local_unnamed_addr"
llvm_amdgpu_device_fun_address_significance = llvm_amdgpu_kernel_address_significance

def compile_hip_source_to_llvm(
    source: str,
    amdgpu_arch: str,
    hip_version_tuple: tuple,
    to_llvm_ir: bool = False,
    extra_opts: str = "",
    comgr_logging: bool = False,
):
    """Compiles a HIP C++ source file to LLVM bitcode or human-readable LLVM IR.

    Args:
        source: str: Contents of the HIP C++ source.
        amdgpu_arch (`str`): An AMD GPU arch identifier such as `gfx90a` (MI200 series) or `gfx942` (MI300 series).
        hip_version_tuple (`tuple`): A tuple of `int` values that contains HIP version major, minor, and patch.
        to_llvm_ir (`bool`): If the compilation result should be LLVM IR (versus LLVM BC). Defaults to `False`.
        extra_opts (`str`, optional): Additional opts to append to the compiler command. Defaults to `""`.
        comgr_logging (`bool`, optional): Enable AMD COMGR logging. Defaults to `False`.

    Returns:
        tuple: A triple consisting of LLVM BC/IR, the log or None, diagnostic information or None.
    """
    (
        llvm_bc_or_ir,
        log,
        diagnostic,
    ) = comgr.ext.compile_hip_to_bc(
        source=source,
        isa_name=f"amdgcn-amd-amdhsa--{amdgpu_arch}",
        hip_version_tuple=hip_version_tuple[:3],
        logging=comgr_logging,
        extra_opts=extra_opts,
    )
    if to_llvm_ir:
        llvm_bc_or_ir = llvmutils.to_ir_from_bc(llvm_bc_or_ir, len(llvm_bc_or_ir))
    return (llvm_bc_or_ir, log, diagnostic)


_DUMMY_KERNEL = """\
extern "C" __attribute__((global)) void MYFUNC ({args}) {{
    return;
}}
"""

_DUMMY_DEVICE_FUN = """\
extern "C" __attribute__((device)) void MYFUNC ({args}) {{
    return;
}}
"""


def _compile_dummy_snippet_to_llvm_ir(source: str, amdgpu_arch: str, args: str):
    (llvm_ir, _, _) = compile_hip_source_to_llvm(
        source=source.format(args=args),
        amdgpu_arch=amdgpu_arch,
        hip_version_tuple=(0, 0, 0),
        to_llvm_ir=True,
    )
    return llvm_ir.decode("utf-8")


def get_dummy_kernel_llvm_ir(amdgpu_arch: str, args: str):
    """Returns LLVM IR for an empty AMD GPU kernel.

    amdgpu_arch (`str`):
            AMD GPU architecture, e.g. ``gfx90a``.
    args (str, optional):
        Arguments to specify for the kernel. Defaults to "".
    """
    return _compile_dummy_snippet_to_llvm_ir(_DUMMY_KERNEL, amdgpu_arch, args)


def get_dummy_device_fun_llvm_ir(amdgpu_arch: str, args: str):
    """Returns LLVM IR for an empty AMD GPU device function.

    amdgpu_arch (`str`):
        AMD GPU architecture, e.g. ``gfx90a``.
    args (str, optional):
        Arguments to specify for the kernel. Defaults to "".
    """
    return _compile_dummy_snippet_to_llvm_ir(_DUMMY_DEVICE_FUN, amdgpu_arch, args)


def parse_llvm_attributes_line(
    attributes_line: str,
    raw: bool = False,
    only_kv: bool = False,
    exclude_patterns=["memory("],
):
    """Parses an LLVM attributes line.

    Parses expressions of the form:

    ``attributes #<attributes-num> = { <attributes> }``

    Args:
        attributes_line (`str`, optional):

        raw (`bool`, optional):
            Return a list with attributes in their raw format.
            See the Returns section for more details.
            Defaults to False.
        only_kv (`bool`, optional):
            Only key-value arguments. Defaults to False.
        exclude_patterns (`list`, optional):
            Exclude all attributes that start with these
            patterns. Defaults to ``["memory("]`` as memory
            is depending on the arguments.
    Returns:
        `tuple` or `list`:
            Depending on argment ``raw`` and ``only_kw_attribs``, either returns:

            * ``raw==False`` and ``only_kw_attribs==False``: a `tuple` consisting of (1) a list that contains
              non-keyword attributes (e.g., 'mustprogress', 'nofree', 'norecurse', ...)
              and (2) a `dict` that stores key and value of key-value attributes.
            * ``raw==False`` and ``only_kw_attribs==True``: a `dict` that stores key and value of key-value attributes.
            * ``raw==True``: a `list` that contains simple attributes (if ``only_kv==False``)
              plus key-value attributes in their raw '"<key>"="<value>"' form.
    """
    values_part = attributes_line.split("=", 1)[1]  # remove 'attributes #<num> = '
    values_part = values_part.strip(
        " {}"
    )  # remove braces and trailing/leading whitespace
    attribs, kwattribs = [], {}

    def excluded_(attrib):
        nonlocal exclude_patterns
        for pattern in exclude_patterns:
            if attrib.startswith(pattern):
                return True
        return False

    for attrib in values_part.split(" "):
        if excluded_(attrib):
            continue
        is_kv_attrib = attrib.startswith('"')
        if not raw and is_kv_attrib:
            key, value = attrib.split("=")
            kwattribs[key.strip('"')] = value.strip('"')
        elif is_kv_attrib:
            attribs.append(attrib)
        elif not only_kv:
            attribs.append(attrib)
    if raw:
        return attribs
    elif only_kv:
        return kwattribs
    else:
        return (attribs, kwattribs)


def get_llvm_kernel_attributes(
    amdgpu_arch: str,
    raw: bool = False,
    only_kv: bool = False,
    exclude_patterns=["memory("],
    args: str = "",
):
    """Return kernel attributes.

    Args:
        attributes_line (`str`, optional):

        raw (`bool`, optional):
            Return a list with attributes in their raw format.
            See the Returns section for more details.
            Defaults to False.
        exclude_patterns (`list`, optional):
            Exclude all attributes that start with these
            patterns. Defaults to ``["memory("]`` as memory
            is depending on the arguments.
        only_kv (`bool`, optional):
            Only key-value arguments. Defaults to False.
    Returns:
        `tuple` or `list`:
            Depending on argment ``raw`` and ``only_kw_attribs``, either returns:

            * ``raw==False`` and ``only_kw_attribs==False``: a `tuple` consisting of (1) a list that contains
              non-keyword attributes (e.g., 'mustprogress', 'nofree', 'norecurse', ...)
              and (2) a `dict` that stores key and value of key-value attributes.
            * ``raw==False`` and ``only_kw_attribs==True``: a `dict` that stores key and value of key-value attributes.
            * ``raw==True``: a `list` that contains simple attributes (if ``only_kv==False``)
              plus key-value attributes in their raw '"<key>"="<value>"' form.
    """
    llvm_ir = _compile_dummy_snippet_to_llvm_ir(_DUMMY_KERNEL, amdgpu_arch, args)
    attributes_0_line = next(
        line for line in llvm_ir.splitlines() if "attributes #0" in line
    )
    return parse_llvm_attributes_line(
        attributes_0_line,
        raw=raw,
        only_kv=only_kv,
        exclude_patterns=exclude_patterns,
    )


def get_llvm_target_features(
    amdgpu_arch: str,
    sort: bool = False,
    as_list: bool = False,
):
    """Return all LLVM target features for the given AMD GPU arch.
    Args:
        amdgpu_arch (`str`):
            AMD GPU architecture, e.g. ``gfx90a``.
        sort (`bool`, optional):
            Return the features in alphabetical order.
            Defaults to False.
        as_list (`bool`, optional):
            Return the features as `list` and not as `str`.
            Defaults to False.
    """
    (_, kwattribs) = get_llvm_kernel_attributes(amdgpu_arch)
    features_raw = kwattribs["target-features"]
    if sort:
        features_list_sorted = sorted(features_raw.split(","))
        if as_list:
            return list(features_list_sorted)
        return ",".join(features_list_sorted)
    else:
        if as_list:
            return features_raw.split(",")
        else:
            return features_raw


def compare_llvm_target_features(
    amdgpu_arch: str, amdgpu_arch_base: str, sort: bool = False, as_list: bool = False
):
    """Compare LLVM target features between two AMD GPU architectures.

    Args:
        amdgpu_arch (`str`):
            AMD GPU architecture, e.g. ``gfx90a``.
        amdgpu_arch_base (`str`):
            AMD GPU architecture to compare to.
        sort (`bool`, optional):
            Return the features in alphabetical order.
            Defaults to False.
        as_list (`bool`, optional):
            Return the feature sets as `list` and not as `str`.
            Defaults to False.

    Returns:
        `dict`:
            A `dict` that contains the following key-value pairs:

            * "identical": the list of identical features,
            * "added": the list of features that have been added for
              `amdgpu_arch` with respect to `amdgpu_arch_base`, and
            * "removed": the list of features that have been removed for
              `amdgpu_arch` with respect to `amdgpu_arch_base.
    """
    features = get_llvm_target_features(amdgpu_arch, as_list=True)
    features_base = get_llvm_target_features(amdgpu_arch_base, as_list=True)
    identical_features = [f for f in features if f in features_base]
    added_features = [f for f in features if f not in features_base]
    removed_features = [f for f in features_base if f not in features]

    def prepare_(features: list):
        if sort:
            features = list(sorted(features))
        if as_list:
            return features
        else:
            return ",".join(features)

    return dict(
        identical=prepare_(identical_features),
        added=prepare_(added_features),
        removed=prepare_(removed_features),
    )


def get_llvm_device_fun_attributes(
    amdgpu_arch: str,
    raw: bool = False,
    only_kv: bool = False,
    exclude_patterns=["memory("],
    args: str = "",
):
    """Return kernel attributes.

    Args:
        attributes_line (`str`, optional):

        raw (`bool`, optional):
            Return a list with attributes in their raw format.
            See the Returns section for more details.
            Defaults to False.
        exclude_patterns (`list`, optional):
            Exclude all attributes that start with these
            patterns. Defaults to ``["memory("]`` as memory
            is depending on the arguments.
        only_kv (`bool`, optional):
            Only key-value arguments. Defaults to False.
    Returns:
        `tuple` or `list`:
            Depending on argment ``raw`` and ``only_kw_attribs``, either returns:

            * ``raw==False`` and ``only_kw_attribs==False``: a `tuple` consisting of (1) a list that contains
              non-keyword attributes (e.g., 'mustprogress', 'nofree', 'norecurse', ...)
              and (2) a `dict` that stores key and value of key-value attributes.
            * ``raw==False`` and ``only_kw_attribs==True``: a `dict` that stores key and value of key-value attributes.
            * ``raw==True``: a `list` that contains simple attributes (if ``only_kv==False``)
              plus key-value attributes in their raw '"<key>"="<value>"' form.
    """
    llvm_ir = _compile_dummy_snippet_to_llvm_ir(_DUMMY_DEVICE_FUN, amdgpu_arch, args)
    attributes_0_line = next(
        line for line in llvm_ir.splitlines() if "attributes #0" in line
    )
    return parse_llvm_attributes_line(
        attributes_0_line,
        raw=raw,
        only_kv=only_kv,
        exclude_patterns=exclude_patterns,
    )
