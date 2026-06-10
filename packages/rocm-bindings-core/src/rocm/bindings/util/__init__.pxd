# Cython package definition for rocm.bindings.util
#
# This marker exists so that Cython's cross-package `cimport` can resolve
# `rocm.bindings.util.types` / `rocm.bindings.util.loader` from the other
# binding packages (hip, libraries, systems, compiler). At Python runtime
# `rocm.bindings.util` is a PEP 420 namespace package (no `__init__.py`);
# the import system ignores this `.pxd`.
