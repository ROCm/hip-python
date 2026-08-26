cmake_minimum_required(VERSION 3.26 FATAL_ERROR)

project(rocm-bindings-libraries LANGUAGES C)

# Helper resolution: try shared helper first (for unified build), then local
set(HIP_PYTHON_SHARED_HELPER "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/HipPythonBuild.cmake")
set(HIP_PYTHON_LOCAL_HELPER "${CMAKE_CURRENT_SOURCE_DIR}/cmake/HipPythonBuild.cmake")

# For sdist: copy shared helper to local cmake/ directory
if(EXISTS "${HIP_PYTHON_SHARED_HELPER}" AND DEFINED SKBUILD_STATE AND SKBUILD_STATE STREQUAL "sdist")
  file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
  configure_file("${HIP_PYTHON_SHARED_HELPER}" "${HIP_PYTHON_LOCAL_HELPER}" COPYONLY)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
if(EXISTS "${HIP_PYTHON_SHARED_HELPER}")
  list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake")
endif()
include(HipPythonBuild)

hip_python_initialize()

hip_python_resolve_version()

hip_python_get_env_default(_hip_python_runtime_linking_default HIP_PYTHON_RUNTIME_LINKING ON)
option(HIP_PYTHON_RUNTIME_LINKING "Whether libraries should resolve ROCm libraries at runtime" ${_hip_python_runtime_linking_default})

# Define all libraries to build (math/FFT/random/sparse).
# RCCL (communication) and ROCTX (tracing) live in the sibling
# rocm-bindings-systems wheel.
#
# hipblaslt and hipsparselt require patched shim headers (their public
# C API headers pull in C++-only includes) — those shims are produced
# by the codegen and prepended to the include path below, so the
# bindings compile as C. hiptensor and hipdnn_backend remain excluded
# (hipdnn_backend also has a missing CMake-generated export header in
# the source tree; see AVAILABLE_GENERATORS in the codegen).
set(HIP_PYTHON_ALL_LIBRARIES hipblas hipsolver hiprand hipfft hipsparse hipblaslt hipsparselt)

# Allow selective building of libraries
hip_python_define_module_options(HIP_PYTHON_ENABLE_LIB HIP_PYTHON_LIBS "Build library module" ${HIP_PYTHON_ALL_LIBRARIES})
hip_python_collect_enabled_modules(HIP_PYTHON_SELECTED_LIBRARIES HIP_PYTHON_ENABLE_LIB ${HIP_PYTHON_ALL_LIBRARIES})

# Collect Cython dependencies
hip_python_collect_cython_depends(LIBRARIES_CYTHON_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/rocm/bindings/*.pxd"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/rocm/bindings/*.pyx"
)

set(LIBRARIES_INCLUDE_DIRS
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

# Patched upstream headers ("shims") produced by the codegen live under
# shim_includes/ (see binding_generator._persist_shim_header). Prepend
# that directory so the gcc compile of the Cython-generated `.c` reads
# the patched copy before the unpatched original under
# ${ROCM_PATH}/include. hip_python_add_cython_module orders these
# include dirs ahead of HIP_PYTHON_ROCM_INCLUDE_DIR, so a plain prepend
# here is sufficient for the shim to win.
set(_shim_includes_dir "${CMAKE_CURRENT_SOURCE_DIR}/shim_includes")
if(EXISTS "${_shim_includes_dir}")
  list(PREPEND LIBRARIES_INCLUDE_DIRS "${_shim_includes_dir}")

  # Surface every shim that is in effect. These bindings only compile
  # because the patched header shadows the on-disk ROCm header; anyone
  # recompiling the generated sources must apply the same shim.
  file(GLOB_RECURSE _shim_headers "${_shim_includes_dir}/*.h")
  foreach(_shim_header IN LISTS _shim_headers)
    file(RELATIVE_PATH _shim_rel "${_shim_includes_dir}" "${_shim_header}")
    message(WARNING
      "hip-python: compiling against patched shim header '${_shim_rel}' "
      "from shim_includes/ instead of \${ROCM_PATH}/include. The generated "
      "Cython bindings only build because this shim is prepended to the "
      "include path; any consumer recompiling these bindings must apply the "
      "same shim. See share/design/UPSTREAM_BUGS/.")
  endforeach()
endif()

# Cross-package cimports: rocm.bindings.util.loader from
# rocm-bindings-core, rocm.bindings.cyhip from rocm-bindings-hip.
hip_python_append_sibling_includes(LIBRARIES_INCLUDE_DIRS)

# Per-module C preprocessor defines. Declared before the availability probe
# below so the probe compiles each header exactly as the module itself will;
# the build loop reads the same variables via `_hp_defs_${_lib}`.
#
# hipsparse: hipsparse.h picks its deprecation marker by feature test. In C
# mode without __has_c_attribute it falls back to HIPSPARSE_DEPRECATED, which
# hipsparse-export.h defines as __declspec(deprecated) for MSVC — and MSVC
# rejects __declspec on an enumerator, which is where hipsparseOrder_t uses
# it. gcc/clang take the __attribute__ or [[deprecated]] path instead and are
# unaffected. Pre-defining the marker empty (hipsparse-export.h guards it with
# #ifndef) drops the deprecation annotations, which a bindings build does not
# need. Keep this MSVC-only so other toolchains compile the header verbatim.
set(_hp_defs_hipsparse)
if(MSVC)
  list(APPEND _hp_defs_hipsparse "HIPSPARSE_DEPRECATED=")
endif()

# Drop libraries the codegen skipped for want of a header (opt-in; see
# hip_python_filter_generated_modules). Ahead of the availability probe below
# because a skipped library has no .pxd for the probe to read a header from.
hip_python_filter_generated_modules(HIP_PYTHON_SELECTED_LIBRARIES
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/rocm/bindings"
  MODULES ${HIP_PYTHON_SELECTED_LIBRARIES}
)

# Skip libraries this ROCm install cannot build (see
# hip_python_filter_available_modules). On Windows this drops hipsparselt,
# whose header ships without the hipsparselt-export.h it includes.
hip_python_filter_available_modules(HIP_PYTHON_SELECTED_LIBRARIES
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/rocm/bindings"
  INCLUDE_DIRS ${LIBRARIES_INCLUDE_DIRS}
  COMPILE_DEFINITIONS ${_hp_defs_hipsparse}
  MODULES ${HIP_PYTHON_SELECTED_LIBRARIES}
)

# Build each enabled library
foreach(_lib IN LISTS HIP_PYTHON_SELECTED_LIBRARIES)
  # Map library name to actual library file name
  if(_lib STREQUAL "hip")
    set(_lib_name amdhip64)
  else()
    set(_lib_name ${_lib})
  endif()

  # Determine link libraries based on runtime linking option
  if(HIP_PYTHON_RUNTIME_LINKING)
    set(_lib_link_libraries)
  else()
    set(_lib_link_libraries ${_lib_name})
  endif()

  # Build C-level bindings (cy*.pyx) - install at rocm/bindings/ level
  hip_python_add_cython_module(
    TARGET rocm_bindings_libraries_cy${_lib}
    MODULE_NAME rocm.bindings.cy${_lib}
    SOURCE src/rocm/bindings/cy${_lib}.pyx
    DESTINATION rocm/bindings
  COMPONENT rocm-bindings-libraries
    CYTHON_INCLUDE_DIRS ${LIBRARIES_INCLUDE_DIRS}
    INCLUDE_DIRS ${LIBRARIES_INCLUDE_DIRS}
    LINK_LIBRARIES ${_lib_link_libraries}
    COMPILE_DEFINITIONS ${_hp_defs_${_lib}}
    DEPENDS ${LIBRARIES_CYTHON_DEPENDS}
  )

  # Build high-level bindings (*.pyx) - install at rocm/bindings/ level
  hip_python_add_cython_module(
    TARGET rocm_bindings_${_lib}
    MODULE_NAME rocm.bindings.${_lib}
    SOURCE src/rocm/bindings/${_lib}.pyx
    DESTINATION rocm/bindings
  COMPONENT rocm-bindings-libraries
    CYTHON_INCLUDE_DIRS ${LIBRARIES_INCLUDE_DIRS}
    INCLUDE_DIRS ${LIBRARIES_INCLUDE_DIRS}
    LINK_LIBRARIES ${_lib_link_libraries}
    COMPILE_DEFINITIONS ${_hp_defs_${_lib}}
    DEPENDS ${LIBRARIES_CYTHON_DEPENDS}
  )

  # Install .pxd files for cimport
  install(
    FILES src/rocm/bindings/cy${_lib}.pxd src/rocm/bindings/${_lib}.pxd
    DESTINATION rocm/bindings
    COMPONENT rocm-bindings-libraries
  )

  # Install generator-emitted .pyi type stubs for the high-level module
  # (cy* modules are cimport-only — no stubs; see plan §B.2). No-op until
  # the interfacegen code generator emits the file.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/rocm/bindings/${_lib}.pyi")
    install(
      FILES src/rocm/bindings/${_lib}.pyi
      DESTINATION rocm/bindings
      COMPONENT rocm-bindings-libraries
    )
  endif()
endforeach()

# NOTE: src/rocm/bindings/ is a PEP 420 implicit namespace package
# It intentionally has NO __init__.py file to allow multiple packages
# (rocm-bindings-core, rocm-bindings-hip, rocm-bindings-libraries)
# to contribute modules to the same namespace.
# See: https://peps.python.org/pep-0420/
# Aggregate target for this package (for build.targets in pyproject.toml)
# Dynamically collect all targets based on selected libraries
set(_all_library_targets)
foreach(_lib IN LISTS HIP_PYTHON_SELECTED_LIBRARIES)
  list(APPEND _all_library_targets
    rocm_bindings_libraries_cy${_lib}
    rocm_bindings_${_lib}
  )
endforeach()

add_custom_target(package_rocm_bindings_libraries
  DEPENDS ${_all_library_targets}
)
