cmake_minimum_required(VERSION 3.26 FATAL_ERROR)

project(rocm-bindings-systems LANGUAGES C)

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

# System-level libraries built into this package: collective communication
# (rccl), tracing/profiling instrumentation (roctx), accelerated file I/O
# (hipfile), and system management interface (amdsmi). Math/FFT/random/
# sparse libraries live in the sibling rocm-bindings-libraries wheel.
#
# NOTE: the hsa binding is intentionally disabled. Its codegen
# registration is commented out (see AVAILABLE_GENERATORS in
# tools/hip-python-generate/.../binding_generator.py), so it is no
# longer produced by the codegen and is excluded from this build too.
set(HIP_PYTHON_ALL_SYSTEMS rccl roctx hipfile amdsmi)

# hipfile is conditional on the package being available — detect upfront so
# the per-module foreach loop can skip it gracefully when not installed.
find_package(hipfile QUIET)
if(hipfile_FOUND)
  message(STATUS "Found hipfile package — hipfile bindings will be built")
  set(HIP_PYTHON_HIPFILE_AVAILABLE TRUE)
  if(TARGET hip::hipfile)
    get_target_property(HIPFILE_INCLUDE_DIR hip::hipfile INTERFACE_INCLUDE_DIRECTORIES)
    get_target_property(HIPFILE_LIBRARY_PATH hip::hipfile IMPORTED_LOCATION)
    if(NOT HIPFILE_LIBRARY_PATH)
      get_target_property(HIPFILE_LIBRARY_PATH hip::hipfile IMPORTED_LOCATION_RELEASE)
    endif()
  endif()
else()
  message(STATUS "hipfile package not found — hipfile bindings will be skipped")
  set(HIP_PYTHON_HIPFILE_AVAILABLE FALSE)
endif()

# Optional bundling: when ON, the resolved libhipfile.so is copied into the
# wheel so end users do not need a system hipfile installation. Default OFF
# to keep wheels lean for the common case.
hip_python_get_env_default(_hip_python_bundle_libhipfile_default HIP_PYTHON_BUNDLE_LIBHIPFILE OFF)
option(HIP_PYTHON_BUNDLE_LIBHIPFILE
  "Bundle libhipfile.so in rocm-bindings-systems wheel"
  ${_hip_python_bundle_libhipfile_default})

# Allow selective building
hip_python_define_module_options(HIP_PYTHON_ENABLE_SYSTEM HIP_PYTHON_SYSTEMS "Build system module" ${HIP_PYTHON_ALL_SYSTEMS})
hip_python_collect_enabled_modules(HIP_PYTHON_SELECTED_SYSTEMS HIP_PYTHON_ENABLE_SYSTEM ${HIP_PYTHON_ALL_SYSTEMS})

# Drop hipfile from the selected list when the package is missing — the user
# may have left HIP_PYTHON_ENABLE_SYSTEM_HIPFILE=ON (the default) without
# realizing find_package failed.
if(NOT HIP_PYTHON_HIPFILE_AVAILABLE)
  list(REMOVE_ITEM HIP_PYTHON_SELECTED_SYSTEMS hipfile)
endif()

# Collect Cython dependencies
hip_python_collect_cython_depends(SYSTEMS_CYTHON_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/rocm/bindings/*.pxd"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/rocm/bindings/*.pyx"
)

set(SYSTEMS_INCLUDE_DIRS
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

# Cross-package cimports: rocm.bindings.util.loader from
# rocm-bindings-core; rccl additionally needs hipStream_t from
# rocm-bindings-hip.
hip_python_append_sibling_includes(SYSTEMS_INCLUDE_DIRS)

# Skip system modules this ROCm install cannot build (see
# hip_python_filter_available_modules). hipfile was already dropped above via
# find_package; this covers the rest. Every module in this package is absent
# from the Windows ROCm SDK, so there the list comes back empty and the
# package builds nothing — see the guard further down.
#
# ENABLE_ESMI_LIB is passed for the same reason the amdsmi compile sets it
# (see the build loop below): without it the preprocessor strips the ESMI
# declarations back out of amd_smi/amdsmi.h.
hip_python_filter_available_modules(HIP_PYTHON_SELECTED_SYSTEMS
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/rocm/bindings"
  INCLUDE_DIRS ${SYSTEMS_INCLUDE_DIRS} ${HIPFILE_INCLUDE_DIR}
  COMPILE_DEFINITIONS ENABLE_ESMI_LIB
  MODULES ${HIP_PYTHON_SELECTED_SYSTEMS}
)

# Build each enabled system module
foreach(_lib IN LISTS HIP_PYTHON_SELECTED_SYSTEMS)
  set(_lib_name ${_lib})

  # Per-module include and linking overrides. hipfile pulls headers from the
  # installed hip::hipfile target rather than the ROCm system include dir.
  set(_extra_includes)
  if(_lib STREQUAL "hipfile")
    if(HIPFILE_INCLUDE_DIR)
      set(_extra_includes ${HIPFILE_INCLUDE_DIR})
    endif()
    if(HIP_PYTHON_RUNTIME_LINKING)
      set(_lib_link_libraries)
    else()
      set(_lib_link_libraries hip::hipfile)
    endif()
  else()
    if(HIP_PYTHON_RUNTIME_LINKING)
      set(_lib_link_libraries)
    else()
      set(_lib_link_libraries ${_lib_name})
    endif()
  endif()

  # Per-module C preprocessor defines. amdsmi's generated cython names
  # the ESMI types/functions because the codegen parses the header with
  # -DENABLE_ESMI_LIB (see interfacegen recipes/hip-python/src/
  # hip_python_codegen/generators_systems.py, generate_amdsmi). gcc must
  # see the same define or the C preprocessor strips those type
  # definitions back out of amd_smi/amdsmi.h and the symbols dangle with
  # "unknown type name". Keep these two sides in parity.
  set(_lib_compile_definitions)
  if(_lib STREQUAL "amdsmi")
    set(_lib_compile_definitions ENABLE_ESMI_LIB)
  endif()

  # Build C-level bindings (cy*.pyx)
  hip_python_add_cython_module(
    TARGET rocm_bindings_systems_cy${_lib}
    MODULE_NAME rocm.bindings.cy${_lib}
    SOURCE src/rocm/bindings/cy${_lib}.pyx
    DESTINATION rocm/bindings
    COMPONENT rocm-bindings-systems
    CYTHON_INCLUDE_DIRS ${SYSTEMS_INCLUDE_DIRS} ${_extra_includes}
    INCLUDE_DIRS ${SYSTEMS_INCLUDE_DIRS} ${_extra_includes}
    LINK_LIBRARIES ${_lib_link_libraries}
    COMPILE_DEFINITIONS ${_lib_compile_definitions}
    DEPENDS ${SYSTEMS_CYTHON_DEPENDS}
  )

  # Build high-level bindings (*.pyx)
  hip_python_add_cython_module(
    TARGET rocm_bindings_${_lib}
    MODULE_NAME rocm.bindings.${_lib}
    SOURCE src/rocm/bindings/${_lib}.pyx
    DESTINATION rocm/bindings
    COMPONENT rocm-bindings-systems
    CYTHON_INCLUDE_DIRS ${SYSTEMS_INCLUDE_DIRS} ${_extra_includes}
    INCLUDE_DIRS ${SYSTEMS_INCLUDE_DIRS} ${_extra_includes}
    LINK_LIBRARIES ${_lib_link_libraries}
    COMPILE_DEFINITIONS ${_lib_compile_definitions}
    DEPENDS ${SYSTEMS_CYTHON_DEPENDS}
  )

  # Install .pxd files for cimport
  install(
    FILES src/rocm/bindings/cy${_lib}.pxd src/rocm/bindings/${_lib}.pxd
    DESTINATION rocm/bindings
    COMPONENT rocm-bindings-systems
  )

  # Install generator-emitted .pyi type stubs for the high-level module
  # (cy* modules are cimport-only — no stubs). No-op until interfacegen
  # emits the file.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/rocm/bindings/${_lib}.pyi")
    install(
      FILES src/rocm/bindings/${_lib}.pyi
      DESTINATION rocm/bindings
      COMPONENT rocm-bindings-systems
    )
  endif()
endforeach()

# NOTE: src/rocm/bindings/ is a PEP 420 implicit namespace package — no
# __init__.py here. The runtime __init__.pxd marker is shipped by
# rocm-bindings-core (the only package that installs it).

# Install the high-level rocm.hipfile sub-package — handcoded Pythonic
# wrapper around the auto-generated rocm.bindings.hipfile bindings.
# Conditional on the hipfile bindings being built (which itself depends
# on libhipfile.so being available); skip otherwise so the wheel doesn't
# ship a wrapper for a binding the runtime can't load.
if(HIP_PYTHON_HIPFILE_AVAILABLE AND "hipfile" IN_LIST HIP_PYTHON_SELECTED_SYSTEMS)
  install(
    FILES
      src/rocm/hipfile/__init__.py
      src/rocm/hipfile/buffer.py
      src/rocm/hipfile/driver.py
      src/rocm/hipfile/enums.py
      src/rocm/hipfile/error.py
      src/rocm/hipfile/file.py
      src/rocm/hipfile/properties.py
    DESTINATION rocm/hipfile
    COMPONENT rocm-bindings-systems
  )
endif()
# Optional libhipfile bundling. MUST come after the foreach loop above
# because bundled/libhipfile/CMakeLists.txt sets RPATH on the cyhipfile /
# hipfile cython module targets created in that loop. The subdirectory
# is always processed; the bundling block inside is conditional on
# HIP_PYTHON_BUNDLE_LIBHIPFILE (default OFF).
add_subdirectory(bundled/libhipfile)


# Aggregate target for this package
set(_all_systems_targets)
foreach(_lib IN LISTS HIP_PYTHON_SELECTED_SYSTEMS)
  list(APPEND _all_systems_targets
    rocm_bindings_systems_cy${_lib}
    rocm_bindings_${_lib}
  )
endforeach()

add_custom_target(package_rocm_bindings_systems
  DEPENDS ${_all_systems_targets}
)

# Report the surviving module list to the unified build so it can skip this
# package's wheel and sdist targets when nothing was built, rather than
# publishing a wheel that advertises bindings it does not contain. The list is
# empty on the Windows ROCm SDK, which ships none of rccl, roctx, hipfile or
# amdsmi. Harmless in a standalone single-package configure, where PARENT_SCOPE
# is just the top level.
set(HIP_PYTHON_SYSTEMS_SELECTED_MODULES "${HIP_PYTHON_SELECTED_SYSTEMS}" PARENT_SCOPE)
