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
# (rccl), tracing/profiling instrumentation (roctx), and accelerated file
# I/O (hipfile). Math/FFT/random/sparse libraries live in the sibling
# rocm-bindings-libraries wheel.
set(HIP_PYTHON_ALL_SYSTEMS rccl roctx hipfile)

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
)

# Use global include dirs if available (unified build), otherwise use local
if(DEFINED HIP_PYTHON_GLOBAL_INCLUDE_DIRS)
  list(APPEND SYSTEMS_INCLUDE_DIRS ${HIP_PYTHON_GLOBAL_INCLUDE_DIRS})
endif()

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

# Configure and install version file
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/_version.py.in"
  "${CMAKE_CURRENT_BINARY_DIR}/_version.py"
  @ONLY
)
install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/_version.py"
  DESTINATION rocm/bindings
  COMPONENT rocm-bindings-systems
)

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
