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

# Define all libraries to build
set(HIP_PYTHON_ALL_LIBRARIES hipblas hipsolver rccl hiprand hipfft hipsparse roctx)

# Allow selective building of libraries
hip_python_define_module_options(HIP_PYTHON_ENABLE_LIB HIP_PYTHON_LIBS "Build library module" ${HIP_PYTHON_ALL_LIBRARIES})
hip_python_collect_enabled_modules(HIP_PYTHON_SELECTED_LIBRARIES HIP_PYTHON_ENABLE_LIB ${HIP_PYTHON_ALL_LIBRARIES})

# Collect Cython dependencies
hip_python_collect_cython_depends(LIBRARIES_CYTHON_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/rocm/bindings/*.pxd"
  "${CMAKE_CURRENT_SOURCE_DIR}/rocm/bindings/*.pyx"
)

set(LIBRARIES_INCLUDE_DIRS
  "${CMAKE_CURRENT_SOURCE_DIR}"
)

# Use global include dirs if available (unified build), otherwise use local
if(DEFINED HIP_PYTHON_GLOBAL_INCLUDE_DIRS)
  list(APPEND LIBRARIES_INCLUDE_DIRS ${HIP_PYTHON_GLOBAL_INCLUDE_DIRS})
endif()

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
    SOURCE rocm/bindings/cy${_lib}.pyx
    DESTINATION rocm/bindings
  COMPONENT rocm-bindings-libraries
    CYTHON_INCLUDE_DIRS ${LIBRARIES_INCLUDE_DIRS}
    INCLUDE_DIRS ${LIBRARIES_INCLUDE_DIRS}
    LINK_LIBRARIES ${_lib_link_libraries}
    DEPENDS ${LIBRARIES_CYTHON_DEPENDS}
  )

  # Build high-level bindings (*.pyx) - install at rocm/bindings/ level
  hip_python_add_cython_module(
    TARGET rocm_bindings_${_lib}
    MODULE_NAME rocm.bindings.${_lib}
    SOURCE rocm/bindings/${_lib}.pyx
    DESTINATION rocm/bindings
  COMPONENT rocm-bindings-libraries
    CYTHON_INCLUDE_DIRS ${LIBRARIES_INCLUDE_DIRS}
    INCLUDE_DIRS ${LIBRARIES_INCLUDE_DIRS}
    LINK_LIBRARIES ${_lib_link_libraries}
    DEPENDS ${LIBRARIES_CYTHON_DEPENDS}
  )

  # Install .pxd files for cimport
  install(
    FILES rocm/bindings/cy${_lib}.pxd rocm/bindings/${_lib}.pxd
    DESTINATION rocm/bindings
    COMPONENT rocm-bindings-libraries
  )

  # Install generator-emitted .pyi type stubs for the high-level module
  # (cy* modules are cimport-only — no stubs; see plan §B.2). No-op until
  # the interfacegen code generator emits the file.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/rocm/bindings/${_lib}.pyi")
    install(
      FILES rocm/bindings/${_lib}.pyi
      DESTINATION rocm/bindings
      COMPONENT rocm-bindings-libraries
    )
  endif()
endforeach()

# NOTE: rocm/bindings/ is a PEP 420 implicit namespace package
# It intentionally has NO __init__.py file to allow multiple packages
# (rocm-bindings-core, rocm-bindings-hip, rocm-bindings-libraries)
# to contribute modules to the same namespace.
# See: https://peps.python.org/pep-0420/

# Configure and install version file
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/_version.py.in"
  "${CMAKE_CURRENT_BINARY_DIR}/_version.py"
  @ONLY
)
install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/_version.py"
  DESTINATION rocm/bindings
  COMPONENT rocm-bindings-libraries
)

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
