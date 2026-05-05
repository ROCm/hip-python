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
# (rccl) and tracing/profiling instrumentation (roctx). Math/FFT/random/
# sparse libraries live in the sibling rocm-bindings-libraries wheel.
set(HIP_PYTHON_ALL_SYSTEMS rccl roctx)

# Allow selective building
hip_python_define_module_options(HIP_PYTHON_ENABLE_SYSTEM HIP_PYTHON_SYSTEMS "Build system module" ${HIP_PYTHON_ALL_SYSTEMS})
hip_python_collect_enabled_modules(HIP_PYTHON_SELECTED_SYSTEMS HIP_PYTHON_ENABLE_SYSTEM ${HIP_PYTHON_ALL_SYSTEMS})

# Collect Cython dependencies
hip_python_collect_cython_depends(SYSTEMS_CYTHON_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/rocm/bindings/*.pxd"
  "${CMAKE_CURRENT_SOURCE_DIR}/rocm/bindings/*.pyx"
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

  # Determine link libraries based on runtime linking option
  if(HIP_PYTHON_RUNTIME_LINKING)
    set(_lib_link_libraries)
  else()
    set(_lib_link_libraries ${_lib_name})
  endif()

  # Build C-level bindings (cy*.pyx)
  hip_python_add_cython_module(
    TARGET rocm_bindings_systems_cy${_lib}
    MODULE_NAME rocm.bindings.cy${_lib}
    SOURCE rocm/bindings/cy${_lib}.pyx
    DESTINATION rocm/bindings
    COMPONENT rocm-bindings-systems
    CYTHON_INCLUDE_DIRS ${SYSTEMS_INCLUDE_DIRS}
    INCLUDE_DIRS ${SYSTEMS_INCLUDE_DIRS}
    LINK_LIBRARIES ${_lib_link_libraries}
    DEPENDS ${SYSTEMS_CYTHON_DEPENDS}
  )

  # Build high-level bindings (*.pyx)
  hip_python_add_cython_module(
    TARGET rocm_bindings_${_lib}
    MODULE_NAME rocm.bindings.${_lib}
    SOURCE rocm/bindings/${_lib}.pyx
    DESTINATION rocm/bindings
    COMPONENT rocm-bindings-systems
    CYTHON_INCLUDE_DIRS ${SYSTEMS_INCLUDE_DIRS}
    INCLUDE_DIRS ${SYSTEMS_INCLUDE_DIRS}
    LINK_LIBRARIES ${_lib_link_libraries}
    DEPENDS ${SYSTEMS_CYTHON_DEPENDS}
  )

  # Install .pxd files for cimport
  install(
    FILES rocm/bindings/cy${_lib}.pxd rocm/bindings/${_lib}.pxd
    DESTINATION rocm/bindings
    COMPONENT rocm-bindings-systems
  )

  # Install generator-emitted .pyi type stubs for the high-level module
  # (cy* modules are cimport-only — no stubs). No-op until interfacegen
  # emits the file.
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/rocm/bindings/${_lib}.pyi")
    install(
      FILES rocm/bindings/${_lib}.pyi
      DESTINATION rocm/bindings
      COMPONENT rocm-bindings-systems
    )
  endif()
endforeach()

# NOTE: rocm/bindings/ is a PEP 420 implicit namespace package — no
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
