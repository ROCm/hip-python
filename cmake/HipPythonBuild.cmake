include_guard(GLOBAL)

include(CMakeParseArguments)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

function(hip_python_get_rocm_path_default out_var)
  if(DEFINED ENV{ROCM_PATH} AND NOT "$ENV{ROCM_PATH}" STREQUAL "")
    set(_value "$ENV{ROCM_PATH}")
  elseif(DEFINED ENV{ROCM_HOME} AND NOT "$ENV{ROCM_HOME}" STREQUAL "")
    set(_value "$ENV{ROCM_HOME}")
  else()
    set(_value "/opt/rocm")
  endif()
  set(${out_var} "${_value}" PARENT_SCOPE)
endfunction()

function(hip_python_get_env_default out_var env_var default_value)
  if(DEFINED ENV{${env_var}} AND NOT "$ENV{${env_var}}" STREQUAL "")
    set(${out_var} "$ENV{${env_var}}" PARENT_SCOPE)
  else()
    set(${out_var} "${default_value}" PARENT_SCOPE)
  endif()
endfunction()

# Resolve the package version string for the calling per-package
# CMakeLists.txt and export it as HIP_PYTHON_VERSION_FULL,
# HIP_PYTHON_VERSION_NAME, and HIP_PYTHON_LONG_VERSION_NAME in
# the parent scope.
#
# Behavior:
# - sdist build (SKBUILD_STATE=sdist): copy repo-root ../../VERSION
#   into ${CMAKE_CURRENT_SOURCE_DIR}/VERSION so the tarball includes it.
#   Repo-root VERSION must exist (FATAL_ERROR if not).
# - Other builds: require ${CMAKE_CURRENT_SOURCE_DIR}/VERSION to exist.
#   It is populated either at unified-CMake configure time (the
#   `python/CMakeLists.txt` configure_file loop) or by an extracted
#   sdist tarball. Missing -> FATAL_ERROR with a clear message.
function(hip_python_resolve_version)
  set(_pkg_version_file "${CMAKE_CURRENT_SOURCE_DIR}/VERSION")
  set(_repo_version_file "${CMAKE_CURRENT_SOURCE_DIR}/../../VERSION")

  if(DEFINED SKBUILD_STATE AND SKBUILD_STATE STREQUAL "sdist")
    if(NOT EXISTS "${_repo_version_file}")
      message(FATAL_ERROR
        "sdist build: repo-root VERSION not found at ${_repo_version_file}.")
    endif()
    configure_file("${_repo_version_file}" "${_pkg_version_file}" COPYONLY)
  endif()

  if(NOT EXISTS "${_pkg_version_file}")
    message(FATAL_ERROR
      "VERSION file not found at ${_pkg_version_file}. "
      "Run the unified configure first: "
      "`cmake -B build -S python` from the repo root, "
      "which populates each per-package VERSION from the canonical "
      "repo-root VERSION.")
  endif()

  file(READ "${_pkg_version_file}" _hp_version)
  string(STRIP "${_hp_version}" _hp_version)
  set(HIP_PYTHON_VERSION_FULL "${_hp_version}" PARENT_SCOPE)
  set(HIP_PYTHON_VERSION_NAME "${_hp_version}" PARENT_SCOPE)
  set(HIP_PYTHON_LONG_VERSION_NAME "${_hp_version}" PARENT_SCOPE)
endfunction()

function(hip_python_initialize)
  hip_python_get_rocm_path_default(_rocm_path_default)
  set(ROCM_PATH "${_rocm_path_default}" CACHE PATH "Path to the ROCm installation")

  if("${ROCM_PATH}" STREQUAL "")
    message(FATAL_ERROR "ROCm path is not set. Set ROCM_PATH or ROCM_HOME.")
  endif()

  if(NOT EXISTS "${ROCM_PATH}/include")
    message(FATAL_ERROR "ROCm include directory not found under ${ROCM_PATH}")
  endif()

  if(EXISTS "${ROCM_PATH}/lib")
    set(HIP_PYTHON_ROCM_LIB_DIR "${ROCM_PATH}/lib" CACHE PATH "ROCm library directory")
  elseif(EXISTS "${ROCM_PATH}/lib64")
    set(HIP_PYTHON_ROCM_LIB_DIR "${ROCM_PATH}/lib64" CACHE PATH "ROCm library directory")
  else()
    message(FATAL_ERROR "ROCm library directory not found under ${ROCM_PATH}")
  endif()

  hip_python_get_env_default(_hip_platform_default HIP_PLATFORM amd)
  set(HIP_PLATFORM "${_hip_platform_default}" CACHE STRING "HIP platform")
  set_property(CACHE HIP_PLATFORM PROPERTY STRINGS amd hcc)

  string(TOLOWER "${HIP_PLATFORM}" _hip_platform)
  if(NOT _hip_platform STREQUAL "amd" AND NOT _hip_platform STREQUAL "hcc")
    message(FATAL_ERROR "Currently only platform 'amd' is supported")
  endif()

  set(HIP_PLATFORM "${_hip_platform}" CACHE STRING "HIP platform" FORCE)
  set(HIP_PYTHON_COMMON_COMPILE_DEFINITIONS
      "__HIP_PLATFORM_AMD__"
      "__half=uint16_t"
      CACHE INTERNAL "Common compile definitions for hip-python extensions" FORCE)
  set(HIP_PYTHON_ROCM_INCLUDE_DIR "${ROCM_PATH}/include" CACHE PATH "ROCm include directory")
endfunction()

function(hip_python_select_modules out_var selection)
  set(_available ${ARGN})
  string(REPLACE " " "" _selection "${selection}")
  if("${_selection}" STREQUAL "" OR "${_selection}" STREQUAL "*")
    set(${out_var} ${_available} PARENT_SCOPE)
    return()
  endif()

  string(SUBSTRING "${_selection}" 0 1 _prefix)
  if(_prefix STREQUAL "^")
    string(SUBSTRING "${_selection}" 1 -1 _trimmed)
    string(REPLACE "," ";" _excluded "${_trimmed}")
    foreach(_name IN LISTS _excluded)
      if(NOT _name IN_LIST _available)
        message(FATAL_ERROR "library name '${_name}' is not valid, use one of: ${_available}")
      endif()
    endforeach()
    set(_selected ${_available})
    list(REMOVE_ITEM _selected ${_excluded})
  else()
    string(REPLACE "," ";" _selected "${_selection}")
    foreach(_name IN LISTS _selected)
      if(NOT _name IN_LIST _available)
        message(FATAL_ERROR "library name '${_name}' is not valid, use one of: ${_available}")
      endif()
    endforeach()
  endif()

  set(${out_var} ${_selected} PARENT_SCOPE)
endfunction()

function(hip_python_define_module_options option_prefix legacy_env_var option_doc)
  set(_modules ${ARGN})
  set(_has_legacy_selection FALSE)
  if(DEFINED ENV{${legacy_env_var}} AND NOT "$ENV{${legacy_env_var}}" STREQUAL "")
    set(_has_legacy_selection TRUE)
    message(DEPRECATION "${legacy_env_var} is deprecated; use individual CMake cache options named ${option_prefix}_<MODULE> instead.")
    hip_python_select_modules(_legacy_selected "$ENV{${legacy_env_var}}" ${_modules})
  endif()

  foreach(_module IN LISTS _modules)
    if(_has_legacy_selection)
      if(_module IN_LIST _legacy_selected)
        set(_default ON)
      else()
        set(_default OFF)
      endif()
    else()
      set(_default ON)
    endif()

    string(TOUPPER "${_module}" _module_upper)
    option(${option_prefix}_${_module_upper} "${option_doc} '${_module}'" ${_default})
  endforeach()
endfunction()

function(hip_python_collect_enabled_modules out_var option_prefix)
  set(_selected)
  foreach(_module IN LISTS ARGN)
    string(TOUPPER "${_module}" _module_upper)
    if(${option_prefix}_${_module_upper})
      list(APPEND _selected "${_module}")
    endif()
  endforeach()
  set(${out_var} ${_selected} PARENT_SCOPE)
endfunction()

function(hip_python_collect_cython_depends out_var)
  set(_files)
  foreach(_pattern IN LISTS ARGN)
    file(GLOB_RECURSE _matched CONFIGURE_DEPENDS "${_pattern}")
    list(APPEND _files ${_matched})
  endforeach()
  list(REMOVE_DUPLICATES _files)
  set(${out_var} ${_files} PARENT_SCOPE)
endfunction()

function(hip_python_resolve_python_package_dir out_var package_name fallback_dir)
  if(EXISTS "${fallback_dir}")
    set(${out_var} "${fallback_dir}" PARENT_SCOPE)
    return()
  endif()

  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import importlib.util, pathlib; spec = importlib.util.find_spec('${package_name}'); print(pathlib.Path(spec.origin).resolve().parent.parent if spec and spec.origin else '', end='')"
    RESULT_VARIABLE _status
    OUTPUT_VARIABLE _resolved_path
    ERROR_VARIABLE _stderr
  )
  if(NOT _status EQUAL 0 OR "${_resolved_path}" STREQUAL "")
    message(FATAL_ERROR "Failed to resolve python package '${package_name}'. Install it or build from the monorepo checkout. ${_stderr}")
  endif()
  set(${out_var} "${_resolved_path}" PARENT_SCOPE)
endfunction()

function(hip_python_add_cython_module)
  set(options)
  set(oneValueArgs TARGET MODULE_NAME SOURCE DESTINATION COMPONENT)
  set(multiValueArgs CYTHON_INCLUDE_DIRS INCLUDE_DIRS LINK_LIBRARIES DEPENDS COMPILE_DEFINITIONS)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  get_filename_component(_source_abs "${ARG_SOURCE}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
  get_filename_component(_source_name "${_source_abs}" NAME_WE)

  set(_generated_dir "${CMAKE_CURRENT_BINARY_DIR}/cython/${ARG_TARGET}")
  file(MAKE_DIRECTORY "${_generated_dir}")
  set(_generated_c "${_generated_dir}/${_source_name}.c")

  set(_cython_command "${Python_EXECUTABLE}" -m cython -3 -X embedsignature=True -o "${_generated_c}")
  foreach(_include_dir IN LISTS ARG_CYTHON_INCLUDE_DIRS)
    list(APPEND _cython_command -I "${_include_dir}")
  endforeach()
  list(APPEND _cython_command "${_source_abs}")

  add_custom_command(
    OUTPUT "${_generated_c}"
    COMMAND ${_cython_command}
    DEPENDS "${_source_abs}" ${ARG_DEPENDS}
    COMMAND_EXPAND_LISTS
    VERBATIM
  )

  Python_add_library(${ARG_TARGET} MODULE WITH_SOABI "${_generated_c}")
  target_compile_definitions(
    ${ARG_TARGET}
    PRIVATE
      ${HIP_PYTHON_COMMON_COMPILE_DEFINITIONS}
      ${ARG_COMPILE_DEFINITIONS}
  )
  target_include_directories(
    ${ARG_TARGET}
    PRIVATE
      ${ARG_INCLUDE_DIRS}
      "${HIP_PYTHON_ROCM_INCLUDE_DIR}"
  )
  if(ARG_LINK_LIBRARIES)
    target_link_directories(${ARG_TARGET} PRIVATE "${HIP_PYTHON_ROCM_LIB_DIR}")
    target_link_libraries(${ARG_TARGET} PRIVATE ${ARG_LINK_LIBRARIES})
  endif()

  string(REPLACE "." ";" _module_parts "${ARG_MODULE_NAME}")
  list(LENGTH _module_parts _module_parts_len)
  math(EXPR _leaf_index "${_module_parts_len} - 1")
  list(GET _module_parts ${_leaf_index} _module_leaf)
  set_target_properties(${ARG_TARGET} PROPERTIES
    OUTPUT_NAME "${_module_leaf}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${ARG_DESTINATION}"
  )

  # Use package-specific component if provided, otherwise default to Python
  if(ARG_COMPONENT)
    install(TARGETS ${ARG_TARGET} LIBRARY DESTINATION "${ARG_DESTINATION}" COMPONENT ${ARG_COMPONENT})
  else()
    install(TARGETS ${ARG_TARGET} LIBRARY DESTINATION "${ARG_DESTINATION}" COMPONENT Python)
  endif()
endfunction()

function(hip_python_add_wheel_target)
  set(options "")
  set(oneValueArgs TARGET PACKAGE_DIR OUTPUT_DIR)
  set(multiValueArgs DEPENDS)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  # Output directory for wheels
  if(NOT ARG_OUTPUT_DIR)
    set(ARG_OUTPUT_DIR "${CMAKE_BINARY_DIR}/dist")
  endif()

  # Use stamp file since wheel filename includes version/platform tags
  set(STAMP_FILE "${CMAKE_CURRENT_BINARY_DIR}/${ARG_TARGET}.stamp")

  # Always use temporary directory for initial wheel build
  set(TEMP_WHEEL_DIR "${CMAKE_CURRENT_BINARY_DIR}/${ARG_TARGET}_temp")

  # Build the wheel command sequence.
  # NOTE: per-package VERSION is populated at unified-CMake configure
  # time by the configure_file() loop in python/CMakeLists.txt, so it
  # already exists in ${ARG_PACKAGE_DIR}/VERSION when this command runs.
  set(WHEEL_COMMANDS
    # Create temporary directory
    COMMAND ${CMAKE_COMMAND} -E make_directory "${TEMP_WHEEL_DIR}"
    # Build wheel to temporary directory.
    # Forward HIP_PYTHON_* CMake options so the per-package scikit-build-core
    # configure (which is a separate CMake invocation) sees the same values
    # as the top-level configure that drives all_wheels.
    COMMAND ${Python_EXECUTABLE} -m build
            --wheel
            --no-isolation
            --outdir=${TEMP_WHEEL_DIR}
            -Ccmake.define.HIP_PYTHON_BUNDLE_LIBLLVM=${HIP_PYTHON_BUNDLE_LIBLLVM}
            -Ccmake.define.HIP_PYTHON_FORCE_BUILD_LIBLLVM=${HIP_PYTHON_FORCE_BUILD_LIBLLVM}
  )

  # Add auditwheel repair or direct copy to output directory
  if(HIP_PYTHON_AUDITWHEEL_REPAIR)
    list(APPEND WHEEL_COMMANDS
      # Create final output directory
      COMMAND ${CMAKE_COMMAND} -E make_directory "${ARG_OUTPUT_DIR}"
      # Repair wheel with --exclude "*" to prevent bundling ROCm libraries
      # Use --allow-pure-python-wheel to handle pure Python wheels gracefully
      # Output goes directly to final output directory
      # Use shell to expand glob pattern
      COMMAND sh -c "${AUDITWHEEL_EXECUTABLE} repair --exclude '*' --allow-pure-python-wheel -w '${ARG_OUTPUT_DIR}' '${TEMP_WHEEL_DIR}'/*.whl"
      # Remove temporary directory (contains original linux wheel)
      COMMAND ${CMAKE_COMMAND} -E rm -rf "${TEMP_WHEEL_DIR}"
    )
  else()
    list(APPEND WHEEL_COMMANDS
      # Create final output directory
      COMMAND ${CMAKE_COMMAND} -E make_directory "${ARG_OUTPUT_DIR}"
      # Copy wheel from temporary to output directory using shell glob
      COMMAND sh -c "cp '${TEMP_WHEEL_DIR}'/*.whl '${ARG_OUTPUT_DIR}/'"
      # Remove temporary directory
      COMMAND ${CMAKE_COMMAND} -E rm -rf "${TEMP_WHEEL_DIR}"
    )
  endif()

  # Add stamp file touch at the end
  list(APPEND WHEEL_COMMANDS
    COMMAND ${CMAKE_COMMAND} -E touch "${STAMP_FILE}"
  )

  # Custom command to build wheel
  add_custom_command(
    OUTPUT "${STAMP_FILE}"
    ${WHEEL_COMMANDS}
    WORKING_DIRECTORY "${ARG_PACKAGE_DIR}"
    DEPENDS ${ARG_DEPENDS}
    COMMENT "Building wheel for ${ARG_TARGET} -> ${ARG_OUTPUT_DIR}"
    VERBATIM
  )

  # Target depends on stamp file
  add_custom_target(${ARG_TARGET}
    DEPENDS "${STAMP_FILE}"
  )
endfunction()