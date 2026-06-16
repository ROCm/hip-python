include_guard(GLOBAL)

include(CMakeParseArguments)

set(Python_FIND_VIRTUALENV FIRST)
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

# Ensure the per-package VERSION file exists for the calling
# per-package CMakeLists.txt (populating it from the repo-root VERSION
# on sdist builds) and export it as HIP_PYTHON_VERSION_FULL in the
# parent scope. scikit-build-core reads the version from the VERSION
# file directly (metadata.version.input = "VERSION" in pyproject.toml).
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

# Append cross-package include paths so cython cimports resolve.
#
# In the unified build (driven by `packages/CMakeLists.txt`),
# `HIP_PYTHON_GLOBAL_INCLUDE_DIRS` already lists every enabled
# package's root + `<root>/src`. Per-package CMakeLists then just
# extend their local INCLUDE_DIRS with that variable.
#
# In the standalone wheel build (`python -m build` subprocess for a
# single package), the unified-build's GLOBAL_INCLUDE_DIRS is not
# forwarded into the scikit-build-core subprocess, so cross-package
# cimports (e.g. `rocm.bindings.util.loader` from rocm-bindings-core,
# `rocm.bindings.cyhip` from rocm-bindings-hip) would fail to resolve.
#
# This helper auto-discovers every sibling package in the
# `packages/<pkg>/src` layout relative to the calling package's
# parent directory and appends each `<sibling-root>` + `<sibling-src>`
# to the named cache variable. No SIBLINGS argument is needed — the
# discovery walks `${CMAKE_CURRENT_SOURCE_DIR}/../*` and includes
# every entry that has an existing `<dir>/src` subdirectory. The
# self-package's own root + src remain on the include list (the
# caller has already added them); duplicates from autodiscovery are
# harmless to Cython.
#
# Usage:
#   hip_python_append_sibling_includes(MY_INCLUDE_DIRS)
#
# Wraps the `if(DEFINED HIP_PYTHON_GLOBAL_INCLUDE_DIRS) ... else() ...
# endif()` idiom so each per-package CMakeLists drops the inline
# branching AND avoids hard-coded sibling lists that drift as the
# package set evolves.
function(hip_python_append_sibling_includes out_var)
  set(_local "${${out_var}}")
  if(DEFINED HIP_PYTHON_GLOBAL_INCLUDE_DIRS)
    list(APPEND _local ${HIP_PYTHON_GLOBAL_INCLUDE_DIRS})
  else()
    # Auto-discover siblings: every dir under packages/ that has its
    # own src/ subdir. Conventionally each hip-python package is laid
    # out as packages/<pkg>/src/rocm/bindings/... or
    # packages/<pkg>/src/cuda/bindings/...
    file(GLOB _sibling_roots
      LIST_DIRECTORIES true
      "${CMAKE_CURRENT_SOURCE_DIR}/../*"
    )
    foreach(_sibling_root IN LISTS _sibling_roots)
      if(IS_DIRECTORY "${_sibling_root}" AND EXISTS "${_sibling_root}/src")
        list(APPEND _local "${_sibling_root}" "${_sibling_root}/src")
      endif()
    endforeach()
  endif()
  set(${out_var} "${_local}" PARENT_SCOPE)
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
  set(options "SKIP_AUDITWHEEL")
  set(oneValueArgs TARGET PACKAGE_DIR OUTPUT_DIR COMPONENT)
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

  if(ARG_COMPONENT)
    # ----------------------------------------------------------------
    # Assemble the wheel from the unified build's compiled output.
    #
    # The unified build has already compiled every Cython extension for
    # this package. Running `python -m build` here would re-invoke
    # scikit-build-core, which compiles every extension a SECOND time in
    # its own per-package build tree (a different CMake source root, so
    # the unified build's object files cannot be reused -- see
    # share/design/BUILDING.md). Instead, the assembler collects the
    # already-compiled install component plus the wheel.packages source
    # overlay and packs an equivalent wheel, so each module is compiled
    # exactly once.
    #
    # `cmake --install` only copies files (no nested build), so there is
    # no jobserver-inheritance hazard and no need to detach from the
    # outer make's MAKEFLAGS. The resulting wheel carries a generic
    # linux_<arch> platform tag; the auditwheel/copy/stamp tail below
    # retags it to manylinux exactly as before.
    #
    # Resolve the assembler from the unified top-level source dir
    # (CMAKE_SOURCE_DIR is `packages/`, so `../cmake` is the repo-root
    # cmake dir where the script lives). We deliberately do NOT use
    # CMAKE_CURRENT_FUNCTION_LIST_DIR: this shared helper is mirrored
    # into each packages/<pkg>/cmake/ and re-included, and one package
    # (rocm-bindings-compiler) includes its local mirror by absolute
    # path. Because that mirror is a different file path,
    # include_guard(GLOBAL) does not block it and it re-defines these
    # functions, leaving FUNCTION_LIST_DIR pointing at a per-package
    # mirror dir that does not contain the assembler script. The
    # assembler is only ever invoked from the unified build, so anchoring
    # to CMAKE_SOURCE_DIR is correct for every caller.
    set(_assemble_script
        "${CMAKE_SOURCE_DIR}/../cmake/hip_python_assemble_wheel.py")
    if(NOT EXISTS "${_assemble_script}")
      message(FATAL_ERROR
        "Wheel assembler script not found at ${_assemble_script}. "
        "It must live in the repo-root cmake/ directory next to "
        "HipPythonBuild.cmake.")
    endif()
    set(WHEEL_COMMANDS
      COMMAND ${CMAKE_COMMAND} -E make_directory "${TEMP_WHEEL_DIR}"
      COMMAND ${Python_EXECUTABLE} "${_assemble_script}"
              --cmake "${CMAKE_COMMAND}"
              --build-dir "${CMAKE_BINARY_DIR}"
              --component "${ARG_COMPONENT}"
              --package-dir "${ARG_PACKAGE_DIR}"
              --output-dir "${TEMP_WHEEL_DIR}"
              --config $<CONFIG>
    )
  else()
    # ----------------------------------------------------------------
    # Pure-Python packages (hip-python) have no compiled extensions and
    # use the setuptools backend, so there is nothing to collect from the
    # unified build -- build them directly with `python -m build`.
    #
    # NOTE: per-package VERSION is populated at unified-CMake configure
    # time by the configure_file() loop in packages/CMakeLists.txt, so it
    # already exists in ${ARG_PACKAGE_DIR}/VERSION when this command runs.
    #
    # We strip MAKEFLAGS/MFLAGS/MAKELEVEL/GNUMAKEFLAGS from the wheel-build
    # subprocess. The outer all_wheels build runs under gmake which exports
    # a jobserver pipe via MAKEFLAGS; scikit-build-core's nested ninja
    # invocation tries to attach to that jobserver, fails to initialize
    # the inherited file descriptors, and then gcc intermittently fails
    # to write the dependency file mid-compile on the largest generated
    # ``.c`` files (core.c is 200k+ lines). Detaching from the jobserver
    # lets the nested ninja schedule its own jobs cleanly.
    set(WHEEL_COMMANDS
      # Create temporary directory
      COMMAND ${CMAKE_COMMAND} -E make_directory "${TEMP_WHEEL_DIR}"
      # Build wheel to temporary directory.
      # Forward HIP_PYTHON_* CMake options so the per-package scikit-build-core
      # configure (which is a separate CMake invocation) sees the same values
      # as the top-level configure that drives all_wheels.
      COMMAND ${CMAKE_COMMAND} -E env
              --unset=MAKEFLAGS --unset=MFLAGS
              --unset=MAKELEVEL --unset=GNUMAKEFLAGS
              ${Python_EXECUTABLE} -m build
              --wheel
              --no-isolation
              --outdir=${TEMP_WHEEL_DIR}
              -Ccmake.define.HIP_PYTHON_BUNDLE_LIBLLVM=${HIP_PYTHON_BUNDLE_LIBLLVM}
              -Ccmake.define.HIP_PYTHON_FORCE_BUILD_LIBLLVM=${HIP_PYTHON_FORCE_BUILD_LIBLLVM}
    )
  endif()

  # Add auditwheel repair or direct copy to output directory.
  #
  # auditwheel is a Linux-only ELF tool, so the repair branch is gated on
  # `NOT WIN32` (in addition to the opt-in HIP_PYTHON_AUDITWHEEL_REPAIR).
  # On Windows the build always falls through to the cross-platform
  # copy_directory path below -- the assembler already emits the correct
  # win_amd64 platform tag, so no retag is needed. The temp dir holds
  # only the freshly produced wheel, so copy_directory needs no glob.
  # SKIP_AUDITWHEEL: some targets produce a platform-tagged wheel with no
  # ELF/shared library (e.g. numba-hip, which setup.py marks non-pure for
  # platlib placement). auditwheel rejects those ("not a platform wheel"),
  # so they must fall through to the plain copy path below.
  if(HIP_PYTHON_AUDITWHEEL_REPAIR AND NOT WIN32 AND NOT ARG_SKIP_AUDITWHEEL)
    list(APPEND WHEEL_COMMANDS
      # Create final output directory
      COMMAND ${CMAKE_COMMAND} -E make_directory "${ARG_OUTPUT_DIR}"
      # Repair wheel with --exclude "*" to prevent bundling ROCm libraries
      # Use --allow-pure-python-wheel to handle pure Python wheels gracefully
      # Output goes directly to final output directory
      # Use shell to expand glob pattern (Linux-only branch)
      COMMAND sh -c "${AUDITWHEEL_EXECUTABLE} repair --exclude '*' --allow-pure-python-wheel -w '${ARG_OUTPUT_DIR}' '${TEMP_WHEEL_DIR}'/*.whl"
      # Remove temporary directory (contains original linux wheel)
      COMMAND ${CMAKE_COMMAND} -E rm -rf "${TEMP_WHEEL_DIR}"
    )
  else()
    list(APPEND WHEEL_COMMANDS
      # Copy the produced wheel to the output directory. The temp dir
      # contains only the single .whl, so a recursive directory copy is
      # equivalent to the old shell glob but works on every platform.
      COMMAND ${CMAKE_COMMAND} -E copy_directory "${TEMP_WHEEL_DIR}" "${ARG_OUTPUT_DIR}"
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


# Add a custom target that runs `python -m build --sdist` from
# ${PACKAGE_DIR}, dropping the resulting .tar.gz into ${OUTPUT_DIR}
# (defaults to ${CMAKE_BINARY_DIR}/dist).
#
# Args:
#   TARGET       Name of the CMake target to create.
#   PACKAGE_DIR  Per-package source dir containing pyproject.toml.
#   OUTPUT_DIR   Destination dir for the sdist tarball (created if
#                missing). Defaults to ${CMAKE_BINARY_DIR}/dist.
#   DEPENDS      Optional CMake target dependencies (rare for sdist;
#                no native build is needed since the sdist just packs
#                source files + the per-package VERSION + cmake helper
#                that the unified configure step has already populated).
function(hip_python_add_sdist_target)
  set(options "")
  set(oneValueArgs TARGET PACKAGE_DIR OUTPUT_DIR)
  set(multiValueArgs DEPENDS)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if(NOT ARG_OUTPUT_DIR)
    set(ARG_OUTPUT_DIR "${CMAKE_BINARY_DIR}/dist")
  endif()

  # Stamp file (sdist filename includes the version string, so we can't
  # use it directly as the OUTPUT of the custom command).
  set(STAMP_FILE "${CMAKE_CURRENT_BINARY_DIR}/${ARG_TARGET}.stamp")
  set(TEMP_SDIST_DIR "${CMAKE_CURRENT_BINARY_DIR}/${ARG_TARGET}_temp")

  # NOTE: per-package VERSION and the shared cmake helper are populated
  # at unified-CMake configure time by the configure_file() loop in
  # python/CMakeLists.txt, so they exist in ${ARG_PACKAGE_DIR}/ when
  # this command runs. The sdist tarball includes them via each
  # per-package pyproject.toml `sdist.include`.
  add_custom_command(
    OUTPUT "${STAMP_FILE}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${TEMP_SDIST_DIR}"
    COMMAND ${Python_EXECUTABLE} -m build
            --sdist
            --no-isolation
            --outdir=${TEMP_SDIST_DIR}
    # Copy the produced sdist to the output dir. The temp dir holds only
    # the single .tar.gz, so a recursive directory copy replaces the old
    # shell glob and works on every platform (no `sh`/`cp`).
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${TEMP_SDIST_DIR}" "${ARG_OUTPUT_DIR}"
    COMMAND ${CMAKE_COMMAND} -E rm -rf "${TEMP_SDIST_DIR}"
    COMMAND ${CMAKE_COMMAND} -E touch "${STAMP_FILE}"
    WORKING_DIRECTORY "${ARG_PACKAGE_DIR}"
    DEPENDS ${ARG_DEPENDS}
    COMMENT "Building sdist for ${ARG_TARGET} -> ${ARG_OUTPUT_DIR}"
    VERBATIM
  )

  add_custom_target(${ARG_TARGET}
    DEPENDS "${STAMP_FILE}"
  )
endfunction()


# Add a developer-only target that runs `mypy stubgen` against a
# handcoded Cython module's compiled extension to (re)generate its
# `.pyi` type stub. The generated `.pyi` is written *into the source
# tree* (next to the `.pyx`), so the dev sees the diff in
# `git status` and can commit it.
#
# This is NOT a build-time step: the per-package wheel/sdist targets
# do not depend on it, end-user `pip install` does not invoke it,
# and `mypy` is NOT a build-system dependency. The dev opts in via
# `-DHIP_PYTHON_ENABLE_STUBGEN=ON` at CMake configure time and runs
# `cmake --build build --target <module>_stub` (or one of the
# aggregate targets) explicitly.
#
# Args:
#   MODULE          Dotted module name (e.g. rocm.bindings.util.types).
#   CYTHON_TARGET   The cython add-module target whose compiled .so
#                   stubgen should introspect. Used for DEPENDS and
#                   for naming the generated <CYTHON_TARGET>_stub
#                   target.
#   SOURCE_PYI_DIR  Absolute path to the directory next to the .pyx
#                   where the generated .pyi should land.
#
# Output target name: ${CYTHON_TARGET}_stub.
function(hip_python_add_stubgen_target)
  set(options "")
  set(oneValueArgs MODULE CYTHON_TARGET SOURCE_PYI_DIR)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "" ${ARGN})

  string(REPLACE "." ";" _parts "${ARG_MODULE}")
  list(GET _parts -1 _leaf)
  set(_pyi_path "${ARG_SOURCE_PYI_DIR}/${_leaf}.pyi")

  # Per-call PYTHONPATH staging dir mirroring the package layout so
  # `python -m mypy.stubgen --module <MODULE>` can resolve the
  # just-built .so. Convert "a.b.c" -> "a/b" (parent path).
  set(_staging "${CMAKE_CURRENT_BINARY_DIR}/_stubgen_staging/${ARG_CYTHON_TARGET}")
  list(REMOVE_AT _parts -1)
  string(REPLACE ";" "/" _module_subdir "${_parts}")

  # stubgen with `--module a.b.c --output OUT` writes
  # OUT/a/b/c.pyi (preserves the dotted path inside OUT). Write to
  # a temp dir per invocation, then move the single leaf .pyi to the
  # source-tree destination.
  set(_stubgen_outdir "${_staging}_out")

  add_custom_target(${ARG_CYTHON_TARGET}_stub
    # Tear down + rebuild the staging tree to keep it in sync with
    # the just-built .so on every invocation.
    COMMAND ${CMAKE_COMMAND} -E rm -rf "${_staging}" "${_stubgen_outdir}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${_staging}/${_module_subdir}"
    # Symlink the just-built .so into the leaf staging directory so
    # `import <module>` works under PYTHONPATH=${_staging}.
    COMMAND ${CMAKE_COMMAND} -E create_symlink
            "$<TARGET_FILE:${ARG_CYTHON_TARGET}>"
            "${_staging}/${_module_subdir}/$<TARGET_FILE_NAME:${ARG_CYTHON_TARGET}>"
    # Make sure the destination dir exists.
    COMMAND ${CMAKE_COMMAND} -E make_directory "${ARG_SOURCE_PYI_DIR}"
    # Run stubgen via the dedicated CLI entry point. `python -m
    # mypy.stubgen` does not work when mypy is installed as a
    # compiled .so (no code object for `-m`); the `stubgen` script
    # in the same venv works regardless of the install layout.
    # --include-docstrings: copy `__doc__` from the compiled module
    # into the generated .pyi. Without this, mypy stubgen emits only
    # signatures, leaving sphinx-autoapi with empty class/method
    # description columns on the rendered docs (rocm.bindings.util.types
    # was the visible regression that motivated this flag).
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${_staging}
            ${HIP_PYTHON_STUBGEN_EXECUTABLE}
            --module ${ARG_MODULE}
            --output ${_stubgen_outdir}
            --include-private
            --include-docstrings
    # Move the generated leaf .pyi from the dotted-path layout into
    # the source-tree destination next to the .pyx.
    COMMAND ${CMAKE_COMMAND} -E copy
            "${_stubgen_outdir}/${_module_subdir}/${_leaf}.pyi"
            "${_pyi_path}"
    COMMAND ${CMAKE_COMMAND} -E rm -rf "${_stubgen_outdir}"
    DEPENDS ${ARG_CYTHON_TARGET}
    BYPRODUCTS "${_pyi_path}"
    COMMENT "stubgen ${ARG_MODULE} -> ${_pyi_path}"
    VERBATIM
  )
endfunction()