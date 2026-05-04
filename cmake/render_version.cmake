function(hip_python_git_output out_var)
  execute_process(
    COMMAND git ${ARGN}
    WORKING_DIRECTORY "${REPO_ROOT}"
    RESULT_VARIABLE git_result
    OUTPUT_VARIABLE git_output
    ERROR_VARIABLE git_error
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(NOT git_result EQUAL 0)
    message(FATAL_ERROR "git ${ARGN} failed: ${git_error}")
  endif()
  set(${out_var} "${git_output}" PARENT_SCOPE)
endfunction()

function(hip_python_render_package_version package_name)
  if(package_name STREQUAL "hip-python")
    set(parent_dir "${REPO_ROOT}/hip-python/hip")
  elseif(package_name STREQUAL "hip-python-as-cuda")
    set(parent_dir "${REPO_ROOT}/hip-python-as-cuda/cuda")
  else()
    message(FATAL_ERROR "unsupported package name: ${package_name}")
  endif()

  hip_python_git_output(current_branch rev-parse --abbrev-ref HEAD)
  hip_python_git_output(current_rev rev-parse --short HEAD)
  hip_python_git_output(current_rev_count rev-list ${current_branch} --count)

  file(READ "${parent_dir}/_version.py.in" rendered_version)
  string(REPLACE "{HIP_PYTHON_VERSION_SHORT}" "${current_rev_count}" rendered_version "${rendered_version}")
  string(REPLACE "{HIP_PYTHON_VERSION}" "${current_rev_count}" rendered_version "${rendered_version}")
  string(REPLACE "{HIP_PYTHON_BRANCH}" "${current_branch}" rendered_version "${rendered_version}")
  string(REPLACE "{HIP_PYTHON_REV}" "${current_rev}" rendered_version "${rendered_version}")
  file(WRITE "${parent_dir}/_version.py" "${rendered_version}")
endfunction()

get_filename_component(REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)

if(NOT DEFINED HP_PACKAGE)
  set(HP_PACKAGE "all")
endif()

if(HP_PACKAGE STREQUAL "all")
  hip_python_render_package_version("hip-python")
  hip_python_render_package_version("hip-python-as-cuda")
else()
  hip_python_render_package_version("${HP_PACKAGE}")
endif()