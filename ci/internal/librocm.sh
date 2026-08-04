#!/usr/bin/env bash
# MIT License
#
# Copyright (c) 2024-2026 Advanced Micro Devices, Inc.
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

# ROCm install and version helpers.
#
# Trimmed copy of aiss-3p-dev-pipelines' public/nodes/rocm/librocm.sh, reduced
# to what hip-python CI needs from a manylinux container. Everything retained
# is byte-identical to upstream so a re-sync stays a readable diff, with two
# exceptions:
#
#   - install_rocm_el replaces upstream's install_rocm_rhel. Only the repository
#     configuration and the package install are kept; upstream additionally
#     installs java, qemu-kvm, dkms, subversion, cmake3 and python39, which
#     would displace the interpreter and toolchain of a manylinux image.
#   - install_rocm rejects the 'therock:gh' and 'internal:...' specifiers.
#     Building TheRock from source and reaching the internal artifact storage
#     need credentials and tooling that only the aiss-3p-dev-pipelines nodes
#     have; nothing else was dropped from the dispatcher.
#
# Upstream's install_rocm_ubuntu is absent for the same reason the workflows no
# longer use a Debian image at all.
#
# Sourced, not executed:
#
#   . ci/internal/librocm.sh
#
# Env read by install_rocm:
#   ROCM_SPECIFIER  required. 'X.Y', 'X.Y.Z', 'therock:X.Y.Z', 'therock:X.Y.ZrcR',
#                   'therock:X.Y.ZaYYYYMMDD', 'therock:*', 'therock:*rc*',
#                   'therock:*a*' or 'preinstalled'.
#   AMDGPU_TARGETS  required for the 'therock:' specifiers; a single target.
#   ROCM_PATH       default '/opt/rocm'.
#   ROCM_PKGS       packages to install for the non-TheRock path; empty
#                   installs the 'rocm' meta package.

# Prints location of the 'rocm_version.h' file.
function __get_rocm_version_header() {
  local rocm_version_h=$(find $(hipconfig --path) -name "rocm_version.h")
  if [[ -z "${rocm_version_h}" ]]; then
    echo "ERROR: Couldn't obtain ROCm versions."
    exit 1
  fi
  printf ${rocm_version_h}
}

# Prints the HIP version.
# Example: 6.2.41133-dd7f95766
function get_hip_version() {
  hipconfig --version
}

# Prints '1' if the hipconfig tool was found.
# Prints nothing otherwise.
# NOTE: Expected to be used as follows: ``[ $(is_hip_installed) ]``.
function is_hip_installed() {
  local hip_version=$(get_hip_version)
  [[ -n "${hip_version}" ]] && printf "1"
}

# Prints '1' if the 'rocm_version.h' header file was found.
# Prints nothing otherwise.
# NOTE: Expected to be used as follows: ``[ $(is_rocm_installed) ]``.
function is_rocm_installed() {
  if [ $(is_hip_installed) ]; then
    local rocm_version_h=$(find $(hipconfig --path) -name "rocm_version.h")
    [[ -n "${rocm_version_h}" ]] && printf "1"
  fi
}

# Prints the >>X<<.Y.Z of the ROCm version X.Y.Z.
function get_rocm_version_major() {
  local rocm_version_h=$(__get_rocm_version_header)
  grep "ROCM_VERSION_MAJOR\s\+[0-9]\+" ${rocm_version_h} | grep -o "[0-9]\+"
}

# Prints the X.>>Y<<.Z of the ROCm version X.Y.Z.
function get_rocm_version_minor() {
  local rocm_version_h=$(__get_rocm_version_header)
  grep "ROCM_VERSION_MINOR\s\+[0-9]\+" ${rocm_version_h} | grep -o "[0-9]\+"
}

# Prints the X.Y.>>Z<< of the ROCm version X.Y.Z.
function get_rocm_version_patch() {
  local rocm_version_h=$(__get_rocm_version_header)
  grep "ROCM_VERSION_PATCH\s\+[0-9]\+" ${rocm_version_h} | grep -o "[0-9]\+"
}

function __make_short_version() {
  local major=${1}
  local minor=${2}
  local patch=${3}
  if [[ "${patch}" == "0" ]]; then
    printf "${major}.${minor}"
  else
    printf "${major}.${minor}.${patch}"
  fi
}

# Prints X.Y if the last number Z in the ROCm version X.Y.Z is 0,
# prints X.Y.Z otherwise.
function get_rocm_version_short() {
  local rocm_version_h=$(__get_rocm_version_header)
  local major=$(grep "ROCM_VERSION_MAJOR\s\+[0-9]\+" ${rocm_version_h} | grep -o "[0-9]\+")
  local minor=$(grep "ROCM_VERSION_MINOR\s\+[0-9]\+" ${rocm_version_h} | grep -o "[0-9]\+")
  local patch=$(grep "ROCM_VERSION_PATCH\s\+[0-9]\+" ${rocm_version_h} | grep -o "[0-9]\+")
  __make_short_version ${major} ${minor} ${patch}
}

# Prints the full ROCm version X.Y.Z.
function get_rocm_version() {
  local rocm_version_h=$(__get_rocm_version_header)
  local major=$(grep "ROCM_VERSION_MAJOR\s\+[0-9]\+" ${rocm_version_h} | grep -o "[0-9]\+")
  local minor=$(grep "ROCM_VERSION_MINOR\s\+[0-9]\+" ${rocm_version_h} | grep -o "[0-9]\+")
  local patch=$(grep "ROCM_VERSION_PATCH\s\+[0-9]\+" ${rocm_version_h} | grep -o "[0-9]\+")
  printf "${major}.${minor}.${patch}"
}

function __validate_rocm_version_expr() {
  local -

  set -xe

  local rocm_specifier="${1}"

  set -u

  # Skip validation for special keywords
  if [[ "${rocm_specifier}" == "preinstalled" ]]; then
    return
  fi

  # See the file header: these two need the aiss-3p-dev-pipelines nodes.
  if [[ "${rocm_specifier}" == "therock:gh" ]] || \
     [[ "${rocm_specifier}" == "internal:"* ]]; then
    printf "ERROR: unsupported ROCM_SPECIFIER '${rocm_specifier}'\n" >&2
    printf "ERROR: building TheRock from source and the internal ROCm builds are only available from the rocm_python* nodes in aiss-3p-dev-pipelines\n" >&2
    return 1
  fi

  # note: therock tarballs use the full version with patch number
  if [[ "${rocm_specifier}" == "therock:"* ]]; then
    local therock_rocm_version="${rocm_specifier#therock:}"
    # Check if therock_rocm_version matches X.Y.Z(.(a|rc)R)? format (where X, Y, Z, R are numbers)
    if [[ ! "${therock_rocm_version}" =~ ^(\*|\*(a|rc)\*|([0-9]+)\.([0-9]+)\.([0-9]+)((a|rc)[0-9]+)?)$ ]]; then
      printf "ERROR: Invalid therock ROCM_SPECIFIER format: '${rocm_specifier}'\n" >&2
      printf "ERROR: Expected formats: 'therock:*', 'therock:*(a|rc)*', 'therock:X.Y.Z(.(a|rc)R)?'\n" >&2
      return 1
    fi
    return
  # Check if ROCM_SPECIFIER matches X.Y or X.Y.Z format (where X, Y, Z are numbers)
  elif [[ "${rocm_specifier}" =~ ^([0-9]+)\.([0-9]+)(\.([0-9]+))?$ ]]; then
    : # pass
  else
    printf "ERROR: Invalid ROCM_SPECIFIER format: '${rocm_specifier}'\n" >&2
    printf "ERROR: Expected format: 'X.Y' or 'X.Y.Z' (numbers), 'preinstalled', 'therock:X.Y.Z', 'therock:X.Y.ZrcR' (R is a number), 'therock:X.Y.ZaD' (D is a date in format 'YYYYMMDD'), 'therock:*', 'therock:*rc*', 'therock:*a*'\n" >&2
    return 1
  fi
}

# Installs ROCm from the repo.radeon.com packages of an enterprise Linux
# (RHEL/AlmaLinux/Rocky) container.
#
# NOTE: Reduced against upstream's install_rocm_rhel; see the file header.
function install_rocm_el() {
  local -

  set -xe

  local ROCM_PKGS="${ROCM_PKGS:-""}"

  set -u

  local el_version_id=$(. /etc/os-release; echo $VERSION_ID)
  local el_version_major=$(echo ${el_version_id} | cut -d '.' -f 1)

  # repo.radeon.com serves X.Y for a '.0' patch release and X.Y.Z otherwise.
  local rocm_patch=$(printf ${ROCM_SPECIFIER} | cut -d "." -f 3)
  local rocm_ver_short
  if [[ "${rocm_patch}" == "0" || "${rocm_patch}" == "" ]]; then
    rocm_ver_short=$(printf ${ROCM_SPECIFIER} | cut -d "." -f 1,2)
  else
    rocm_ver_short=${ROCM_SPECIFIER}
  fi

  # note: update list of ids via https://repo.radeon.com/graphics/latest/el/
  local graphics_el_ids=(
    "8"
    "8.10"
    "9.4"
    "9.6"
    "9.7"
    "10"
    "10.1"
  )

  local graphics_el_id=""
  for version in "${graphics_el_ids[@]}"; do
    if [[ "${el_version_id}" == "${version}" ]]; then
      graphics_el_id="${el_version_id}"
      break
    fi
  done
  if [[ -z "${graphics_el_id}" ]]; then
    for version in "${graphics_el_ids[@]}"; do
      if [[ "${el_version_major}" == "${version}" ]]; then
        graphics_el_id="${el_version_major}"
        printf "INFO: EL version ${el_version_id} not explicitly supported, using major version ${el_version_major}\n"
        break
      fi
    done
  fi
  if [[ -z "${graphics_el_id}" ]]; then
    printf "ERROR: EL version ${el_version_id} (major: ${el_version_major}) has no graphics repository\n" >&2
    printf "ERROR: Supported ids: ${graphics_el_ids[*]}\n" >&2
    return 1
  fi

  local rocm_ver_short_graphics=${rocm_ver_short}
  if [[ ${rocm_ver_short_graphics} == "7.2.2" ]]; then
    # 7.2.2 release uses the 7.2.1 graphics repo, not 7.2.2
    rocm_ver_short_graphics="7.2.1"
  fi

  tee /etc/yum.repos.d/rocm.repo <<EOF
[rocm]
name=ROCm ${rocm_ver_short} repository
baseurl=https://repo.radeon.com/rocm/el${el_version_major}/${rocm_ver_short}/main
enabled=1
priority=50
gpgcheck=1
gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key

[amdgraphics]
name=AMD Graphics ${rocm_ver_short_graphics} repository
baseurl=https://repo.radeon.com/graphics/${rocm_ver_short_graphics}/el/${graphics_el_id}/main/x86_64/
enabled=1
priority=50
gpgcheck=1
gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
EOF
  dnf clean all

  # install ROCm (full vs individual packages, no driver)
  dnf install -y ${ROCM_PKGS:-rocm}

  # make ld aware about rocm install dir
  tee --append /etc/ld.so.conf.d/rocm.conf <<EOF
/opt/rocm/lib
/opt/rocm/lib64
EOF
  ldconfig # update ld cache
}

function install_rocm() {
  local -

  set -xe

  local ROCM_PATH="${ROCM_PATH:-"/opt/rocm"}"

  set -u

  __validate_rocm_version_expr "${ROCM_SPECIFIER}"

  if [[ "${ROCM_SPECIFIER}" == "therock:"* ]]; then
    install_therock_from_tarball
  elif [[ "${ROCM_SPECIFIER}" == "preinstalled" ]]; then
    printf "INFO: Use pre-installed ROCm."
  else
    local os_id=$(. /etc/os-release; echo $ID)
    local os_id_like=$(. /etc/os-release; echo $ID_LIKE)
    if [[ "${os_id}" == "rhel" ]] || [[ "${os_id_like}" == *"rhel"* ]]; then
      install_rocm_el
    else
      printf "ERROR: no package-based ROCm install for '${os_id}' (ID_LIKE '${os_id_like}')\n" >&2
      printf "ERROR: use a 'therock:...' ROCM_SPECIFIER, or an enterprise Linux container\n" >&2
      return 1
    fi
  fi

  # NOTE: We want PATH to be updated on caller site too, so that rocm-smi,
  # hipconfig, etc. are available after install_rocm returns
  export PATH="${ROCM_PATH}/bin:${PATH}"

  rocm-smi --showdriverversion || true
  rocm-smi --showhw || true
  rocm-smi --showtopo || true
  rocm-smi || true

  rocminfo || true

  hipconfig || true

  hipcc --version || true
}

# Returns all applicable TheRock artifact groups for the given amdgpu_target.
# Args:
#   $1 - amdgpu_target, e.g. gfx90a, gfx1100, gfx1150
#         If not provided, the AMDGPU_TARGETS env var is used.
# Returns:
#   space-separated list of artifact group strings, e.g. "gfx90X-dcgpu gfx94X-dcgpu"
function amdgpu_target_to_therock_artifact_group() {
  local -

  set -xeu

  local amdgpu_target="${1:-${AMDGPU_TARGETS}}"

  if [[ "${amdgpu_target}" == *";"* ]]; then
    printf "ERROR: Multiple amdgpu_target values detected ('${amdgpu_target}'). Only a single target is supported for therock artifact group mapping.\n" >&2
    return 1
  fi

  # note: groups identified via https://rocm.prereleases.amd.com/tarball/,
  # https://rocm.nightlies.amd.com/tarball/, https://repo.amd.com/rocm/tarball/
  #
  # NOTE: -all must come before -dgpu as they are the more recent endings
  local artifact_groups=(
    gfx110X-all
    gfx120X-all
    gfx101X-dgpu
    gfx103X-dgpu
    gfx110X-dgpu
    gfx90X-dcgpu
    gfx94X-dcgpu
    gfx950-dcgpu
  )

  local matched_groups=()

  # Check if amdgpu_target is directly in artifact_groups
  for group in "${artifact_groups[@]}"; do
    if [[ "${amdgpu_target}" == "${group}" ]]; then
      matched_groups+=("${group}")
    fi
  done

  # Try matching with last char replaced by X and different suffixes
  local base_target="${amdgpu_target:0:-1}X"  # Replace last char with X
  local suffixes=("-all" "-dcgpu" "-dgpu")

  for suffix in "${suffixes[@]}"; do
    local candidate="${base_target}${suffix}"
    for group in "${artifact_groups[@]}"; do
      if [[ "${candidate}" == "${group}" ]]; then
        # Add if not already in matched_groups
        local found=0
        for matched in "${matched_groups[@]}"; do
          if [[ "${matched}" == "${group}" ]]; then
            found=1
            break
          fi
        done
        if [[ ${found} -eq 0 ]]; then
          matched_groups+=("${group}")
        fi
      fi
    done
  done

  # Return matched groups if any
  if [[ ${#matched_groups[@]} -gt 0 ]]; then
    printf "%s" "${matched_groups[*]}"
    return 0
  fi

  # No match found, emit error
  printf "ERROR: amdgpu_target '${amdgpu_target}' does not match any artifact group\n" >&2
  printf "ERROR: Available artifact groups: ${artifact_groups[*]}\n" >&2
  return 1
}

# Prints the latest available TheRock wheel version for the given list type and artifact group.
# Args:
#   $1 - list type: nightlies, prereleases, releases
#   $2 - artifact group, e.g. gfx90X-dcgpu, gfx110X-dgpu, gfx1150, gfx120X-all
# Returns:
#   latest version string, e.g. 5.4.0rc3
function get_latest_therock_wheel_version() {
  local -

  set -xeu

  local list_type="${1}"
  local artifact_group="${2}"
  local pkg="rocm-sdk-libraries-${artifact_group,,}"  # ,, => lower case

  local lists=(
    "nightlies https://rocm.nightlies.amd.com/v2"
    "prereleases https://rocm.prereleases.amd.com/whl"
    "releases https://repo.amd.com/rocm/whl"
  )

  local list=""
  for entry in "${lists[@]}"; do
    if [[ "${entry}" == "${list_type} "* ]]; then
      parts=(${entry})
      list="${parts[1]}"
      break
    fi
  done
  if [[ -z "${list}" ]]; then
    printf "ERROR: Unknown list type '${list_type}'. Supported types: nightlies, prereleases, releases\n" >&2
    false
  fi

  local tmp_dir=$(mktemp -d)
  cd ${tmp_dir}
    python3 -m venv _venv
    source _venv/bin/activate
    pip install --upgrade pip >&2
    local json_file="versions.json"
    pip index versions --json --pre --index-url \
        ${list}/${artifact_group}/ ${pkg}\
          >> ${json_file} && sync

    cat ${json_file} >&2
    local latest_version=$(python3 -c "\
import json
print(json.load(open('${json_file}'))['latest'])")

    deactivate
    rm -rf ${tmp_dir}
    printf "${latest_version}"
}

# Installs TheRock ROCm distribution from tarball.
# Args:
#   $1 - therock_rocm_version (optional), e.g. "7.10.0rc2", "7.10.0", "*", "*a*"
#        "*rc*", ....
#        if not provided, the version is extracted from ROCM_SPECIFIER env var.
#   $2 - artifact_group (optional), e.g. gfx90X-dcgpu, gfx110X-dgpu, gfx1150,
#        gfx120X-all
#        if not provided, tries AMDGPU_TARGETS directly, then all applicable
#        artifact groups derived from AMDGPU_TARGETS env var.
function install_therock_from_tarball() {
  local -

  set -xe

  local ROCM_PATH="${ROCM_PATH:-"/opt/rocm"}"

  set -u

  local therock_rocm_version="${1:-${ROCM_SPECIFIER#therock:}}"

  # Validate AMDGPU_TARGETS is a single entry
  if [[ "${AMDGPU_TARGETS}" == *";"* ]]; then
    printf "ERROR: Multiple AMDGPU_TARGETS values detected ('${AMDGPU_TARGETS}'). Only a single target is supported.\n" >&2
    return 1
  fi

  # Build list of targets to try: AMDGPU_TARGETS first, then artifact groups
  local targets_to_try=("${AMDGPU_TARGETS}")

  # Get applicable artifact groups
  local artifact_groups_str="$(amdgpu_target_to_therock_artifact_group)"
  read -ra artifact_groups_array <<< "${artifact_groups_str}"
  targets_to_try+=("${artifact_groups_array[@]}")

  local os="linux"
  local list=""
  local first_artifact_group="${artifact_groups_array[0]}"

  # Determine list URL and resolve wildcards
  if [[ "${therock_rocm_version}" == *"rc"* ]]; then
    list="rocm.prereleases.amd.com"
    if [[ "${therock_rocm_version}" == "*rc*" ]]; then
      therock_rocm_version=$(get_latest_therock_wheel_version "prereleases" "${first_artifact_group}")
    fi
  elif [[ "${therock_rocm_version}" == *"a"* ]]; then
    list="rocm.nightlies.amd.com"
    if [[ "${therock_rocm_version}" == "*a*" ]]; then
      therock_rocm_version=$(get_latest_therock_wheel_version "nightlies" "${first_artifact_group}")
    fi
  else
    if [[ "${therock_rocm_version}" == "*" ]]; then
      therock_rocm_version=$(get_latest_therock_wheel_version "releases" "${first_artifact_group}")
    fi
    list="repo.amd.com/rocm"
  fi

  local tmp_dir=$(mktemp -d)
  cd ${tmp_dir}
    local download_successful=0
    local tried_targets=""

    # Try each target in order
    for target in "${targets_to_try[@]}"; do
      local url="https://${list}/tarball-multi-arch/therock-dist-${os}-${target}-${therock_rocm_version}.tar.gz"
      printf "INFO: Trying to download: ${url}\n" >&2

      if wget --spider "${url}" 2>/dev/null; then
        printf "INFO: Found tarball for target: ${target}\n" >&2
        wget "${url}"
        download_successful=1
        break
      else
        printf "WARN: Tarball not found for target: ${target}\n" >&2
        tried_targets+="${target} "
      fi
    done

    if [[ ${download_successful} -eq 0 ]]; then
      printf "ERROR: Failed to download tarball for any of the tried targets: ${tried_targets}\n" >&2
      rm -r ${tmp_dir}
      return 1
    fi

    mkdir -p ${ROCM_PATH}
    tar -xf *.tar.gz -C ${ROCM_PATH}
  rm -r ${tmp_dir}

  # Configure ROCm PATH. Make sure you're in the therock-tarball directory before proceeding.
  tee /etc/profile.d/set-rocm-env.sh << EOF
export ROCM_PATH=${ROCM_PATH}
export PATH=\$PATH:\$ROCM_PATH/bin:\$ROCM_PATH/lib/llvm/bin
export LD_LIBRARY_PATH=\$ROCM_PATH/lib
EOF
  chmod +x /etc/profile.d/set-rocm-env.sh
}
