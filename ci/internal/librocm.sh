#!/usr/bin/env bash
# MIT License
#
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
#                   'therock:*a*', 'therock:*dev*' or 'preinstalled'. Wildcards
#                   may also be written 'therock:?', 'therock:?rc?',
#                   'therock:?a?' and 'therock:?dev?'; see
#                   __canonicalize_rocm_specifier.
#   AMDGPU_TARGETS  required for the 'therock:' specifiers; a single target.
#                   'gfx942', an artifact group such as 'gfx94X-dcgpu', or a
#                   spelling that carries a board suffix, a model letter or
#                   feature flags: install_rocm reduces each of those to the
#                   target, see __normalize_amdgpu_targets.
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
    if [[ "${therock_rocm_version}" =~ ^(\*|\*(a|rc|dev)\*)$ ]]; then
      return
    elif [[ "${therock_rocm_version}" =~ ^[0-9]+\.[0-9]+\.[0-9]+((a|rc)[0-9]+)?$ ]]; then
      return
    elif [[ "${therock_rocm_version}" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.dev[0-9]+\+[0-9a-f]{40}$ ]]; then
      return
    else
      printf "ERROR: Invalid therock ROCM_SPECIFIER format: '${rocm_specifier}'\n" >&2
      printf "ERROR: Expected formats: 'therock:*', 'therock:*(a|rc|dev)*', 'therock:X.Y.Z(.(a|rc)R)?', 'therock:X.Y.Z.devN+<sha>'\n" >&2
      printf "ERROR: Wildcards may also be written 'therock:?', 'therock:?a?', 'therock:?rc?', 'therock:?dev?'\n" >&2
      return 1
    fi
  # Check if ROCM_SPECIFIER matches X.Y or X.Y.Z format (where X, Y, Z are numbers)
  elif [[ "${rocm_specifier}" =~ ^([0-9]+)\.([0-9]+)(\.([0-9]+))?$ ]]; then
    : # pass
  else
    printf "ERROR: Invalid ROCM_SPECIFIER format: '${rocm_specifier}'\n" >&2
    printf "ERROR: Expected format: 'X.Y' or 'X.Y.Z' (numbers), 'preinstalled', 'therock:X.Y.Z', 'therock:X.Y.ZrcR', 'therock:X.Y.ZaD', 'therock:*', 'therock:*rc*', 'therock:*a*', 'therock:*dev*' (also 'therock:?', 'therock:?rc?', 'therock:?a?', 'therock:?dev?')\n" >&2
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

# Rewrites the '?' spellings of the three therock wildcards to the '*' ones,
# in place, and exports the result.
#
# A specifier is typically typed into a GitHub comment, where '*a*' is read as
# emphasis and renders as 'a': the text a reader sees is then not the specifier
# that ran. '?a?' survives the rendering and means the same thing, so it is
# rewritten here, once, ahead of everything that reads ROCM_SPECIFIER --
# validation, the tarball URL, the env file the later steps source, and the
# step summary all see the one canonical spelling.
#
# The patterns are quoted because an unquoted '?' in a case pattern is a glob
# that matches any single character.
function __canonicalize_rocm_specifier() {
  case "${ROCM_SPECIFIER:-}" in
    'therock:?')     ROCM_SPECIFIER='therock:*'     ;;
    'therock:?a?')   ROCM_SPECIFIER='therock:*a*'   ;;
    'therock:?rc?')  ROCM_SPECIFIER='therock:*rc*'  ;;
    'therock:?dev?') ROCM_SPECIFIER='therock:*dev*' ;;
  esac
  export ROCM_SPECIFIER
}

# Reduces AMDGPU_TARGETS to the gfx target the TheRock path can use, in place,
# and exports the result.
#
# Only a 'therock:' specifier reads it, to pick which distribution is
# downloaded, and it arrives from wherever a caller happens to keep the
# architecture: a workflow's constant, a SLURM GRES type name, the 'gfx90a'
# default of ci/internal/prepare-container.sh, or a shell where the same
# variable also drives a device-code compile and so carries a board suffix or
# feature flags. Every one of those spellings names one target here.
#
# This is insurance rather than a check on an input: without it
# 'gfx942-mi300x' reaches amdgpu_target_to_therock_artifact_group, which derives
# a group by replacing the last character with 'X' and so reports that
# 'gfx942-mi300x' "does not match any artifact group" -- true, and no help at
# all in finding the cause.
#
# A model name such as 'mi300x' is an error rather than something to resolve.
# Translating one is the caller's job, done before a job is even submitted,
# where a name we do not know can still be reported as a bad request.
function __normalize_amdgpu_targets() {
  local target="${AMDGPU_TARGETS:-}"

  # Nothing to normalise, and nothing to complain about either: the package and
  # 'preinstalled' paths never read it. The TheRock path does, so an empty value
  # there is worth saying plainly rather than leaving to 'unbound variable'.
  if [[ -z "${target}" ]]; then
    if [[ "${ROCM_SPECIFIER:-}" == "therock:"* ]]; then
      printf "ERROR: AMDGPU_TARGETS is empty; a 'therock:' ROCM_SPECIFIER needs a target like gfx942 to pick a distribution\n" >&2
      return 1
    fi
    return 0
  fi

  # 'gfx942:sramecc+:xnack-' -> 'gfx942'. The feature flags belong to a
  # compiler invocation and have no bearing on which tarball is downloaded.
  target=${target%%:*}

  case "${target,,}" in
    # Already an artifact group. install_therock_from_tarball tries the value
    # verbatim before any derived name and matches it against a list of
    # capital-X spellings, so this one keeps the case it came in with.
    gfx*-all|gfx*-dcgpu|gfx*-dgpu) ;;
    # 'gfx942-mi300x' -> 'gfx942'. A board suffix says which card, which the
    # distribution does not distinguish.
    gfx*-*) target=${target%%-*} ; target=${target,,} ;;
    gfx*) target=${target,,} ;;
    *)
      printf "ERROR: AMDGPU_TARGETS='%s' does not name a gfx target; pass one like gfx942\n" "${AMDGPU_TARGETS}" >&2
      return 1 ;;
  esac

  # 'gfx1100p' -> 'gfx1100'. SLURM's RDNA variant names put the model in a
  # single trailing letter rather than behind a dash, so gfx1100p and gfx1100w
  # are two cards of the gfx1100 architecture and gfx1101v is one of gfx1101.
  # Only a four-digit number can carry such a letter: gfx90a's is part of the
  # architecture, which is why this asks for the digits rather than stripping
  # any trailing letter it finds.
  if [[ ${target} =~ ^gfx[0-9]{4}[a-z]$ ]]; then
    target=${target:0:-1}
  fi

  export AMDGPU_TARGETS=${target}
}

function install_rocm() {
  local -

  set -xe

  local ROCM_PATH="${ROCM_PATH:-"/opt/rocm"}"

  set -u

  __canonicalize_rocm_specifier
  __normalize_amdgpu_targets
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

  # note: groups identified via https://stable.repo.amd.com/rocm/core/tarball/,
  # https://nightly.repo.amd.com/rocm/core/tarball/,
  # https://dev.repo.amd.com/rocm/core/tarball/
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
    # The Strix APUs publish per target rather than per family: the index lists
    # gfx1150 and gfx1151 themselves, and no gfx115X spelling exists for the
    # derivation below to find.
    gfx1150
    gfx1151
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

# Returns whl_index and tarball_base for a TheRock release channel.
# Args: $1 - list type: releases, nightlies, devreleases, prereleases
function __therock_repo_bases() {
  case "${1}" in
    releases)
      printf '%s %s\n' \
        'https://stable.repo.amd.com/rocm/core/whl-next' \
        'https://stable.repo.amd.com/rocm/core/tarball/' ;;
    nightlies)
      printf '%s %s\n' \
        'https://nightly.repo.amd.com/rocm/core/whl-next' \
        'https://nightly.repo.amd.com/rocm/core/tarball/' ;;
    devreleases)
      printf '%s %s\n' \
        'https://dev.repo.amd.com/rocm/core/whl-next' \
        'https://dev.repo.amd.com/rocm/core/tarball/' ;;
    prereleases)
      printf '%s %s\n' \
        'https://rc.repo.amd.com/rocm/core/whl-next' \
        'https://rc.repo.amd.com/rocm/core/tarball/' ;;
    *)
      printf "ERROR: Unknown list type '${1}'. Supported: releases, nightlies, devreleases, prereleases\n" >&2
      return 1 ;;
  esac
}

# Rewrites '?' wildcard spellings to '*' (therock:?dev? -> therock:*dev*, etc.).
function __therock_list_type_for_version() {
  local version="${1}"
  if [[ "${version}" == "*dev*" || "${version}" == *.dev* ]]; then
    printf "devreleases"
  elif [[ "${version}" == "*rc*" || "${version}" =~ rc[0-9] ]]; then
    printf "prereleases"
  elif [[ "${version}" == "*a*" || "${version}" =~ [0-9]a[0-9]{8} ]]; then
    printf "nightlies"
  else
    printf "releases"
  fi
}

function __therock_gfx_target_for_whl() {
  local artifact_group="${1}"
  local target="${AMDGPU_TARGETS:-}"

  if [[ -n "${target}" && "${target}" =~ ^gfx[0-9a-z]+$ ]]; then
    printf "${target}"
    return 0
  fi
  if [[ "${artifact_group}" =~ ^gfx[0-9][0-9a-z]*$ ]]; then
    printf "${artifact_group}"
    return 0
  fi
  printf "ERROR: cannot resolve gfx target for wheel lookup (AMDGPU_TARGETS='${AMDGPU_TARGETS}', artifact_group='${artifact_group}')\n" >&2
  return 1
}

function __therock_whl_pkg_for_target() {
  local artifact_group="${1}"
  local target
  target="$(__therock_gfx_target_for_whl "${artifact_group}")" || return 1
  printf "rocm-sdk-device-${target}"
}

function __therock_version_from_pip() {
  local whl_index="${1}"
  local pkg="${2}"
  local tmp_dir
  tmp_dir=$(mktemp -d)
  (
    cd "${tmp_dir}"
    python3 -m venv _venv >/dev/null 2>&1
    source _venv/bin/activate
    pip install --upgrade pip >/dev/null 2>&1
    pip index versions --pre --index-url "${whl_index}/" "${pkg}" 2>/dev/null \
      | sed -n 's/.*(\([^)]*\)).*/\1/p' | head -1
    deactivate
  )
  rm -rf "${tmp_dir}"
}

function __therock_version_from_whl_html() {
  local whl_index="${1}"
  local pkg="${2}"
  python3 - "${whl_index}" "${pkg}" <<'PY'
import re, sys, urllib.request
whl_index, pkg = sys.argv[1:3]
url = f"{whl_index}/{pkg}/"
html = urllib.request.urlopen(url).read().decode()
wheel_prefix = pkg.replace("-", "_") + "-"
versions = []
for m in re.finditer(r'href="([^"]+\.whl)"', html):
    name = m.group(1)
    if not name.startswith(wheel_prefix):
        continue
    rest = name[len(wheel_prefix):]
    version = rest.split("-py3-none-")[0].replace("%2B", "+")
    versions.append(version)
if not versions:
    sys.exit(1)
print(versions[0])
PY
}

function __therock_latest_dev_version_from_tarball() {
  local tarball_base="${1}"
  local target="${2}"
  local artifact_group="${3:-}"
  python3 - "${tarball_base}" "${target}" "${artifact_group}" <<'PY'
import json, re, sys, urllib.request
tarball_base, target, artifact_group = sys.argv[1:4]
html = urllib.request.urlopen(tarball_base).read().decode()
m = re.search(r'const files = (\[.*?\]);', html, re.S)
if not m:
    sys.exit(1)
files = sorted(json.loads(m.group(1)), key=lambda f: -f["mtime"])
candidates = []
for value in (target, artifact_group, "multiarch"):
    if value and value not in candidates:
        candidates.append(value)
for f in files:
    name = f["name"]
    if "-tests-" in name or not name.endswith(".tar.gz"):
        continue
    if not name.startswith("therock-dist-linux-"):
        continue
    body = name[len("therock-dist-linux-"):-len(".tar.gz")]
    for cand in candidates:
        prefix = cand + "-"
        if body.startswith(prefix):
            print(body[len(cand) + 1:])
            sys.exit(0)
sys.exit(1)
PY
}

function __therock_legacy_wheel_version() {
  local list_type="${1}"
  local artifact_group="${2}"
  local pkg="rocm-sdk-libraries-${artifact_group,,}"
  local list=""
  case "${list_type}" in
    prereleases) list="https://rocm.prereleases.amd.com/whl" ;;
    releases) list="https://repo.amd.com/rocm/whl" ;;
    nightlies) list="https://rocm.nightlies.amd.com/v2" ;;
    *) return 1 ;;
  esac
  local tmp_dir version
  tmp_dir=$(mktemp -d)
  version=$(
    cd "${tmp_dir}"
    python3 -m venv _venv >/dev/null 2>&1
    source _venv/bin/activate
    pip install --upgrade pip >/dev/null 2>&1
    pip index versions --pre --index-url "${list}/${artifact_group}/" "${pkg}" 2>/dev/null \
      | sed -n 's/.*(\([^)]*\)).*/\1/p' | head -1
    deactivate
  )
  rm -rf "${tmp_dir}"
  if [[ -n "${version}" ]]; then
    printf "${version}"
  else
    return 1
  fi
}


# Prints the latest available TheRock wheel version for the given list type and artifact group.
# Args:
#   $1 - list type: nightlies, prereleases, releases, devreleases
#   $2 - artifact group, e.g. gfx90X-dcgpu, gfx110X-dgpu, gfx1150, gfx120X-all
# Returns:
#   latest version string, e.g. 10.0.0, 10.1.0a20260828, 10.1.0.dev0+<sha>
function get_latest_therock_wheel_version() {
  local -

  set -xeu

  local list_type="${1}"
  local artifact_group="${2}"
  local whl_index tarball_base
  read whl_index tarball_base < <(__therock_repo_bases "${list_type}")

  if [[ "${list_type}" == "devreleases" ]]; then
    local target version
    target="$(__therock_gfx_target_for_whl "${artifact_group}")" || return 1
    version="$(__therock_latest_dev_version_from_tarball "${tarball_base}" "${target}" "${artifact_group}")" || {
      printf "ERROR: no dev build found for target '${target}' on ${tarball_base}\n" >&2
      return 1
    }
    printf "${version}"
    return 0
  fi

  local pkg target latest_version=""
  pkg="$(__therock_whl_pkg_for_target "${artifact_group}")" || return 1
  latest_version="$(__therock_version_from_pip "${whl_index}" "${pkg}")" || true
  if [[ -z "${latest_version}" ]]; then
    latest_version="$(__therock_version_from_whl_html "${whl_index}" "${pkg}")" || true
  fi
  if [[ -z "${latest_version}" ]]; then
    latest_version="$(__therock_legacy_wheel_version "${list_type}" "${artifact_group}")" || true
  fi
  if [[ -z "${latest_version}" ]]; then
    printf "ERROR: no version found for ${pkg} on ${list_type} channel\n" >&2
    return 1
  fi
  printf "${latest_version}"
}

# Installs TheRock ROCm distribution from tarball.
# Args:
#   $1 - therock_rocm_version (optional), e.g. "10.0.0", "*", "*a*", "*rc*", "*dev*"
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

  if [[ "${AMDGPU_TARGETS}" == *";"* ]]; then
    printf "ERROR: Multiple AMDGPU_TARGETS values detected ('${AMDGPU_TARGETS}'). Only a single target is supported.\n" >&2
    return 1
  fi

  local targets_to_try=("${AMDGPU_TARGETS}")
  local artifact_groups_str="$(amdgpu_target_to_therock_artifact_group)"
  read -ra artifact_groups_array <<< "${artifact_groups_str}"
  targets_to_try+=("${artifact_groups_array[@]}")

  local os="linux"
  local first_artifact_group="${artifact_groups_array[0]:-}"
  local list_type tarball_base _whl_index
  list_type="$(__therock_list_type_for_version "${therock_rocm_version}")"
  read _whl_index tarball_base < <(__therock_repo_bases "${list_type}")

  if [[ -z "${first_artifact_group}" && "${therock_rocm_version}" == *"*"* ]]; then
    printf "ERROR: AMDGPU_TARGETS='%s' matches no TheRock artifact group, so '%s' cannot be resolved\n" "${AMDGPU_TARGETS}" "${therock_rocm_version}" >&2
    return 1
  fi

  case "${therock_rocm_version}" in
    "*dev*")
      therock_rocm_version=$(get_latest_therock_wheel_version "devreleases" "${first_artifact_group}") ;;
    "*rc*")
      therock_rocm_version=$(get_latest_therock_wheel_version "prereleases" "${first_artifact_group}") ;;
    "*a*")
      therock_rocm_version=$(get_latest_therock_wheel_version "nightlies" "${first_artifact_group}") ;;
    "*")
      therock_rocm_version=$(get_latest_therock_wheel_version "releases" "${first_artifact_group}") ;;
  esac

  local encoded_version="${therock_rocm_version//+/%2B}"
  local tmp_dir=$(mktemp -d)
  cd ${tmp_dir}
    local download_successful=0
    local tried_targets=""

    for target in "${targets_to_try[@]}"; do
      local url="${tarball_base}therock-dist-${os}-${target}-${encoded_version}.tar.gz"
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

  tee /etc/profile.d/set-rocm-env.sh << EOF
export ROCM_PATH=${ROCM_PATH}
export PATH=\$PATH:\$ROCM_PATH/bin:\$ROCM_PATH/lib/llvm/bin
export LD_LIBRARY_PATH=\$ROCM_PATH/lib
EOF
  chmod +x /etc/profile.d/set-rocm-env.sh
}


