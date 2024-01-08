#!/usr/bin/env bash
MAIN_BRANCH=${MAIN_BRANCH:-"origin/new_main"}
git fetch origin
git checkout ${MAIN_BRANCH} -- ci
git checkout ${MAIN_BRANCH} -- build_hip_python_pkgs.sh
git checkout ${MAIN_BRANCH} -- _render_update_version.py
git checkout ${MAIN_BRANCH} -- docs
git checkout ${MAIN_BRANCH} -- examples
# unstage
git reset -q HEAD ci build_hip_python_pkgs.sh _render_update_version.py docs examples
