#!/usr/bin/env bash
git fetch origin
git checkout origin/new_main -- ci
git checkout origin/new_main -- build_hip_python_pkgs.sh
git checkout origin/new_main -- _render_update_version.py
git checkout origin/new_main -- docs
git checkout origin/new_main -- examples
# unstage
git reset -q HEAD ci build_hip_python_pkgs.sh _render_update_version.py docs examples
