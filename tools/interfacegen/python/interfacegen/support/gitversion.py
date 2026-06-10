# MIT License
#
# Copyright (c) 2023-2024 Advanced Micro Devices, Inc.
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

import subprocess


def git_describe(always=True):
    """Call git describe.

    Let's git describe this commit based on release tags.
    If ``always`` is specified, this will return
    the short version of the hash.
    """
    cmd = ["git", "describe"]
    if always:
        cmd.append("--always")
    return subprocess.check_output(cmd).decode("utf-8").strip()


def git_rev(short=True):
    """Returns the git revision."""
    cmd = ["git", "rev-parse"]
    if short:
        cmd.append("--short")
    cmd.append("HEAD")
    return subprocess.check_output(cmd).decode("utf-8").strip()


def git_current_branch():
    """Return the name of the current branch."""
    return (
        subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        .decode("utf-8")
        .strip()
    )


def git_is_clean():
    """Checks if the repository is clean.

    Checks if there are no uncommitted changes to indexed files.
    Unindexed files are not considered.
    """
    try:
        subprocess.check_output(["git", "update-index", "--really-refresh"])
        subprocess.check_output(["git", "diff-index", "--quiet", "HEAD"])
        return True
    except subprocess.CalledProcessError:
        return False
