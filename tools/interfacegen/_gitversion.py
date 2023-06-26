# MIT License
# 
# Copyright (c) 2023 Advanced Micro Devices, Inc.
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
import datetime

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
    """Returns the git revision.
    """
    cmd = ["git", "rev-parse"]
    if short:
        cmd.append("--short")
    cmd.append("HEAD")
    return subprocess.check_output(cmd).decode("utf-8").strip()

def git_branch_rev_count(branch):
    """Count the number of revisions on branch 'branch'.
    """    
    return int(subprocess.check_output(["git","rev-list",branch,"--count"]).decode("utf-8").strip())

def git_head_rev_count():
    """Count the number of revisions of the HEAD.
    """    
    return git_branch_rev_count("HEAD")

def git_current_branch():
    """Return the name of the current branch."""
    return subprocess.check_output(["git","branch","--show-current"]).decode("utf-8").strip()

def git_is_main_branch():
    """If the current branch is 'main'."""
    return git_current_branch() == "main"

def git_upstream_commits_vs_local():
    """Returns max(0,#upstream - #local)
    """
    return int(subprocess.check_output(["git","rev-list","@..@{u}","--count"]).decode("utf-8").strip())

def git_local_commits_vs_upstream():
    """Returns max(0,#local - #upstream)
    """
    return int(subprocess.check_output(["git","rev-list","@{u}..@","--count"]).decode("utf-8").strip())

def git_is_clean():
    """Checks if the repository is clean.

    Checks if there are no uncommitted changes to indexed files.
    Unindexed files are not considered.
    """
    try:
        subprocess.check_output(["git","update-index","--really-refresh"])
        subprocess.check_output(["git","diff-index","--quiet","HEAD"])
        return True
    except subprocess.CalledProcessError:
        return False

def version(append_hash=False,append_date=False):
    """Version number of the code generator.

    Takes revision count of origin branch 'main' as version number.
    Appends `.dev{num}` if the head of the current branch/local version of 'main'
    deviates `{num}` revisions from origin 'main'.
    Further appends the revision hash in this case.
    Finally appends a date if the local repository is not clean.
    """
    is_clean = git_is_clean()
    tag = git_describe()
    rev_hash = git_rev()
    branch = git_current_branch()
    is_main = git_is_main_branch()
    main_rev_count = git_branch_rev_count("origin/main")
    if is_main:
        vs_local = git_upstream_commits_vs_local()
        vs_upstream = git_local_commits_vs_upstream()
        distance = vs_local + vs_upstream
    else:
        branch_rev_count = git_branch_rev_count(branch)
        distance = abs(branch_rev_count-main_rev_count)
        
    yyyymmdd = datetime.date.today().strftime("%Y%m%d")
    if distance == 0 and is_clean:
        result =  f"{main_rev_count}"
    elif distance and is_clean:
        result =  f"{main_rev_count}.dev{distance}+g{rev_hash}"
    elif distance == 0 and not is_clean:
        result =  f"{main_rev_count}+d{yyyymmdd}"
    else: # if distance > 0 and not is_clean:
        result =  f"{main_rev_count}.dev{distance}+g{rev_hash}+d{yyyymmdd}"
    if not append_hash:
        result = result.replace(f"+g{rev_hash}","")
    if not append_date:
        result = result.replace(f"+d{yyyymmdd}","")
    return result
