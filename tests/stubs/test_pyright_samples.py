# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
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

"""Run pyright over annotated sample scripts, the consumer's view.

The other suite here reads a stub as text. This one asks a type checker
what the *installed* wheels are worth: whether the PEP 561 markers are in
place, whether the `.pyi` files next to the extensions are found through
them, and whether ordinary annotated code against `hip`, `rocm.bindings`
and `cuda.bindings` comes out clean. None of the samples run, so no
device and no HIP runtime are involved.

`samples/unknown_attribute.py` is the control. Without it a checker that
resolved nothing -- the state every one of these packages shipped in
until the markers landed -- would pass the other three.

`pyrightconfig.json` turns `useLibraryCodeForTypes` off. Left on, as it
is by default, pyright infers types from the installed *sources* and
every sample here passes with all markers and stubs deleted, which would
make this suite decorative. Off, a package counts as typed only through
the artifacts the wheels ship.

The config also turns unnecessary suppressions into errors. The samples
suppress what the generated stubs do not cover yet, enum members above
all, so the suite speaks up when that is fixed.
"""

__author__ = "Advanced Micro Devices, Inc."

import json
import os
import shutil
import subprocess
import sys

import pytest

#: Sample -> the distribution that has to be installed for it to mean
#: anything. A Windows install without the interop wheel skips its case.
SAMPLES = {
    "uses_hip.py": "hip",
    "uses_rocm_bindings.py": "rocm.bindings",
    "uses_cuda_bindings.py": "cuda.bindings",
}

#: Generous: the first pyright run of a session downloads a node runtime.
TIMEOUT_S = 600

_SUITE_DIR = os.path.dirname(os.path.abspath(__file__))
_SAMPLES_DIR = os.path.join(_SUITE_DIR, "samples")


@pytest.fixture(scope="module")
def pyright():
    """The pyright command line, or a skip where it is not usable here."""
    executable = shutil.which("pyright")
    command = [executable] if executable else [sys.executable, "-m", "pyright"]
    try:
        probe = subprocess.run(
            command + ["--version"],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired) as err:
        pytest.skip(f"pyright is not usable here: {err}")
    if probe.returncode != 0:
        # Typically an offline runner: the wheel is a launcher that fetches
        # its node runtime on first use.
        pytest.skip(f"pyright is not usable here: {probe.stderr.strip()}")
    return command


def _check(pyright, sample):
    """Diagnostics pyright reports for one sample, as (count, text)."""
    completed = subprocess.run(
        pyright
        + [
            "--project",
            _SUITE_DIR,
            "--outputjson",
            # Check against the interpreter running the suite, which is the
            # venv the wheels were installed into.
            "--pythonpath",
            sys.executable,
            os.path.join(_SAMPLES_DIR, sample),
        ],
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
    )
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError:
        raise AssertionError(
            f"pyright produced no report for {sample}:\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    diagnostics = [
        "{}:{} {} [{}]".format(
            os.path.basename(entry["file"]),
            entry["range"]["start"]["line"] + 1,
            entry["message"].splitlines()[0],
            entry.get("rule", "-"),
        )
        for entry in report["generalDiagnostics"]
        if entry["severity"] == "error"
    ]
    return report["summary"]["errorCount"], "\n".join(diagnostics)


@pytest.mark.parametrize("sample", sorted(SAMPLES))
def test_sample_checks_clean(pyright, sample):
    pytest.importorskip(SAMPLES[sample])
    errors, diagnostics = _check(pyright, sample)
    assert errors == 0, f"pyright rejects {sample}:\n{diagnostics}"


def test_the_control_sample_is_rejected(pyright):
    """The suite has to be able to fail."""
    pytest.importorskip("hip")
    errors, diagnostics = _check(pyright, "unknown_attribute.py")
    assert errors, "pyright accepted a call to a function that does not exist"
    assert "hipMallocTypo" in diagnostics
