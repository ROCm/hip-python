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

# Run the hip-python test suites on Windows.
#
# PowerShell counterpart to ci/internal/test.sh. It runs the same suites against
# the wheels produced by ci\internal\build-wheels.ps1, and deviates from the bash
# script only where Linux assumptions do not hold:
#
#   * No numba-hip suite. build-wheels.ps1 configures with
#     -DHIP_PYTHON_BUILD_NUMBA_HIP=OFF by default, so no numba_hip wheel exists
#     to test. numba.hip itself runs on Windows as long as the compiler wheel
#     was built with -DHIP_PYTHON_BUNDLE_LIBLLVM=ON; its suite passes there.
#   * Suites are not fatal individually. On Linux every ROCm component is
#     present, so any failure is a real defect; on Windows ROCm ships no AMD SMI,
#     RCCL, ROCTX, hipFile, hipSPARSELt or hipTensor, and the suites skip the
#     affected tests. Each suite's exit code is collected and reported together
#     at the end so one suite's failure does not hide the others' results.
#
# The suites skip rather than fail when a backing ROCm library is missing, so
# -rs (report skip reasons) is passed throughout: it is the only thing that makes
# a Windows log distinguishable from a run that silently tested nothing.
#
# ROCm itself comes from one of two places. By default the environment supplies
# it, which suits a system install or an unpacked tarball -- a single tree that
# ROCM_PATH can name. Pass -UseRocmSdkWheels to install it into the test venv
# from the rocm_sdk wheels instead, which is how most users get ROCm and spreads
# it over several trees that only rocm_sdk.find_libraries can tell apart.
#
# Usage:
#   ci\internal\test.ps1                                 # wheels from the default build dir
#   ci\internal\test.ps1 -BuildArtifactsDir packages\build\dist
#   ci\internal\test.ps1 -TestVenv _test_venv            # reuse a venv across runs
#   ci\internal\test.ps1 -UseRocmSdkWheels               # provide ROCm from wheels too

[CmdletBinding()]
param(
    # Directory holding the wheels built by ci\internal\build-wheels.ps1.
    [string] $BuildArtifactsDir,

    # Venv to create and install the wheels into. A temporary directory that is
    # removed on success by default; name one to keep it for debugging.
    [string] $TestVenv,

    # Python used to create the venv. Must be the interpreter the wheels were
    # built against, because the extension modules are ABI-tagged for it --
    # unless they were built with build-wheels.ps1 -UseSabi, in which case any
    # interpreter from that floor upwards can install them.
    [string] $Python = "python",

    # Install ROCm itself into the test venv from the rocm_sdk wheels, which is
    # how most users get it. Libraries then resolve through
    # rocm_sdk.find_libraries, which is the only thing that knows which of
    # _rocm_sdk_core, _rocm_sdk_libraries and _rocm_sdk_devel owns a given
    # library. That distinction matters: the devel tree holds a second copy of
    # libhipblaslt.dll beside an incomplete set of Tensile kernel files, and
    # hipBLASLt looks for kernels next to whichever copy was loaded, so reaching
    # that one faults inside the algorithm search instead of reporting a missing
    # kernel. Never hand a bare _rocm_sdk_devel path to ROCM_PATH for that reason.
    [switch] $UseRocmSdkWheels,

    # GPU target for the rocm-sdk-device-* wheel, e.g. gfx1103. Queried from the
    # GPU when omitted; pass it to test a target this machine does not have.
    [string] $GfxArch,

    # ROCm version for those wheels. Defaults to the version the bindings were
    # generated against, read back from the installed wheels, so the SDK cannot
    # silently disagree with them.
    [string] $RocmSdkVersion,

    [string] $RocmSdkIndexUrl = "https://repo.amd.com/rocm/whl-multi-arch/"
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path

# Run a native command, failing on its exit code rather than on its stderr.
# pytest and pip both write progress to stderr, which $ErrorActionPreference =
# 'Stop' would turn into a terminating error as soon as a caller merges the
# streams to capture a log. See the same helper in build-wheels.ps1.
function Invoke-Native {
    param(
        [Parameter(Mandatory)] [string] $Exe,
        [Parameter(ValueFromRemainingArguments)] [string[]] $Arguments
    )
    $ErrorActionPreference = "Continue"
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Exe $($Arguments -join ' ') failed with exit code $LASTEXITCODE"
    }
}

### Inputs

if (-not $BuildArtifactsDir) {
    $BuildArtifactsDir = Join-Path $repoRoot "packages\build\dist"
}
if (-not (Test-Path -LiteralPath $BuildArtifactsDir)) {
    throw "No wheel directory at $BuildArtifactsDir. Run ci\internal\build-wheels.ps1 " +
          "first, or pass -BuildArtifactsDir."
}
$BuildArtifactsDir = (Resolve-Path -LiteralPath $BuildArtifactsDir).Path

$keepVenv = [bool] $TestVenv
if (-not $TestVenv) {
    $TestVenv = Join-Path ([System.IO.Path]::GetTempPath()) "hip_python_test_$PID"
}

### Shared venv + dependency install

# The examples are copied out of the source tree and run from the copy, which
# keeps the suite from importing a `rocm` package out of the repo instead of the
# installed wheels.
$examplesBuildDir = Join-Path ([System.IO.Path]::GetTempPath()) "hip_python_examples_$PID"
New-Item -ItemType Directory -Path $examplesBuildDir -Force | Out-Null
Copy-Item -Path (Join-Path $repoRoot "examples") `
          -Destination (Join-Path $examplesBuildDir "examples") -Recurse -Force

Invoke-Native $Python -m venv $TestVenv
$venvPython = Join-Path $TestVenv "Scripts\python.exe"

Invoke-Native $venvPython -m pip install --upgrade pip pytest cffi
Invoke-Native $venvPython -m pip install -r (Join-Path $examplesBuildDir "examples\requirements.txt")

# numba_hip is deliberately absent from the wheels installed here (see the
# header).
$wheels = @(
    Get-ChildItem -Path $BuildArtifactsDir -Filter "rocm_bindings_*.whl"
    Get-ChildItem -Path $BuildArtifactsDir -Filter "hip_python_interop*.whl"
    Get-ChildItem -Path $BuildArtifactsDir -Filter "hip_python-*.whl"
) | ForEach-Object { $_.FullName }

if ($wheels.Count -eq 0) {
    throw "No hip-python wheels found under $BuildArtifactsDir"
}
Write-Host "Installing $($wheels.Count) wheel(s):"
$wheels | ForEach-Object { Write-Host "  $(Split-Path $_ -Leaf)" }
Invoke-Native $venvPython -m pip install @wheels

### ROCm from wheels (optional)

if ($UseRocmSdkWheels) {
    if (-not $RocmSdkVersion) {
        $RocmSdkVersion = (& $venvPython -c "import rocm.version; print(rocm.version.ROCM_VERSION_NAME)")
        if ($LASTEXITCODE -ne 0 -or -not $RocmSdkVersion) {
            throw "Could not read ROCM_VERSION_NAME from the installed wheels. Pass -RocmSdkVersion."
        }
        $RocmSdkVersion = $RocmSdkVersion.Trim()
    }

    # The base packages come first so that a device wheel is chosen for the GPU
    # actually present: querying the GPU needs a HIP runtime, which these
    # provide, and the device wheel carries only kernels, so nothing here needs
    # it yet.
    Write-Host ""
    Write-Host "Installing ROCm $RocmSdkVersion from $RocmSdkIndexUrl"
    Invoke-Native $venvPython -m pip install --index-url $RocmSdkIndexUrl `
        "rocm[libraries]==$RocmSdkVersion"

    # Point ROCM_PATH at the wheels' own tree, so that a tree left over from the
    # build cannot supply headers that disagree with the libraries under test.
    # The Cython examples need it: their setup.py takes its include directory from
    # ROCM_PATH and falls back to /opt/rocm, so without this they compile with no
    # HIP headers at all and fail on structs the .pyx declares. Library resolution
    # is unaffected either way, because get_library_path consults
    # rocm_sdk.find_libraries before it looks at ROCM_PATH.
    #
    # The core package is the tree to name: it carries include/, lib/ and bin/,
    # and unlike the devel tree it is present without rocm[devel].
    $rocmSdkRoot = (& $venvPython -c @'
from pathlib import Path
from rocm_sdk import find_libraries
print(Path(find_libraries('amdhip64')[0]).parent.parent)
'@)
    if ($LASTEXITCODE -ne 0 -or -not $rocmSdkRoot) {
        throw "Could not locate the rocm_sdk tree after installing the ROCm wheels."
    }
    $rocmSdkRoot = $rocmSdkRoot.Trim()
    $env:ROCM_PATH = $rocmSdkRoot
    $env:HIP_PATH = $rocmSdkRoot
    Remove-Item Env:\ROCM_HOME -ErrorAction SilentlyContinue
    Write-Host "ROCM_PATH set to $rocmSdkRoot"

    if (-not $GfxArch) {
        # gcnArchName carries target features on some GPUs (gfx1103:xnack-),
        # which are not part of a wheel name.
        #
        # Quote with '' only: PowerShell strips double quotes out of arguments on
        # their way to a native executable, so "" here reaches python unquoted.
        $GfxArch = (& $venvPython -c @'
from rocm.bindings import hip
err, props = hip.hipGetDeviceProperties(0)
if int(err) != 0:
    raise SystemExit('hipGetDeviceProperties failed: ' + str(err))
name = props.gcnArchName
name = name.decode() if isinstance(name, (bytes, bytearray)) else bytes(name).decode()
print(name.split(chr(0))[0].split(':')[0])
'@)
        if ($LASTEXITCODE -ne 0 -or -not $GfxArch) {
            throw "Could not query the GPU target. Pass -GfxArch (e.g. -GfxArch gfx1103)."
        }
        $GfxArch = $GfxArch.Trim()
        Write-Host "Detected GPU target: $GfxArch"
    }

    Invoke-Native $venvPython -m pip install --index-url $RocmSdkIndexUrl `
        "rocm[libraries,device-$GfxArch]==$RocmSdkVersion"

    # `rocm-sdk init` expands the devel tree and links the device wheels' files
    # into it, and errors out when rocm[devel] is absent. The runtime libraries
    # the suites need come from the core and libraries packages, which
    # find_libraries locates without it, so devel is not installed above and this
    # runs only for a caller who added it.
    $hasDevel = (& $venvPython -c @'
import importlib.util
print(1 if importlib.util.find_spec('rocm_sdk_devel') else 0)
'@)
    if ($LASTEXITCODE -eq 0 -and $hasDevel.Trim() -eq "1") {
        Invoke-Native $venvPython -m rocm_sdk init
    }

    # Records which tree each library came from, the one thing that distinguishes
    # a correctly wired run from one that silently found the wrong copy. Called
    # directly rather than through Invoke-Native, which does not pass a multi-line
    # argument through intact.
    $resolved = (& $venvPython -c @'
from rocm.bindings.util.paths import get_library_path
for shortname in ('amdhip64', 'hipblaslt', 'amd_comgr'):
    try:
        print('  %-12s %s' % (shortname, get_library_path(shortname).decode()))
    except Exception as exc:
        print('  %-12s unresolved (%s)' % (shortname, exc))
'@)
    Write-Host "ROCm resolved from:"
    $resolved | ForEach-Object { Write-Host $_ }
}

# The examples suite asks the interop shim to report CUDA error codes that have
# no HIP equivalent instead of raising; see hip-python-interop's docs.
$env:HIP_PYTHON_cudaError_t_HALLUCINATE = "1"

### Suites

# Suite 1 - hip-python examples.
# Suite 2 - hip-python-interop pynvml/NVML shim unit tests (mocked, no GPU).
# Suite 3 - rocm-bindings unit tests (core + compiler), GPU-free.
# Suite 4 - handcoded-Cython stubs: checks that the hand-maintained
#           cuda.bindings.cufile stub still covers the installed module.
#
# Suites 2 to 4 live outside the importable packages (tests/, not under src/) so
# they exercise the *installed* wheels.
$suites = [ordered] @{
    "examples"                = Join-Path $examplesBuildDir "examples"
    "hip-python-interop"      = Join-Path $repoRoot "tests\hip-python-interop"
    "rocm-bindings-core"      = Join-Path $repoRoot "tests\rocm-bindings-core"
    "rocm-bindings-compiler"  = Join-Path $repoRoot "tests\rocm-bindings-compiler"
    "stubs"                   = Join-Path $repoRoot "tests\stubs"
}

# pytest exits 5 when it collected nothing to run. On Windows that is the
# expected outcome for a suite whose every module is platform-skipped -- the
# whole hip-python-interop suite is, since ROCm ships neither AMD SMI, ROCTX nor
# hipFile there -- so it is reported apart from both success and failure rather
# than being mistaken for either.
$PYTEST_NO_TESTS = 5

$results = [ordered] @{}
foreach ($name in $suites.Keys) {
    Write-Host ""
    Write-Host "=== suite: $name ==="
    $ErrorActionPreference = "Continue"
    & $venvPython -m pytest -v -rs $suites[$name]
    $results[$name] = $LASTEXITCODE
    $ErrorActionPreference = "Stop"
}

### Verdict

Write-Host ""
Write-Host "=== summary ==="
foreach ($name in $results.Keys) {
    $code = $results[$name]
    $verdict = switch ($code) {
        0                 { "PASSED" }
        $PYTEST_NO_TESTS  { "NO TESTS (every module skipped on this platform)" }
        default           { "FAILED (exit $code)" }
    }
    Write-Host ("{0,-24} {1}" -f $name, $verdict)
}

$failed = @($results.Keys | Where-Object {
    $results[$_] -ne 0 -and $results[$_] -ne $PYTEST_NO_TESTS
})

if ($failed.Count -eq 0 -and -not $keepVenv) {
    Remove-Item -Recurse -Force $TestVenv, $examplesBuildDir -ErrorAction SilentlyContinue
} else {
    Write-Host ""
    Write-Host "venv kept at $TestVenv"
    Write-Host "examples kept at $examplesBuildDir"
}

if ($failed.Count -ne 0) {
    throw "$($failed.Count) suite(s) failed: $($failed -join ', ')"
}
Write-Host "All suites passed."
