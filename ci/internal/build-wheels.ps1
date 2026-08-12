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

# Build hip-python wheels on Windows.
#
# PowerShell counterpart to ci/internal/build-wheels.sh. It drives the same
# unified packages/CMakeLists.txt aggregate target, and deviates from the bash
# script only where Linux assumptions do not hold:
#
#   * No auditwheel/patchelf. auditwheel is a Linux ELF retagger;
#     packages/CMakeLists.txt already ignores HIP_PYTHON_AUDITWHEEL_REPAIR on
#     Windows because the wheel assembler emits a native win_amd64 tag.
#   * Ninja instead of "Unix Makefiles". The bash script forces make purely to
#     dodge a GCC jobserver bug on the largest generated .c files; that does not
#     apply here, and scikit-build-core expects ninja.
#
# libLLVM bundling and numba-hip are both on, as they are on Linux. ROCm ships no
# shared LLVM on Windows, so bundling links one from the static archives and adds
# ~75 MB to the compiler wheel; that is the price of rocm.bindings.llvm.* working
# at all, and numba.hip needs those bindings, so the two travel together. Decline
# them with -NoBundleLibLLVM (which implies -NoNumbaHip) or -NoNumbaHip alone.
#
# The MSVC environment is imported automatically when the script is not already
# running inside a Developer shell, so it behaves the same from a plain
# PowerShell, a CI runner, or a preconfigured terminal. That also avoids picking
# up Git's Unix link.exe instead of MSVC's linker.
#
# No ROCm version appears anywhere: everything is derived from -RocmPath.
#
# Usage:
#   ci\internal\build-wheels.ps1                      # ROCM_PATH from the environment
#   ci\internal\build-wheels.ps1 -RocmPath C:\rocm
#   ci\internal\build-wheels.ps1 -Light               # core + hip + compiler only
#   ci\internal\build-wheels.ps1 -UseRocmClang        # amdclang-cl instead of MSVC
#   ci\internal\build-wheels.ps1 -UseSabi 3.11        # one cp311-abi3 wheel set
#   ci\internal\build-wheels.ps1 -NoBundleLibLLVM     # smaller wheel, no numba.hip

[CmdletBinding()]
param(
    # ROCm root. Defaults to ROCM_PATH / ROCM_HOME from the environment
    # (see ci\internal\env-rocm.ps1).
    [string] $RocmPath = $(if ($env:ROCM_PATH) { $env:ROCM_PATH } else { $env:ROCM_HOME }),

    # Build only core + hip + compiler, mirroring LIGHT_MODE in the bash script.
    # Chooses which packages are built, not how the compiler wheel is
    # configured, so pair it with -NoBundleLibLLVM for the quickest build of
    # all: linking the shared LLVM is the longest single step here.
    [switch] $Light,

    # Skip rocm-bindings-compiler, which needs a usable LLVM CMake package.
    [switch] $NoCompiler,

    # Do not link a shared LLVM into the compiler wheel. Saves ~75 MB, at the
    # cost of rocm.bindings.llvm.*, which then imports but raises on first use.
    # numba.hip is built on those bindings, so this implies -NoNumbaHip.
    [switch] $NoBundleLibLLVM,

    # Skip the pure-Python numba-hip wheel.
    [switch] $NoNumbaHip,

    # Compile with ROCm's clang (amdclang-cl, the MSVC-compatible driver)
    # instead of MSVC. Useful if ROCm headers need clang extensions.
    [switch] $UseRocmClang,

    # Build limited-API (abi3) wheels against the CPython stable ABI, with this
    # CPython version as the floor, e.g. "3.11". One such wheel set loads on
    # every interpreter from the floor upwards, so the build no longer has to be
    # repeated per Python version. "no" (the default) builds version-specific
    # wheels. Reads USE_SABI from the environment, the same name the bash script
    # uses, so CI can set it once for both platforms.
    [string] $UseSabi = $(if ($env:USE_SABI) { $env:USE_SABI } else { "no" }),

    [string] $BuildDir = "build",
    [string] $WheelOutputDir,
    [string] $Target = "all_wheels",
    [int]    $MaxJobs = 0,
    [string[]] $ExtraCMakeArgs = @()
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path

# Run a native command, failing on its exit code rather than on its stderr.
#
# CMake reports warnings and some progress on stderr, and with
# $ErrorActionPreference = 'Stop' PowerShell turns a native command's stderr into
# a terminating error the moment a caller merges the streams (`build-wheels.ps1
# ... 2>&1 | Tee-Object`, which is how one would capture a build log). The build
# would then abort on the first CMake warning with a NativeCommandError that
# says nothing about the warning. Exit codes carry the real verdict, so the
# preference is relaxed for the duration of the call only.
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

### Stable-ABI (abi3) floor
#
# Validated before anything expensive runs, because the failure modes further
# down are unhelpful: CMake reports a malformed floor only after the toolchain
# probe, and a floor below 3.11 gets all the way into the C compiler, where the
# bindings fail on the buffer-protocol calls in rocm-bindings-core's types.pyx
# (PyObject_GetBuffer and friends entered the limited API in 3.11).
#
# The floor is independent of the interpreter running the build, except that it
# cannot exceed it -- you cannot target a newer stable ABI than the headers you
# compile against. HipPythonBuild.cmake enforces that side, since it is the one
# that knows which Python was found.
$abi3Floor = ""
if ($UseSabi -ne "no") {
    if ($UseSabi -notmatch '^3\.[0-9]{2}$') {
        throw "-UseSabi must be 'no' or a CPython floor version like 3.11; got '$UseSabi'."
    }
    if ([version] $UseSabi -lt [version] "3.11") {
        throw "-UseSabi floor must be at least 3.11 (the bindings use the buffer " +
              "protocol, which the stable ABI only exposes from 3.11 on); got '$UseSabi'."
    }
    $abi3Floor = $UseSabi
}

### MSVC environment

function Initialize-MsvcEnvironment {
    # Ask for the compiler rather than for the marker variable VSCMD_VER: a
    # process can inherit that variable while its PATH carries none of the MSVC
    # directories, which is what happens when a tool spawns a shell from a
    # Developer PowerShell with a trimmed environment. Trusting the marker there
    # skips the bootstrap and fails on the cl.exe check below instead.
    if ($env:VSCMD_VER -and (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
        Write-Host "MSVC environment already present (VSCMD_VER=$env:VSCMD_VER)"
        return
    }

    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path -LiteralPath $vswhere)) {
        throw "vswhere.exe not found at $vswhere. Install Visual Studio with the " +
              "C++ build tools, or run this script from a Developer PowerShell."
    }

    $vsPath = & $vswhere -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath
    if (-not $vsPath) {
        throw "No Visual Studio installation with the x86/x64 C++ tools was found."
    }

    $devShell = Join-Path $vsPath "Common7\Tools\Launch-VsDevShell.ps1"
    if (-not (Test-Path -LiteralPath $devShell)) {
        throw "Launch-VsDevShell.ps1 not found under $vsPath."
    }

    Write-Host "Importing MSVC environment from $vsPath"
    # -SkipAutomaticLocation keeps the caller's working directory.
    & $devShell -Arch amd64 -HostArch amd64 -SkipAutomaticLocation | Out-Null
}

Initialize-MsvcEnvironment

if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    throw "cl.exe is still not on PATH after importing the MSVC environment."
}

### ROCm location

if (-not $RocmPath) {
    throw "No ROCm root. Pass -RocmPath, or dot-source ci\internal\env-rocm.ps1 first."
}
$RocmPath = (Resolve-Path -LiteralPath $RocmPath).Path
if (-not (Test-Path -LiteralPath (Join-Path $RocmPath "include"))) {
    throw "No 'include' directory under $RocmPath - hip_python_initialize() will fail."
}

### Build tooling

foreach ($tool in @("cmake", "ninja")) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        throw "$tool not found. Install the build requirements: " +
              "python -m pip install -r ci\internal\requirements-build.txt"
    }
}

if (-not $WheelOutputDir) {
    $WheelOutputDir = Join-Path $repoRoot "packages\$BuildDir\dist"
}
if ($MaxJobs -le 0) {
    $MaxJobs = [Environment]::ProcessorCount
}

### CMake arguments

$buildLibraries = if ($Light) { "OFF" } else { "ON" }
$buildSystems   = if ($Light) { "OFF" } else { "ON" }
$buildInterop   = if ($Light) { "OFF" } else { "ON" }
$buildCompiler  = if ($NoCompiler) { "OFF" } else { "ON" }

# Bundling and numba-hip travel together. numba.hip is built on the
# rocm.bindings.llvm.* modules, and those only work against a bundled shared
# LLVM, so anything that removes the bundle removes numba-hip too: -NoCompiler
# drops the wheel that carries the bindings, and -Light drops
# hip-python-interop, which numba-hip declares as a dependency and so cannot be
# installed without.
$bundleLibLLVM = if ($NoBundleLibLLVM) { "OFF" } else { "ON" }
$buildNumbaHip = if ($NoNumbaHip -or $NoBundleLibLLVM -or $NoCompiler -or $Light) {
    "OFF"
} else {
    "ON"
}
if ($buildNumbaHip -eq "OFF" -and -not $NoNumbaHip) {
    Write-Host ("Not building numba-hip: it needs the bundled shared LLVM and " +
                "hip-python-interop, which this configuration leaves out.")
}

# Pin the host compiler explicitly. ROCm's own clang.exe sits in
# $RocmPath\lib\llvm\bin, which env-rocm.ps1 prepends to PATH so the bindings
# can find the version-suffixed ROCm DLLs at import time. CMake's Ninja
# generator would then pick that clang up ahead of MSVC, i.e. PATH order --
# not intent -- would decide the toolchain. Naming the compiler makes the
# choice deterministic regardless of how the environment was set up.
#
# MSVC is the default because the host CPython is built with it, so its CRT
# and ABI are what the extension modules must match. amdclang-cl (below) is
# the MSVC-compatible clang driver, so it keeps that same ABI; it is worth
# reaching for as a second opinion because it diagnoses portability problems
# MSVC stays quiet about (it is what caught the LLP64 pointer truncation in
# rocm-bindings-core's types.pyx).
if ($UseRocmClang) {
    $hostCompiler = Join-Path $RocmPath "lib\llvm\bin\amdclang-cl.exe"
    if (-not (Test-Path -LiteralPath $hostCompiler)) {
        throw "amdclang-cl.exe not found at $hostCompiler."
    }
} else {
    $hostCompiler = "cl"
}

$cmakeArgs = @(
    "-G", "Ninja",
    "-S", "packages",
    "-B", "packages/$BuildDir",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_C_COMPILER=$hostCompiler",
    "-DCMAKE_CXX_COMPILER=$hostCompiler",
    "-DROCM_PATH=$RocmPath",
    "-DHIP_PLATFORM=amd",
    "-DHIP_PYTHON_BUILD_CORE=ON",
    "-DHIP_PYTHON_BUILD_HIP=ON",
    "-DHIP_PYTHON_BUILD_LIBRARIES=$buildLibraries",
    "-DHIP_PYTHON_BUILD_SYSTEMS=$buildSystems",
    "-DHIP_PYTHON_BUILD_COMPILER=$buildCompiler",
    "-DHIP_PYTHON_BUILD_INTEROP=$buildInterop",
    "-DHIP_PYTHON_BUNDLE_LIBLLVM=$bundleLibLLVM",
    "-DHIP_PYTHON_BUILD_NUMBA_HIP=$buildNumbaHip",
    # auditwheel is Linux-only; wheels already carry a win_amd64 tag.
    "-DHIP_PYTHON_AUDITWHEEL_REPAIR=OFF",
    "-DHIP_PYTHON_WHEEL_OUTPUT_DIR=$WheelOutputDir"
)

# Quote the whole -D argument: Windows PowerShell 5.1 splits an unquoted
# `-DFOO=3.11` into `-DFOO=3` and `.11` on its way to a native command, and CMake
# then rejects the floor as '3' while warning about a stray path.
if ($abi3Floor) {
    $cmakeArgs += "-DHIP_PYTHON_ABI3_FLOOR=$abi3Floor"
    Write-Host ("Building against the CPython stable ABI, floor $abi3Floor; " +
                "wheels will be tagged cp$($abi3Floor -replace '\.','')-abi3.")
}

$cmakeArgs += $ExtraCMakeArgs

Push-Location $repoRoot
try {
    Write-Host "cmake $($cmakeArgs -join ' ')"
    Invoke-Native cmake @cmakeArgs
    Invoke-Native cmake --build "packages/$BuildDir" --target $Target --parallel $MaxJobs
}
finally {
    Pop-Location
}

Write-Host "Wheels written to $WheelOutputDir"
