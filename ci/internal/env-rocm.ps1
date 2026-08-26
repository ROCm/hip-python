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

# Point the current PowerShell session at a ROCm installation.
#
# Dot-source it:  . ci\internal\env-rocm.ps1 -Root C:\path\to\rocm
#
# CMake resolves the ROCm location from ROCM_PATH, then ROCM_HOME, and
# otherwise falls back to the Linux default /opt/rocm (see
# hip_python_get_rocm_path_default in cmake/HipPythonBuild.cmake), so
# ROCM_PATH is the variable that actually matters for the build. HIP_PATH /
# LLVM_PATH / HIP_DEVICE_LIB_PATH are what the HIP tools themselves read.
#
# Two ROCm layouts are supported and both are auto-detected:
#
#   * A "flat" install -- an unpacked ROCm SDK tarball or a system HIP SDK.
#     Pass its root with -Root.
#   * A pip/TheRock rocm-sdk install. Omit -Root and the devel tree is
#     resolved from the active interpreter via `rocm_sdk path --root`.
#
# No ROCm version appears anywhere below: everything is derived from the
# resolved root, so the same script works for any release.

[CmdletBinding()]
param(
    # Root of a flat ROCm install. When omitted, the rocm-sdk wheel install
    # in the active Python environment is used instead.
    [string] $Root,

    # Interpreter used to locate a rocm-sdk wheel install. Defaults to the
    # active environment's python.
    [string] $Python = "python"
)

$ErrorActionPreference = "Stop"

if (-not $Root) {
    Write-Output "No -Root given; querying $Python for a rocm-sdk install..."
    $Root = & $Python -m rocm_sdk path --root 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $Root) {
        throw "Could not resolve a ROCm root. Pass -Root <path> for a flat " +
              "install, or activate an environment with the 'rocm' wheels installed."
    }
    $Root = $Root.Trim()
}

$Root = (Resolve-Path -LiteralPath $Root).Path

# The build's own precondition: hip_python_initialize() fails without these.
if (-not (Test-Path -LiteralPath (Join-Path $Root "include"))) {
    throw "No 'include' directory under $Root - this is not a ROCm root. " +
          "For a pip install, point at the rocm-sdk-devel tree " +
          "(python -m rocm_sdk path --root), not the venv."
}

$llvm = Join-Path $Root "lib\llvm"

$env:ROCM_PATH = $Root
$env:ROCM_HOME = $Root
$env:HIP_PATH = $Root
$env:HIP_PLATFORM = "amd"
if (Test-Path -LiteralPath $llvm) {
    $env:LLVM_PATH = $llvm
    $bitcode = Join-Path $llvm "amdgcn\bitcode"
    if (Test-Path -LiteralPath $bitcode) {
        $env:HIP_DEVICE_LIB_PATH = $bitcode
    }
}

# Prepend the ROCm binary directories, so that ROCm's own tools resolve and so
# that a loaded DLL's siblings are reachable.
#
# Note that this is NOT what lets the bindings find the runtime: ROCm's Windows
# DLLs are version-suffixed (amdhip64_7.dll, hiprtc<ver>.dll), so handing the
# loader the plain amdhip64.dll that the other platforms use would fail no
# matter what PATH holds. rocm.bindings.util.paths resolves the real filename
# from ROCM_PATH\bin instead (and walks PATH itself when no ROCm tree is set).
#
# Side effect worth knowing about: lib\llvm\bin also holds ROCm's clang.exe,
# so a bare `cmake -G Ninja` run in this session would select that clang as
# the host compiler ahead of MSVC. ci\internal\build-wheels.ps1 therefore names
# CMAKE_C_COMPILER explicitly instead of letting PATH order decide.
$binDirs = @((Join-Path $Root "bin"))
if (Test-Path -LiteralPath $llvm) { $binDirs += (Join-Path $llvm "bin") }
foreach ($dir in $binDirs) {
    if ((Test-Path -LiteralPath $dir) -and ($env:PATH -notlike "*$dir*")) {
        $env:PATH = "$dir;$env:PATH"
    }
}

Write-Output "ROCM_PATH  = $env:ROCM_PATH"
Write-Output "LLVM_PATH  = $env:LLVM_PATH"
Write-Output "Verify the install with: hipInfo.exe"
