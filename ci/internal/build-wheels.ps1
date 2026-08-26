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
#   * numba-hip is off. numba.hip raises NotImplementedError on Windows.
#   * HIP_PYTHON_FORCE_BUILD_LIBLLVM is not set. CI enables it for numba-hip's
#     benefit and it relies on GNU --whole-archive semantics.
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

[CmdletBinding()]
param(
    # ROCm root. Defaults to ROCM_PATH / ROCM_HOME from the environment
    # (see ci\internal\env-rocm.ps1).
    [string] $RocmPath = $(if ($env:ROCM_PATH) { $env:ROCM_PATH } else { $env:ROCM_HOME }),

    # Build only core + hip + compiler, mirroring LIGHT_MODE in the bash script.
    [switch] $Light,

    # Skip rocm-bindings-compiler, which needs a usable LLVM CMake package.
    [switch] $NoCompiler,

    # Compile with ROCm's clang (amdclang-cl, the MSVC-compatible driver)
    # instead of MSVC. Useful if ROCm headers need clang extensions.
    [switch] $UseRocmClang,

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
              "python -m pip install -r ci\requirements-build.txt"
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
    # numba.hip does not support Windows.
    "-DHIP_PYTHON_BUILD_NUMBA_HIP=OFF",
    # auditwheel is Linux-only; wheels already carry a win_amd64 tag.
    "-DHIP_PYTHON_AUDITWHEEL_REPAIR=OFF",
    "-DHIP_PYTHON_WHEEL_OUTPUT_DIR=$WheelOutputDir"
)

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
