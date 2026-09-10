[CmdletBinding()]
param(
  [string]$BuildDirectory = "build-windows",
  [ValidateSet("Release", "RelWithDebInfo", "Debug")]
  [string]$Configuration = "Release",
  [switch]$Clean,
  [switch]$Sanitizers
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Build = Join-Path $Root $BuildDirectory
if ($Clean -and (Test-Path $Build)) { Remove-Item -Recurse -Force $Build }
$cmakeArgs = @("-S", $Root, "-B", $Build, "-G", "Visual Studio 17 2022", "-A", "x64", "-DCNRL_WARNINGS_AS_ERRORS=ON")
if ($Sanitizers) { $cmakeArgs += "-DCNRL_ENABLE_SANITIZERS=ON" }
& cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed" }
& cmake --build $Build --config $Configuration --parallel
if ($LASTEXITCODE -ne 0) { throw "Build failed" }
& ctest --test-dir $Build -C $Configuration --output-on-failure
if ($LASTEXITCODE -ne 0) { throw "Tests failed" }
Write-Output (Join-Path $Build $Configuration)
