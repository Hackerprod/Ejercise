[CmdletBinding()]
param(
  [string]$BuildDirectory = "build-windows",
  [ValidateSet("Release", "RelWithDebInfo", "Debug")]
  [string]$Configuration = "Release",
  [ValidateSet("Auto", "Ninja", "VisualStudio")]
  [string]$Generator = "Auto",
  [string]$VisualStudioGenerator = "",
  [switch]$Clean,
  [switch]$Sanitizers
)
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")
$Root = Split-Path -Parent $PSScriptRoot
$Build = Resolve-CnrlBuildDirectory -Root $Root -BuildDirectory $BuildDirectory
if ($Clean -and (Test-Path -LiteralPath $Build)) {
  Remove-Item -Recurse -Force -LiteralPath $Build
}

$CMakeCommand = Get-Command cmake.exe -ErrorAction SilentlyContinue
if ($null -eq $CMakeCommand) { $CMakeCommand = Get-Command cmake -ErrorAction SilentlyContinue }
if ($null -eq $CMakeCommand) { throw "cmake was not found in PATH" }
$CMake = $CMakeCommand.Source

$SelectedGenerator = $Generator
$Ninja = $null
if ($SelectedGenerator -eq "Auto" -or $SelectedGenerator -eq "Ninja") {
  Import-CnrlVcVars -Architecture x64
  $Ninja = Find-CnrlNinja
  if ($SelectedGenerator -eq "Ninja" -and [string]::IsNullOrWhiteSpace($Ninja)) {
    throw "Ninja was requested but ninja.exe was not found"
  }
  if ($SelectedGenerator -eq "Auto") {
    $SelectedGenerator = if ([string]::IsNullOrWhiteSpace($Ninja)) { "VisualStudio" } else { "Ninja" }
  }
}

$cmakeArgs = @("-S", $Root, "-B", $Build, "-DCNRL_WARNINGS_AS_ERRORS=ON")
if ($Sanitizers) { $cmakeArgs += "-DCNRL_ENABLE_SANITIZERS=ON" }

if ($SelectedGenerator -eq "Ninja") {
  $NinjaDirectory = Split-Path -Parent $Ninja
  if ($env:PATH -notlike "*$NinjaDirectory*") { $env:PATH = "$NinjaDirectory;$env:PATH" }
  $cmakeArgs += @("-G", "Ninja", "-DCMAKE_BUILD_TYPE=$Configuration")
} else {
  $ResolvedVisualStudioGenerator = $VisualStudioGenerator
  if ([string]::IsNullOrWhiteSpace($ResolvedVisualStudioGenerator)) {
    $ResolvedVisualStudioGenerator = Find-CnrlLatestVisualStudioGenerator
  }
  $cmakeArgs += @("-G", $ResolvedVisualStudioGenerator, "-A", "x64")
}

& $CMake @cmakeArgs
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed" }

$CachePath = Join-Path $Build "CMakeCache.txt"
$ActualGenerator = Get-CnrlCacheValue -CachePath $CachePath -Name "CMAKE_GENERATOR"
if ([string]::IsNullOrWhiteSpace($ActualGenerator)) {
  throw "CMAKE_GENERATOR was not found in $CachePath"
}
$ConfigurationTypes = Get-CnrlCacheValue -CachePath $CachePath -Name "CMAKE_CONFIGURATION_TYPES"
$IsMultiConfig = -not [string]::IsNullOrWhiteSpace($ConfigurationTypes)

$buildArgs = @("--build", $Build, "--parallel")
if ($IsMultiConfig) { $buildArgs += @("--config", $Configuration) }
& $CMake @buildArgs
if ($LASTEXITCODE -ne 0) { throw "Build failed" }

$CTestCommand = Get-Command ctest.exe -ErrorAction SilentlyContinue
if ($null -eq $CTestCommand) { $CTestCommand = Get-Command ctest -ErrorAction SilentlyContinue }
if ($null -eq $CTestCommand) { throw "ctest was not found in PATH" }
$testArgs = @("--test-dir", $Build, "--output-on-failure")
if ($IsMultiConfig) { $testArgs += @("-C", $Configuration) }
& $CTestCommand.Source @testArgs
if ($LASTEXITCODE -ne 0) { throw "Tests failed" }

$Gate = Resolve-CnrlBinary -Root $Root -BuildDirectory $BuildDirectory `
  -Configuration $Configuration -Name "cnrl_gate"
$Bin = Split-Path -Parent $Gate
$Compiler = Get-CnrlCacheValue -CachePath $CachePath -Name "CMAKE_CXX_COMPILER"
$BuildInfo = [ordered]@{
  project_version = "0.4.1"
  requested_generator = $Generator
  generator = $ActualGenerator
  multi_config = $IsMultiConfig
  configuration = $Configuration
  build_directory = $Build
  binary_directory = $Bin
  compiler = $Compiler
  ninja = $Ninja
}
$BuildInfo | ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $Build "cnrl-build-info.json")
Write-Output $Bin
