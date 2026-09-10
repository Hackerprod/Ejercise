[CmdletBinding()]
param(
  [string]$BuildDirectory = "build-windows",
  [string]$Configuration = "Release",
  [ValidateSet("Auto", "Ninja", "VisualStudio")]
  [string]$Generator = "Auto",
  [string]$VisualStudioGenerator = "",
  [string]$Python = "python",
  [switch]$SkipExactStandaloneBridges,
  [switch]$SkipBuild,
  [switch]$SkipTransitionMicrobench
)
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")
$Root = Split-Path -Parent $PSScriptRoot
if (-not $SkipBuild) {
  & (Join-Path $PSScriptRoot "build_windows.ps1") -BuildDirectory $BuildDirectory `
    -Configuration $Configuration -Generator $Generator `
    -VisualStudioGenerator $VisualStudioGenerator
  if ($LASTEXITCODE -ne 0) { throw "Build script failed" }
}
$Topology = Resolve-CnrlBinary -Root $Root -BuildDirectory $BuildDirectory -Configuration $Configuration -Name "cnrl_topology"
$Tests = Resolve-CnrlBinary -Root $Root -BuildDirectory $BuildDirectory -Configuration $Configuration -Name "cnrl_tests"
$Bandwidth = Resolve-CnrlBinary -Root $Root -BuildDirectory $BuildDirectory -Configuration $Configuration -Name "cnrl_bandwidth"
$Calibrate = Resolve-CnrlBinary -Root $Root -BuildDirectory $BuildDirectory -Configuration $Configuration -Name "cnrl_calibrate"
$Gate = Resolve-CnrlBinary -Root $Root -BuildDirectory $BuildDirectory -Configuration $Configuration -Name "cnrl_gate"
$TransitionBench = Resolve-CnrlBinary -Root $Root -BuildDirectory $BuildDirectory -Configuration $Configuration -Name "cnrl_transition_bench"
$Results = Join-Path $Root "results"
if (-not (Test-Path $Results)) { $null = New-Item -ItemType Directory -Path $Results }
& $Topology --json | Set-Content -Encoding utf8 (Join-Path $Results "topology.json")
if ($LASTEXITCODE -ne 0) { throw "Topology discovery failed" }
& $Tests | Set-Content -Encoding utf8 (Join-Path $Results "tests.txt")
if ($LASTEXITCODE -ne 0) { throw "Self-tests failed" }
& $Python (Join-Path $Root "tools/audit_source.py") | Set-Content -Encoding utf8 (Join-Path $Results "source_audit.txt")
if ($LASTEXITCODE -ne 0) { throw "Source invariant audit failed" }
& (Join-Path $PSScriptRoot "audit_windows_assembly.ps1") `
  -BuildDirectory (Resolve-CnrlBuildDirectory -Root $Root -BuildDirectory $BuildDirectory) -Configuration $Configuration `
  -OutputFile (Join-Path $Results "kernels.dumpbin.txt") |
  Set-Content -Encoding utf8 (Join-Path $Results "assembly_audit.txt")
if ($LASTEXITCODE -ne 0) { throw "MSVC assembly audit failed" }
& (Join-Path $PSScriptRoot "audit_windows_transitions.ps1") `
  -BuildDirectory (Resolve-CnrlBuildDirectory -Root $Root -BuildDirectory $BuildDirectory) -Configuration $Configuration `
  -OutputFile (Join-Path $Results "transitions.dumpbin.txt") |
  Set-Content -Encoding utf8 (Join-Path $Results "transition_assembly_audit.txt")
if ($LASTEXITCODE -ne 0) { throw "MSVC transition assembly audit failed" }
& $Bandwidth --mode read --mib 256 --repetitions 4 | Set-Content -Encoding ascii (Join-Path $Results "bandwidth_read.csv")
if ($LASTEXITCODE -ne 0) { throw "Read-only bandwidth benchmark failed" }
& $Bandwidth --mode copy --mib 256 --repetitions 4 | Set-Content -Encoding ascii (Join-Path $Results "bandwidth_copy.csv")
if ($LASTEXITCODE -ne 0) { throw "Copy bandwidth benchmark failed" }
$CalibrationPath = Join-Path $Results "calibration.csv"
& $Calibrate | Set-Content -Encoding ascii $CalibrationPath
if ($LASTEXITCODE -ne 0) { throw "Calibration executable failed" }
$calibration = @(Import-Csv $CalibrationPath | Where-Object { $_.valid -eq "true" })
if ($calibration.Count -eq 0) { throw "Calibration produced no valid rows" }
$Cpus = ($calibration | ForEach-Object { $_.logical_cpu }) -join ','
$Rates = ($calibration | ForEach-Object { $_.mac_per_second }) -join ','
if (-not $SkipTransitionMicrobench) {
  & (Join-Path $PSScriptRoot "run_transition_bench.ps1") `
    -Executable $TransitionBench -Output (Join-Path $Results "transitions.csv") `
    -Cpus $Cpus -Rates $Rates
  & $Python (Join-Path $PSScriptRoot "analyze_transition_results.py") `
    (Join-Path $Results "transitions.csv") --strict `
    --output (Join-Path $Results "transition_analysis.md")
  if ($LASTEXITCODE -ne 0) { throw "Transition analysis rejected the run" }
}
& (Join-Path $PSScriptRoot "run_t0r.ps1") -Executable $Gate -Output (Join-Path $Results "t0r.csv") -Cpus $Cpus -Rates $Rates
& (Join-Path $PSScriptRoot "run_t0m.ps1") -Executable $Gate -Output (Join-Path $Results "t0m.csv") -Cpus $Cpus -Rates $Rates
& (Join-Path $PSScriptRoot "run_t0rm.ps1") -Executable $Gate -Output (Join-Path $Results "t0rm.csv") -Cpus $Cpus -Rates $Rates
if (-not $SkipExactStandaloneBridges) {
  & (Join-Path $PSScriptRoot "run_exact_bridges.ps1") -Executable $Gate `
    -OutputDirectory $Results -Cpus $Cpus -Rates $Rates -D 1472
}
& $Python (Join-Path $PSScriptRoot "analyze_results.py") $Results --output (Join-Path $Results "analysis.md") --strict-structure
if ($LASTEXITCODE -ne 0) { throw "Structural analysis rejected the run" }
Write-Output $Results
