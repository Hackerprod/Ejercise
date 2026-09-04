[CmdletBinding()]
param(
  [string]$BuildDirectory = "build-windows",
  [string]$Configuration = "Release",
  [string]$Python = "python",
  [switch]$SkipBuild,
  [switch]$SkipTransitionMicrobench
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if (-not $SkipBuild) { & (Join-Path $PSScriptRoot "build_windows.ps1") -BuildDirectory $BuildDirectory -Configuration $Configuration }
$Bin = Join-Path (Join-Path $Root $BuildDirectory) $Configuration
$Results = Join-Path $Root "results"
if (-not (Test-Path $Results)) { $null = New-Item -ItemType Directory -Path $Results }
& (Join-Path $Bin "cnrl_topology.exe") --json | Set-Content -Encoding utf8 (Join-Path $Results "topology.json")
if ($LASTEXITCODE -ne 0) { throw "Topology discovery failed" }
& (Join-Path $Bin "cnrl_tests.exe") | Set-Content -Encoding utf8 (Join-Path $Results "tests.txt")
if ($LASTEXITCODE -ne 0) { throw "Self-tests failed" }
& $Python (Join-Path $Root "tools/audit_source.py") | Set-Content -Encoding utf8 (Join-Path $Results "source_audit.txt")
if ($LASTEXITCODE -ne 0) { throw "Source invariant audit failed" }
& (Join-Path $PSScriptRoot "audit_windows_assembly.ps1") `
  -BuildDirectory (Join-Path $Root $BuildDirectory) -Configuration $Configuration `
  -OutputFile (Join-Path $Results "kernels.dumpbin.txt") |
  Set-Content -Encoding utf8 (Join-Path $Results "assembly_audit.txt")
if ($LASTEXITCODE -ne 0) { throw "MSVC assembly audit failed" }
& (Join-Path $PSScriptRoot "audit_windows_transitions.ps1") `
  -BuildDirectory (Join-Path $Root $BuildDirectory) -Configuration $Configuration `
  -OutputFile (Join-Path $Results "transitions.dumpbin.txt") |
  Set-Content -Encoding utf8 (Join-Path $Results "transition_assembly_audit.txt")
if ($LASTEXITCODE -ne 0) { throw "MSVC transition assembly audit failed" }
& (Join-Path $Bin "cnrl_bandwidth.exe") --mode read --mib 256 --repetitions 4 | Set-Content -Encoding ascii (Join-Path $Results "bandwidth_read.csv")
if ($LASTEXITCODE -ne 0) { throw "Read-only bandwidth benchmark failed" }
& (Join-Path $Bin "cnrl_bandwidth.exe") --mode copy --mib 256 --repetitions 4 | Set-Content -Encoding ascii (Join-Path $Results "bandwidth_copy.csv")
if ($LASTEXITCODE -ne 0) { throw "Copy bandwidth benchmark failed" }
$CalibrationPath = Join-Path $Results "calibration.csv"
& (Join-Path $Bin "cnrl_calibrate.exe") | Set-Content -Encoding ascii $CalibrationPath
if ($LASTEXITCODE -ne 0) { throw "Calibration executable failed" }
$calibration = @(Import-Csv $CalibrationPath | Where-Object { $_.valid -eq "true" })
if ($calibration.Count -eq 0) { throw "Calibration produced no valid rows" }
$Cpus = ($calibration | ForEach-Object { $_.logical_cpu }) -join ','
$Rates = ($calibration | ForEach-Object { $_.mac_per_second }) -join ','
$Gate = Join-Path $Bin "cnrl_gate.exe"
if (-not $SkipTransitionMicrobench) {
  & (Join-Path $PSScriptRoot "run_transition_bench.ps1") `
    -Executable (Join-Path $Bin "cnrl_transition_bench.exe") `
    -Output (Join-Path $Results "transitions.csv") -Cpus $Cpus -Rates $Rates
}
& (Join-Path $PSScriptRoot "run_t0r.ps1") -Executable $Gate -Output (Join-Path $Results "t0r.csv") -Cpus $Cpus -Rates $Rates
& (Join-Path $PSScriptRoot "run_t0m.ps1") -Executable $Gate -Output (Join-Path $Results "t0m.csv") -Cpus $Cpus -Rates $Rates
& (Join-Path $PSScriptRoot "run_t0rm.ps1") -Executable $Gate -Output (Join-Path $Results "t0rm.csv") -Cpus $Cpus -Rates $Rates
& $Python (Join-Path $PSScriptRoot "analyze_results.py") $Results --output (Join-Path $Results "analysis.md") --strict-structure
if ($LASTEXITCODE -ne 0) { throw "Structural analysis rejected the run" }
Write-Output $Results
