[CmdletBinding()]
param(
  [string]$BuildDirectory = "build-windows",
  [string]$Configuration = "Release",
  [ValidateSet("Auto", "Ninja", "VisualStudio")]
  [string]$Generator = "Auto",
  [string]$VisualStudioGenerator = "",
  [string]$Python = "python",
  [string]$Cpus = "0,2,4,6",
  [string]$Rates = "42.9,30.4,33.7,33.7",
  [string]$OutputDirectory = "results-v041-patch",
  [switch]$SkipBuild
)
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")
$Root = Split-Path -Parent $PSScriptRoot
if (-not $SkipBuild) {
  & (Join-Path $PSScriptRoot "build_windows.ps1") -BuildDirectory $BuildDirectory `
    -Configuration $Configuration -Generator $Generator `
    -VisualStudioGenerator $VisualStudioGenerator -Clean
}
$Build = Resolve-CnrlBuildDirectory -Root $Root -BuildDirectory $BuildDirectory
$Gate = Resolve-CnrlBinary -Root $Root -BuildDirectory $BuildDirectory `
  -Configuration $Configuration -Name "cnrl_gate"
$TransitionBench = Resolve-CnrlBinary -Root $Root -BuildDirectory $BuildDirectory `
  -Configuration $Configuration -Name "cnrl_transition_bench"
$Results = if ([IO.Path]::IsPathRooted($OutputDirectory)) {
  $OutputDirectory
} else {
  Join-Path $Root $OutputDirectory
}
if (Test-Path -LiteralPath $Results) { Remove-Item -Recurse -Force -LiteralPath $Results }
$null = New-Item -ItemType Directory -Path $Results

& $Python (Join-Path $Root "tools/audit_source.py") |
  Set-Content -Encoding utf8 (Join-Path $Results "source_audit.txt")
if ($LASTEXITCODE -ne 0) { throw "Source invariant audit failed" }

& (Join-Path $PSScriptRoot "audit_windows_assembly.ps1") `
  -BuildDirectory $Build -Configuration $Configuration `
  -OutputFile (Join-Path $Results "kernels.dumpbin.txt") |
  Set-Content -Encoding utf8 (Join-Path $Results "assembly_audit.txt")
& (Join-Path $PSScriptRoot "audit_windows_transitions.ps1") `
  -BuildDirectory $Build -Configuration $Configuration `
  -OutputFile (Join-Path $Results "transitions.dumpbin.txt") |
  Set-Content -Encoding utf8 (Join-Path $Results "transition_assembly_audit.txt")

& (Join-Path $PSScriptRoot "run_exact_bridges.ps1") -Executable $Gate `
  -OutputDirectory $Results -Cpus $Cpus -Rates $Rates -D 1472 `
  -Depths @(4,8) -Slots @(1,8,16) -ExternalRepeats 4 -InternalRepetitions 4 -Warmup 2

& (Join-Path $PSScriptRoot "run_t0rm.ps1") -Executable $Gate `
  -Output (Join-Path $Results "t0rm_v041.csv") -Cpus $Cpus -Rates $Rates `
  -D 1472 -Slots @(1,8,16) -Depths @(8) -ExternalRepeats 4 `
  -InternalRepetitions 4 -Warmup 2 -FixedProjectionShift 14 -RmsProjectionShift 12

& (Join-Path $PSScriptRoot "run_transition_bench.ps1") -Executable $TransitionBench `
  -Output (Join-Path $Results "transitions_v041.csv") -Cpus $Cpus -Rates $Rates `
  -Dimensions @(1472) -Slots @(1,8,16) -ChainLengths @(1,8) `
  -WarmupChains 25 -Chains 250 -FixedProjectionShift 14 -RmsProjectionShift 12

& $Python (Join-Path $PSScriptRoot "analyze_transition_results.py") `
  (Join-Path $Results "transitions_v041.csv") --strict `
  --output (Join-Path $Results "transition_analysis.md")
if ($LASTEXITCODE -ne 0) { throw "Transition analysis rejected the patch run" }

& (Join-Path $PSScriptRoot "run_fixed_shift_sweep.ps1") -Executable $Gate `
  -Output (Join-Path $Results "fixed_shift_sweep.csv") -Cpus $Cpus -Rates $Rates `
  -D 1472 -Slots @(1,8,16) -Depths @(8) -ProjectionShifts @(12,13,14,15) `
  -ExternalRepeats 4 -InternalRepetitions 4 -Warmup 2

& $Python (Join-Path $PSScriptRoot "analyze_results.py") $Results `
  --output (Join-Path $Results "analysis.md") --strict-structure
if ($LASTEXITCODE -ne 0) { throw "v0.4.1 patch validation was rejected" }
Write-Output $Results
