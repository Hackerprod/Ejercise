[CmdletBinding()]
param(
  [string]$Executable = "",
  [string]$OutputDirectory = "results",
  [string]$Cpus = "0,2,4,6",
  [string]$Rates = "19.3,18.1,10.9,17.0",
  [int]$D = 1472,
  [int[]]$Depths = @(4,8),
  [int[]]$Slots = @(1,8,16),
  [int]$ExternalRepeats = 6,
  [int]$InternalRepetitions = 4,
  [int]$Warmup = 2
)
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")
$Root = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($Executable)) {
  $Executable = Resolve-CnrlDefaultExecutable -Root $Root -Name "cnrl_gate"
}
$Directory = if ([IO.Path]::IsPathRooted($OutputDirectory)) {
  $OutputDirectory
} else {
  Join-Path $Root $OutputDirectory
}
if (-not (Test-Path -LiteralPath $Directory -PathType Container)) {
  $null = New-Item -ItemType Directory -Path $Directory
}

$T0r = Join-Path $Directory "t0r_bridge_D$D.csv"
$T0m = Join-Path $Directory "t0m_bridge_D$D.csv"
& (Join-Path $PSScriptRoot "run_t0r.ps1") -Executable $Executable -Output $T0r `
  -Cpus $Cpus -Rates $Rates -D $D -SquareOutput -Depths $Depths `
  -ExternalRepeats $ExternalRepeats -InternalRepetitions $InternalRepetitions -Warmup $Warmup
if ($LASTEXITCODE -ne 0) { throw "Exact T0-R bridge failed" }
& (Join-Path $PSScriptRoot "run_t0m.ps1") -Executable $Executable -Output $T0m `
  -Cpus $Cpus -Rates $Rates -D $D -SquareOutput -Depths $Depths -Slots $Slots `
  -ExternalRepeats $ExternalRepeats -InternalRepetitions $InternalRepetitions -Warmup $Warmup
if ($LASTEXITCODE -ne 0) { throw "Exact T0-M bridge failed" }
Write-Output $T0r
Write-Output $T0m
