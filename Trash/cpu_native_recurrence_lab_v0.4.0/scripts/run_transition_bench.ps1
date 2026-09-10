[CmdletBinding()]
param(
  [string]$Executable = "",
  [string]$Output = "results/transitions.csv",
  [string]$Cpus = "0,2,4,6",
  [string]$Rates = "19.3,18.1,10.9,17.0",
  [int[]]$Dimensions = @(512,1472,1600),
  [int[]]$Slots = @(1,4,8,16),
  [string[]]$Transitions = @("fixed","group-rms","global-rms"),
  [int]$Warmup = 100,
  [int]$Repetitions = 1000,
  [int]$ProjectionShift = 12
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($Executable)) {
  $Executable = Join-Path $Root "build-windows/Release/cnrl_transition_bench.exe"
}
if (-not (Test-Path $Executable -PathType Leaf)) { throw "Executable not found: $Executable" }
$OutputPath = if ([IO.Path]::IsPathRooted($Output)) { $Output } else { Join-Path $Root $Output }
$Directory = Split-Path -Parent $OutputPath
if (-not (Test-Path $Directory)) { $null = New-Item -ItemType Directory -Path $Directory }
Remove-Item $OutputPath -Force -ErrorAction SilentlyContinue
$HeaderWritten = $false
foreach ($dimension in $Dimensions) {
  foreach ($slot in $Slots) {
    foreach ($transition in $Transitions) {
      $lines = @(& $Executable --D $dimension --S $slot --transition $transition `
        --cpus $Cpus --rates $Rates --projection-shift $ProjectionShift `
        --warmup $Warmup --repetitions $Repetitions |
        Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
      if ($LASTEXITCODE -ne 0 -or $lines.Count -ne 2) {
        throw "Transition benchmark failed: D=$dimension S=$slot transition=$transition"
      }
      if (-not $HeaderWritten) {
        Set-Content -Encoding ascii -Path $OutputPath -Value $lines[0]
        $HeaderWritten = $true
      }
      Add-Content -Encoding ascii -Path $OutputPath -Value $lines[1]
    }
  }
}
Write-Output $OutputPath
