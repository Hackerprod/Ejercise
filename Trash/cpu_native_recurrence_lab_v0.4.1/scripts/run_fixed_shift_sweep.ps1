[CmdletBinding()]
param(
  [string]$Executable = "",
  [string]$Output = "results/fixed_shift_sweep.csv",
  [string]$Cpus = "0,2,4,6",
  [string]$Rates = "42.9,30.4,33.7,33.7",
  [int]$D = 1472,
  [int[]]$Slots = @(1,8,16),
  [int[]]$Depths = @(4,8),
  [int[]]$ProjectionShifts = @(12,13,14,15),
  [int]$ExternalRepeats = 4,
  [int]$InternalRepetitions = 4,
  [int]$Warmup = 2
)
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")
$Root = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($Executable)) {
  $Executable = Resolve-CnrlDefaultExecutable -Root $Root -Name "cnrl_gate"
}
if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
  throw "Executable not found: $Executable"
}
$OutputPath = if ([IO.Path]::IsPathRooted($Output)) { $Output } else { Join-Path $Root $Output }
$Directory = Split-Path -Parent $OutputPath
if (-not (Test-Path -LiteralPath $Directory -PathType Container)) {
  $null = New-Item -ItemType Directory -Path $Directory
}
$Log = [IO.Path]::ChangeExtension($OutputPath, ".stderr.log")
Remove-Item -LiteralPath $OutputPath,$Log -Force -ErrorAction SilentlyContinue
$HeaderWritten = $false
$Order = 0
for ($repeat = 1; $repeat -le $ExternalRepeats; $repeat++) {
  $shiftOrder = @($ProjectionShifts)
  if (($repeat % 2) -eq 0) { [array]::Reverse($shiftOrder) }
  foreach ($shift in $shiftOrder) {
    foreach ($slot in $Slots) {
      foreach ($depth in $Depths) {
        $variants = if (($repeat % 2) -eq 1) { @("shared","clone") } else { @("clone","shared") }
        foreach ($variant in $variants) {
          $Order++
          $arguments = @(
            "--gate","t0rm","--D","$D","--S","$slot","--R","$depth",
            "--kernel","fused","--slot-tile","4","--variant",$variant,
            "--transition","fixed","--cpus",$Cpus,"--rates",$Rates,
            "--projection-shift","$shift","--target-rms","32",
            "--warmup","$Warmup","--repetitions","$InternalRepetitions"
          )
          $lines = @(& $Executable @arguments 2>> $Log |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
          if ($LASTEXITCODE -ne 0 -or $lines.Count -ne 2) {
            throw "Fixed shift sweep failed: shift=$shift S=$slot R=$depth variant=$variant"
          }
          if (-not $HeaderWritten) {
            Set-Content -Encoding ascii -LiteralPath $OutputPath `
              -Value ($lines[0] + ",batch_repeat,variant_order")
            $HeaderWritten = $true
          }
          Add-Content -Encoding ascii -LiteralPath $OutputPath `
            -Value ($lines[1] + ",$repeat,$Order")
        }
      }
    }
  }
}
Write-Output $OutputPath
