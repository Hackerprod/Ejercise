[CmdletBinding()]
param(
  [string]$Executable = "",
  [string]$Output = "results/t0r.csv",
  [string]$Cpus = "0,2,4,6",
  [string]$Rates = "19.3,18.1,10.9,17.0",
  [int]$D = 512,
  [switch]$SquareOutput,
  [int[]]$SizesKiB = @(384,512,640,768),
  [int[]]$Depths = @(1,4,8,16),
  [int]$ExternalRepeats = 6,
  [int]$InternalRepetitions = 5,
  [int]$Warmup = 2
)
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")
$Root = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($Executable)) { $Executable = Resolve-CnrlDefaultExecutable -Root $Root -Name "cnrl_gate" }
if (-not (Test-Path $Executable -PathType Leaf)) { throw "Executable not found: $Executable" }
$OutputPath = if ([IO.Path]::IsPathRooted($Output)) { $Output } else { Join-Path $Root $Output }
$Dir = Split-Path -Parent $OutputPath
if (-not (Test-Path $Dir)) { $null = New-Item -ItemType Directory -Path $Dir }
$Log = [IO.Path]::ChangeExtension($OutputPath, ".stderr.log")
Remove-Item $OutputPath,$Log -Force -ErrorAction SilentlyContinue
$HeaderWritten = $false
function Add-Run([string[]]$Arguments,[int]$Repeat,[int]$Order) {
  $lines = @(& $Executable @Arguments 2>> $Log | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
  if ($LASTEXITCODE -ne 0 -or $lines.Count -ne 2) { throw "Gate invocation failed: $($Arguments -join ' ')" }
  if (-not $script:HeaderWritten) {
    Set-Content -Encoding ascii -Path $OutputPath -Value ($lines[0] + ",batch_repeat,variant_order")
    $script:HeaderWritten = $true
  }
  Add-Content -Encoding ascii -Path $OutputPath -Value ($lines[1] + ",$Repeat,$Order")
}
$EffectiveSizes = if ($SquareOutput) { @(0) } else { @($SizesKiB) }
for ($repeat=1; $repeat -le $ExternalRepeats; $repeat++) {
  foreach ($size in $EffectiveSizes) {
    foreach ($depth in $Depths) {
      $variants = if (($repeat % 2) -eq 1) { @("shared","clone") } else { @("clone","shared") }
      for ($position=0; $position -lt $variants.Count; $position++) {
        $Arguments = @(
          "--gate","t0r","--D","$D","--S","1","--R","$depth",
          "--kernel","fused","--slot-tile","4","--variant",$variants[$position],
          "--cpus",$Cpus,"--rates",$Rates,
          "--warmup","$Warmup","--repetitions","$InternalRepetitions"
        )
        if ($SquareOutput) { $Arguments += "--square-output" }
        else { $Arguments += @("--average-weight-kib-per-core","$size") }
        Add-Run $Arguments $repeat ($position+1)
      }
    }
  }
}
# Cold is a causal control. clflush is explicitly outside the timed round window.
$ColdSizes = if ($SquareOutput) { @(0) } else { @(512,768) }
foreach ($size in $ColdSizes) {
  $Arguments = @(
    "--gate","t0r","--D","$D","--S","1","--R","16","--kernel","fused",
    "--slot-tile","4","--variant","cold","--timing","round","--cpus",$Cpus,
    "--rates",$Rates,"--warmup","1","--repetitions","3"
  )
  if ($SquareOutput) { $Arguments += "--square-output" }
  else { $Arguments += @("--average-weight-kib-per-core","$size") }
  Add-Run $Arguments 0 0
}
Write-Output $OutputPath
