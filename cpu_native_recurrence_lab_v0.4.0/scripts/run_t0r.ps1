[CmdletBinding()]
param(
  [string]$Executable = "",
  [string]$Output = "results/t0r.csv",
  [string]$Cpus = "0,2,4,6",
  [string]$Rates = "19.3,18.1,10.9,17.0",
  [int[]]$SizesKiB = @(384,512,640,768),
  [int[]]$Depths = @(1,4,8,16),
  [int]$ExternalRepeats = 6,
  [int]$InternalRepetitions = 5,
  [int]$Warmup = 2
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($Executable)) { $Executable = Join-Path $Root "build-windows/Release/cnrl_gate.exe" }
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
for ($repeat=1; $repeat -le $ExternalRepeats; $repeat++) {
  foreach ($size in $SizesKiB) {
    foreach ($depth in $Depths) {
      $variants = if (($repeat % 2) -eq 1) { @("shared","clone") } else { @("clone","shared") }
      for ($position=0; $position -lt $variants.Count; $position++) {
        Add-Run @(
          "--gate","t0r","--D","512","--S","1","--R","$depth",
          "--kernel","fused","--slot-tile","4","--variant",$variants[$position],
          "--cpus",$Cpus,"--rates",$Rates,"--average-weight-kib-per-core","$size",
          "--warmup","$Warmup","--repetitions","$InternalRepetitions"
        ) $repeat ($position+1)
      }
    }
  }
}
# Cold is a causal control. clflush is explicitly outside the timed round window.
foreach ($size in @(512,768)) {
  Add-Run @(
    "--gate","t0r","--D","512","--S","1","--R","16","--kernel","fused",
    "--slot-tile","4","--variant","cold","--timing","round","--cpus",$Cpus,
    "--rates",$Rates,"--average-weight-kib-per-core","$size","--warmup","1","--repetitions","3"
  ) 0 0
}
Write-Output $OutputPath
