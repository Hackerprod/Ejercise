[CmdletBinding()]
param(
  [string]$Executable = "",
  [string]$Output = "results/t0m.csv",
  [string]$Cpus = "0,2,4,6",
  [string]$Rates = "19.3,18.1,10.9,17.0",
  [int]$D = 512,
  [switch]$SquareOutput,
  [int[]]$SizesKiB = @(512,768),
  [int[]]$Depths = @(1,8,16),
  [int[]]$Slots = @(1,2,4,8,16),
  [int]$ExternalRepeats = 6,
  [int]$InternalRepetitions = 4,
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

function Get-CycledOrder([object[]]$Items,[int]$Repeat) {
  $copy = @($Items)
  if ($copy.Count -le 1) { return $copy }
  switch (($Repeat - 1) % 4) {
    0 { return $copy }
    1 { [array]::Reverse($copy); return $copy }
    2 { return @($copy[1..($copy.Count-1)]) + @($copy[0]) }
    3 { return @($copy[$copy.Count-1]) + @($copy[0..($copy.Count-2)]) }
  }
}

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
  $orderCounter = 0
  $EffectiveSizes = if ($SquareOutput) { @(0) } else { @($SizesKiB) }
  $sizeOrder = @(Get-CycledOrder $EffectiveSizes $repeat)
  $depthOrder = @(Get-CycledOrder $Depths (($repeat + 1)))
  $slotOrder = @(Get-CycledOrder $Slots (($repeat + 2)))
  foreach ($size in $sizeOrder) {
    foreach ($depth in $depthOrder) {
      foreach ($slot in $slotOrder) {
        $basePairs = @(
          [pscustomobject]@{ Variant = "shared"; Kernel = "repeat" },
          [pscustomobject]@{ Variant = "shared"; Kernel = "fused" },
          [pscustomobject]@{ Variant = "clone"; Kernel = "repeat" },
          [pscustomobject]@{ Variant = "clone"; Kernel = "fused" }
        )
        switch (($repeat - 1) % 4) {
          0 { $pairs = @($basePairs[0],$basePairs[1],$basePairs[2],$basePairs[3]) }
          1 { $pairs = @($basePairs[3],$basePairs[2],$basePairs[1],$basePairs[0]) }
          2 { $pairs = @($basePairs[1],$basePairs[3],$basePairs[0],$basePairs[2]) }
          3 { $pairs = @($basePairs[2],$basePairs[0],$basePairs[3],$basePairs[1]) }
        }
        foreach ($pair in $pairs) {
          $orderCounter++
          $Arguments = @(
            "--gate","t0m","--D","$D","--S","$slot","--R","$depth",
            "--kernel",$pair.Kernel,"--slot-tile","4","--variant",$pair.Variant,
            "--cpus",$Cpus,"--rates",$Rates,
            "--warmup","$Warmup","--repetitions","$InternalRepetitions"
          )
          if ($SquareOutput) { $Arguments += "--square-output" }
          else { $Arguments += @("--average-weight-kib-per-core","$size") }
          Add-Run $Arguments $repeat $orderCounter
        }
      }
    }
  }
}
Write-Output $OutputPath
