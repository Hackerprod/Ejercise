[CmdletBinding()]
param(
  [string]$Executable = "",
  [string]$Output = "results/t0rm.csv",
  [string]$Cpus = "0,2,4,6",
  [string]$Rates = "19.3,18.1,10.9,17.0",
  [int]$D = 1472,
  [int[]]$Slots = @(1,8,16),
  [int[]]$Depths = @(4,8),
  [string[]]$Transitions = @("fixed","group-rms","global-rms"),
  [int]$ExternalRepeats = 6,
  [int]$InternalRepetitions = 4,
  [int]$Warmup = 2,
  [int]$ProjectionShift = 12
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
$script:OrderCounter = 0

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

function Add-Run([string[]]$Arguments,[int]$Repeat) {
  $script:OrderCounter++
  $lines = @(& $Executable @Arguments 2>> $Log | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
  if ($LASTEXITCODE -ne 0 -or $lines.Count -ne 2) { throw "Gate invocation failed: $($Arguments -join ' ')" }
  if (-not $script:HeaderWritten) {
    Set-Content -Encoding ascii -Path $OutputPath -Value ($lines[0] + ",batch_repeat,variant_order")
    $script:HeaderWritten = $true
  }
  Add-Content -Encoding ascii -Path $OutputPath -Value ($lines[1] + ",$Repeat,$script:OrderCounter")
}

function Invoke-BridgeCell([int]$Repeat,[int]$Slot,[int]$Depth) {
  $variants = if (($Repeat % 2) -eq 1) { @("shared","clone") } else { @("clone","shared") }
  foreach ($variant in $variants) {
    $kernels = if (($Repeat % 2) -eq 1) { @("repeat","fused") } else { @("fused","repeat") }
    foreach ($kernel in $kernels) {
      Add-Run @(
        "--gate","t0m","--D","$D","--S","$Slot","--R","$Depth","--square-output",
        "--kernel",$kernel,"--slot-tile","4","--variant",$variant,"--cpus",$Cpus,
        "--rates",$Rates,"--warmup","$Warmup","--repetitions","$InternalRepetitions"
      ) $Repeat
    }
  }
}

function Invoke-RecurrentCell([int]$Repeat,[int]$Slot,[int]$Depth) {
  $transitionOrder = @(Get-CycledOrder $Transitions $Repeat)
  foreach ($transition in $transitionOrder) {
    $variants = if (($Repeat % 2) -eq 1) { @("shared","clone") } else { @("clone","shared") }
    foreach ($variant in $variants) {
      Add-Run @(
        "--gate","t0rm","--D","$D","--S","$Slot","--R","$Depth",
        "--kernel","fused","--slot-tile","4","--variant",$variant,
        "--transition",$transition,"--cpus",$Cpus,"--rates",$Rates,
        "--projection-shift","$ProjectionShift","--target-rms","32",
        "--warmup","$Warmup","--repetitions","$InternalRepetitions"
      ) $Repeat
    }
  }
}

for ($repeat=1; $repeat -le $ExternalRepeats; $repeat++) {
  $script:OrderCounter = 0
  $slotOrder = @(Get-CycledOrder $Slots $repeat)
  $depthOrder = @(Get-CycledOrder $Depths ($repeat + 1))
  foreach ($slot in $slotOrder) {
    foreach ($depth in $depthOrder) {
      # Interleave the frozen bridge and recurrent cell to avoid a whole-phase thermal bias.
      if (($repeat % 2) -eq 1) {
        Invoke-BridgeCell $repeat $slot $depth
        Invoke-RecurrentCell $repeat $slot $depth
      } else {
        Invoke-RecurrentCell $repeat $slot $depth
        Invoke-BridgeCell $repeat $slot $depth
      }
    }
  }
}
Write-Output $OutputPath
