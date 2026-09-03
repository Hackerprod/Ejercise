[CmdletBinding()]
param(
  [string]$Executable = "",
  [string]$OutputDirectory = "",
  [int]$TimedRepetitions = 5,
  [int]$Warmup = 2,
  [int]$PilotRepeats = 3
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($Executable)) {
  $Executable = Join-Path $projectRoot "build\t0m_int8_probe.exe"
}
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
  $OutputDirectory = Join-Path $projectRoot "sweep-output\t0m-phase3"
}
if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
  throw "Executable not found: $Executable"
}
if ($TimedRepetitions -le 0 -or $Warmup -lt 0 -or $PilotRepeats -le 0) {
  throw "TimedRepetitions and PilotRepeats must be positive; Warmup cannot be negative"
}
if (Test-Path -LiteralPath (Join-Path $OutputDirectory "t0m_phase3.summary.txt") -PathType Leaf) {
  throw "Refusing to repeat completed Phase 3 campaign: $OutputDirectory"
}
if (-not (Test-Path -LiteralPath $OutputDirectory -PathType Container)) {
  $null = New-Item -ItemType Directory -Path $OutputDirectory
}

$D = 512
$cpus = @(0, 2, 4, 6)
$workers = $cpus.Count
$sizes = @(384, 512, 640, 768)
$slots = @(1, 2, 4, 8, 16)
$depths = @(1, 2, 4, 8, 16)
$modes = @("fused", "repeat")
$variants = @("A", "B")
$controlSizes = @(512, 768)
$controlSlots = @(1, 8, 16)
$pilotSlots = 8
$pilotDepth = 16
$targetRows = @{}
foreach ($size in $sizes) {
  $targetRows[$size] = [int][math]::Floor(($size * 1024) / $D)
}

$machineCsv = Join-Path $OutputDirectory "t0m_phase3.machine.csv"
$stderrLog = Join-Path $OutputDirectory "t0m_phase3.stderr.log"
$summaryPath = Join-Path $OutputDirectory "t0m_phase3.summary.txt"
$commandLog = Join-Path $OutputDirectory "t0m_phase3.commands.log"
$script:failedInvocations = 0
$script:invocationNumber = 0
$script:records = @()
$script:probeHeader = $null

Set-Content -LiteralPath $stderrLog -Encoding ascii -Value @(
  "T0-M Phase 3 stderr and correction-gated command log"
  "executable=$Executable"
  "D=$D; bytes_per_int8=1; physical_cpus=$($cpus -join ','); workers=$workers"
  "target_kib_per_worker=$($sizes -join ','); S=$($slots -join ','); R=$($depths -join ','); modes=$($modes -join ','); variants=A,B"
  "timed_repetitions=$TimedRepetitions; warmup=$Warmup; pilot_repeats=$PilotRepeats"
  "Phase 4 constant-work sweep=NOT RUN"
)
Set-Content -LiteralPath $commandLog -Encoding ascii -Value "T0-M Phase 3 exact commands"

function Format-Number([double]$Value) {
  return $Value.ToString("R", [Globalization.CultureInfo]::InvariantCulture)
}

function Get-Median([double[]]$Values) {
  if ($Values.Count -eq 0) { throw "Cannot calculate median of empty values" }
  $ordered = @($Values | Sort-Object)
  $middle = [int][math]::Floor($ordered.Count / 2)
  if (($ordered.Count % 2) -eq 1) { return [double]$ordered[$middle] }
  return ([double]$ordered[$middle - 1] + [double]$ordered[$middle]) / 2.0
}

function Get-Mean([double[]]$Values) {
  if ($Values.Count -eq 0) { throw "Cannot calculate mean of empty values" }
  return [double](($Values | Measure-Object -Average).Average)
}

function Get-ExpectedBytes([int]$Rows, [int]$RecurrentDepth, [string]$Variant) {
  $depthBlocks = if ($Variant -eq "B") { $RecurrentDepth } else { 1 }
  return [int64]$Rows * $D * $depthBlocks
}

function Get-ExpectedRows([int]$Rows) {
  return "$Rows,$Rows,$Rows,$Rows"
}

function Get-ExpectedBytesList([int]$Rows, [int]$RecurrentDepth, [string]$Variant) {
  $bytes = Get-ExpectedBytes $Rows $RecurrentDepth $Variant
  return "$bytes,$bytes,$bytes,$bytes"
}

function Invoke-Probe([string]$Campaign, [int]$Repeat, [int]$DValue, [int]$SValue,
                      [int]$RValue, [int]$Rows, [string]$ModeValue, [string]$VariantValue,
                      [int]$Tile, [string]$Context) {
  $args = @(
    "--D", $DValue, "--S", $SValue, "--R", $RValue,
    "--mode", $ModeValue, "--variant", $VariantValue, "--S-tile", $Tile,
    "--workers", $workers, "--cpus", ($cpus -join ','),
    "--rows-per-worker", (Get-ExpectedRows $Rows),
    "--iterations", 1, "--timed-repetitions", $TimedRepetitions, "--warmup", $Warmup
  )
  $script:invocationNumber++
  $id = $script:invocationNumber
  $command = '"' + $Executable + '" ' + ($args -join ' ')
  Add-Content -LiteralPath $commandLog -Encoding ascii -Value "command[$id] $command"
  Add-Content -LiteralPath $stderrLog -Encoding ascii -Value "command[$id] $command"

  $startInfo = New-Object System.Diagnostics.ProcessStartInfo
  $startInfo.FileName = $Executable
  $startInfo.Arguments = ($args -join ' ')
  $startInfo.UseShellExecute = $false
  $startInfo.CreateNoWindow = $true
  $startInfo.RedirectStandardOutput = $true
  $startInfo.RedirectStandardError = $true
  $process = New-Object System.Diagnostics.Process
  $process.StartInfo = $startInfo
  try {
    if (-not $process.Start()) { throw "Could not start probe" }
    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()
    $process.WaitForExit()
    $stdout = $stdoutTask.Result
    $stderr = $stderrTask.Result
    Add-Content -LiteralPath $stderrLog -Encoding ascii -Value "stdout[$id] $($stdout.TrimEnd())"
    if ($stderr) { Add-Content -LiteralPath $stderrLog -Encoding ascii -Value "stderr[$id] $($stderr.TrimEnd())" }
    if ($process.ExitCode -ne 0) {
      $script:failedInvocations++
      throw "Probe failed with exit code $($process.ExitCode): $Context"
    }
  } catch {
    if ($process.ExitCode -ne 0 -and $script:failedInvocations -eq 0) { $script:failedInvocations++ }
    throw
  } finally {
    $process.Dispose()
  }

  if ($stderr -notmatch "T0-M correction passed") {
    $script:failedInvocations++
    throw "Correction gate evidence missing: $Context"
  }
  $lines = @($stdout -split '\r?\n' | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
  if ($lines.Count -ne 2) {
    $script:failedInvocations++
    throw "Probe did not produce one CSV row: $Context"
  }
  if ($null -eq $script:probeHeader) {
    $script:probeHeader = $lines[0]
  } elseif ($lines[0] -ne $script:probeHeader) {
    $script:failedInvocations++
    throw "Probe CSV header changed: $Context"
  }
  $row = $lines[1] | ConvertFrom-Csv -Header ($lines[0] -split ',')
  $expectedBytes = Get-ExpectedBytesList $Rows $RValue $VariantValue
  $expectedEviction = if ($VariantValue -eq "C") { "67108864" } else { "0" }
  $checks = @(
    @($row.D, "$D", "D"), @($row.S, "$SValue", "S"), @($row.R, "$RValue", "R"),
    @($row.O_i, (Get-ExpectedRows $Rows), "O_i"), @($row.B_i, $expectedBytes, "B_i"),
    @($row.worker_count, "$workers", "worker_count"), @($row.worker_list, ($cpus -join ','), "worker_list"),
    @($row.affinity_succeeded, "true", "affinity_succeeded"), @($row.affinity, "true,true,true,true", "affinity"),
    @($row.affinity_error, "0,0,0,0", "affinity_error"), @($row.timed_repetitions_exact, "true", "timed_repetitions_exact"),
    @($row.avx2_supported, "true", "avx2_supported"), @($row.kernel_used, "avx2", "kernel_used"),
    @($row.timed_repetitions, "$TimedRepetitions", "timed_repetitions"), @($row.eviction_bytes, $expectedEviction, "eviction_bytes"),
    @($row.mode, $ModeValue, "mode"), @($row.variant, $VariantValue, "variant"), @($row.S_tile, "$Tile", "S_tile")
  )
  foreach ($check in $checks) {
    if ([string]$check[0] -ne [string]$check[1]) {
      $script:failedInvocations++
      throw "Invalid row $($check[2])=$($check[0]), expected $($check[1]): $Context"
    }
  }
  if ($VariantValue -eq "C" -and [uint64]$row.eviction_checksum -eq 0) {
    $script:failedInvocations++
    throw "C eviction checksum is zero: $Context"
  }
  $record = [ordered]@{ campaign = $Campaign; campaign_repeat = $Repeat; invocation = $id }
  foreach ($property in $row.psobject.Properties) { $record[$property.Name] = $property.Value }
  $script:records += [pscustomobject]$record
  return $row
}

function Get-MainRow([int]$Size, [int]$SValue, [int]$RValue, [string]$ModeValue, [string]$VariantValue) {
  $matches = @($script:mainRows | Where-Object {
      $_.D -eq "$D" -and $_.S -eq "$SValue" -and $_.R -eq "$RValue" -and
      $_.mode -eq $ModeValue -and $_.variant -eq $VariantValue -and $_.target_kib -eq "$Size"
    })
  if ($matches.Count -ne 1) { throw "Expected one main row for size=$Size S=$SValue R=$RValue mode=$ModeValue variant=$VariantValue" }
  return $matches[0]
}

$pilotRows = @()
foreach ($pilotRepeat in 1..$PilotRepeats) {
  foreach ($tile in @(2, 4, 8)) {
    foreach ($mode in $modes) {
      foreach ($variant in $variants) {
        $pilotRows += Invoke-Probe "pilot" $pilotRepeat $D $pilotSlots $pilotDepth $targetRows[512] $mode $variant $tile `
          "pilot repeat=$pilotRepeat tile=$tile mode=$mode variant=$variant"
      }
    }
  }
}

$pilotScores = @()
foreach ($tile in @(2, 4, 8)) {
  $fusedMedians = @()
  foreach ($variant in $variants) {
    $values = @($pilotRows | Where-Object { $_.S_tile -eq "$tile" -and $_.mode -eq "fused" -and $_.variant -eq $variant } |
      ForEach-Object { [double]$_.mac_per_second })
    $fusedMedians += Get-Median $values
  }
  $pilotScores += [pscustomobject]@{
    S_tile = $tile
    A_fused_median = $fusedMedians[0]
    B_fused_median = $fusedMedians[1]
    score = Get-Mean ([double[]]$fusedMedians)
  }
}
$selectedTile = ($pilotScores | Sort-Object @{Expression = { $_.score }; Descending = $true }, @{Expression = { $_.S_tile }; Descending = $false} | Select-Object -First 1).S_tile

$script:mainRows = @()
$mainCount = 0
foreach ($size in $sizes) {
  foreach ($SValue in $slots) {
    foreach ($RValue in $depths) {
      foreach ($mode in $modes) {
        foreach ($variant in $variants) {
          $mainCount++
          $row = Invoke-Probe "main" 1 $D $SValue $RValue $targetRows[$size] $mode $variant $selectedTile `
            "main=$mainCount size=$size S=$SValue R=$RValue mode=$mode variant=$variant"
          $row | Add-Member -NotePropertyName target_kib -NotePropertyValue $size
          $script:mainRows += $row
        }
      }
    }
  }
}
if ($mainCount -ne 400) { throw "Expected 400 A/B main invocations, got $mainCount" }

$controlRows = @()
$controlCount = 0
foreach ($size in $controlSizes) {
  foreach ($SValue in $controlSlots) {
    foreach ($mode in $modes) {
      $controlCount++
      $row = Invoke-Probe "control-C" 1 $D $SValue 16 $targetRows[$size] $mode "C" $selectedTile `
        "control-C=$controlCount size=$size S=$SValue R=16 mode=$mode"
      $row | Add-Member -NotePropertyName target_kib -NotePropertyValue $size
      $controlRows += $row
    }
  }
}
if ($controlCount -ne 12) { throw "Expected 12 C control invocations, got $controlCount" }

foreach ($control in $controlRows) {
  $matchingA = Get-MainRow ([int]$control.target_kib) ([int]$control.S) 16 $control.mode "A"
  if ([uint64]$control.checksum -ne [uint64]$matchingA.checksum) {
    throw "C checksum differs from A: size=$($control.target_kib) S=$($control.S) mode=$($control.mode)"
  }
}

$invariantLines = @()
$invariantPass = $true
foreach ($size in @(512, 768)) {
  $ratios = @()
  foreach ($RValue in $depths) {
    $a = Get-MainRow $size 1 $RValue "fused" "A"
    $b = Get-MainRow $size 1 $RValue "fused" "B"
    $bOverA = [double]$b.mac_per_second / [double]$a.mac_per_second
    $aOverB = [double]$a.mac_per_second / [double]$b.mac_per_second
    $ratios += $aOverB
    $r1Gate = $RValue -ne 1 -or ($bOverA -ge 0.97 -and $bOverA -le 1.03)
    $invariantLines += "invariant target_kib=$size R=$RValue mode=fused B_over_A=$(Format-Number $bOverA) A_over_B=$(Format-Number $aOverB) R1_B_over_A_gate=$r1Gate"
    if (-not $r1Gate) { $invariantPass = $false }
  }
  for ($index = 1; $index -lt $ratios.Count; $index++) {
    if ($ratios[$index] -lt ($ratios[$index - 1] * 0.85)) { $invariantPass = $false }
  }
  $r16 = $ratios[$ratios.Count - 1]
  $r16Gate = $r16 -ge 2.5 -and $r16 -le 2.9
  $invariantLines += "invariant target_kib=$size gradual_rule=each A_over_B(R) >= prior*0.85 R16_A_over_B=$(Format-Number $r16) R16_gate=$r16Gate"
  if (-not $r16Gate) { $invariantPass = $false }
}

$machineCsvContent = @($script:records | ConvertTo-Csv -NoTypeInformation)
Set-Content -LiteralPath $machineCsv -Encoding ascii -Value $machineCsvContent
if (-not $invariantPass) {
  Set-Content -LiteralPath $summaryPath -Encoding ascii -Value @(
    "T0-M Phase 3 report"
    "status=STOP_INVARIANT_GATE"
    "scope=Phase 3 only; interpretation and G/F aggregation blocked"
    "machine_csv=$machineCsv"
    "stderr_log=$stderrLog"
    "command_log=$commandLog"
    "failed_invocations=$script:failedInvocations; total_invocations=$script:invocationNumber"
    "D=$D; bytes_per_int8=1; workers=$workers; physical_cpus=$($cpus -join ','); selected_S_tile=$selectedTile"
    "row_policy=400 A/B main rows + 12 C controls + $($PilotRepeats * 12) pilot rows; machine_rows=$($script:records.Count)"
    ($pilotScores | ForEach-Object { "pilot S_tile=$($_.S_tile) A_fused_median=$(Format-Number $_.A_fused_median) B_fused_median=$(Format-Number $_.B_fused_median) score=$(Format-Number $_.score)" })
    $invariantLines
    "Phase 4 constant-work sweep=NOT RUN"
  )
  throw "T0-R invariant gate failed; Phase 3 interpretation blocked"
}

$metricLines = @()
$maxG8 = 0.0
$maxG16 = 0.0
foreach ($size in $sizes) {
  foreach ($RValue in $depths) {
    foreach ($variant in $variants) {
      $base = [double](Get-MainRow $size 1 $RValue "fused" $variant).mac_per_second
      $gValues = @{}
      foreach ($SValue in @(2, 4, 8, 16)) {
        $gValues[$SValue] = [double](Get-MainRow $size $SValue $RValue "fused" $variant).mac_per_second / $base
      }
      $maxG8 = [math]::Max($maxG8, $gValues[8])
      $maxG16 = [math]::Max($maxG16, $gValues[16])
      $metricLines += "G target_kib=$size R=$RValue variant=$variant G2=$(Format-Number $gValues[2]) G4=$(Format-Number $gValues[4]) G8=$(Format-Number $gValues[8]) G16=$(Format-Number $gValues[16]) max_G8_G16=$(Format-Number ([math]::Max($gValues[8], $gValues[16])))"

      $fValues = @{}
      foreach ($SValue in @(4, 8, 16)) {
        $fused = [double](Get-MainRow $size $SValue $RValue "fused" $variant).mac_per_second
        $repeat = [double](Get-MainRow $size $SValue $RValue "repeat" $variant).mac_per_second
        $fValues[$SValue] = $fused / $repeat
      }
      $rises = $fValues[8] -gt $fValues[4] -and $fValues[16] -gt $fValues[8]
      $metricLines += "F target_kib=$size R=$RValue variant=$variant F4=$(Format-Number $fValues[4]) F8=$(Format-Number $fValues[8]) F16=$(Format-Number $fValues[16]) F_rises_4_8_16=$rises"
    }
  }
}

$pilotText = $pilotScores | ForEach-Object {
  "pilot S_tile=$($_.S_tile) A_fused_median=$(Format-Number $_.A_fused_median) B_fused_median=$(Format-Number $_.B_fused_median) score=$(Format-Number $_.score)"
}
$controlText = $controlRows | ForEach-Object {
  "control-C target_kib=$($_.target_kib) S=$($_.S) R=$($_.R) mode=$($_.mode) checksum=$($_.checksum) eviction_bytes=$($_.eviction_bytes) eviction_checksum=$($_.eviction_checksum) mac_per_second=$($_.mac_per_second)"
}
$status = if ($invariantPass -and $maxG8 -ge 1.5) { "PASS" } else { "STOP_G_GATE" }
$summary = @(
  "T0-M Phase 3 report"
  "status=$status"
  "scope=Phase 3 only; Phase 4 constant-work sweep=NOT RUN; MRDL/Q4/later recurrent Requantize/Norm/Residual=untouched"
  "executable=$Executable"
  "machine_csv=$machineCsv"
  "stderr_log=$stderrLog"
  "command_log=$commandLog"
  "D=$D; bytes_per_int8=1; workers=$workers; physical_cpus=$($cpus -join ','); affinity_policy=explicit four-worker CPUs 0,2,4,6"
  "size_policy=equal per-worker target bytes; O_i=floor(target_bytes/(D bytes/int8)); same four-shard O_i rows"
  "target_mapping=384KiB:O_i=768,B_i(A/C)=393216; 512KiB:O_i=1024,B_i(A/C)=524288; 640KiB:O_i=1280,B_i(A/C)=655360; 768KiB:O_i=1536,B_i(A/C)=786432"
  "B_i variant B=O_i*D*R; variant A shares one weight block across R; variant B uses distinct blocks by R; C=A/shared plus 64MiB eviction and clflush outside timed units"
  "timed_repetitions=$TimedRepetitions; warmup=$Warmup; prep_timing=excluded; failed_invocations=$script:failedInvocations; total_invocations=$script:invocationNumber"
  "row_policy=400 A/B main rows + 12 C controls + $($PilotRepeats * 12) pilot rows; main exact matrix=4 sizes*5 S*5 R*2 modes*2 variants"
  "selected_S_tile=$selectedTile; pilot_rule=per candidate, median over $PilotRepeats fused invocations for A and B, then mean of A/B medians; repeat mode also measured as required but excluded from tile score"
  $pilotText
  "invariant_gate=$invariantPass; invariant_rule=R1 B_over_A in [0.97,1.03], each later A_over_B >= prior*0.85, R16 A_over_B in [2.5,2.9]; interpretation blocked if false"
  $invariantLines
  "G_gate=max(G8,G16)>=1.5; max_G8=$(Format-Number $maxG8); max_G16=$(Format-Number $maxG16); max_G8_or_G16=$(Format-Number ([math]::Max($maxG8, $maxG16)))"
  "metrics=G(S)=MAC/s_Fused(S)/MAC/s_Fused(1); F(S)=MAC/s_Fused(S)/MAC/s_Repeat(S); G/F include A/B only, never C"
  $metricLines
  "C_controls_raw_values_only=true"
  $controlText
  "checksum_policy=all Y[S x O_i] cells included; C checksums equal corresponding A rows; correction gate passed before every speed row"
  "AVX2_policy=every row avx2_supported=true and kernel_used=avx2; four-worker affinity and exact repetition fields validated"
  "next=Do not start Phase 4 until Phase 3 interpretation is approved"
)
Set-Content -LiteralPath $summaryPath -Encoding ascii -Value $summary
Write-Output $summaryPath
