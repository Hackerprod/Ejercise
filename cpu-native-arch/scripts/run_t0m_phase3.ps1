[CmdletBinding()]
param(
  [string]$Executable = "",
  [string]$OutputDirectory = "",
  [int]$TimedRepetitions = 8,
  [int]$Warmup = 2,
  [int]$IndependentRuns = 10
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
if ($TimedRepetitions -le 0 -or $Warmup -lt 0 -or $IndependentRuns -ne 10) {
  throw "TimedRepetitions must be positive, Warmup cannot be negative, and IndependentRuns must equal 10"
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
$pilotTiles = @(2, 4, 8)
$pilotSlots = 8
$pilotDepth = 16
$targetRows = @{
  384 = [int[]]@(908, 852, 512, 800)
  512 = [int[]]@(1211, 1136, 683, 1066)
  640 = [int[]]@(1514, 1420, 854, 1332)
  768 = [int[]]@(1816, 1705, 1023, 1600)
}
foreach ($size in $sizes) {
  $rows = [int[]]$targetRows[$size]
  $expectedTotalRows = [int](($size * 1024 * $workers) / $D)
  $actualTotalRows = [int](($rows | Measure-Object -Sum).Sum)
  if ($rows.Count -ne $workers -or @($rows | Where-Object { $_ -le 0 }).Count -gt 0 -or $actualTotalRows -ne $expectedTotalRows) {
    throw "Invalid proportional rows for size=$size`: expected $expectedTotalRows positive rows across $workers workers, got $($rows -join ',')"
  }
}

$machineCsv = Join-Path $OutputDirectory "t0m_phase3.machine.csv"
$aggregateCsv = Join-Path $OutputDirectory "t0m_phase3.aggregate.csv"
$metricsCsv = Join-Path $OutputDirectory "t0m_phase3.metrics.csv"
$stderrLog = Join-Path $OutputDirectory "t0m_phase3.stderr.log"
$commandLog = Join-Path $OutputDirectory "t0m_phase3.commands.log"
$preflightStdout = Join-Path $OutputDirectory "t0m_phase3.preflight.stdout.log"
$preflightStderr = Join-Path $OutputDirectory "t0m_phase3.preflight.stderr.log"
$summaryPath = Join-Path $OutputDirectory "t0m_phase3.summary.txt"
$script:invocationNumber = 0
$script:failedInvocations = 0
$script:records = New-Object -TypeName 'System.Collections.Generic.List[object]'

$csvHeader = "D,S,R,O_i,B_i,rows_per_worker,bytes_per_worker,mode,variant,S_tile,iterations,timed_repetitions,warmup,worker_count,worker_list,affinity,affinity_error,affinity_succeeded,timed_repetitions_exact,avx2_supported,kernel_used,eviction_bytes,eviction_checksum,elapsed_seconds,mac_total,mac_per_second,checksum"
$logHeader = @(
  "T0-M Phase 3 stderr and bounded process-output log"
  "executable=$Executable"
  "D=$D; bytes_per_int8=1; physical_cpus=$($cpus -join ','); workers=$workers"
  "target_kib=$($sizes -join ','); S=$($slots -join ','); R=$($depths -join ','); modes=$($modes -join ','); variants=A,B"
  "timed_repetitions=$TimedRepetitions; warmup=$Warmup; independent_runs=$IndependentRuns"
  "normal_speed_correction=forbidden; explicit self-test is logged separately"
  "Phase 4 constant-work sweep=NOT RUN"
)
Set-Content -LiteralPath $stderrLog -Encoding ascii -Value $logHeader
Set-Content -LiteralPath $commandLog -Encoding ascii -Value @(
  "T0-M Phase 3 exact commands"
  "preflight=single explicit --self-test; speed rows never receive --self-test"
)

function Format-Number([double]$Value) {
  return $Value.ToString("R", [Globalization.CultureInfo]::InvariantCulture)
}

function Limit-Text([string]$Text, [int]$Maximum = 2000) {
  if ($null -eq $Text) { return "" }
  $normalized = $Text.TrimEnd()
  if ($normalized.Length -le $Maximum) { return $normalized }
  return $normalized.Substring(0, $Maximum) + "...[truncated]"
}

function Invoke-CapturedProcess([string]$FileName, [string[]]$Arguments) {
  $startInfo = New-Object System.Diagnostics.ProcessStartInfo
  $startInfo.FileName = $FileName
  $startInfo.Arguments = ($Arguments -join " ")
  $startInfo.UseShellExecute = $false
  $startInfo.CreateNoWindow = $true
  $startInfo.RedirectStandardOutput = $true
  $startInfo.RedirectStandardError = $true
  $process = New-Object System.Diagnostics.Process
  $process.StartInfo = $startInfo
  $started = $false
  try {
    if (-not $process.Start()) { throw "Could not start process: $FileName" }
    $started = $true
    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()
    $process.WaitForExit()
    return [pscustomobject]@{
      ExitCode = $process.ExitCode
      Stdout = $stdoutTask.Result
      Stderr = $stderrTask.Result
    }
  } finally {
    $process.Dispose()
  }
}

function Get-ExpectedRows([int[]]$Rows) {
  return ($Rows -join ',')
}

function Get-ExpectedBytes([int[]]$Rows, [int]$RecurrentDepth, [string]$Variant) {
  $depthBlocks = if ($Variant -eq "B") { $RecurrentDepth } else { 1 }
  return (($Rows | ForEach-Object { [int64]$_ * $D * $depthBlocks }) -join ',')
}

function Get-ExpectedMacTotal([int[]]$Rows, [int]$SValue, [int]$RValue) {
  $totalRows = [int64](($Rows | Measure-Object -Sum).Sum)
  return $totalRows * $D * $SValue * $RValue * $TimedRepetitions
}

function Invoke-Preflight {
  $arguments = @("--self-test")
  Add-Content -LiteralPath $commandLog -Encoding ascii -Value "preflight_command `"$Executable`" --self-test"
  $result = Invoke-CapturedProcess $Executable $arguments
  Set-Content -LiteralPath $preflightStdout -Encoding ascii -Value (Limit-Text $result.Stdout)
  Set-Content -LiteralPath $preflightStderr -Encoding ascii -Value (Limit-Text $result.Stderr)
  Add-Content -LiteralPath $stderrLog -Encoding ascii -Value "preflight_stdout $(Limit-Text $result.Stdout)"
  Add-Content -LiteralPath $stderrLog -Encoding ascii -Value "preflight_stderr $(Limit-Text $result.Stderr)"
  if ($result.ExitCode -ne 0) {
    throw "T0-M self-test preflight failed with exit code $($result.ExitCode)"
  }
  if ((($result.Stdout + "`n" + $result.Stderr) -notmatch "(?i)T0-M correction passed")) {
    throw "T0-M self-test preflight did not report correction pass"
  }
}

function Invoke-Probe([string]$Campaign, [int]$Repeat, [int]$OrderIndex, [int]$TargetKiB,
                       [int]$SValue, [int]$RValue, [int[]]$Rows, [string]$ModeValue,
                       [string]$VariantValue, [int]$Tile, [string]$Context) {
  $arguments = @(
    "--D", $D, "--S", $SValue, "--R", $RValue,
    "--mode", $ModeValue, "--variant", $VariantValue, "--S-tile", $Tile,
    "--workers", $workers, "--cpus", ($cpus -join ','),
    "--rows-per-worker", (Get-ExpectedRows $Rows),
    "--iterations", 1, "--timed-repetitions", $TimedRepetitions, "--warmup", $Warmup
  )
  $script:invocationNumber++
  $id = $script:invocationNumber
  $command = '"' + $Executable + '" ' + ($arguments -join ' ')
  Add-Content -LiteralPath $commandLog -Encoding ascii -Value "command[$id] campaign=$Campaign repeat=$Repeat order_index=$OrderIndex target_kib=$TargetKiB $command"
  Add-Content -LiteralPath $stderrLog -Encoding ascii -Value "command[$id] campaign=$Campaign repeat=$Repeat order_index=$OrderIndex target_kib=$TargetKiB $command"

  $result = Invoke-CapturedProcess $Executable $arguments
  Add-Content -LiteralPath $stderrLog -Encoding ascii -Value "stdout[$id] $(Limit-Text $result.Stdout)"
  Add-Content -LiteralPath $stderrLog -Encoding ascii -Value "stderr[$id] $(Limit-Text $result.Stderr)"
  if ($result.ExitCode -ne 0) {
    $script:failedInvocations++
    throw "Probe failed with exit code $($result.ExitCode): $Context"
  }
  if ($result.Stderr -match "(?i)correction") {
    $script:failedInvocations++
    throw "Normal speed stderr contains forbidden correction message: $Context"
  }

  $lines = @($result.Stdout -split '\r?\n' | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
  if ($lines.Count -ne 2 -or $lines[0] -ne $csvHeader) {
    $script:failedInvocations++
    throw "Probe CSV schema/output invalid: $Context"
  }
  $row = $lines[1] | ConvertFrom-Csv -Header ($csvHeader -split ',')
  $expectedRows = Get-ExpectedRows $Rows
  $expectedBytes = Get-ExpectedBytes $Rows $RValue $VariantValue
  $expectedEviction = if ($VariantValue -eq "C") { "67108864" } else { "0" }
  $expectedMacTotal = Get-ExpectedMacTotal $Rows $SValue $RValue
  $checks = @(
    @($row.D, "$D", "D"), @($row.S, "$SValue", "S"), @($row.R, "$RValue", "R"),
    @($row.O_i, $expectedRows, "O_i"), @($row.B_i, $expectedBytes, "B_i"),
    @($row.rows_per_worker, $expectedRows, "rows_per_worker"), @($row.bytes_per_worker, $expectedBytes, "bytes_per_worker"),
    @($row.worker_count, "$workers", "worker_count"), @($row.worker_list, ($cpus -join ','), "worker_list"),
    @($row.affinity_succeeded, "true", "affinity_succeeded"), @($row.affinity, "true,true,true,true", "affinity"),
    @($row.affinity_error, "0,0,0,0", "affinity_error"), @($row.timed_repetitions_exact, "true", "timed_repetitions_exact"),
    @($row.avx2_supported, "true", "avx2_supported"), @($row.kernel_used, "avx2", "kernel_used"),
    @($row.iterations, "1", "iterations"), @($row.timed_repetitions, "$TimedRepetitions", "timed_repetitions"),
    @($row.warmup, "$Warmup", "warmup"), @($row.eviction_bytes, $expectedEviction, "eviction_bytes"),
    @($row.mode, $ModeValue, "mode"), @($row.variant, $VariantValue, "variant"), @($row.S_tile, "$Tile", "S_tile"),
    @($row.mac_total, "$expectedMacTotal", "mac_total")
  )
  foreach ($check in $checks) {
    if ([string]$check[0] -ne [string]$check[1]) {
      $script:failedInvocations++
      throw "Invalid row $($check[2])=$($check[0]), expected $($check[1]): $Context"
    }
  }
  if ([uint64]$row.checksum -eq 0 -or [double]$row.mac_total -le 0 -or [double]$row.mac_per_second -le 0) {
    $script:failedInvocations++
    throw "Zero checksum or MAC result: $Context"
  }
  if ($VariantValue -eq "C" -and [uint64]$row.eviction_checksum -eq 0) {
    $script:failedInvocations++
    throw "C eviction checksum is zero: $Context"
  }

  $record = [ordered]@{
    campaign = $Campaign
    campaign_repeat = $Repeat
    order_index = $OrderIndex
    target_kib = $TargetKiB
    mode = $ModeValue
    variant = $VariantValue
    S = $SValue
    R = $RValue
    tile = $Tile
    invocation = $id
  }
  foreach ($property in $row.psobject.Properties) { $record[$property.Name] = $property.Value }
  [void]$script:records.Add([object]([pscustomobject]$record))
  return [pscustomobject]$record
}

function Get-ExecutionOrder([int]$Repeat, [bool]$Control) {
  if ($Control) {
    if (($Repeat % 2) -eq 0) {
      return @(
        [pscustomobject]@{ mode = "fused"; variant = "C" }
        [pscustomobject]@{ mode = "repeat"; variant = "C" }
      )
    }
    return @(
      [pscustomobject]@{ mode = "repeat"; variant = "C" }
      [pscustomobject]@{ mode = "fused"; variant = "C" }
    )
  }
  if (($Repeat % 2) -eq 0) {
    return @(
      [pscustomobject]@{ mode = "fused"; variant = "A" }
      [pscustomobject]@{ mode = "fused"; variant = "B" }
      [pscustomobject]@{ mode = "repeat"; variant = "A" }
      [pscustomobject]@{ mode = "repeat"; variant = "B" }
    )
  }
  return @(
    [pscustomobject]@{ mode = "repeat"; variant = "B" }
    [pscustomobject]@{ mode = "repeat"; variant = "A" }
    [pscustomobject]@{ mode = "fused"; variant = "B" }
    [pscustomobject]@{ mode = "fused"; variant = "A" }
  )
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

function New-Aggregates([object[]]$InputRows, [bool]$IncludeTile) {
  $groups = @{}
  foreach ($inputRow in $InputRows) {
    $keyParts = @($inputRow.target_kib, $inputRow.S, $inputRow.R, $inputRow.mode, $inputRow.variant)
    if ($IncludeTile) { $keyParts += $inputRow.tile }
    $key = ($keyParts -join '|')
    if (-not $groups.ContainsKey($key)) { $groups[$key] = New-Object -TypeName 'System.Collections.Generic.List[object]' }
    [void]$groups[$key].Add([object]$inputRow)
  }
  $aggregates = New-Object -TypeName 'System.Collections.Generic.List[object]'
  foreach ($key in $groups.Keys) {
    $group = @($groups[$key] | ForEach-Object { $_ })
    if ($group.Count -ne $IndependentRuns) {
      throw "Expected $IndependentRuns rows for aggregate key=$key, got $($group.Count)"
    }
    $values = [double[]]@($group | ForEach-Object { [double]$_.mac_per_second })
    $sumSquares = 0.0
    $mean = Get-Mean $values
    foreach ($value in $values) { $sumSquares += ($value - $mean) * ($value - $mean) }
    $checksums = @($group | ForEach-Object { [string]$_.checksum })
    $uniqueChecksums = @($checksums | Sort-Object -Unique)
    $aggregate = [ordered]@{
      campaign = [string]$group[0].campaign
      target_kib = [int]$group[0].target_kib
      S = [int]$group[0].S
      R = [int]$group[0].R
      mode = [string]$group[0].mode
      variant = [string]$group[0].variant
      tile = if ($IncludeTile) { [int]$group[0].tile } else { "" }
      n = $group.Count
      mean = $mean
      median = Get-Median $values
      min = [double](($values | Measure-Object -Minimum).Minimum)
      max = [double](($values | Measure-Object -Maximum).Maximum)
      population_sd = [math]::Sqrt($sumSquares / $values.Count)
      checksum = if ($uniqueChecksums.Count -eq 1) { $uniqueChecksums[0] } else { "NONDETERMINISTIC" }
      checksum_deterministic = ($uniqueChecksums.Count -eq 1)
      checksum_values = ($uniqueChecksums -join '|')
    }
    [void]$aggregates.Add([object]([pscustomobject]$aggregate))
  }
  return $aggregates.ToArray()
}

function Get-MainRow([int]$Size, [int]$SValue, [int]$RValue, [string]$ModeValue, [string]$VariantValue) {
  $matches = @($script:mainAggregates | Where-Object {
      [int]$_.target_kib -eq $Size -and [int]$_.S -eq $SValue -and [int]$_.R -eq $RValue -and
      $_.mode -eq $ModeValue -and $_.variant -eq $VariantValue
    })
  if ($matches.Count -ne 1 -or [int]$matches[0].n -ne $IndependentRuns) {
    throw "Expected one median main row with n=$IndependentRuns for size=$Size S=$SValue R=$RValue mode=$ModeValue variant=$VariantValue"
  }
  return $matches[0]
}

function Get-ControlRow([int]$Size, [int]$SValue, [string]$ModeValue) {
  $matches = @($script:controlAggregates | Where-Object {
      [int]$_.target_kib -eq $Size -and [int]$_.S -eq $SValue -and [int]$_.R -eq 16 -and
      $_.mode -eq $ModeValue -and $_.variant -eq "C"
    })
  if ($matches.Count -ne 1 -or [int]$matches[0].n -ne $IndependentRuns) {
    throw "Expected one median control row with n=$IndependentRuns for size=$Size S=$SValue mode=$ModeValue"
  }
  return $matches[0]
}

Invoke-Preflight

$pilotRaw = New-Object -TypeName 'System.Collections.Generic.List[object]'
foreach ($pilotRepeat in 1..$IndependentRuns) {
  foreach ($tile in $pilotTiles) {
    $orderIndex = 0
    foreach ($entry in @(Get-ExecutionOrder $pilotRepeat $false)) {
      $orderIndex++
      $pilotRecord = Invoke-Probe "pilot" $pilotRepeat $orderIndex 512 $pilotSlots $pilotDepth $targetRows[512] $entry.mode $entry.variant $tile `
        "pilot repeat=$pilotRepeat tile=$tile mode=$($entry.mode) variant=$($entry.variant)"
      [void]$pilotRaw.Add([object]$pilotRecord)
    }
  }
}
$pilotCount = $pilotRaw.Count
if ($pilotCount -ne ($pilotTiles.Count * 4 * $IndependentRuns)) {
  throw "Expected $($pilotTiles.Count * 4 * $IndependentRuns) pilot invocations, got $pilotCount"
}
$pilotAggregates = New-Aggregates $pilotRaw.ToArray() $true
$pilotScores = New-Object -TypeName 'System.Collections.Generic.List[object]'
foreach ($tile in $pilotTiles) {
  $fusedMedians = New-Object -TypeName 'System.Collections.Generic.List[double]'
  foreach ($variant in $variants) {
    $pilotAggregate = @($pilotAggregates | Where-Object {
        [int]$_.target_kib -eq 512 -and [int]$_.S -eq $pilotSlots -and [int]$_.R -eq $pilotDepth -and
        [int]$_.tile -eq $tile -and $_.mode -eq "fused" -and $_.variant -eq $variant
      })
    if ($pilotAggregate.Count -ne 1) { throw "Missing pilot aggregate for tile=$tile variant=$variant" }
    $fusedMedians.Add([double]$pilotAggregate[0].median)
  }
  [void]$pilotScores.Add([object]([pscustomobject]@{
    S_tile = $tile
    A_fused_median = $fusedMedians[0]
    B_fused_median = $fusedMedians[1]
    score = Get-Mean ([double[]]$fusedMedians)
  }))
}
$selectedTile = [int](($pilotScores | Sort-Object @{Expression = { $_.score }; Descending = $true }, @{Expression = { $_.S_tile }; Descending = $false} | Select-Object -First 1).S_tile)

$mainRaw = New-Object -TypeName 'System.Collections.Generic.List[object]'
foreach ($size in $sizes) {
  foreach ($SValue in $slots) {
    foreach ($RValue in $depths) {
      for ($repeat = 1; $repeat -le $IndependentRuns; $repeat++) {
        $orderIndex = 0
        foreach ($entry in @(Get-ExecutionOrder $repeat $false)) {
          $orderIndex++
          $mainRecord = Invoke-Probe "main" $repeat $orderIndex $size $SValue $RValue $targetRows[$size] $entry.mode $entry.variant $selectedTile `
            "main size=$size S=$SValue R=$RValue repeat=$repeat mode=$($entry.mode) variant=$($entry.variant)"
          [void]$mainRaw.Add([object]$mainRecord)
        }
      }
    }
  }
}
$mainCount = $mainRaw.Count
if ($mainCount -ne (4 * 5 * 5 * 4 * $IndependentRuns)) {
  throw "Expected $((4 * 5 * 5 * 4 * $IndependentRuns)) main invocations, got $mainCount"
}

$controlRaw = New-Object -TypeName 'System.Collections.Generic.List[object]'
foreach ($size in $controlSizes) {
  foreach ($SValue in $controlSlots) {
    for ($repeat = 1; $repeat -le $IndependentRuns; $repeat++) {
      $orderIndex = 0
      foreach ($entry in @(Get-ExecutionOrder $repeat $true)) {
        $orderIndex++
        $controlRecord = Invoke-Probe "control-C" $repeat $orderIndex $size $SValue 16 $targetRows[$size] $entry.mode "C" $selectedTile `
          "control-C size=$size S=$SValue repeat=$repeat mode=$($entry.mode)"
        [void]$controlRaw.Add([object]$controlRecord)
      }
    }
  }
}
$controlCount = $controlRaw.Count
if ($controlCount -ne (2 * 3 * 2 * $IndependentRuns)) {
  throw "Expected $((2 * 3 * 2 * $IndependentRuns)) C control invocations, got $controlCount"
}

$script:mainAggregates = New-Aggregates $mainRaw.ToArray() $false
$script:controlAggregates = New-Aggregates $controlRaw.ToArray() $false
$checksumLines = New-Object -TypeName 'System.Collections.Generic.List[string]'
$checksumPass = $true
foreach ($controlAggregate in $script:controlAggregates) {
  $matchingA = Get-MainRow ([int]$controlAggregate.target_kib) ([int]$controlAggregate.S) 16 $controlAggregate.mode "A"
  $matchingC = Get-ControlRow ([int]$controlAggregate.target_kib) ([int]$controlAggregate.S) $controlAggregate.mode
  $aRawChecksums = @($mainRaw | Where-Object {
      [int]$_.target_kib -eq [int]$controlAggregate.target_kib -and [int]$_.S -eq [int]$controlAggregate.S -and
      [int]$_.R -eq 16 -and $_.mode -eq $controlAggregate.mode -and $_.variant -eq "A"
    } | ForEach-Object { [string]$_.checksum } | Sort-Object -Unique)
  $cRawChecksums = @($controlRaw | Where-Object {
      [int]$_.target_kib -eq [int]$controlAggregate.target_kib -and [int]$_.S -eq [int]$controlAggregate.S -and
      [int]$_.R -eq 16 -and $_.mode -eq $controlAggregate.mode -and $_.variant -eq "C"
    } | ForEach-Object { [string]$_.checksum } | Sort-Object -Unique)
  $deterministic = [bool]$matchingA.checksum_deterministic -and [bool]$matchingC.checksum_deterministic
  if ($deterministic) {
    $equal = [string]$matchingA.checksum -eq [string]$matchingC.checksum
    $evidence = "aggregate checksum"
  } else {
    $equal = (($aRawChecksums -join '|') -eq ($cRawChecksums -join '|'))
    $evidence = "raw checksum equality (aggregate checksum nondeterministic)"
  }
  [void]$checksumLines.Add("C_CHECKSUM target_kib=$($controlAggregate.target_kib) S=$($controlAggregate.S) mode=$($controlAggregate.mode) equal=$equal evidence=$evidence A=$($matchingA.checksum) C=$($matchingC.checksum)")
  if (-not $equal) { $checksumPass = $false }
}
if (-not $checksumPass) { throw "C checksum-vs-A checksum validation failed" }

$metricRecords = New-Object -TypeName 'System.Collections.Generic.List[object]'
$maxG8 = 0.0
$maxG16 = 0.0
$flatWarningLines = New-Object -TypeName 'System.Collections.Generic.List[string]'
foreach ($size in $sizes) {
  foreach ($RValue in $depths) {
    foreach ($variant in $variants) {
      $base = [double](Get-MainRow $size 1 $RValue "fused" $variant).median
      $gValues = @{}
      foreach ($SValue in @(2, 4, 8, 16)) {
        $gValues[$SValue] = [double](Get-MainRow $size $SValue $RValue "fused" $variant).median / $base
      }
      $maxG8 = [math]::Max($maxG8, $gValues[8])
      $maxG16 = [math]::Max($maxG16, $gValues[16])
      $fValues = @{}
      foreach ($SValue in @(4, 8, 16)) {
        $fused = [double](Get-MainRow $size $SValue $RValue "fused" $variant).median
        $repeat = [double](Get-MainRow $size $SValue $RValue "repeat" $variant).median
        $fValues[$SValue] = $fused / $repeat
      }
      $flat = [math]::Abs($fValues[4] - 1.0) -le 0.05 -and
              [math]::Abs($fValues[8] - 1.0) -le 0.05 -and
              [math]::Abs($fValues[16] - 1.0) -le 0.05
      if ($flat) {
        $flatWarningLines.Add("F_FLAT_WARNING target_kib=$size R=$RValue variant=$variant fused/repeat medians effectively equal")
      }
      [void]$metricRecords.Add([object]([pscustomobject][ordered]@{
        target_kib = $size
        R = $RValue
        variant = $variant
        G2 = $gValues[2]
        G4 = $gValues[4]
        G8 = $gValues[8]
        G16 = $gValues[16]
        max_G8_G16 = [math]::Max($gValues[8], $gValues[16])
        F4 = $fValues[4]
        F8 = $fValues[8]
        F16 = $fValues[16]
        F_rises_4_8_16 = ($fValues[8] -gt $fValues[4] -and $fValues[16] -gt $fValues[8])
        F_flat_warning = $flat
      }))
    }
  }
}
$maxG = [math]::Max($maxG8, $maxG16)
$status = if ($maxG -ge 2.0) { "PASS_STRONG" } elseif ($maxG -ge 1.5) { "PASS" } elseif ($maxG -ge 1.2) { "AMBIGUOUS" } else { "NEGATIVE_FLAT" }
$interpretation = if ($maxG -ge 2.0) { "strong aggregate G evidence" } elseif ($maxG -ge 1.5) { "aggregate G gate passed" } elseif ($maxG -ge 1.2) { "aggregate G evidence ambiguous" } else { "aggregate G evidence negative/flat" }
$metricText = $metricRecords | ForEach-Object {
  "metrics target_kib=$($_.target_kib) R=$($_.R) variant=$($_.variant) G2=$(Format-Number $_.G2) G4=$(Format-Number $_.G4) G8=$(Format-Number $_.G8) G16=$(Format-Number $_.G16) max_G8_G16=$(Format-Number $_.max_G8_G16) F4=$(Format-Number $_.F4) F8=$(Format-Number $_.F8) F16=$(Format-Number $_.F16) F_rises_4_8_16=$($_.F_rises_4_8_16)"
}

$rawCsvRows = @($script:records | ConvertTo-Csv -NoTypeInformation)
Set-Content -LiteralPath $machineCsv -Encoding ascii -Value $rawCsvRows
$aggregateRows = @($pilotAggregates) + @($script:mainAggregates) + @($script:controlAggregates)
Set-Content -LiteralPath $aggregateCsv -Encoding ascii -Value @($aggregateRows | ConvertTo-Csv -NoTypeInformation)
Set-Content -LiteralPath $metricsCsv -Encoding ascii -Value @($metricRecords | ConvertTo-Csv -NoTypeInformation)

$pilotText = $pilotScores | ForEach-Object {
  "pilot S_tile=$($_.S_tile) A_fused_median=$(Format-Number $_.A_fused_median) B_fused_median=$(Format-Number $_.B_fused_median) score=$(Format-Number $_.score)"
}
$summary = @(
  "T0-M Phase 3 report"
  "status=$status"
  "scope=Phase 3 only; Phase 4 constant-work sweep=NOT RUN; MRDL/Q4/later recurrent Requantize/Norm/Residual=untouched"
  "executive_summary=$interpretation; gate=max(G8,G16)>=1.5; strong_threshold=2.0; ambiguous_range=1.2-1.5"
  "executable=$Executable"
  "preflight=single explicit --self-test; exit=0; correction_pass_required=true; speed_rows_never_receive_self_test=true"
  "machine_csv=$machineCsv"
  "aggregate_csv=$aggregateCsv"
  "metrics_csv=$metricsCsv"
  "stderr_log=$stderrLog"
  "command_log=$commandLog"
  "preflight_stdout=$preflightStdout"
  "preflight_stderr=$preflightStderr"
  "D=$D; bytes_per_int8=1; workers=$workers; physical_cpus=$($cpus -join ','); affinity_policy=explicit four-worker CPUs 0,2,4,6"
  "size_policy=validated proportional rows across four workers; O_i/B_i exact arrays validated on every row"
  "target_mapping=384KiB:O_i=[908,852,512,800],sum=3072; 512KiB:O_i=[1211,1136,683,1066],sum=4096; 640KiB:O_i=[1514,1420,854,1332],sum=5120; 768KiB:O_i=[1816,1705,1023,1600],sum=6144"
  "B_i=O_i*D*R for variant B; variant A/C use one weight block across R; C adds 64MiB eviction outside timed units"
  "timed_repetitions=$TimedRepetitions; warmup=$Warmup; independent_runs=$IndependentRuns; failed_invocations=$script:failedInvocations; speed_processes=$script:invocationNumber; total_processes_including_preflight=$($script:invocationNumber + 1)"
  "exact_counts=pilot=$pilotCount (3 tiles*2 modes*2 variants*10); main=$mainCount (4 sizes*5 S*5 R*2 modes*2 variants*10); controls=$controlCount (2 sizes*3 S*2 modes*10); total_speed=$($pilotCount + $mainCount + $controlCount); total_processes_including_preflight=$($pilotCount + $mainCount + $controlCount + 1)"
  "selected_S_tile=$selectedTile; pilot_rule=fused median over 10 independent repetitions per A/B tile, then mean of A/B medians"
  $pilotText
  "C_checksum_policy=aggregate checksums compared when deterministic; otherwise raw checksum equality documented"
  $checksumLines
  "G_definition=median fused(S)/median fused(1), per size/R/variant; F_definition=median fused(S)/median repeat(S), per size/R/variant"
  $metricText
  "G_gate=max(G8,G16)>=1.5; max_G8=$(Format-Number $maxG8); max_G16=$(Format-Number $maxG16); max_G8_or_G16=$(Format-Number $maxG)"
  $flatWarningLines
  "interpretation=$interpretation"
  "A/B_raw_ratio_gate=removed; raw A/B ratio is not primary gate"
  "metrics_csv_rows=$($metricRecords.Count); reports=G2/G4/G8/G16,max_G8/G16,F4/F8/F16,F_rises_4_8_16"
  "AVX2_policy=every row avx2_supported=true and kernel_used=avx2; affinity, exact repetitions, nonzero checksum, and MAC total validated"
  "next=Do not start Phase 4 until Phase 3 interpretation is approved"
)
Set-Content -LiteralPath $summaryPath -Encoding ascii -Value $summary
Write-Output $summaryPath
