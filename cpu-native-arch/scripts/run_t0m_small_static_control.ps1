[CmdletBinding()]
param(
  [string]$Executable = "",
  [string]$OutputDirectory = ""
)

$ErrorActionPreference = "Stop"
$invariant = [Globalization.CultureInfo]::InvariantCulture
[System.Threading.Thread]::CurrentThread.CurrentCulture = $invariant
[System.Threading.Thread]::CurrentThread.CurrentUICulture = $invariant

$projectRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($Executable)) {
  $Executable = Join-Path $projectRoot "build\t0m_int8_probe.exe"
}
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
  $OutputDirectory = Join-Path $projectRoot "sweep-output\t0m-small-static-control-d512"
}

$D = 512
$SValues = @(1, 2, 4, 8, 16)
$RValues = @(1, 2, 4, 8, 16)
$cpus = @(0, 2, 4, 6)
$workers = 4
$rows = @(128, 128, 128, 128)
$rowText = $rows -join ','
$cpuText = $cpus -join ','
$variant = "A"
$tile = 8
$iterations = 1
$timedRepetitions = 8
$warmup = 2
$independentRuns = 10
$modes = @("fused", "repeat")

$summaryPath = Join-Path $OutputDirectory "summary.txt"
if (Test-Path -LiteralPath $summaryPath -PathType Leaf) {
  throw "Refusing rerun: summary exists at $summaryPath"
}
if (-not (Test-Path -LiteralPath $OutputDirectory -PathType Container)) {
  $null = New-Item -ItemType Directory -Path $OutputDirectory
}

if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
  $buildDirectory = Join-Path $projectRoot "build"
  if (-not (Test-Path -LiteralPath $buildDirectory -PathType Container)) {
    throw "Executable not found and build directory unavailable: $Executable"
  }
  & cmake --build $buildDirectory --target t0m_int8_probe
  if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
    throw "Could not build required executable: $Executable"
  }
}

$machinePath = Join-Path $OutputDirectory "machine.csv"
$aggregatePath = Join-Path $OutputDirectory "aggregate.csv"
$metricsPath = Join-Path $OutputDirectory "metrics.csv"
$comparisonPath = Join-Path $OutputDirectory "comparison.csv"
$commandsPath = Join-Path $OutputDirectory "commands.log"
$stderrPath = Join-Path $OutputDirectory "stderr.log"
$preflightStdoutPath = Join-Path $OutputDirectory "preflight.stdout"
$preflightStderrPath = Join-Path $OutputDirectory "preflight.stderr"
$validationPath = Join-Path $OutputDirectory "validation.txt"

Set-Content -LiteralPath $commandsPath -Encoding ascii -Value @(
  "T0-M small static control exact commands"
  "executable=$Executable"
  "static_path=original non-recurrent T0-M; fixed activations/state; no Norm/Requantize/residual"
  "gate=D=512; workers=4; cpus=0,2,4,6; rows_per_worker=128,128,128,128; variant=A(shared weights); S_tile=8"
  "S=1,2,4,8,16; R=1,2,4,8,16; modes=fused,repeat; iterations=1; timed_repetitions=8; warmup=2"
  "independent_runs=10; even run fused then repeat; odd run repeat then fused; pair-adjacent; no concurrency"
  "preflight=one explicit --self-test; speed rows never receive --self-test"
)
Set-Content -LiteralPath $stderrPath -Encoding ascii -Value @(
  "T0-M small static control stderr log"
  "executable=$Executable"
  "speed correction/autotest text is forbidden"
)

$script:records = New-Object 'System.Collections.Generic.List[object]'
$script:invocations = 0
$script:failedInvocations = 0

function Format-Number([double]$Value) {
  return $Value.ToString("R", $invariant)
}

function Limit-Text([string]$Text, [int]$Maximum = 4000) {
  if ($null -eq $Text) { return "" }
  $value = $Text.TrimEnd()
  if ($value.Length -le $Maximum) { return $value }
  return $value.Substring(0, $Maximum) + "...[truncated]"
}

function Invoke-CapturedProcess([string]$FileName, [string[]]$Arguments) {
  $startInfo = New-Object System.Diagnostics.ProcessStartInfo
  $startInfo.FileName = $FileName
  $startInfo.Arguments = (($Arguments | ForEach-Object {
      if ($_ -match '[\s"]') { '"' + ($_ -replace '"', '\"') + '"' } else { [string]$_ }
    }) -join ' ')
  $startInfo.UseShellExecute = $false
  $startInfo.CreateNoWindow = $true
  $startInfo.RedirectStandardOutput = $true
  $startInfo.RedirectStandardError = $true
  $process = New-Object System.Diagnostics.Process
  $process.StartInfo = $startInfo
  try {
    if (-not $process.Start()) { throw "Could not start process: $FileName" }
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

function Get-Median([double[]]$Values) {
  if ($Values.Count -ne $independentRuns) { throw "Median requires $independentRuns values, got $($Values.Count)" }
  $ordered = @($Values | Sort-Object)
  return ([double]$ordered[4] + [double]$ordered[5]) / 2.0
}

function Get-PopulationSd([double[]]$Values) {
  $mean = [double](($Values | Measure-Object -Average).Average)
  $sum = 0.0
  foreach ($value in $Values) { $sum += ($value - $mean) * ($value - $mean) }
  return [math]::Sqrt($sum / $Values.Count)
}

function Require-Equal([string]$Name, [string]$Actual, [string]$Expected, [string]$Context) {
  if ($Actual -ne $Expected) { throw "Invalid ${Name}=$Actual, expected ${Expected}: $Context" }
}

function Require-Positive([string]$Name, [double]$Value, [string]$Context) {
  if ([double]::IsNaN($Value) -or [double]::IsInfinity($Value) -or $Value -le 0) {
    throw "Invalid nonpositive ${Name}=$Value`: $Context"
  }
}

function Invoke-Preflight {
  $arguments = @("--self-test")
  Add-Content -LiteralPath $commandsPath -Encoding ascii -Value "preflight_command `"$Executable`" --self-test"
  $result = Invoke-CapturedProcess $Executable $arguments
  Set-Content -LiteralPath $preflightStdoutPath -Encoding ascii -Value (Limit-Text $result.Stdout)
  Set-Content -LiteralPath $preflightStderrPath -Encoding ascii -Value (Limit-Text $result.Stderr)
  Add-Content -LiteralPath $stderrPath -Encoding ascii -Value "preflight exit=$($result.ExitCode) stdout=$(Limit-Text $result.Stdout) stderr=$(Limit-Text $result.Stderr)"
  if ($result.ExitCode -ne 0) { throw "Static probe self-test preflight failed with exit code $($result.ExitCode)" }
  if ((($result.Stdout + "`n" + $result.Stderr) -notmatch "(?i)T0-M correction passed")) {
    throw "Static probe self-test preflight did not report correction pass"
  }
}

function Invoke-Speed([int]$Run, [int]$OrderIndex, [int]$S, [int]$R, [string]$Mode) {
  $context = "S=$S R=$R run=$Run mode=$Mode order_index=$OrderIndex"
  $arguments = @(
    "--D", "$D", "--S", "$S", "--R", "$R", "--mode", $Mode,
    "--variant", $variant, "--S-tile", "$tile", "--workers", "$workers",
    "--cpus", $cpuText, "--rows-per-worker", $rowText, "--iterations", "$iterations",
    "--timed-repetitions", "$timedRepetitions", "--warmup", "$warmup"
  )
  $script:invocations++
  $id = $script:invocations
  $command = '"' + $Executable + '" ' + ($arguments -join ' ')
  Add-Content -LiteralPath $commandsPath -Encoding ascii -Value "[$id] speed $context $command"
  $result = Invoke-CapturedProcess $Executable $arguments
  Add-Content -LiteralPath $stderrPath -Encoding ascii -Value "[$id] $context exit=$($result.ExitCode) stdout=$(Limit-Text $result.Stdout) stderr=$(Limit-Text $result.Stderr)"
  if ($result.ExitCode -ne 0) {
    $script:failedInvocations++
    throw "Nonzero speed exit $($result.ExitCode): $context"
  }
  if (($result.Stdout + "`n" + $result.Stderr) -match "(?i)(correction|autotest|self.test|test passed|test failed)") {
    $script:failedInvocations++
    throw "Forbidden correction/autotest text in speed output: $context"
  }

  $header = "D,S,R,O_i,B_i,rows_per_worker,bytes_per_worker,mode,variant,S_tile,iterations,timed_repetitions,warmup,worker_count,worker_list,affinity,affinity_error,affinity_succeeded,timed_repetitions_exact,avx2_supported,kernel_used,eviction_bytes,eviction_checksum,elapsed_seconds,mac_total,mac_per_second,checksum"
  $lines = @($result.Stdout -split '\r?\n' | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
  if ($lines.Count -ne 2 -or $lines[0] -ne $header) {
    $script:failedInvocations++
    throw "Bad static CSV schema/output: $context"
  }
  try { $row = $lines[1] | ConvertFrom-Csv -Header ($header -split ',') } catch {
    $script:failedInvocations++
    throw "Bad static CSV parse: $context"
  }

  $expectedBytes = "65536,65536,65536,65536"
  $expectedMac = [int64]$rows[0] * $workers * $D * $S * $R * $iterations * $timedRepetitions
  $checks = @(
    @($row.D, "$D", "D"), @($row.S, "$S", "S"), @($row.R, "$R", "R"),
    @($row.O_i, $rowText, "O_i"), @($row.B_i, $expectedBytes, "B_i"),
    @($row.rows_per_worker, $rowText, "rows_per_worker"), @($row.bytes_per_worker, $expectedBytes, "bytes_per_worker"),
    @($row.mode, $Mode, "mode"), @($row.variant, $variant, "variant"), @($row.S_tile, "$tile", "S_tile"),
    @($row.iterations, "$iterations", "iterations"), @($row.timed_repetitions, "$timedRepetitions", "timed_repetitions"),
    @($row.warmup, "$warmup", "warmup"), @($row.worker_count, "$workers", "worker_count"),
    @($row.worker_list, $cpuText, "worker_list"), @($row.affinity, "true,true,true,true", "affinity"),
    @($row.affinity_error, "0,0,0,0", "affinity_error"), @($row.affinity_succeeded, "true", "affinity_succeeded"),
    @($row.timed_repetitions_exact, "true", "timed_repetitions_exact"), @($row.avx2_supported, "true", "avx2_supported"),
    @($row.kernel_used, "avx2", "kernel_used"), @($row.eviction_bytes, "0", "eviction_bytes"),
    @($row.iterations, "1", "iterations"), @($row.mac_total, "$expectedMac", "mac_total")
  )
  foreach ($check in $checks) { Require-Equal $check[2] ([string]$check[0]) ([string]$check[1]) $context }
  if ([uint64]$row.checksum -eq 0) { throw "Checksum zero: $context" }
  Require-Positive "elapsed_seconds" ([double]$row.elapsed_seconds) $context
  Require-Positive "mac_per_second" ([double]$row.mac_per_second) $context

  return [pscustomobject][ordered]@{
    run = $Run; order_index = $OrderIndex; D = [int]$row.D; S = [int]$row.S; R = [int]$row.R
    rows_per_worker = [string]$row.rows_per_worker; O_i = [string]$row.O_i; B_i = [string]$row.B_i
    bytes_per_worker = [string]$row.bytes_per_worker; mode = [string]$row.mode; variant = [string]$row.variant
    S_tile = [int]$row.S_tile; iterations = [int]$row.iterations; timed_repetitions = [int]$row.timed_repetitions
    warmup = [int]$row.warmup; worker_count = [int]$row.worker_count; worker_list = [string]$row.worker_list
    affinity = [string]$row.affinity; affinity_error = [string]$row.affinity_error
    affinity_succeeded = [string]$row.affinity_succeeded; timed_repetitions_exact = [string]$row.timed_repetitions_exact
    avx2_supported = [string]$row.avx2_supported; kernel_used = [string]$row.kernel_used
    eviction_bytes = [int64]$row.eviction_bytes; eviction_checksum = [uint64]$row.eviction_checksum
    elapsed_seconds = [double]$row.elapsed_seconds; mac_total = [uint64]$row.mac_total
    mac_per_second = [double]$row.mac_per_second; checksum = [uint64]$row.checksum
  }
}

function Get-Aggregates([object[]]$InputRows) {
  $aggregates = New-Object 'System.Collections.Generic.List[object]'
  foreach ($S in $SValues) {
    foreach ($R in $RValues) {
      foreach ($Mode in $modes) {
        $group = @($InputRows | Where-Object { $_.S -eq $S -and $_.R -eq $R -and $_.mode -eq $Mode })
        if ($group.Count -ne $independentRuns) { throw "Expected $independentRuns raw rows for S=$S R=$R mode=$Mode, got $($group.Count)" }
        $checksums = @($group | ForEach-Object { [string]$_.checksum } | Sort-Object -Unique)
        if ($checksums.Count -ne 1) { throw "Nondeterministic checksum for S=$S R=$R mode=$Mode" }
        $values = [double[]]@($group | ForEach-Object { $_.mac_per_second })
        $elapsed = [double[]]@($group | ForEach-Object { $_.elapsed_seconds })
        [void]$aggregates.Add([pscustomobject][ordered]@{
          D = $D; S = $S; R = $R; mode = $Mode; variant = $variant; S_tile = $tile
          rows_per_worker = $rowText; worker_count = $workers; worker_list = $cpuText
          n = $group.Count; median_mac_per_second = Get-Median $values; mean_mac_per_second = [double](($values | Measure-Object -Average).Average)
          min_mac_per_second = [double](($values | Measure-Object -Minimum).Minimum); max_mac_per_second = [double](($values | Measure-Object -Maximum).Maximum)
          population_sd_mac_per_second = Get-PopulationSd $values; median_elapsed_seconds = Get-Median $elapsed
          mac_total = [uint64]$group[0].mac_total; checksum = $checksums[0]; checksum_deterministic = $true; checksum_values = $checksums[0]
          all_affinity_succeeded = (($group | Where-Object { $_.affinity_succeeded -ne "true" }).Count -eq 0)
          all_timed_repetitions_exact = (($group | Where-Object { $_.timed_repetitions_exact -ne "true" }).Count -eq 0)
        })
      }
    }
  }
  if ($aggregates.Count -ne 50) { throw "Expected 50 aggregate rows, got $($aggregates.Count)" }
  return $aggregates.ToArray()
}

function Get-Aggregate([object[]]$Aggregates, [int]$S, [int]$R, [string]$Mode) {
  $matches = @($Aggregates | Where-Object { $_.S -eq $S -and $_.R -eq $R -and $_.mode -eq $Mode })
  if ($matches.Count -ne 1) { throw "Expected one aggregate for S=$S R=$R mode=$Mode, got $($matches.Count)" }
  return $matches[0]
}

Invoke-Preflight

foreach ($S in $SValues) {
  foreach ($R in $RValues) {
    for ($run = 1; $run -le $independentRuns; $run++) {
      $order = if (($run % 2) -eq 0) { @("fused", "repeat") } else { @("repeat", "fused") }
      $orderIndex = 0
      foreach ($Mode in $order) {
        $orderIndex++
        [void]$script:records.Add((Invoke-Speed $run $orderIndex $S $R $Mode))
      }
    }
  }
}

if ($script:records.Count -ne 500) { throw "Expected 500 raw rows, got $($script:records.Count)" }
$aggregates = Get-Aggregates $script:records.ToArray()
$metrics = New-Object 'System.Collections.Generic.List[object]'
foreach ($R in $RValues) {
  $base = [double](Get-Aggregate $aggregates 1 $R "fused").median_mac_per_second
  foreach ($S in $SValues) {
    $fused = Get-Aggregate $aggregates $S $R "fused"
    $repeat = Get-Aggregate $aggregates $S $R "repeat"
    $f = if ($S -ge 4) { [double]$fused.median_mac_per_second / [double]$repeat.median_mac_per_second } else { $null }
    [void]$metrics.Add([pscustomobject][ordered]@{
      D = $D; S = $S; R = $R; variant = $variant; S_tile = $tile
      fused_median_mac_per_second = [double]$fused.median_mac_per_second
      repeat_median_mac_per_second = [double]$repeat.median_mac_per_second
      G_S = [double]$fused.median_mac_per_second / $base; F_S = $f
      G_definition = "median fused(S)/median fused(1), per R"
      F_definition = if ($S -ge 4) { "median fused(S)/median repeat(S), per R" } else { "not defined for S<4" }
    })
  }
}
if ($metrics.Count -ne 25) { throw "Expected 25 normalized metric rows, got $($metrics.Count)" }

function Get-MaxMetric([string]$Name, [int[]]$AllowedS) {
  $values = @($metrics | Where-Object { $AllowedS -contains [int]$_.S } | ForEach-Object { [double]$_.$Name })
  if ($values.Count -ne ($AllowedS.Count * $RValues.Count)) { throw "Unexpected metric count for $Name" }
  return [double](($values | Measure-Object -Maximum).Maximum)
}

$controlMax = [ordered]@{
  G8 = Get-MaxMetric "G_S" @(8)
  G16 = Get-MaxMetric "G_S" @(16)
  F4 = Get-MaxMetric "F_S" @(4)
  F8 = Get-MaxMetric "F_S" @(8)
  F16 = Get-MaxMetric "F_S" @(16)
}

$recurrenceComparisonPath = Join-Path $projectRoot "sweep-output\t0m-recurrence-vectorization-comparison-exact\comparison-pre-post.csv"
if (-not (Test-Path -LiteralPath $recurrenceComparisonPath -PathType Leaf)) {
  throw "Required recurrent comparison artifact not found: $recurrenceComparisonPath"
}
$recurrenceRows = @(Import-Csv -LiteralPath $recurrenceComparisonPath)
$comparisonRows = New-Object 'System.Collections.Generic.List[object]'
$phases = @("pre-vectorization", "post-vectorization")
$recurrenceColumns = [ordered]@{ G8 = "recurrence_G8_norm"; G16 = "recurrence_G16_norm"; F4 = "recurrence_F4_norm"; F8 = "recurrence_F8_norm"; F16 = "recurrence_F16_norm" }
foreach ($phase in $phases) {
  $phaseRows = @($recurrenceRows | Where-Object { $_.phase -eq $phase -and [int]$_.D -eq $D -and [int]$_.target_kib -eq $D })
  if ($phaseRows.Count -ne 5) { throw "Expected five recurrent rows for phase=$phase, got $($phaseRows.Count)" }
  $row = [ordered]@{ record_type = "max_normalized_comparison"; D = $D; target_kib = $D; phase = $phase; static_variant = $variant; static_control = "small fixed-activation/state non-recurrent T0-M; no Norm/Requantize/residual"; recurrent_artifact = $recurrenceComparisonPath }
  foreach ($name in $recurrenceColumns.Keys) {
    $recurrentMax = [double](($phaseRows | ForEach-Object { [double]$_.$($recurrenceColumns[$name]) } | Measure-Object -Maximum).Maximum)
    $staticMax = [double]$controlMax[$name]
    $row["static_control_max_$name"] = $staticMax
    $row["recurrent_${name}_max"] = $recurrentMax
    $row["control_minus_recurrent_$name"] = $staticMax - $recurrentMax
    $row["control_over_recurrent_$name"] = $staticMax / $recurrentMax
  }
  $row.interpretation = "Static control max normalized scaling is compared directly with recurrent $phase max over R; raw MAC/s remains separate."
  [void]$comparisonRows.Add([pscustomobject]$row)
}
if ($comparisonRows.Count -ne 2) { throw "Expected two direct recurrent comparison rows" }

Set-Content -LiteralPath $machinePath -Encoding ascii -Value @($script:records | ConvertTo-Csv -NoTypeInformation)
Set-Content -LiteralPath $aggregatePath -Encoding ascii -Value @($aggregates | ConvertTo-Csv -NoTypeInformation)
Set-Content -LiteralPath $metricsPath -Encoding ascii -Value @($metrics | ConvertTo-Csv -NoTypeInformation)
Set-Content -LiteralPath $comparisonPath -Encoding ascii -Value @($comparisonRows | ConvertTo-Csv -NoTypeInformation)

$validation = @(
  "status=PASS"
  "raw_rows=$($script:records.Count); expected_raw_rows=500"
  "aggregate_rows=$($aggregates.Count); expected_aggregate_rows=50"
  "normalized_metric_rows=$($metrics.Count); expected_normalized_metric_rows=25"
  "cells=25; modes_per_cell=2; runs_per_mode=10; speed_processes=$($script:invocations); preflight_processes=1; total_processes=$($script:invocations + 1)"
  "D_S_R_validation=PASS; D=512; S=1,2,4,8,16; R=1,2,4,8,16"
  "rows_validation=PASS; rows_per_worker=128,128,128,128; total_rows=512"
  "mode_variant_tile_validation=PASS; modes=fused,repeat; variant=A(shared weights); S_tile=8"
  "affinity_validation=PASS; workers=4; cpus=0,2,4,6; affinity=true,true,true,true; affinity_error=0,0,0,0; affinity_succeeded=true"
  "repetition_validation=PASS; iterations=1; timed_repetitions=8; warmup=2; timed_repetitions_exact=true"
  "avx2_validation=PASS; avx2_supported=true; kernel_used=avx2"
  "checksum_validation=PASS; every raw checksum nonzero and deterministic per S,R,mode"
  "speed_stderr_correction_autotest_text=PASS absent; speed_invocations_have_no_self_test=PASS"
  "exit_validation=PASS; failed_invocations=$script:failedInvocations"
  "comparison_validation=PASS; recurrent pre-vectorization and post-vectorization artifact rows=5 each; direct max G8/G16/F4/F8/F16 comparison"
)
Set-Content -LiteralPath $validationPath -Encoding ascii -Value $validation

$summary = @(
  "status=PASS"
  "executive_summary=Original non-recurrent T0-M small static control completed with fixed activations/state, shared weights across R, and no Norm/Requantize/residual path."
  "scope=ONLY requested control experiment; no recurrence changes; no later investigations; no large static target-size comparison"
  "executable=$Executable; implementation=static t0m_int8_probe.exe"
  "D=$D; workers=$workers; cpus=$cpuText; rows_per_worker=$rowText; total_rows=512"
  "S=$($SValues -join ','); R=$($RValues -join ','); modes=fused,repeat; variant=A(shared weights); S_tile=$tile"
  "iterations=$iterations; timed_repetitions=$timedRepetitions; warmup=$warmup; independent_runs=$independentRuns"
  "execution_order=even run fused then repeat; odd run repeat then fused; each S,R pair adjacent; no concurrency"
  "preflight=exactly one explicit --self-test before speed rows; speed rows never receive --self-test"
  "counts=raw_rows=$($script:records.Count); aggregate_rows=$($aggregates.Count); normalized_metric_rows=$($metrics.Count); comparison_rows=$($comparisonRows.Count); speed_processes=$($script:invocations); total_processes=$($script:invocations + 1)"
  "maxima_static_control_G8=$(Format-Number $controlMax.G8); maxima_static_control_G16=$(Format-Number $controlMax.G16); maxima_static_control_F4=$(Format-Number $controlMax.F4); maxima_static_control_F8=$(Format-Number $controlMax.F8); maxima_static_control_F16=$(Format-Number $controlMax.F16)"
  "raw_units=aggregate median_mac_per_second and median_elapsed_seconds are raw throughput/timing; G/F metrics and comparison values are dimensionless"
  "comparison=direct against recurrent pre-vectorization and post-vectorization normalized comparison artifact; comparison.csv contains max G8/G16/F4/F8/F16 and control-minus-recurrent/control-over-recurrent values"
  "artifacts=machine.csv,aggregate.csv,metrics.csv,comparison.csv,summary.txt,validation.txt,commands.log,stderr.log,preflight.stdout,preflight.stderr"
  "validation=PASS; failed_invocations=$script:failedInvocations"
)
Set-Content -LiteralPath $summaryPath -Encoding ascii -Value $summary
Write-Output $summaryPath
