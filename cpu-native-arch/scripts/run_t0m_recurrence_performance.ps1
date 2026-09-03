[CmdletBinding()]
param(
  [string]$Executable = "",
  [string]$OutputDirectory = "",
  [string]$CTestExecutable = "",
  [int]$TimedRepetitions = 8,
  [int]$Warmup = 2,
  [int]$IndependentRuns = 10
)

$ErrorActionPreference = "Stop"
[System.Threading.Thread]::CurrentThread.CurrentCulture = [Globalization.CultureInfo]::InvariantCulture
[System.Threading.Thread]::CurrentThread.CurrentUICulture = [Globalization.CultureInfo]::InvariantCulture
$projectRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($Executable)) { $Executable = Join-Path $projectRoot "build\t0m_recurrence_probe.exe" }
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) { $OutputDirectory = Join-Path $projectRoot "sweep-output\t0m-recurrence-performance-d512" }
if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) { throw "Executable not found: $Executable" }
if ($TimedRepetitions -ne 8 -or $Warmup -ne 2 -or $IndependentRuns -ne 10) {
  throw "Approved gate requires --timed-repetitions 8, --warmup 2, and 10 independent runs"
}
$summaryPath = Join-Path $OutputDirectory "summary.txt"
if (Test-Path -LiteralPath $summaryPath -PathType Leaf) { throw "Refusing rerun: summary exists at $summaryPath" }
if (-not (Test-Path -LiteralPath $OutputDirectory -PathType Container)) { $null = New-Item -ItemType Directory -Path $OutputDirectory }

$D = 512
$SValues = @(1, 2, 4, 8, 16)
$RValues = @(1, 2, 4, 8, 16)
$cpus = @(0, 2, 4, 6)
$workers = 4
$rows = @(128, 128, 128, 128)
$rowText = $rows -join ','
$cpuText = $cpus -join ','
$machinePath = Join-Path $OutputDirectory "machine.csv"
$aggregatePath = Join-Path $OutputDirectory "aggregate.csv"
$metricsPath = Join-Path $OutputDirectory "metrics.csv"
$commandsPath = Join-Path $OutputDirectory "commands.log"
$stderrPath = Join-Path $OutputDirectory "stderr.log"
$preflightStdoutPath = Join-Path $OutputDirectory "preflight.stdout"
$preflightStderrPath = Join-Path $OutputDirectory "preflight.stderr"
$validationPath = Join-Path $OutputDirectory "validation.txt"
$script:records = New-Object 'System.Collections.Generic.List[object]'
$script:invocations = 0

$nativeHeader = "D,S,R,rows_per_worker,component,mode,kernel,elapsed_seconds,elapsed_per_timed_step,qpc_ticks_per_timed_step,tsc_cycles_per_timed_step,tsc_supported,mac_total,mac_per_second,checksum_kind,validation_invariant,final_checksum,per_round_checksums,per_round_finite,per_round_overflow,per_round_clipped_cells,per_round_clipping_rates,clipped_cells,clipping_rate,all_rounds_valid,worker_count,cpus,affinity,affinity_errors,affinity_succeeded,timed_repetitions,warmup,timed_repetitions_exact"
Set-Content -LiteralPath $commandsPath -Encoding ascii -Value @(
  "T0-M recurrence performance gate exact commands"
  "executable=$Executable"
  "preflight: CTest recurrence correction, then one explicit --self-test; speed rows never receive --self-test"
  "gate: D=512; workers=4; cpus=0,2,4,6; rows-per-worker=128,128,128,128; S=1,2,4,8,16; R=1,2,4,8,16; modes=fused,repeat"
  "timed-repetitions=8; warmup=2; independent-runs=10; order=even fused then repeat, odd repeat then fused; pair-adjacent; no concurrency"
)
Set-Content -LiteralPath $stderrPath -Encoding ascii -Value @(
  "T0-M recurrence performance gate stderr log"
  "executable=$Executable"
  "speed stderr correction/autotest text is forbidden"
)

function Format-Number([double]$Value) { return $Value.ToString("R", [Globalization.CultureInfo]::InvariantCulture) }
function Limit-Text([string]$Text, [int]$Maximum = 4000) {
  if ($null -eq $Text) { return "" }
  $value = $Text.TrimEnd()
  if ($value.Length -le $Maximum) { return $value }
  return $value.Substring(0, $Maximum) + "...[truncated]"
}
function Invoke-CapturedProcess([string]$FileName, [string[]]$Arguments) {
  $startInfo = New-Object System.Diagnostics.ProcessStartInfo
  $startInfo.FileName = $FileName
  $startInfo.Arguments = (($Arguments | ForEach-Object { if ($_ -match '[\s"]') { '"' + ($_ -replace '"', '\"') + '"' } else { [string]$_ } }) -join ' ')
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
    return [pscustomobject]@{ ExitCode = $process.ExitCode; Stdout = $stdoutTask.Result; Stderr = $stderrTask.Result }
  } finally { $process.Dispose() }
}
function Invoke-Logged([string]$Label, [string[]]$Arguments) {
  $script:invocations++
  $id = $script:invocations
  $command = '"' + $Executable + '" ' + ($Arguments -join ' ')
  Add-Content -LiteralPath $commandsPath -Encoding ascii -Value "[$id] $Label $command"
  $result = Invoke-CapturedProcess $Executable $Arguments
  Add-Content -LiteralPath $stderrPath -Encoding ascii -Value "[$id] $Label exit=$($result.ExitCode) stdout=$(Limit-Text $result.Stdout) stderr=$(Limit-Text $result.Stderr)"
  return [pscustomobject]@{ Id = $id; Command = $command; Result = $result }
}
function Require-Equal([string]$Name, [string]$Actual, [string]$Expected, [string]$Context) {
  if ($Actual -ne $Expected) { throw "Invalid ${Name}=$Actual, expected ${Expected}: $Context" }
}
function Require-BoolList([string]$Text, [bool]$Expected, [int]$Count, [string]$Name, [string]$Context) {
  $values = @($Text -split ';')
  if ($values.Count -ne $Count -or @($values | Where-Object { $_ -ne ($(if ($Expected) { 'true' } else { 'false' }) ) }).Count -gt 0) {
    throw "Invalid ${Name}=$Text`: expected $Count $(if ($Expected) { 'true' } else { 'false' }): $Context"
  }
}
function Require-Positive([string]$Name, [double]$Value, [string]$Context) {
  if ([double]::IsNaN($Value) -or [double]::IsInfinity($Value) -or $Value -le 0) { throw "Invalid nonpositive ${Name}=$Value`: $Context" }
}
function Get-Median([double[]]$Values) {
  $ordered = @($Values | Sort-Object)
  if ($ordered.Count -ne 10) { throw "Median requires 10 values, got $($ordered.Count)" }
  return ([double]$ordered[4] + [double]$ordered[5]) / 2.0
}
function Get-PopulationSd([double[]]$Values) {
  $mean = [double](($Values | Measure-Object -Average).Average)
  $sum = 0.0
  foreach ($value in $Values) { $sum += ($value - $mean) * ($value - $mean) }
  return [math]::Sqrt($sum / $Values.Count)
}

if ([string]::IsNullOrWhiteSpace($CTestExecutable)) {
  $ctestCommand = Get-Command ctest.exe -ErrorAction SilentlyContinue
  if ($null -ne $ctestCommand) { $CTestExecutable = $ctestCommand.Path }
}
if ([string]::IsNullOrWhiteSpace($CTestExecutable)) {
  $knownCTestPaths = @(
    "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\ctest.exe",
    "C:\Program Files\CMake\bin\ctest.exe",
    "C:\vkd\tools\cmake-4.3.3-windows\cmake-4.3.3-windows-x86_64\bin\ctest.exe"
  )
  $CTestExecutable = @($knownCTestPaths | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf })[0]
}
if ([string]::IsNullOrWhiteSpace($CTestExecutable) -or -not (Test-Path -LiteralPath $CTestExecutable -PathType Leaf)) {
  throw "ctest.exe not found; provide -CTestExecutable and run CTest recurrence correction before sweep"
}
$ctestArgs = @("--test-dir", (Join-Path $projectRoot "build"), "-R", "t0m_recurrence_correction", "--output-on-failure")
Add-Content -LiteralPath $commandsPath -Encoding ascii -Value ("ctest_command `"$CTestExecutable`" " + ($ctestArgs -join ' '))
$ctestResult = Invoke-CapturedProcess $CTestExecutable $ctestArgs
Set-Content -LiteralPath (Join-Path $OutputDirectory "preflight.ctest.stdout") -Encoding ascii -Value (Limit-Text $ctestResult.Stdout)
Add-Content -LiteralPath (Join-Path $OutputDirectory "preflight.ctest.stderr") -Encoding ascii -Value (Limit-Text $ctestResult.Stderr)
if ($ctestResult.ExitCode -ne 0) { throw "CTest recurrence correction failed with exit code $($ctestResult.ExitCode)" }

$preflight = Invoke-Logged "preflight_self_test" @("--self-test")
Set-Content -LiteralPath $preflightStdoutPath -Encoding ascii -Value (Limit-Text $preflight.Result.Stdout)
Set-Content -LiteralPath $preflightStderrPath -Encoding ascii -Value (Limit-Text $preflight.Result.Stderr)
if ($preflight.Result.ExitCode -ne 0 -or (($preflight.Result.Stdout + "`n" + $preflight.Result.Stderr) -notmatch "(?i)T0-M recurrence correction passed")) {
  throw "Explicit recurrence self-test preflight failed or did not report correction pass"
}

foreach ($S in $SValues) {
  foreach ($R in $RValues) {
    for ($run = 1; $run -le $IndependentRuns; $run++) {
      $order = if (($run % 2) -eq 0) { @("fused", "repeat") } else { @("repeat", "fused") }
      $orderIndex = 0
      foreach ($mode in $order) {
        $orderIndex++
        $context = "S=$S R=$R run=$run mode=$mode order_index=$orderIndex"
        $arguments = @("--D", "$D", "--S", "$S", "--R", "$R", "--mode", $mode,
          "--workers", "$workers", "--cpus", $cpuText, "--rows-per-worker", $rowText,
          "--timed-repetitions", "$TimedRepetitions", "--warmup", "$Warmup")
        $logged = Invoke-Logged "speed $context" $arguments
        $result = $logged.Result
        if ($result.ExitCode -ne 0) { throw "Nonzero speed exit $($result.ExitCode): $context" }
        if ($result.Stderr -match "(?i)(correction|autotest|self.test|test passed|test failed)") { throw "Forbidden correction/autotest text in speed stderr: $context" }
        $lines = @($result.Stdout -split '\r?\n' | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
        if ($lines.Count -ne 2 -or $lines[0] -ne $nativeHeader) { throw "Bad CSV schema/row count: $context" }
        try { $row = $lines[1] | ConvertFrom-Csv -Header ($nativeHeader -split ',') } catch { throw "Bad CSV parse: $context" }
        Require-Equal "D" ([string]$row.D) "$D" $context
        Require-Equal "S" ([string]$row.S) "$S" $context
        Require-Equal "R" ([string]$row.R) "$R" $context
        Require-Equal "rows_per_worker" ([string]$row.rows_per_worker) $rowText $context
        Require-Equal "mode" ([string]$row.mode) $mode $context
        Require-Equal "worker_count" ([string]$row.worker_count) "$workers" $context
        Require-Equal "cpus" ([string]$row.cpus) $cpuText $context
        Require-Equal "affinity" ([string]$row.affinity) "1,1,1,1" $context
        Require-Equal "affinity_errors" ([string]$row.affinity_errors) "0,0,0,0" $context
        Require-Equal "affinity_succeeded" ([string]$row.affinity_succeeded) "true" $context
        Require-Equal "timed_repetitions" ([string]$row.timed_repetitions) "$TimedRepetitions" $context
        Require-Equal "warmup" ([string]$row.warmup) "$Warmup" $context
        Require-Equal "timed_repetitions_exact" ([string]$row.timed_repetitions_exact) "true" $context
        Require-Equal "all_rounds_valid" ([string]$row.all_rounds_valid) "true" $context
        Require-BoolList ([string]$row.per_round_finite) $true $R "per_round_finite" $context
        Require-BoolList ([string]$row.per_round_overflow) $false $R "per_round_overflow" $context
        $expectedMac = [int64]$D * $D * $S * $R * $TimedRepetitions
        Require-Equal "mac_total" ([string]$row.mac_total) "$expectedMac" $context
        $checksum = [uint64]$row.final_checksum
        if ($checksum -eq 0) { throw "Checksum zero: $context" }
        Require-Positive "elapsed_seconds" ([double]$row.elapsed_seconds) $context
        Require-Positive "mac_per_second" ([double]$row.mac_per_second) $context
        $roundChecksums = @(([string]$row.per_round_checksums) -split ';')
        if ($roundChecksums.Count -ne $R -or @($roundChecksums | Where-Object { [uint64]$_ -eq 0 }).Count -gt 0) { throw "Invalid per-round checksums: $context" }
        [void]$script:records.Add([pscustomobject][ordered]@{
          run = $run; order_index = $orderIndex; metric = if ($mode -eq "fused") { "G(S)" } else { "F(S)" }; native_mode = $mode
          D = [int]$row.D; S = [int]$row.S; R = [int]$row.R; rows_per_worker = [string]$row.rows_per_worker
          mode = [string]$row.mode; kernel = [string]$row.kernel; elapsed_seconds = [double]$row.elapsed_seconds
          mac_total = [uint64]$row.mac_total; mac_per_second = [double]$row.mac_per_second; final_checksum = $checksum
          per_round_checksums = [string]$row.per_round_checksums; per_round_finite = [string]$row.per_round_finite
          per_round_overflow = [string]$row.per_round_overflow; per_round_clipped_cells = [string]$row.per_round_clipped_cells
          per_round_clipping_rates = [string]$row.per_round_clipping_rates; clipped_cells = [uint64]$row.clipped_cells
          clipping_rate = [double]$row.clipping_rate; all_rounds_valid = [string]$row.all_rounds_valid; worker_count = [int]$row.worker_count
          cpus = [string]$row.cpus; affinity = [string]$row.affinity; affinity_errors = [string]$row.affinity_errors
          affinity_succeeded = [string]$row.affinity_succeeded; timed_repetitions = [int]$row.timed_repetitions; warmup = [int]$row.warmup
          timed_repetitions_exact = [string]$row.timed_repetitions_exact
        })
      }
    }
  }
}

if ($script:records.Count -ne 500) { throw "Expected 500 raw rows, got $($script:records.Count)" }
$aggregates = New-Object 'System.Collections.Generic.List[object]'
foreach ($S in $SValues) {
  foreach ($R in $RValues) {
    foreach ($mode in @("fused", "repeat")) {
      $group = @($script:records | Where-Object { $_.S -eq $S -and $_.R -eq $R -and $_.native_mode -eq $mode })
      if ($group.Count -ne 10) { throw "Expected 10 rows in S=$S R=$R mode=$mode, got $($group.Count)" }
      $values = [double[]]@($group | ForEach-Object { $_.mac_per_second })
      $checksums = @($group | ForEach-Object { [string]$_.final_checksum } | Sort-Object -Unique)
      if ($checksums.Count -ne 1) { throw "Nondeterministic checksum in S=$S R=$R mode=$mode" }
      [void]$aggregates.Add([pscustomobject][ordered]@{
        metric = if ($mode -eq "fused") { "G(S)" } else { "F(S)" }; native_mode = $mode; D = $D; S = $S; R = $R; rows_per_worker = $rowText
        n = $group.Count; median = Get-Median $values; min = ($values | Measure-Object -Minimum).Minimum; max = ($values | Measure-Object -Maximum).Maximum
        population_sd = Get-PopulationSd $values; checksum = $checksums[0]; checksum_deterministic = $true; checksum_values = ($checksums -join '|')
        elapsed_median_seconds = Get-Median ([double[]]@($group | ForEach-Object { $_.elapsed_seconds })); mac_total = [uint64]$group[0].mac_total
        mac_per_second_median = Get-Median $values; all_rounds_valid = (($group | Where-Object { $_.all_rounds_valid -ne "true" }).Count -eq 0)
        timed_repetitions = $TimedRepetitions; warmup = $Warmup; worker_count = $workers; cpus = $cpuText
      })
    }
  }
}
if ($aggregates.Count -ne 50) { throw "Expected 50 aggregate rows, got $($aggregates.Count)" }

$metrics = New-Object 'System.Collections.Generic.List[object]'
foreach ($S in $SValues) {
  foreach ($R in $RValues) {
    $fused = @($aggregates | Where-Object { $_.S -eq $S -and $_.R -eq $R -and $_.native_mode -eq "fused" })[0]
    $repeat = @($aggregates | Where-Object { $_.S -eq $S -and $_.R -eq $R -and $_.native_mode -eq "repeat" })[0]
    [void]$metrics.Add([pscustomobject][ordered]@{
      D = $D; S = $S; R = $R; G_S = [double]$fused.median; F_S = [double]$repeat.median; G_metric = "G(S)"; F_metric = "F(S)"
      G_native_mode = "fused"; F_native_mode = "repeat"; G_over_F = [double]$fused.median / [double]$repeat.median
      recurrence_path = "Norm+Requantize+residual between every round"; rows_per_worker = $rowText; workers = $workers; cpus = $cpuText
      timed_repetitions = $TimedRepetitions; warmup = $Warmup; independent_runs = $IndependentRuns
    })
  }
}

Set-Content -LiteralPath $machinePath -Encoding ascii -Value @($script:records | ConvertTo-Csv -NoTypeInformation)
Set-Content -LiteralPath $aggregatePath -Encoding ascii -Value @($aggregates | ConvertTo-Csv -NoTypeInformation)
Set-Content -LiteralPath $metricsPath -Encoding ascii -Value @($metrics | ConvertTo-Csv -NoTypeInformation)
$validation = @(
  "status=PASS"
  "raw_rows=$($script:records.Count); expected_raw_rows=500"
  "aggregate_rows=$($aggregates.Count); expected_aggregate_rows=50"
  "cells=25; modes_per_cell=2; runs_per_mode=10"
  "checksum_determinism=PASS for every S,R,mode cell"
  "all_rounds_valid=PASS for every raw row; per_round_finite=true and per_round_overflow=false"
  "affinity=PASS for every raw row; cpus=0,2,4,6; affinity=1,1,1,1; affinity_errors=0,0,0,0"
  "repetition_validation=PASS; timed_repetitions=8; warmup=2; timed_repetitions_exact=true"
  "speed_stderr_correction_autotest_text=PASS absent"
  "checksum_zero=PASS none; nonzero_exit=PASS none; bad_csv=PASS none"
)
Set-Content -LiteralPath $validationPath -Encoding ascii -Value $validation
$summary = @(
  "status=PASS"
  "executive_summary=Approved T0-M recurrence performance gate completed; G(S)=fused and F(S)=repeat over D=512, 25 S/R cells, 10 independent process runs per mode."
  "executable=$Executable"
  "recurrence_path=Norm+Requantize+residual between every round; native mode names preserved as fused/repeat"
  "D=$D; workers=$workers; cpus=$cpuText; rows_per_worker=$rowText"
  "S=$($SValues -join ','); R=$($RValues -join ','); modes=fused,repeat"
  "timed_repetitions=$TimedRepetitions; warmup=$Warmup; independent_runs=$IndependentRuns"
  "execution_order=even run fused then repeat; odd run repeat then fused; each S,R pair adjacent; modes never concurrent"
  "counts=raw_rows=500; aggregate_rows=50; metrics_rows=$($metrics.Count); speed_processes=$($script:invocations - 1); preflight_processes=1; total_processes=$script:invocations"
  "aggregation=median of 10; min/max; population SD; checksum determinism; measured elapsed; MAC totals; MAC/s"
  "artifacts=machine.csv,aggregate.csv,metrics.csv,commands.log,stderr.log,preflight.stdout,preflight.stderr,preflight.ctest.stdout,preflight.ctest.stderr,summary.txt,validation.txt"
  "validation=PASS; see validation.txt"
)
Set-Content -LiteralPath $summaryPath -Encoding ascii -Value $summary
Write-Output $summaryPath
