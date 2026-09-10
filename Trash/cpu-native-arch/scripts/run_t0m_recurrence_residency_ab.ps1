[CmdletBinding()]
param(
  [string]$Executable = "",
  [string]$CalibrationExecutable = "",
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
if ([string]::IsNullOrWhiteSpace($CalibrationExecutable)) { $CalibrationExecutable = Join-Path $projectRoot "build\int8_probe.exe" }
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) { $OutputDirectory = Join-Path $projectRoot "sweep-output\t0m-recurrence-residency-ab-d1472-r16" }
if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) { throw "Executable not found: $Executable" }
if (-not (Test-Path -LiteralPath $CalibrationExecutable -PathType Leaf)) { throw "Calibration executable not found: $CalibrationExecutable" }
if ($TimedRepetitions -ne 8 -or $Warmup -ne 2 -or $IndependentRuns -ne 10) {
  throw "Campaign requires --timed-repetitions 8, --warmup 2, and 10 independent runs"
}
$summaryPath = Join-Path $OutputDirectory "summary.txt"
if (Test-Path -LiteralPath $summaryPath -PathType Leaf) { throw "Refusing rerun: summary exists at $summaryPath" }
if (-not (Test-Path -LiteralPath $OutputDirectory -PathType Container)) { $null = New-Item -ItemType Directory -Path $OutputDirectory }

$D = 1472; $S = 1; $R = 16; $workers = 4
$cpus = @(0, 2, 4, 6)
$cpuText = $cpus -join ','
$calibrationPath = Join-Path $OutputDirectory "calibration.csv"
$machinePath = Join-Path $OutputDirectory "machine.csv"
$aggregatePath = Join-Path $OutputDirectory "aggregate.csv"
$comparisonPath = Join-Path $OutputDirectory "comparison.csv"
$commandsPath = Join-Path $OutputDirectory "commands.log"
$stderrPath = Join-Path $OutputDirectory "stderr.log"
$validationPath = Join-Path $OutputDirectory "validation.txt"
$preflightCtestStdoutPath = Join-Path $OutputDirectory "preflight.ctest.stdout.log"
$preflightCtestStderrPath = Join-Path $OutputDirectory "preflight.ctest.stderr.log"
$preflightSelfTestStdoutPath = Join-Path $OutputDirectory "preflight.self-test.stdout.log"
$preflightSelfTestStderrPath = Join-Path $OutputDirectory "preflight.self-test.stderr.log"
$script:records = New-Object 'System.Collections.Generic.List[object]'
$script:invocations = 0
$nativeHeader = "D,S,R,variant,rows_per_worker,component,mode,kernel,elapsed_seconds,elapsed_per_timed_step,qpc_ticks_per_timed_step,tsc_cycles_per_timed_step,tsc_supported,mac_total,mac_per_second,checksum_kind,validation_invariant,final_checksum,per_round_checksums,per_round_finite,per_round_overflow,per_round_clipped_cells,per_round_clipping_rates,clipped_cells,clipping_rate,all_rounds_valid,worker_count,cpus,affinity,affinity_errors,affinity_succeeded,timed_repetitions,warmup,timed_repetitions_exact"

Set-Content -LiteralPath $commandsPath -Encoding ascii -Value @(
  "T0-M recurrence residency control A/B"
  "scope=D=1472; S=1; R=16; mode=fused only; variants=A(shared),B(per-round)"
  "workers=4; cpus=0,2,4,6; rows_per_worker=calibrated proportional plan"
  "timed_repetitions=8; warmup=2; independent_runs=10; no concurrency"
  "order=even run A then B; odd run B then A; pair-adjacent"
  "B seed=0xC001CAFE ^ 0x9E3779B9*(shard_index+1) ^ 0x85EBCA6B*(round+1); xorshift32; value=(next%255)-127"
  "A seed=0xC001CAFE ^ 0x9E3779B9*(shard_index+1); same xorshift32/value mapping; block shared across rounds"
  "preflight=one CTest correction plus one explicit --self-test; speed invocations never receive --self-test"
  "calibration=original T0-R mechanism: int8_probe --cpu CPU --m 64 --K 64 --depth 64 --iterations 4 --repetitions 20 --warmup 5 --kernel avx2; total recurrence rows=1472; largest-remainder proportional allocation"
)
Set-Content -LiteralPath $stderrPath -Encoding ascii -Value @(
  "T0-M recurrence residency control stderr log"
  "speed stderr/stdout correction/autotest text is forbidden"
)

function Limit-Text([string]$Text, [int]$Maximum = 4000) {
  if ($null -eq $Text) { return "" }
  $value = $Text.TrimEnd()
  if ($value.Length -le $Maximum) { return $value }
  return $value.Substring(0, $Maximum) + "...[truncated]"
}
function Format-Number([double]$Value) { return $Value.ToString("R", [Globalization.CultureInfo]::InvariantCulture) }
function Invoke-CapturedProcess([string]$FileName, [string[]]$Arguments) {
  $startInfo = New-Object System.Diagnostics.ProcessStartInfo
  $startInfo.FileName = $FileName
  $startInfo.Arguments = (($Arguments | ForEach-Object { if ($_ -match '[\s"]') { '"' + ($_ -replace '"', '\"') + '"' } else { [string]$_ } }) -join ' ')
  $startInfo.UseShellExecute = $false; $startInfo.CreateNoWindow = $true
  $startInfo.RedirectStandardOutput = $true; $startInfo.RedirectStandardError = $true
  $process = New-Object System.Diagnostics.Process; $process.StartInfo = $startInfo
  try {
    if (-not $process.Start()) { throw "Could not start process: $FileName" }
    $stdoutTask = $process.StandardOutput.ReadToEndAsync(); $stderrTask = $process.StandardError.ReadToEndAsync()
    $process.WaitForExit()
    return [pscustomobject]@{ ExitCode = $process.ExitCode; Stdout = $stdoutTask.Result; Stderr = $stderrTask.Result }
  } finally { $process.Dispose() }
}
function Require-Equal([string]$Name, [string]$Actual, [string]$Expected, [string]$Context) {
  if ($Actual -ne $Expected) { throw "Invalid ${Name}=$Actual, expected ${Expected}: $Context" }
}
function Require-BoolList([string]$Text, [bool]$Expected, [int]$Count, [string]$Name, [string]$Context) {
  $values = @($Text -split ';'); $expectedText = if ($Expected) { 'true' } else { 'false' }
  if ($values.Count -ne $Count -or @($values | Where-Object { $_ -ne $expectedText }).Count -gt 0) {
    throw "Invalid ${Name}=$Text`: expected $Count $expectedText`: $Context"
  }
}
function Require-Positive([string]$Name, [double]$Value, [string]$Context) {
  if ([double]::IsNaN($Value) -or [double]::IsInfinity($Value) -or $Value -le 0) { throw "Invalid nonpositive ${Name}=$Value`: $Context" }
}
function Require-NonNegativeFinite([string]$Name, [double]$Value, [string]$Context) {
  if ([double]::IsNaN($Value) -or [double]::IsInfinity($Value) -or $Value -lt 0) { throw "Invalid ${Name}=$Value`: $Context" }
}
function Get-Median([double[]]$Values) {
  $ordered = @($Values | Sort-Object)
  if ($ordered.Count -ne 10) { throw "Median requires 10 values, got $($ordered.Count)" }
  return ([double]$ordered[4] + [double]$ordered[5]) / 2.0
}
function Get-PopulationSd([double[]]$Values) {
  $mean = [double](($Values | Measure-Object -Average).Average); $sum = 0.0
  foreach ($value in $Values) { $sum += ($value - $mean) * ($value - $mean) }
  return [math]::Sqrt($sum / $Values.Count)
}
function Get-ProportionalRows($Calibration, [int]$TotalRows) {
  $throughputTotal = ($Calibration | Measure-Object -Property mac_per_second -Sum).Sum
  if ($throughputTotal -le 0) { throw "Calibration throughput must be positive" }
  $shares = @($Calibration | ForEach-Object {
      $exact = $TotalRows * $_.mac_per_second / $throughputTotal
      [pscustomobject]@{ cpu = $_.logical_cpu_index; rows = [math]::Floor($exact); fraction = $exact - [math]::Floor($exact) }
    })
  $remaining = $TotalRows - (($shares | Measure-Object -Property rows -Sum).Sum)
  foreach ($share in ($shares | Sort-Object fraction -Descending | Select-Object -First $remaining)) { $share.rows++ }
  return @($shares | Sort-Object cpu | ForEach-Object { [int]$_.rows })
}
function Invoke-Speed([string]$Variant, [int]$Run, [int]$OrderIndex) {
  $script:invocations++
  $context = "variant=$Variant run=$Run order_index=$OrderIndex"
  $arguments = @("--D", "$D", "--S", "$S", "--R", "$R", "--variant", $Variant, "--mode", "fused",
    "--workers", "$workers", "--cpus", $cpuText, "--rows-per-worker", $rowText,
    "--timed-repetitions", "$TimedRepetitions", "--warmup", "$Warmup")
  Add-Content -LiteralPath $commandsPath -Encoding ascii -Value "[$($script:invocations)] speed $context `"$Executable`" $($arguments -join ' ')"
  $result = Invoke-CapturedProcess $Executable $arguments
  Add-Content -LiteralPath $stderrPath -Encoding ascii -Value "[$($script:invocations)] $context exit=$($result.ExitCode) stdout=$(Limit-Text $result.Stdout) stderr=$(Limit-Text $result.Stderr)"
  if ($result.ExitCode -ne 0) { throw "Nonzero speed exit $($result.ExitCode): $context" }
  if (($result.Stdout + "`n" + $result.Stderr) -match "(?i)(correction|autotest|self.test|test passed|test failed)") {
    throw "Forbidden correction/autotest text in speed output: $context"
  }
  $lines = @($result.Stdout -split '\r?\n' | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
  if ($lines.Count -ne 2 -or $lines[0] -ne $nativeHeader) { throw "Bad CSV schema/row count: $context" }
  try { $row = $lines[1] | ConvertFrom-Csv -Header ($nativeHeader -split ',') } catch { throw "Bad CSV parse: $context" }
  Require-Equal "D" ([string]$row.D) "$D" $context; Require-Equal "S" ([string]$row.S) "$S" $context
  Require-Equal "R" ([string]$row.R) "$R" $context; Require-Equal "variant" ([string]$row.variant) $Variant $context
  Require-Equal "rows_per_worker" ([string]$row.rows_per_worker) $rowText $context; Require-Equal "component" ([string]$row.component) "full" $context
  Require-Equal "mode" ([string]$row.mode) "fused" $context; Require-Equal "kernel" ([string]$row.kernel) "avx2_fused" $context
  Require-Equal "worker_count" ([string]$row.worker_count) "$workers" $context; Require-Equal "cpus" ([string]$row.cpus) $cpuText $context
  Require-Equal "affinity" ([string]$row.affinity) "1,1,1,1" $context; Require-Equal "affinity_errors" ([string]$row.affinity_errors) "0,0,0,0" $context
  Require-Equal "affinity_succeeded" ([string]$row.affinity_succeeded) "true" $context; Require-Equal "timed_repetitions" ([string]$row.timed_repetitions) "$TimedRepetitions" $context
  Require-Equal "warmup" ([string]$row.warmup) "$Warmup" $context; Require-Equal "timed_repetitions_exact" ([string]$row.timed_repetitions_exact) "true" $context
  Require-Equal "all_rounds_valid" ([string]$row.all_rounds_valid) "true" $context
  Require-BoolList ([string]$row.per_round_finite) $true $R "per_round_finite" $context
  Require-BoolList ([string]$row.per_round_overflow) $false $R "per_round_overflow" $context
  $expectedMac = [int64]$D * $D * $S * $R * $TimedRepetitions; Require-Equal "mac_total" ([string]$row.mac_total) "$expectedMac" $context
  $checksum = [uint64]$row.final_checksum; if ($checksum -eq 0) { throw "Checksum zero: $context" }
  Require-Positive "elapsed_seconds" ([double]$row.elapsed_seconds) $context; Require-Positive "mac_per_second" ([double]$row.mac_per_second) $context
  $roundChecksums = @(([string]$row.per_round_checksums) -split ';')
  if ($roundChecksums.Count -ne $R -or @($roundChecksums | Where-Object { [uint64]$_ -eq 0 }).Count -gt 0) { throw "Invalid per-round checksums: $context" }
  Require-NonNegativeFinite "clipping_rate" ([double]$row.clipping_rate) $context
  [void]$script:records.Add([pscustomobject][ordered]@{
    run = $Run; order_index = $OrderIndex; variant = [string]$row.variant; D = [int]$row.D; S = [int]$row.S; R = [int]$row.R
    rows_per_worker = [string]$row.rows_per_worker; component = [string]$row.component; mode = [string]$row.mode; kernel = [string]$row.kernel
    elapsed_seconds = [double]$row.elapsed_seconds; elapsed_per_timed_step = [double]$row.elapsed_per_timed_step; mac_total = [uint64]$row.mac_total
    mac_per_second = [double]$row.mac_per_second; final_checksum = $checksum; per_round_checksums = [string]$row.per_round_checksums
    per_round_finite = [string]$row.per_round_finite; per_round_overflow = [string]$row.per_round_overflow
    per_round_clipped_cells = [string]$row.per_round_clipped_cells; clipped_cells = [uint64]$row.clipped_cells; clipping_rate = [double]$row.clipping_rate
    all_rounds_valid = [string]$row.all_rounds_valid; worker_count = [int]$row.worker_count; cpus = [string]$row.cpus
    affinity = [string]$row.affinity; affinity_errors = [string]$row.affinity_errors; affinity_succeeded = [string]$row.affinity_succeeded
    timed_repetitions = [int]$row.timed_repetitions; warmup = [int]$row.warmup; timed_repetitions_exact = [string]$row.timed_repetitions_exact
  })
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
if ([string]::IsNullOrWhiteSpace($CTestExecutable) -or -not (Test-Path -LiteralPath $CTestExecutable -PathType Leaf)) { throw "ctest.exe not found" }
$ctestArgs = @("--test-dir", (Join-Path $projectRoot "build"), "-R", "t0m_recurrence_correction", "--output-on-failure")
Add-Content -LiteralPath $commandsPath -Encoding ascii -Value ("ctest_command `"$CTestExecutable`" " + ($ctestArgs -join ' '))
$ctestResult = Invoke-CapturedProcess $CTestExecutable $ctestArgs
Set-Content -LiteralPath $preflightCtestStdoutPath -Encoding ascii -Value (Limit-Text $ctestResult.Stdout)
Set-Content -LiteralPath $preflightCtestStderrPath -Encoding ascii -Value (Limit-Text $ctestResult.Stderr)
if ($ctestResult.ExitCode -ne 0) { throw "CTest recurrence correction failed with exit code $($ctestResult.ExitCode)" }
$selfTestArgs = @("--self-test")
Add-Content -LiteralPath $commandsPath -Encoding ascii -Value ("self_test_command `"$Executable`" --self-test")
$selfTestResult = Invoke-CapturedProcess $Executable $selfTestArgs
Set-Content -LiteralPath $preflightSelfTestStdoutPath -Encoding ascii -Value (Limit-Text $selfTestResult.Stdout)
Set-Content -LiteralPath $preflightSelfTestStderrPath -Encoding ascii -Value (Limit-Text $selfTestResult.Stderr)
if ($selfTestResult.ExitCode -ne 0 -or (($selfTestResult.Stdout + "`n" + $selfTestResult.Stderr) -notmatch "(?i)T0-M recurrence correction passed")) {
  throw "Explicit recurrence self-test preflight failed or did not report correction pass"
}

$calibration = @()
foreach ($cpu in $cpus) {
  $calibrationArgs = @("--cpu", "$cpu", "--m", 64, "--K", 64, "--depth", 64,
    "--iterations", 4, "--repetitions", 20, "--warmup", 5, "--kernel", "avx2")
  Add-Content -LiteralPath $commandsPath -Encoding ascii -Value ("calibration cpu=$cpu `"$CalibrationExecutable`" " + ($calibrationArgs -join ' '))
  $calibrationResult = Invoke-CapturedProcess $CalibrationExecutable $calibrationArgs
  if ($calibrationResult.ExitCode -ne 0) { throw "Calibration failed for CPU $cpu" }
  $calibrationLines = @($calibrationResult.Stdout -split '\r?\n' | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
  if ($calibrationLines.Count -ne 2) { throw "Calibration did not produce one CSV row for CPU $cpu" }
  $calibrationRow = $calibrationLines[1] | ConvertFrom-Csv -Header ($calibrationLines[0] -split ',')
  if ([string]$calibrationRow.affinity_succeeded -ne "true") { throw "Calibration affinity failed for CPU $cpu" }
  $calibration += [pscustomobject]@{ logical_cpu_index = $cpu; mac_per_second = [double]$calibrationRow.mac_per_second; elapsed_seconds = [double]$calibrationRow.elapsed_seconds; affinity_succeeded = [string]$calibrationRow.affinity_succeeded }
}
$rows = Get-ProportionalRows $calibration $D
$rowText = $rows -join ','
Set-Content -LiteralPath $calibrationPath -Encoding ascii -Value "logical_cpu_index,mac_per_second,elapsed_seconds,affinity_succeeded"
foreach ($calibrationRow in $calibration) {
  Add-Content -LiteralPath $calibrationPath -Encoding ascii -Value "$($calibrationRow.logical_cpu_index),$($calibrationRow.mac_per_second),$($calibrationRow.elapsed_seconds),$($calibrationRow.affinity_succeeded)"
}
Add-Content -LiteralPath $commandsPath -Encoding ascii -Value "calibrated_rows D=1472 cpus=$cpuText rows_per_worker=$rowText"

for ($run = 1; $run -le $IndependentRuns; $run++) {
  $order = if (($run % 2) -eq 0) { @("A", "B") } else { @("B", "A") }
  $orderIndex = 0
  foreach ($variant in $order) { $orderIndex++; Invoke-Speed $variant $run $orderIndex }
}
if ($script:records.Count -ne 20) { throw "Expected 20 raw rows, got $($script:records.Count)" }

$aggregates = New-Object 'System.Collections.Generic.List[object]'
foreach ($variant in @("A", "B")) {
  $group = @($script:records | Where-Object { $_.variant -eq $variant })
  if ($group.Count -ne 10) { throw "Expected 10 rows for variant $variant, got $($group.Count)" }
  $values = [double[]]@($group | ForEach-Object { $_.mac_per_second })
  $checksums = @($group | ForEach-Object { [string]$_.final_checksum } | Sort-Object -Unique)
  if ($checksums.Count -ne 1) { throw "Nondeterministic checksum for variant $variant" }
  [void]$aggregates.Add([pscustomobject][ordered]@{
    variant = $variant; D = $D; S = $S; R = $R; mode = "fused"; rows_per_worker = $rowText; n = $group.Count
    median_mac_per_second = Get-Median $values; min_mac_per_second = ($values | Measure-Object -Minimum).Minimum
    max_mac_per_second = ($values | Measure-Object -Maximum).Maximum; population_sd_mac_per_second = Get-PopulationSd $values
    checksum = $checksums[0]; checksum_deterministic = $true; all_rounds_valid = (($group | Where-Object { $_.all_rounds_valid -ne "true" }).Count -eq 0)
    affinity = (($group | Where-Object { $_.affinity_succeeded -ne "true" }).Count -eq 0); timed_repetitions = $TimedRepetitions; warmup = $Warmup
  })
}
if ($aggregates.Count -ne 2) { throw "Expected 2 aggregate rows, got $($aggregates.Count)" }
$a = @($aggregates | Where-Object { $_.variant -eq "A" })[0]; $b = @($aggregates | Where-Object { $_.variant -eq "B" })[0]
$bOverA = [double]$b.median_mac_per_second / [double]$a.median_mac_per_second; $aOverB = 1.0 / $bOverA

$t0rSummaryPath = Join-Path $projectRoot "sweep-output\t0r-int8-sharded\t0r_int8_sharded.summary.txt"
$t0rReference = $null
if (Test-Path -LiteralPath $t0rSummaryPath -PathType Leaf) {
  $t0rText = Get-Content -LiteralPath $t0rSummaryPath -Raw
  $match = [regex]::Match($t0rText, "target_kib=512 depth=16 B_over_A=([0-9.eE+-]+)")
  if ($match.Success) { $t0rReference = [double]$match.Groups[1].Value }
}
$staticControlPath = Join-Path $projectRoot "sweep-output\t0m-small-static-control-d512\aggregate.csv"
$staticControlMedian = $null
if (Test-Path -LiteralPath $staticControlPath -PathType Leaf) {
  $staticRows = @(Import-Csv -LiteralPath $staticControlPath | Where-Object { $_.D -eq "512" -and $_.S -eq "1" -and $_.R -eq "16" -and $_.mode -eq "fused" -and $_.variant -eq "A" })
  if ($staticRows.Count -eq 1) { $staticControlMedian = [double]$staticRows[0].median_mac_per_second }
}
$comparison = [pscustomobject][ordered]@{
  record_type = "residency_ab"; campaign = "T0-M recurrence real GEMV+Norm/Requantize/residual+depth barrier"
  D = $D; S = $S; R = $R; mode = "fused"; A_median_mac_per_second = [double]$a.median_mac_per_second; B_median_mac_per_second = [double]$b.median_mac_per_second
  B_over_A = $bOverA; A_over_B = $aOverB; t0r_static_reference_path = $t0rSummaryPath
  t0r_static_reference_B_over_A = if ($null -eq $t0rReference) { "NOT_AVAILABLE" } else { Format-Number $t0rReference }
  t0r_static_reference_A_over_B = if ($null -eq $t0rReference) { "NOT_AVAILABLE" } else { Format-Number (1.0 / $t0rReference) }
  t0r_reference_comparison = if ($null -eq $t0rReference) { "NOT_AVAILABLE" } else { "Descriptive ratio only; T0-R static sharded campaign is not equivalent workload" }
  recent_static_control_path = $staticControlPath
  recent_static_control_A_fused_median_mac_per_second = if ($null -eq $staticControlMedian) { "NOT_AVAILABLE" } else { Format-Number $staticControlMedian }
  recurrence_A_over_recent_static_control = if ($null -eq $staticControlMedian) { "NOT_AVAILABLE" } else { Format-Number ([double]$a.median_mac_per_second / $staticControlMedian) }
  recent_static_control_comparison = "Descriptive only; static T0-M uses fixed activation/state and no Norm/Requantize/residual, so not equivalent"
}

Set-Content -LiteralPath $machinePath -Encoding ascii -Value @($script:records | ConvertTo-Csv -NoTypeInformation)
Set-Content -LiteralPath $aggregatePath -Encoding ascii -Value @($aggregates | ConvertTo-Csv -NoTypeInformation)
Set-Content -LiteralPath $comparisonPath -Encoding ascii -Value @($comparison | ConvertTo-Csv -NoTypeInformation)
$validation = @(
  "status=PASS"
  "raw_rows=$($script:records.Count); expected_raw_rows=20"
  "aggregate_rows=$($aggregates.Count); expected_aggregate_rows=2"
  "variants=A,B; rows_per_variant=10; mode=fused only; no concurrent processes"
  "checksum_determinism=PASS for A and B; nonzero final/per-round checksums"
  "finite_no_overflow=PASS for every raw row; per_round_finite=true; per_round_overflow=false; all_rounds_valid=true"
  "affinity=PASS for every raw row; cpus=0,2,4,6; affinity=1,1,1,1; affinity_errors=0,0,0,0"
  "calibration=PASS; original T0-R per-core calibration on CPUs 0,2,4,6; proportional rows=$rowText; D=1472"
  "repetition_validation=PASS; timed_repetitions=8; warmup=2; timed_repetitions_exact=true"
  "speed_stderr_stdout_correction_autotest_text=PASS absent"
  "preflight=PASS; exactly one CTest correction and one explicit self-test before speed rows"
  "seed_formula=PASS; documented in commands.log and summary.txt"
)
Set-Content -LiteralPath $validationPath -Encoding ascii -Value $validation
$summary = @(
  "status=PASS"
  "executive_summary=T0-M recurrence residency control at D=1472 completed for real GEMV plus Norm/Requantize/residual and depth barrier; A shared weights versus B per-round weights."
  "executable=$Executable"
  "D=$D; S=$S; R=$R; workers=$workers; cpus=$cpuText; rows_per_worker=$rowText; mode=fused only"
  "calibration_path=$calibrationPath; calibration_executable=$CalibrationExecutable; rows are largest-remainder shares of measured mac_per_second"
  "variants=A(shared),B(per-round); timed_repetitions=$TimedRepetitions; warmup=$Warmup; independent_runs=$IndependentRuns"
  "execution_order=even run A then B; odd run B then A; each pair adjacent; never concurrent"
  "A_median_mac_per_second=$(Format-Number ([double]$a.median_mac_per_second)); B_median_mac_per_second=$(Format-Number ([double]$b.median_mac_per_second))"
  "B_over_A=$(Format-Number $bOverA); A_over_B=$(Format-Number $aOverB)"
  "seed_formula=B: 0xC001CAFE ^ 0x9E3779B9*(shard_index+1) ^ 0x85EBCA6B*(round+1); A omits round term and shares one block"
  "counts=raw_rows=20; aggregate_rows=2; speed_processes=20; preflight=one_ctest_plus_one_self_test"
  "static_references=T0-R path=$t0rSummaryPath; recent_static_control_path=$staticControlPath; comparisons descriptive and non-equivalent"
  "artifacts=calibration.csv,machine.csv,aggregate.csv,comparison.csv,summary.txt,validation.txt,commands.log,stderr.log,preflight.ctest.stdout.log,preflight.ctest.stderr.log,preflight.self-test.stdout.log,preflight.self-test.stderr.log"
  "validation=PASS; see validation.txt"
)
Set-Content -LiteralPath $summaryPath -Encoding ascii -Value $summary
Write-Output $summaryPath
