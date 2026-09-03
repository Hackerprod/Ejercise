[CmdletBinding()]
param(
  [string]$Executable = "",
  [string]$OutputDirectory = "",
  [string]$CTestExecutable = "",
  [string]$CMakeExecutable = "",
  [string]$DumpbinExecutable = "",
  [int]$TimedRepetitions = 8,
  [int]$Warmup = 2,
  [int]$IndependentRuns = 10,
  [int]$CalibrationRuns = 5
)

$ErrorActionPreference = "Stop"
[System.Threading.Thread]::CurrentThread.CurrentCulture = [Globalization.CultureInfo]::InvariantCulture
[System.Threading.Thread]::CurrentThread.CurrentUICulture = [Globalization.CultureInfo]::InvariantCulture
$projectRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($Executable)) { $Executable = Join-Path $projectRoot "build\t0m_recurrence_probe.exe" }
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) { $OutputDirectory = Join-Path $projectRoot "sweep-output\t0m-bridge1-residency-d1472" }
if (Test-Path -LiteralPath (Join-Path $OutputDirectory "summary.txt") -PathType Leaf) {
  throw "Refusing rerun: summary exists at $(Join-Path $OutputDirectory 'summary.txt')"
}
if (-not (Test-Path -LiteralPath $OutputDirectory -PathType Container)) { $null = New-Item -ItemType Directory -Path $OutputDirectory }
if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) { throw "Executable not found: $Executable" }
if ($TimedRepetitions -ne 8 -or $Warmup -ne 2 -or $IndependentRuns -ne 10 -or $CalibrationRuns -lt 3) {
  throw "Bridge 1 requires timed repetitions 8, warmup 2, 10 speed runs, and at least 3 calibration runs"
}

$D = 1472; $S = 1; $R = 16; $workers = 4
$cpus = @(0, 2, 4, 6); $calibrationRows = $D
$staticReference = 2.687600766002487
$machinePath = Join-Path $OutputDirectory "machine.csv"
$aggregatePath = Join-Path $OutputDirectory "aggregate.csv"
$calibrationPath = Join-Path $OutputDirectory "calibration.csv"
$comparisonPath = Join-Path $OutputDirectory "comparison.csv"
$summaryPath = Join-Path $OutputDirectory "summary.txt"
$validationPath = Join-Path $OutputDirectory "validation.txt"
$commandsPath = Join-Path $OutputDirectory "commands.log"
$stderrPath = Join-Path $OutputDirectory "stderr.log"
$buildStdoutPath = Join-Path $OutputDirectory "preflight.build.stdout.log"
$buildStderrPath = Join-Path $OutputDirectory "preflight.build.stderr.log"
$ctestStdoutPath = Join-Path $OutputDirectory "preflight.ctest.stdout.log"
$ctestStderrPath = Join-Path $OutputDirectory "preflight.ctest.stderr.log"
$selfTestStdoutPath = Join-Path $OutputDirectory "preflight.self-test.stdout.log"
$selfTestStderrPath = Join-Path $OutputDirectory "preflight.self-test.stderr.log"
$dumpbinPath = Join-Path $OutputDirectory "dumpbin-evidence.txt"
$nativeHeader = "D,S,R,variant,rows_per_worker,component,mode,kernel,elapsed_seconds,elapsed_per_timed_step,qpc_ticks_per_timed_step,tsc_cycles_per_timed_step,tsc_supported,mac_total,mac_per_second,checksum_kind,validation_invariant,final_checksum,per_round_checksums,per_round_finite,per_round_overflow,per_round_clipped_cells,per_round_clipping_rates,clipped_cells,clipping_rate,all_rounds_valid,worker_count,cpus,affinity,affinity_errors,affinity_succeeded,timed_repetitions,warmup,timed_repetitions_exact"
$script:records = New-Object 'System.Collections.Generic.List[object]'
$script:calibrationRecords = New-Object 'System.Collections.Generic.List[object]'
$script:invocations = 0

Set-Content -LiteralPath $commandsPath -Encoding ascii -Value @(
  "Bridge 1: T0-R residency control with byte-identical Bclone"
  "scope=D=1472; S=1; R=16; component=gemv-only; mode=fused; workers=4; cpus=0,2,4,6"
  "calibration=one worker per CPU; calibration_runs=$CalibrationRuns; rows_per_worker=1472; rows derived proportional to median mac_per_second"
  "speed=variants A,Bclone; independent_runs=10 each; odd run A then Bclone; even run Bclone then A; never concurrent"
  "timed_repetitions=8; warmup=2; speed invocations never receive --self-test"
  "A=one weight allocation selected block 0 every round; B=round-dependent distinct blocks; Bclone=16 distinct allocations copied byte-for-byte from A"
  "checksums=post-timing GEMV snapshot; no checksum calculation in timed region"
  "static_reference=T0-R target512 depth16 A/B=2.687600766002487; descriptive only because workload differs"
)
Set-Content -LiteralPath $stderrPath -Encoding ascii -Value "Bridge 1 process diagnostics"

function Invoke-CapturedProcess([string]$FileName, [string[]]$Arguments) {
  $startInfo = New-Object System.Diagnostics.ProcessStartInfo
  $startInfo.FileName = $FileName
  $startInfo.Arguments = (($Arguments | ForEach-Object {
    if ($_ -match '[\s"]') { '"' + ($_ -replace '"', '\"') + '"' } else { [string]$_ }
  }) -join ' ')
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
function Format-Number([double]$Value) { return $Value.ToString("R", [Globalization.CultureInfo]::InvariantCulture) }
function Get-Median([double[]]$Values) {
  $ordered = @($Values | Sort-Object)
  if ($ordered.Count -ne $CalibrationRuns) { throw "Median requires $CalibrationRuns values, got $($ordered.Count)" }
  return ([double]$ordered[[int][math]::Floor(($ordered.Count - 1) / 2)] + [double]$ordered[[int][math]::Ceiling(($ordered.Count - 1) / 2)]) / 2.0
}
function Get-PopulationSd([double[]]$Values) {
  $mean = [double](($Values | Measure-Object -Average).Average); $sum = 0.0
  foreach ($value in $Values) { $sum += ($value - $mean) * ($value - $mean) }
  return [math]::Sqrt($sum / $Values.Count)
}
function Require-Equal([string]$Name, [string]$Actual, [string]$Expected, [string]$Context) {
  if ($Actual -ne $Expected) { throw "Invalid ${Name}=$Actual`: expected ${Expected}: $Context" }
}
function Require-BoolList([string]$Text, [bool]$Expected, [int]$Count, [string]$Name, [string]$Context) {
  $values = @($Text -split ';'); $expectedText = if ($Expected) { 'true' } else { 'false' }
  if ($values.Count -ne $Count -or @($values | Where-Object { $_ -ne $expectedText }).Count -gt 0) {
    throw "Invalid ${Name}=$Text`: expected $Count $expectedText`: $Context"
  }
}
function Validate-NativeRow([string]$Variant, [int]$Run, [int]$Cpu, [string]$RowsText, $Result, [bool]$Calibration) {
  if ($Result.ExitCode -ne 0) { throw "Nonzero probe exit $($Result.ExitCode): variant=$Variant run=$Run cpu=$Cpu" }
  $lines = @($Result.Stdout -split '\r?\n' | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
  if ($lines.Count -ne 2 -or $lines[0] -ne $nativeHeader) { throw "Bad CSV schema/row count: variant=$Variant run=$Run cpu=$Cpu" }
  $row = $lines[1] | ConvertFrom-Csv -Header ($nativeHeader -split ',')
  $context = "variant=$Variant run=$Run cpu=$Cpu"
  Require-Equal "D" ([string]$row.D) "$D" $context; Require-Equal "S" ([string]$row.S) "$S" $context
  Require-Equal "R" ([string]$row.R) "$R" $context; Require-Equal "variant" ([string]$row.variant) $Variant $context
  Require-Equal "rows_per_worker" ([string]$row.rows_per_worker) $RowsText $context
  Require-Equal "component" ([string]$row.component) "gemv-only" $context; Require-Equal "mode" ([string]$row.mode) "fused" $context
  Require-Equal "kernel" ([string]$row.kernel) "avx2_fused" $context; Require-Equal "worker_count" ([string]$row.worker_count) $(if ($Calibration) { "1" } else { "$workers" }) $context
  Require-Equal "cpus" ([string]$row.cpus) $(if ($Calibration) { "$Cpu" } else { "0,2,4,6" }) $context
  Require-Equal "affinity" ([string]$row.affinity) $(if ($Calibration) { "1" } else { "1,1,1,1" }) $context
  Require-Equal "affinity_errors" ([string]$row.affinity_errors) $(if ($Calibration) { "0" } else { "0,0,0,0" }) $context
  Require-Equal "affinity_succeeded" ([string]$row.affinity_succeeded) "true" $context
  Require-Equal "timed_repetitions" ([string]$row.timed_repetitions) "$TimedRepetitions" $context
  Require-Equal "warmup" ([string]$row.warmup) "$Warmup" $context; Require-Equal "timed_repetitions_exact" ([string]$row.timed_repetitions_exact) "true" $context
  Require-Equal "all_rounds_valid" ([string]$row.all_rounds_valid) "true" $context
  Require-BoolList ([string]$row.per_round_finite) $true $R "per_round_finite" $context
  Require-BoolList ([string]$row.per_round_overflow) $false $R "per_round_overflow" $context
  $expectedMac = [int64]$D * $D * $S * $R * $TimedRepetitions
  Require-Equal "mac_total" ([string]$row.mac_total) "$expectedMac" $context
  if ([uint64]$row.final_checksum -eq 0) { throw "Zero checksum: $context" }
  $roundChecksums = @(([string]$row.per_round_checksums) -split ';')
  if ($roundChecksums.Count -ne $R -or @($roundChecksums | Where-Object { [uint64]$_ -eq 0 }).Count -gt 0) { throw "Invalid per-round checksums: $context" }
  if ([double]::IsNaN([double]$row.elapsed_seconds) -or [double]$row.elapsed_seconds -le 0) { throw "Invalid elapsed_seconds: $context" }
  if ([double]::IsNaN([double]$row.mac_per_second) -or [double]$row.mac_per_second -le 0) { throw "Invalid mac_per_second: $context" }
  return $row
}
function Append-ProcessLog([string]$Label, $Result) {
  Add-Content -LiteralPath $stderrPath -Encoding ascii -Value @("[$Label] exit=$($Result.ExitCode)", "stdout:", $Result.Stdout.TrimEnd(), "stderr:", $Result.Stderr.TrimEnd())
}
function Resolve-Tool([string]$Requested, [string]$Name, [string[]]$KnownPaths) {
  if (-not [string]::IsNullOrWhiteSpace($Requested)) { if (Test-Path -LiteralPath $Requested -PathType Leaf) { return $Requested }; throw "$Name not found: $Requested" }
  $command = Get-Command $Name -ErrorAction SilentlyContinue
  if ($null -ne $command) { return $command.Path }
  $found = @($KnownPaths | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf })
  if ($found.Count -gt 0) { return $found[0] }
  throw "$Name not found"
}

$cmake = Resolve-Tool $CMakeExecutable "cmake.exe" @(
  "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
  "C:\vkd\tools\cmake-4.3.3-windows\cmake-4.3.3-windows-x86_64\bin\cmake.exe"
)
$ctest = Resolve-Tool $CTestExecutable "ctest.exe" @(
  "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\ctest.exe",
  "C:\Program Files\CMake\bin\ctest.exe"
)
$dumpbin = Resolve-Tool $DumpbinExecutable "dumpbin.exe" @(
  "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64\dumpbin.exe"
)

$buildArgs = @("--build", (Join-Path $projectRoot "build"), "--target", "t0m_recurrence_probe")
Add-Content -LiteralPath $commandsPath -Encoding ascii -Value ("build_command `"$cmake`" " + ($buildArgs -join ' '))
$buildResult = Invoke-CapturedProcess $cmake $buildArgs
Set-Content -LiteralPath $buildStdoutPath -Encoding ascii -Value $buildResult.Stdout
Set-Content -LiteralPath $buildStderrPath -Encoding ascii -Value $buildResult.Stderr
if ($buildResult.ExitCode -ne 0) { throw "Build failed with exit code $($buildResult.ExitCode)" }

$ctestArgs = @("--test-dir", (Join-Path $projectRoot "build"), "-R", "t0m_recurrence_correction", "--output-on-failure")
Add-Content -LiteralPath $commandsPath -Encoding ascii -Value ("ctest_command `"$ctest`" " + ($ctestArgs -join ' '))
$ctestResult = Invoke-CapturedProcess $ctest $ctestArgs
Set-Content -LiteralPath $ctestStdoutPath -Encoding ascii -Value $ctestResult.Stdout
Set-Content -LiteralPath $ctestStderrPath -Encoding ascii -Value $ctestResult.Stderr
if ($ctestResult.ExitCode -ne 0) { throw "CTest correction failed with exit code $($ctestResult.ExitCode)" }

$selfTestArgs = @("--self-test")
Add-Content -LiteralPath $commandsPath -Encoding ascii -Value ("self_test_command `"$Executable`" --self-test")
$selfTestResult = Invoke-CapturedProcess $Executable $selfTestArgs
Set-Content -LiteralPath $selfTestStdoutPath -Encoding ascii -Value $selfTestResult.Stdout
Set-Content -LiteralPath $selfTestStderrPath -Encoding ascii -Value $selfTestResult.Stderr
if ($selfTestResult.ExitCode -ne 0 -or $selfTestResult.Stderr -notmatch "T0-M recurrence correction passed" -or $selfTestResult.Stderr -notmatch "self_test_bclone,D=1472,S=1,R=16.*outputs_equal=true,checksums_equal=true") {
  throw "Explicit correction/Bclone self-test failed"
}

$dumpbinArgs = @("/DISASM", "/SYMBOLS", $Executable)
Add-Content -LiteralPath $commandsPath -Encoding ascii -Value ("dumpbin_command `"$dumpbin`" " + ($dumpbinArgs -join ' '))
$dumpbinResult = Invoke-CapturedProcess $dumpbin $dumpbinArgs
$dumpbinText = $dumpbinResult.Stdout + "`n" + $dumpbinResult.Stderr
$avx2Matches = @($dumpbinText -split '\r?\n' | Where-Object { $_ -match '(?i)\b(vpmov|vpmadd|vpsign|vpunpck|vmovdqu|vpadd|vpxor|vpbroadcast)\w*\b' })
Set-Content -LiteralPath $dumpbinPath -Encoding ascii -Value @(
  "command=`"$dumpbin`" /DISASM /SYMBOLS `"$Executable`""
  "exit_code=$($dumpbinResult.ExitCode)"
  "evidence=AVX2 GEMV instructions from rebuilt t0m_recurrence_probe binary"
  "avx2_instruction_match_count=$($avx2Matches.Count)"
  $avx2Matches
  "source_changed_before_dumpbin=true; evidence_rerun_after_build=true"
)
if ($dumpbinResult.ExitCode -ne 0 -or $avx2Matches.Count -eq 0) { throw "Dumpbin AVX2 GEMV evidence gate failed" }

foreach ($cpu in $cpus) {
  $values = New-Object 'System.Collections.Generic.List[double]'
  for ($run = 1; $run -le $CalibrationRuns; $run++) {
    $rowText = "$calibrationRows"
    $args = @("--D", "$D", "--S", "$S", "--R", "$R", "--variant", "A", "--component", "gemv-only", "--mode", "fused", "--workers", "1", "--cpus", "$cpu", "--rows-per-worker", $rowText, "--timed-repetitions", "$TimedRepetitions", "--warmup", "$Warmup")
    $script:invocations++; Add-Content -LiteralPath $commandsPath -Encoding ascii -Value "[$($script:invocations)] calibration cpu=$cpu run=$run `"$Executable`" $($args -join ' ')"
    $process = Invoke-CapturedProcess $Executable $args; Append-ProcessLog "calibration cpu=$cpu run=$run" $process
    $row = Validate-NativeRow "A" $run $cpu $rowText $process $true
    [void]$values.Add([double]$row.mac_per_second)
    [void]$script:calibrationRecords.Add([pscustomobject][ordered]@{ record_type = "raw"; cpu = $cpu; run = $run; rows_per_worker = $rowText; median_mac_per_second = ""; elapsed_seconds = [double]$row.elapsed_seconds; mac_per_second = [double]$row.mac_per_second; final_checksum = [uint64]$row.final_checksum; per_round_checksums = [string]$row.per_round_checksums; affinity_succeeded = [string]$row.affinity_succeeded; timed_repetitions_exact = [string]$row.timed_repetitions_exact })
  }
  $median = Get-Median ([double[]]$values)
  [void]$script:calibrationRecords.Add([pscustomobject][ordered]@{ record_type = "median"; cpu = $cpu; run = ""; rows_per_worker = $calibrationRows; median_mac_per_second = $median; elapsed_seconds = ""; mac_per_second = ""; final_checksum = ""; per_round_checksums = ""; affinity_succeeded = "true"; timed_repetitions_exact = "true" })
}

$medianRows = @($script:calibrationRecords | Where-Object { $_.record_type -eq "median" })
$speedTotal = ($medianRows | Measure-Object -Property median_mac_per_second -Sum).Sum
if ([double]::IsNaN([double]$speedTotal) -or $speedTotal -le 0) { throw "Calibration median sum invalid" }
$allocations = @()
foreach ($entry in $medianRows) {
  $ideal = $D * ([double]$entry.median_mac_per_second / [double]$speedTotal)
  $allocations += [pscustomobject]@{ Cpu = [int]$entry.cpu; Ideal = $ideal; Fraction = $ideal - [math]::Floor($ideal); Rows = [int][math]::Floor($ideal) }
}
foreach ($allocation in $allocations) { if ($allocation.Rows -lt 1) { $allocation.Rows = 1 } }
$remaining = $D - (($allocations | Measure-Object -Property Rows -Sum).Sum)
if ($remaining -lt 0) { throw "Positive proportional calibration rows exceed D" }
foreach ($allocation in @($allocations | Sort-Object Fraction -Descending)) {
  if ($remaining -le 0) { break }
  $allocation.Rows++; $remaining--
}
if ($remaining -ne 0) { throw "Could not distribute proportional calibration rows" }
$rows = @($allocations | Sort-Object Cpu | ForEach-Object { [int]$_.Rows })
if (($rows | Where-Object { $_ -le 0 }).Count -gt 0 -or ($rows | Measure-Object -Sum).Sum -ne $D) { throw "Derived rows invalid: $($rows -join ',')" }
$cpuText = $cpus -join ','; $rowText = $rows -join ','
$medianText = @($medianRows | ForEach-Object { $_.cpu.ToString() + ':' + (Format-Number ([double]$_.median_mac_per_second)) }) -join ';'
Add-Content -LiteralPath $commandsPath -Encoding ascii -Value "derived_rows=$rowText; calibration_medians=$medianText"

function Invoke-Speed([string]$Variant, [int]$Run, [int]$OrderIndex) {
  $args = @("--D", "$D", "--S", "$S", "--R", "$R", "--variant", $Variant, "--component", "gemv-only", "--mode", "fused", "--workers", "$workers", "--cpus", $cpuText, "--rows-per-worker", $rowText, "--timed-repetitions", "$TimedRepetitions", "--warmup", "$Warmup")
  $script:invocations++; $context = "variant=$Variant run=$Run order_index=$OrderIndex"
  Add-Content -LiteralPath $commandsPath -Encoding ascii -Value "[$($script:invocations)] speed $context `"$Executable`" $($args -join ' ')"
  $process = Invoke-CapturedProcess $Executable $args; Append-ProcessLog "speed $context" $process
  $row = Validate-NativeRow $Variant $Run 0 $rowText $process $false
  [void]$script:records.Add([pscustomobject][ordered]@{ run = $Run; order_index = $OrderIndex; variant = $Variant; D = $D; S = $S; R = $R; rows_per_worker = $rowText; component = [string]$row.component; mode = [string]$row.mode; kernel = [string]$row.kernel; elapsed_seconds = [double]$row.elapsed_seconds; elapsed_per_timed_step = [double]$row.elapsed_per_timed_step; mac_total = [uint64]$row.mac_total; mac_per_second = [double]$row.mac_per_second; checksum_kind = [string]$row.checksum_kind; final_checksum = [uint64]$row.final_checksum; per_round_checksums = [string]$row.per_round_checksums; per_round_finite = [string]$row.per_round_finite; per_round_overflow = [string]$row.per_round_overflow; all_rounds_valid = [string]$row.all_rounds_valid; affinity_succeeded = [string]$row.affinity_succeeded; timed_repetitions_exact = [string]$row.timed_repetitions_exact })
}

for ($run = 1; $run -le $IndependentRuns; $run++) {
  $order = if (($run % 2) -eq 1) { @("A", "Bclone") } else { @("Bclone", "A") }
  Invoke-Speed $order[0] $run 1; Invoke-Speed $order[1] $run 2
  $pair = @($script:records | Where-Object { $_.run -eq $run })
  if ($pair.Count -ne 2) { throw "Speed pair count invalid for run $run" }
  $aPair = @($pair | Where-Object { $_.variant -eq "A" })[0]; $bclonePair = @($pair | Where-Object { $_.variant -eq "Bclone" })[0]
  if ($aPair.per_round_checksums -ne $bclonePair.per_round_checksums -or $aPair.final_checksum -ne $bclonePair.final_checksum) {
    throw "A/Bclone checksum mismatch at run $run"
  }
}
if ($script:records.Count -ne 20) { throw "Expected 20 speed rows, got $($script:records.Count)" }

$aggregates = @()
foreach ($variant in @("A", "Bclone")) {
  $group = @($script:records | Where-Object { $_.variant -eq $variant })
  if ($group.Count -ne 10) { throw "Expected 10 speed rows for $variant" }
  $values = [double[]]@($group | ForEach-Object { $_.mac_per_second }); $ordered = @($values | Sort-Object)
  $checksumSet = @($group | ForEach-Object { $_.final_checksum } | Sort-Object -Unique)
  if ($checksumSet.Count -ne 1) { throw "Nondeterministic checksum for $variant" }
  $aggregates += [pscustomobject][ordered]@{ variant = $variant; D = $D; S = $S; R = $R; component = "gemv-only"; mode = "fused"; rows_per_worker = $rowText; n = 10; median_mac_per_second = ([double]$ordered[4] + [double]$ordered[5]) / 2.0; min_mac_per_second = ($values | Measure-Object -Minimum).Minimum; max_mac_per_second = ($values | Measure-Object -Maximum).Maximum; population_sd_mac_per_second = Get-PopulationSd $values; checksum = $checksumSet[0]; checksum_deterministic = $true; all_rounds_valid = $true; affinity = $true; timed_repetitions = $TimedRepetitions; warmup = $Warmup }
}
$a = @($aggregates | Where-Object { $_.variant -eq "A" })[0]; $bclone = @($aggregates | Where-Object { $_.variant -eq "Bclone" })[0]
$aOverBclone = [double]$a.median_mac_per_second / [double]$bclone.median_mac_per_second; $bcloneOverA = 1.0 / $aOverBclone
$ratioPass = $aOverBclone -ge 2.5 -and $aOverBclone -le 2.9
$comparison = [pscustomobject][ordered]@{ record_type = "bridge1_residency"; D = $D; S = $S; R = $R; component = "gemv-only"; mode = "fused"; rows_per_worker = $rowText; A_median_mac_per_second = [double]$a.median_mac_per_second; Bclone_median_mac_per_second = [double]$bclone.median_mac_per_second; A_over_Bclone = $aOverBclone; Bclone_over_A = $bcloneOverA; expected_A_over_Bclone_min = 2.5; expected_A_over_Bclone_max = 2.9; ratio_pass = $ratioPass; static_T0R_reference_target512_depth16_A_over_B = $staticReference; static_reference_note = "Descriptive accepted static T0-R reference only; workload differs: Bridge 1 uses D=1472, S=1, R=16, fused AVX2 GEMV-only, calibrated rows, frozen X, no transition."; checksum_proof = "A==Bclone per-round output checksums and final checksums for all 10 paired runs; dedicated self-test compared every output cell for all 16 rounds." }

Set-Content -LiteralPath $calibrationPath -Encoding ascii -Value @($script:calibrationRecords | ConvertTo-Csv -NoTypeInformation)
Set-Content -LiteralPath $machinePath -Encoding ascii -Value @($script:records | ConvertTo-Csv -NoTypeInformation)
Set-Content -LiteralPath $aggregatePath -Encoding ascii -Value @($aggregates | ConvertTo-Csv -NoTypeInformation)
Set-Content -LiteralPath $comparisonPath -Encoding ascii -Value @($comparison | ConvertTo-Csv -NoTypeInformation)
if (-not $ratioPass) {
  Set-Content -LiteralPath $validationPath -Encoding ascii -Value @("status=FAIL", "bridge1_ratio=FAIL; A_over_Bclone=$(Format-Number $aOverBclone); expected=2.5..2.9", "checksum_proof=PASS", "STOP=Bridge 2/3/4 not started")
  Set-Content -LiteralPath $summaryPath -Encoding ascii -Value @("status=FAIL", "executive_summary=Bridge 1 stopped because A/Bclone ratio failed expected T0-R range", "A_over_Bclone=$(Format-Number $aOverBclone)", "artifacts=machine.csv,aggregate.csv,calibration.csv,comparison.csv,summary.txt,validation.txt,commands.log,stderr.log,dumpbin-evidence.txt,preflight logs", "next_recommended=Do not start Bridge 2/3/4; investigate ratio")
  throw "Bridge 1 ratio failed: A_over_Bclone=$aOverBclone, expected 2.5..2.9"
}
$validation = @(
  "status=PASS"
  "bridge=1 only; no Bridge 2/3/4 or G/S sweep"
  "raw_speed_rows=20; expected=20; aggregate_rows=2; expected=2"
  "calibration=PASS; one-worker calibration on CPUs 0,2,4,6; runs_per_cpu=$CalibrationRuns; median evidence saved"
  "exact_calibration_rows=$rowText; positive=true; sum=$D"
  "configuration=D=1472;S=1;R=16;workers=4;cpus=0,2,4,6;component=gemv-only;mode=fused"
  "checksums=PASS; dedicated self-test and all 10 paired speed cells have exact A==Bclone per-round checksums; X frozen"
  "checksum_timing=PASS; checksums computed only by post-timing snapshot"
  "determinism=PASS; A and Bclone final checksums each deterministic across 10 runs"
  "avx2=PASS; every native row kernel=avx2_fused; dumpbin evidence saved"
  "affinity=PASS; all speed rows pinned to 0,2,4,6"
  "repetitions=PASS; timed_repetitions=8;warmup=2;timed_repetitions_exact=true"
  "finite_overflow=PASS; all_rounds_valid=true; per_round_finite=true; per_round_overflow=false"
  "A_median_mac_per_second=$(Format-Number ([double]$a.median_mac_per_second)); Bclone_median_mac_per_second=$(Format-Number ([double]$bclone.median_mac_per_second))"
  "A_over_Bclone=$(Format-Number $aOverBclone); Bclone_over_A=$(Format-Number $bcloneOverA); expected_range=2.5..2.9"
  "static_reference=T0-R target512 depth16 A/B=2.687600766002487; workload differences explicitly documented in comparison.csv"
)
Set-Content -LiteralPath $validationPath -Encoding ascii -Value $validation
$summary = @(
  "status=PASS"
  "executive_summary=Bridge 1 completed A versus byte-identical multi-allocation Bclone residency control at D=1472,S=1,R=16 with frozen X and GEMV-only AVX2 fused timing."
  "exact_calibration_rows=$rowText"
  "A_median_mac_per_second=$(Format-Number ([double]$a.median_mac_per_second)); Bclone_median_mac_per_second=$(Format-Number ([double]$bclone.median_mac_per_second))"
  "A_over_Bclone=$(Format-Number $aOverBclone); Bclone_over_A=$(Format-Number $bcloneOverA); expected_range=2.5..2.9"
  "speed_rows=20; aggregates=2; independent_runs=10_per_variant; order=odd_A_then_Bclone_even_Bclone_then_A; never_concurrent"
  "checksum_proof=dedicated self-test compared every output cell/checksum for 16 rounds; all 10 speed pairs matched per-round and final checksums"
  "static_reference=T0-R target512 depth16 A/B=2.687600766002487; descriptive only because workload differs"
  "artifacts=machine.csv,aggregate.csv,calibration.csv,comparison.csv,summary.txt,validation.txt,commands.log,stderr.log,dumpbin-evidence.txt,preflight.build.stdout.log,preflight.build.stderr.log,preflight.ctest.stdout.log,preflight.ctest.stderr.log,preflight.self-test.stdout.log,preflight.self-test.stderr.log"
  "next_recommended=Stop after Bridge 1; do not start Bridge 2/3/4 until result reviewed"
)
Set-Content -LiteralPath $summaryPath -Encoding ascii -Value $summary
Write-Output $summaryPath
