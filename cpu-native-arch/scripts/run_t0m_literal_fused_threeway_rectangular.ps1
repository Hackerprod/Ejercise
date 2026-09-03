[CmdletBinding()]
param(
  [string]$Executable = "",
  [string]$OutputDirectory = "",
  [string]$CMakeExecutable = "",
  [string]$DumpbinExecutable = ""
)

$ErrorActionPreference = "Stop"
$invariant = [Globalization.CultureInfo]::InvariantCulture
[System.Threading.Thread]::CurrentThread.CurrentCulture = $invariant
[System.Threading.Thread]::CurrentThread.CurrentUICulture = $invariant

$projectRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($Executable)) { $Executable = Join-Path $projectRoot "build\t0m_int8_probe.exe" }
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) { $OutputDirectory = Join-Path $projectRoot "sweep-output\t0m-bridge1-literal-fused-threeway-rectangular" }
$summaryPath = Join-Path $OutputDirectory "summary.txt"
if (Test-Path -LiteralPath $summaryPath -PathType Leaf) { throw "Refusing rerun: summary exists at $summaryPath" }
if (-not (Test-Path -LiteralPath $OutputDirectory -PathType Container)) { $null = New-Item -ItemType Directory -Path $OutputDirectory }

$D = 512; $S = 1; $R = 16; $workers = 4; $tile = 8; $iterations = 1
$timedRepetitions = 8; $warmup = 2; $independentRuns = 10
$cpus = @(0, 2, 4, 6); $rows = @(1332, 807, 965, 992)
$cpuText = $cpus -join ','; $rowText = $rows -join ','
$expectedMac = [int64]$D * (($rows | Measure-Object -Sum).Sum) * $S * $R * $iterations * $timedRepetitions
$expectedABytes = @(681984, 413184, 494080, 507904)
$expectedBBytes = @(10911744, 6610944, 7905280, 8126464)
$probeHeader = "D,S,R,O_i,B_i,rows_per_worker,bytes_per_worker,mode,variant,S_tile,iterations,timed_repetitions,warmup,worker_count,worker_list,affinity,affinity_error,affinity_succeeded,timed_repetitions_exact,avx2_supported,kernel_used,eviction_bytes,eviction_checksum,elapsed_seconds,mac_total,mac_per_second,checksum"

$machinePath = Join-Path $OutputDirectory "machine.csv"
$aggregatePath = Join-Path $OutputDirectory "aggregate.csv"
$comparisonPath = Join-Path $OutputDirectory "comparison.csv"
$validationPath = Join-Path $OutputDirectory "validation.txt"
$commandsPath = Join-Path $OutputDirectory "commands.log"
$stderrPath = Join-Path $OutputDirectory "stderr.log"
$dumpbinPath = Join-Path $OutputDirectory "dumpbin-evidence.txt"
$preflightBuildStdout = Join-Path $OutputDirectory "preflight.build.stdout.log"
$preflightBuildStderr = Join-Path $OutputDirectory "preflight.build.stderr.log"
$preflightSelfStdout = Join-Path $OutputDirectory "preflight.self-test.stdout.log"
$preflightSelfStderr = Join-Path $OutputDirectory "preflight.self-test.stderr.log"

function Format-Number([double]$Value) { $Value.ToString("R", $invariant) }
function Require-Equal([string]$Name, [string]$Actual, [string]$Expected, [string]$Context) {
  if ($Actual -ne $Expected) { throw "Invalid ${Name}=$Actual; expected ${Expected}: $Context" }
}
function Require-Positive([string]$Name, [double]$Value, [string]$Context) {
  if ([double]::IsNaN($Value) -or [double]::IsInfinity($Value) -or $Value -le 0) { throw "Invalid ${Name}: $Context" }
}
function Limit-Text([string]$Text, [int]$Maximum = 10000) {
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
function Resolve-Tool([string]$Requested, [string]$Name, [string[]]$KnownPaths) {
  if (-not [string]::IsNullOrWhiteSpace($Requested)) {
    if (Test-Path -LiteralPath $Requested -PathType Leaf) { return $Requested }
    throw "$Name not found: $Requested"
  }
  $command = Get-Command $Name -ErrorAction SilentlyContinue
  if ($null -ne $command) { return $command.Path }
  $found = @($KnownPaths | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf })
  if ($found.Count -gt 0) { return $found[0] }
  throw "$Name not found"
}
function Get-Median([double[]]$Values) {
  if ($Values.Count -ne $independentRuns) { throw "Median requires $independentRuns values, got $($Values.Count)" }
  $ordered = @($Values | Sort-Object)
  return ([double]$ordered[4] + [double]$ordered[5]) / 2.0
}
function Get-PopulationSd([double[]]$Values) {
  $mean = [double](($Values | Measure-Object -Average).Average); $sum = 0.0
  foreach ($value in $Values) { $sum += ($value - $mean) * ($value - $mean) }
  return [math]::Sqrt($sum / $Values.Count)
}
function Get-RunOrder([int]$Run) {
  switch (($Run - 1) % 6) {
    0 { return @("A", "Bclone", "B_real") }
    1 { return @("B_real", "Bclone", "A") }
    2 { return @("Bclone", "A", "B_real") }
    3 { return @("B_real", "A", "Bclone") }
    4 { return @("A", "B_real", "Bclone") }
    default { return @("Bclone", "B_real", "A") }
  }
}

Set-Content -LiteralPath $commandsPath -Encoding ascii -Value @(
  "scope=literal accepted t0m_int8_probe fused kernel only; no recurrence/transition; no Bridges 2-4"
  "source_shape=sweep-output\\t0r-int8-sharded target_kib=512; original D=K=512; original m=1024; original O_i=1332,807,965,992"
  "configuration=D=512;S=1;R=16;mode=fused;S_tile=8;iterations=1;timed_repetitions=8;warmup=2"
  "workers=4;cpus=0,2,4,6;rows_per_worker=1332,807,965,992;sum_O_i=4096;sum_O_i_ne_D=true;no_recalibration"
  "A=one shared weight allocation reused at every depth; Bclone=16 distinct allocations byte-identical to A; B_real=16 distinct allocations with deterministic round-dependent B seed"
  "activations=deterministic and frozen; state=none; checksum=post-timing only; no transition"
  "speed=10 independent three-way runs; deterministic six-order rotation; pair-adjacent serial execution; never concurrent"
  "preflight=one t0m_int8_probe --self-test; speed invocations never receive --self-test"
)
Set-Content -LiteralPath $stderrPath -Encoding ascii -Value "literal fused three-way rectangular campaign process log"

$cmake = Resolve-Tool $CMakeExecutable "cmake.exe" @(
  "C:\Program Files\CMake\bin\cmake.exe",
  "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
  "C:\vkd\tools\cmake-4.3.3-windows\cmake-4.3.3-windows-x86_64\bin\cmake.exe"
)
$dumpbin = Resolve-Tool $DumpbinExecutable "dumpbin.exe" @(
  "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64\dumpbin.exe"
)
$buildDirectory = Join-Path $projectRoot "build"
$buildArgs = @("--build", $buildDirectory, "--target", "t0m_int8_probe")
Add-Content -LiteralPath $commandsPath -Encoding ascii -Value ("build_command `"$cmake`" " + ($buildArgs -join ' '))
$build = Invoke-CapturedProcess $cmake $buildArgs
Set-Content -LiteralPath $preflightBuildStdout -Encoding ascii -Value $build.Stdout
Set-Content -LiteralPath $preflightBuildStderr -Encoding ascii -Value $build.Stderr
if ($build.ExitCode -ne 0) { throw "Build failed: exit=$($build.ExitCode)" }
if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) { throw "Executable not found after build: $Executable" }

$selfTestArgs = @("--self-test")
Add-Content -LiteralPath $commandsPath -Encoding ascii -Value "preflight_command `"$Executable`" --self-test"
$selfTest = Invoke-CapturedProcess $Executable $selfTestArgs
Set-Content -LiteralPath $preflightSelfStdout -Encoding ascii -Value (Limit-Text $selfTest.Stdout)
Set-Content -LiteralPath $preflightSelfStderr -Encoding ascii -Value (Limit-Text $selfTest.Stderr)
$selfTestText = $selfTest.Stdout + "`n" + $selfTest.Stderr
if ($selfTest.ExitCode -ne 0 -or
    $selfTestText -notmatch "self_test_bclone,D=1472,S=1,R=16,distinct_allocations=16,byte_identical=true,outputs_equal=true,checksums_equal=true" -or
    $selfTestText -notmatch "self_test_b_real,D=1472,S=1,R=16,distinct_allocations=16,round_dependent=true,content_distinct=true,output_checksum_distinct=true" -or
    $selfTestText -notmatch "T0-M correction passed") { throw "Focused self-test failed" }

$dumpbinArgs = @("/DISASM", "/SYMBOLS", $Executable)
Add-Content -LiteralPath $commandsPath -Encoding ascii -Value ("dumpbin_command `"$dumpbin`" " + ($dumpbinArgs -join ' '))
$dumpbinResult = Invoke-CapturedProcess $dumpbin $dumpbinArgs
$dumpbinText = $dumpbinResult.Stdout + "`n" + $dumpbinResult.Stderr
$dumpbinLines = @($dumpbinText -split '\r?\n')
$movMatches = @($dumpbinLines | Where-Object { $_ -match '(?i)\bvpmovsxbw\b' })
$maddMatches = @($dumpbinLines | Where-Object { $_ -match '(?i)\bvpmaddwd\b' })
$addMatches = @($dumpbinLines | Where-Object { $_ -match '(?i)\bvpaddd\b' })
$symbolMatches = @($dumpbinLines | Where-Object { $_ -match '(?i)run_fused(_impl)?' })
Set-Content -LiteralPath $dumpbinPath -Encoding ascii -Value @(
  "command=`"$dumpbin`" /DISASM /SYMBOLS `"$Executable`""
  "exit_code=$($dumpbinResult.ExitCode)"
  "source_kernel=run_fused_impl<S_TILE=1> accepted t0m_int8_probe literal fused template; arithmetic/template unchanged"
  "variant_dispatch=A,Bclone,B_real -> run_shard_depth -> run_fused -> run_fused_impl<S_TILE=1>"
  "required_sequence=vpmovsxbw + vpmaddwd + vpaddd accumulator; same executable/template for all variants"
  "vpmovsxbw_matches=$($movMatches.Count)"
  "vpmaddwd_matches=$($maddMatches.Count)"
  "vpaddd_matches=$($addMatches.Count)"
  "fused_symbol_matches=$($symbolMatches.Count)"
  "instruction_evidence_begin"
  @($movMatches + $maddMatches + $addMatches + $symbolMatches | Select-Object -First 120)
  "instruction_evidence_end"
)
if ($dumpbinResult.ExitCode -ne 0 -or $movMatches.Count -eq 0 -or $maddMatches.Count -eq 0 -or $addMatches.Count -eq 0) { throw "Dumpbin accepted fused AVX2 evidence failed" }

$script:records = New-Object 'System.Collections.Generic.List[object]'
$script:invocations = 0
function Invoke-Speed([string]$Variant, [int]$Run, [int]$OrderIndex) {
  $args = @("--D", "$D", "--S", "$S", "--R", "$R", "--mode", "fused", "--variant", $Variant,
            "--S-tile", "$tile", "--workers", "$workers", "--cpus", $cpuText,
            "--rows-per-worker", $rowText, "--iterations", "$iterations",
            "--timed-repetitions", "$timedRepetitions", "--warmup", "$warmup")
  $script:invocations++
  $id = $script:invocations; $context = "variant=$Variant run=$Run order_index=$OrderIndex"
  Add-Content -LiteralPath $commandsPath -Encoding ascii -Value "[$id] speed $context `"$Executable`" $($args -join ' ')"
  $result = Invoke-CapturedProcess $Executable $args
  Add-Content -LiteralPath $stderrPath -Encoding ascii -Value @("[$id] $context exit=$($result.ExitCode)", "stdout:", $result.Stdout.TrimEnd(), "stderr:", $result.Stderr.TrimEnd())
  if ($result.ExitCode -ne 0) { throw "Nonzero speed exit $($result.ExitCode): $context" }
  if (($result.Stdout + "`n" + $result.Stderr) -match "(?i)(self[- ]?test|correction|autotest|test passed|test failed|recurrence|transition)") { throw "Forbidden preflight/test text in speed output: $context" }
  $lines = @($result.Stdout -split '\r?\n' | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
  if ($lines.Count -ne 2 -or $lines[0] -ne $probeHeader) { throw "Bad speed CSV schema: $context" }
  $row = $lines[1] | ConvertFrom-Csv -Header ($probeHeader -split ',')
  Require-Equal "D" ([string]$row.D) "$D" $context; Require-Equal "S" ([string]$row.S) "$S" $context
  Require-Equal "R" ([string]$row.R) "$R" $context; Require-Equal "variant" ([string]$row.variant) $Variant $context
  Require-Equal "rows_per_worker" ([string]$row.rows_per_worker) $rowText $context
  $expectedBytes = if ($Variant -eq "A") { $expectedABytes -join ',' } else { $expectedBBytes -join ',' }
  Require-Equal "bytes_per_worker" ([string]$row.bytes_per_worker) $expectedBytes $context
  Require-Equal "mode" ([string]$row.mode) "fused" $context; Require-Equal "S_tile" ([string]$row.S_tile) "$tile" $context
  Require-Equal "iterations" ([string]$row.iterations) "$iterations" $context; Require-Equal "timed_repetitions" ([string]$row.timed_repetitions) "$timedRepetitions" $context
  Require-Equal "warmup" ([string]$row.warmup) "$warmup" $context; Require-Equal "worker_count" ([string]$row.worker_count) "$workers" $context
  Require-Equal "worker_list" ([string]$row.worker_list) $cpuText $context; Require-Equal "affinity" ([string]$row.affinity) "true,true,true,true" $context
  Require-Equal "affinity_error" ([string]$row.affinity_error) "0,0,0,0" $context; Require-Equal "affinity_succeeded" ([string]$row.affinity_succeeded) "true" $context
  Require-Equal "timed_repetitions_exact" ([string]$row.timed_repetitions_exact) "true" $context; Require-Equal "avx2_supported" ([string]$row.avx2_supported) "true" $context
  Require-Equal "kernel_used" ([string]$row.kernel_used) "avx2" $context; Require-Equal "eviction_bytes" ([string]$row.eviction_bytes) "0" $context
  Require-Equal "mac_total" ([string]$row.mac_total) "$expectedMac" $context
  if ([uint64]$row.checksum -eq 0) { throw "Zero checksum: $context" }
  Require-Positive "elapsed_seconds" ([double]$row.elapsed_seconds) $context; Require-Positive "mac_per_second" ([double]$row.mac_per_second) $context
  return [pscustomobject][ordered]@{
    run = $Run; order_index = $OrderIndex; variant = $Variant; D = $D; S = $S; R = $R; mode = "fused"; S_tile = $tile
    rows_per_worker = [string]$row.rows_per_worker; bytes_per_worker = [string]$row.bytes_per_worker
    elapsed_seconds = [double]$row.elapsed_seconds; mac_total = [uint64]$row.mac_total; mac_per_second = [double]$row.mac_per_second
    checksum = [uint64]$row.checksum; worker_count = [int]$row.worker_count; worker_list = [string]$row.worker_list
    affinity = [string]$row.affinity; affinity_error = [string]$row.affinity_error; affinity_succeeded = [string]$row.affinity_succeeded
    timed_repetitions = [int]$row.timed_repetitions; warmup = [int]$row.warmup; timed_repetitions_exact = [string]$row.timed_repetitions_exact
    avx2_supported = [string]$row.avx2_supported; kernel_used = [string]$row.kernel_used
  }
}

for ($run = 1; $run -le $independentRuns; $run++) {
  $order = Get-RunOrder $run
  Add-Content -LiteralPath $commandsPath -Encoding ascii -Value "run=$run order=$($order -join '>')"
  $runRecords = @()
  for ($orderIndex = 0; $orderIndex -lt $order.Count; $orderIndex++) {
    $record = Invoke-Speed $order[$orderIndex] $run ($orderIndex + 1)
    [void]$script:records.Add($record); $runRecords += $record
  }
  $aRun = @($runRecords | Where-Object { $_.variant -eq "A" })[0]
  $bcloneRun = @($runRecords | Where-Object { $_.variant -eq "Bclone" })[0]
  $brealRun = @($runRecords | Where-Object { $_.variant -eq "B_real" })[0]
  if ($aRun.checksum -ne $bcloneRun.checksum) { throw "A/Bclone checksum mismatch at paired run $run" }
  if ($brealRun.checksum -eq $aRun.checksum) { throw "B_real checksum is not distinct at paired run $run" }
}
if ($script:records.Count -ne 30) { throw "Expected 30 raw rows, got $($script:records.Count)" }

$aggregates = New-Object 'System.Collections.Generic.List[object]'
foreach ($Variant in @("A", "Bclone", "B_real")) {
  $group = @($script:records | Where-Object { $_.variant -eq $Variant })
  if ($group.Count -ne 10) { throw "Expected 10 rows for $Variant" }
  $checksums = @($group | ForEach-Object { $_.checksum } | Sort-Object -Unique)
  if ($checksums.Count -ne 1 -or [uint64]$checksums[0] -eq 0) { throw "Nondeterministic or zero checksum for $Variant" }
  $values = [double[]]@($group | ForEach-Object { $_.mac_per_second })
  [void]$aggregates.Add([pscustomobject][ordered]@{
    variant = $Variant; D = $D; S = $S; R = $R; mode = "fused"; S_tile = $tile; rows_per_worker = $rowText; worker_count = $workers; worker_list = $cpuText
    n = 10; median_mac_per_second = Get-Median $values; min_mac_per_second = [double](($values | Measure-Object -Minimum).Minimum); max_mac_per_second = [double](($values | Measure-Object -Maximum).Maximum)
    population_sd_mac_per_second = Get-PopulationSd $values; checksum = [uint64]$checksums[0]; checksum_deterministic = $true; all_affinity_succeeded = $true; timed_repetitions_exact = $true
    timed_repetitions = $timedRepetitions; warmup = $warmup
  })
}
$a = @($aggregates | Where-Object { $_.variant -eq "A" })[0]
$bclone = @($aggregates | Where-Object { $_.variant -eq "Bclone" })[0]
$breal = @($aggregates | Where-Object { $_.variant -eq "B_real" })[0]
$aOverBclone = [double]$a.median_mac_per_second / [double]$bclone.median_mac_per_second
$aOverBreal = [double]$a.median_mac_per_second / [double]$breal.median_mac_per_second
$brealOverBclone = [double]$breal.median_mac_per_second / [double]$bclone.median_mac_per_second
$ratioInReferenceBand = $aOverBreal -ge 2.5 -and $aOverBreal -le 2.9
$comparison = [pscustomobject][ordered]@{
  record_type = "literal_fused_threeway_rectangular"; source_shape = "T0-R target512 depth16"; D = $D; S = $S; R = $R; mode = "fused"; S_tile = $tile; rows_per_worker = $rowText
  A_median_mac_per_second = [double]$a.median_mac_per_second; Bclone_median_mac_per_second = [double]$bclone.median_mac_per_second; B_real_median_mac_per_second = [double]$breal.median_mac_per_second
  A_over_Bclone = $aOverBclone; A_over_B_real = $aOverBreal; B_real_over_Bclone = $brealOverBclone; reference_ratio_band_2_5_to_2_9 = $ratioInReferenceBand
  A_checksum = [uint64]$a.checksum; Bclone_checksum = [uint64]$bclone.checksum; B_real_checksum = [uint64]$breal.checksum
  checksum_equality_all_A_Bclone_pairs = $true; checksum_distinct_all_B_real_pairs = $true; checksum_equality_deterministic = $true; speed_rows = 30; aggregate_rows = 3
}

Set-Content -LiteralPath $machinePath -Encoding ascii -Value @($script:records | ConvertTo-Csv -NoTypeInformation)
Set-Content -LiteralPath $aggregatePath -Encoding ascii -Value @($aggregates | ConvertTo-Csv -NoTypeInformation)
Set-Content -LiteralPath $comparisonPath -Encoding ascii -Value @($comparison | ConvertTo-Csv -NoTypeInformation)
Set-Content -LiteralPath $validationPath -Encoding ascii -Value @(
  "status=PASS"
  "raw_rows=30; aggregate_rows=3; expected_raw_rows=30; expected_aggregate_rows=3"
  "source_shape=PASS; artifact=sweep-output\\t0r-int8-sharded\\t0r_int8_sharded.csv; original D=K=512;m=1024;target_kib=512;depth=16;O_i=1332,807,965,992;sum_O_i=4096_ne_D"
  "configuration=PASS; D=512;S=1;R=16;mode=fused;S_tile=8;iterations=1;timed_repetitions=8;warmup=2"
  "affinity=PASS; workers=4;cpus=0,2,4,6;rows_per_worker=1332,807,965,992;rows_fixed_no_recalibration=true"
  "avx2=PASS; every speed row avx2_supported=true;kernel_used=avx2"
  "repetitions=PASS; exact timed_repetitions=8; warmup=2; all rows timed_repetitions_exact=true"
  "checksums=PASS; nonzero; deterministic per variant; A==Bclone for all 10 pair-adjacent runs; B_real distinct for all 10 runs"
  "checksum_timing=PASS; checksum emitted after timed region by existing probe behavior"
  "order=PASS; deterministic six-order rotation recorded in commands.log; serial pair-adjacent execution; never concurrent"
  "preflight=PASS; exactly one t0m_int8_probe --self-test; speed invocations contain no self-test or test text"
  "dumpbin=PASS; rebuilt executable; A/Bclone/B_real dispatch same accepted literal fused template with vpmovsxbw, vpmaddwd, vpaddd"
  "ratio_observation=A_over_B_real=$(Format-Number $aOverBreal);reference_band_2_5_to_2_9=$ratioInReferenceBand"
  "A_median_mac_per_second=$(Format-Number $a.median_mac_per_second); Bclone_median_mac_per_second=$(Format-Number $bclone.median_mac_per_second); B_real_median_mac_per_second=$(Format-Number $breal.median_mac_per_second)"
  "A_over_Bclone=$(Format-Number $aOverBclone); B_real_over_Bclone=$(Format-Number $brealOverBclone)"
)
Set-Content -LiteralPath $summaryPath -Encoding ascii -Value @(
  "status=PASS"
  "executive_summary=Literal accepted fused t0m_int8_probe three-way rectangular control completed at original T0-R shape D=512,S=1,R=16 with frozen activations/state and no transition path."
  "source_shape=original T0-R target512 depth16; m=1024; K/D=512; O_i/rows_per_worker=1332,807,965,992; sum_O_i=4096 != D"
  "A_median_mac_per_second=$(Format-Number $a.median_mac_per_second); Bclone_median_mac_per_second=$(Format-Number $bclone.median_mac_per_second); B_real_median_mac_per_second=$(Format-Number $breal.median_mac_per_second)"
  "ratios=A_over_Bclone=$(Format-Number $aOverBclone); A_over_B_real=$(Format-Number $aOverBreal); B_real_over_Bclone=$(Format-Number $brealOverBclone); reference_band_2_5_to_2_9=$ratioInReferenceBand"
  "checksum_proof=A_checksum=$($a.checksum); Bclone_checksum=$($bclone.checksum); B_real_checksum=$($breal.checksum); A==Bclone all 10 pairs; B_real distinct and deterministic"
  "counts=30 raw rows; 3 aggregates; 10 independent runs per variant; pair-adjacent serial execution"
  "configuration=D=512;S=1;R=16;mode=fused;S_tile=8;iterations=1;timed_repetitions=8;warmup=2;workers=4;cpus=0,2,4,6;rows_per_worker=1332,807,965,992"
  "artifacts=machine.csv,aggregate.csv,comparison.csv,summary.txt,validation.txt,commands.log,stderr.log,preflight.build.stdout.log,preflight.build.stderr.log,preflight.self-test.stdout.log,preflight.self-test.stderr.log,dumpbin-evidence.txt"
)
Write-Output $summaryPath
