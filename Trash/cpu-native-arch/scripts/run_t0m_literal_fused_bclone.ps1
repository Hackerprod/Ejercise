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
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) { $OutputDirectory = Join-Path $projectRoot "sweep-output\t0m-bridge1-literal-fused-d1472" }
$summaryPath = Join-Path $OutputDirectory "summary.txt"
if (Test-Path -LiteralPath $summaryPath -PathType Leaf) { throw "Refusing rerun: summary exists at $summaryPath" }
if (-not (Test-Path -LiteralPath $OutputDirectory -PathType Container)) { $null = New-Item -ItemType Directory -Path $OutputDirectory }
if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) { throw "Executable not found: $Executable" }

$D = 1472; $S = 1; $R = 16; $workers = 4; $tile = 8; $iterations = 1
$timedRepetitions = 8; $warmup = 2; $independentRuns = 10
$cpus = @(0, 2, 4, 6); $rows = @(457, 297, 347, 371)
$cpuText = $cpus -join ','; $rowText = $rows -join ','
$expectedMac = [int64]$D * $D * $S * $R * $iterations * $timedRepetitions
$header = "D,S,R,O_i,B_i,rows_per_worker,bytes_per_worker,mode,variant,S_tile,iterations,timed_repetitions,warmup,worker_count,worker_list,affinity,affinity_error,affinity_succeeded,timed_repetitions_exact,avx2_supported,kernel_used,eviction_bytes,eviction_checksum,elapsed_seconds,mac_total,mac_per_second,checksum"

$machinePath = Join-Path $OutputDirectory "machine.csv"
$aggregatePath = Join-Path $OutputDirectory "aggregate.csv"
$comparisonPath = Join-Path $OutputDirectory "comparison.csv"
$validationPath = Join-Path $OutputDirectory "validation.txt"
$commandsPath = Join-Path $OutputDirectory "commands.log"
$stderrPath = Join-Path $OutputDirectory "stderr.log"
$buildStdoutPath = Join-Path $OutputDirectory "preflight.build.stdout.log"
$buildStderrPath = Join-Path $OutputDirectory "preflight.build.stderr.log"
$selfTestStdoutPath = Join-Path $OutputDirectory "preflight.self-test.stdout.log"
$selfTestStderrPath = Join-Path $OutputDirectory "preflight.self-test.stderr.log"
$dumpbinPath = Join-Path $OutputDirectory "dumpbin-evidence.txt"

function Format-Number([double]$Value) { $Value.ToString("R", $invariant) }
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
function Require-Equal([string]$Name, [string]$Actual, [string]$Expected, [string]$Context) {
  if ($Actual -ne $Expected) { throw "Invalid ${Name}=$Actual; expected ${Expected}: $Context" }
}
function Require-Positive([string]$Name, [double]$Value, [string]$Context) {
  if ([double]::IsNaN($Value) -or [double]::IsInfinity($Value) -or $Value -le 0) { throw "Invalid ${Name}: $Context" }
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

Set-Content -LiteralPath $commandsPath -Encoding ascii -Value @(
  "scope=literal accepted t0m_int8_probe fused kernel only; no recurrence/transition"
  "configuration=D=1472;S=1;R=16;mode=fused;S_tile=8;iterations=1;timed_repetitions=8;warmup=2"
  "workers=4;cpus=0,2,4,6;rows_per_worker=457,297,347,371;rows fixed from prior calibration; no recalibration"
  "A=one shared weight allocation selected at every depth; Bclone=16 distinct allocations, every block byte-identical to A"
  "activations=deterministic and frozen; output checksum is post-timing only; no checksum work in timed kernel"
  "speed=10 independent paired runs; even A then Bclone; odd Bclone then A; pair-adjacent; never concurrent"
  "preflight=one t0m_int8_probe --self-test; speed invocations never receive --self-test"
)
Set-Content -LiteralPath $stderrPath -Encoding ascii -Value "literal fused Bclone campaign process log"

$cmake = Resolve-Tool $CMakeExecutable "cmake.exe" @(
  "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
  "C:\vkd\tools\cmake-4.3.3-windows\cmake-4.3.3-windows-x86_64\bin\cmake.exe"
)
$dumpbin = Resolve-Tool $DumpbinExecutable "dumpbin.exe" @(
  "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64\dumpbin.exe"
)

$vsDevCmd = Resolve-Tool "" "VsDevCmd.bat" @(
  "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\Tools\VsDevCmd.bat"
)
$buildDirectory = Join-Path $projectRoot "build"
$buildCommand = "call C:\PROGRA~2\MICROS~2\18\BUILDT~1\Common7\Tools\VsDevCmd.bat -arch=x64 && C:\PROGRA~2\MICROS~2\18\BUILDT~1\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe --build $buildDirectory --target t0m_int8_probe"
$buildArgs = @("/d", "/s", "/c", $buildCommand)
Add-Content -LiteralPath $commandsPath -Encoding ascii -Value ("build_command cmd.exe " + ($buildArgs -join ' '))
$build = Invoke-CapturedProcess "cmd.exe" $buildArgs
Set-Content -LiteralPath $buildStdoutPath -Encoding ascii -Value $build.Stdout
Set-Content -LiteralPath $buildStderrPath -Encoding ascii -Value $build.Stderr
if ($build.ExitCode -ne 0) { throw "Build failed: exit=$($build.ExitCode)" }

$selfTestArgs = @("--self-test")
Add-Content -LiteralPath $commandsPath -Encoding ascii -Value "preflight_command `"$Executable`" --self-test"
$selfTest = Invoke-CapturedProcess $Executable $selfTestArgs
Set-Content -LiteralPath $selfTestStdoutPath -Encoding ascii -Value (Limit-Text $selfTest.Stdout)
Set-Content -LiteralPath $selfTestStderrPath -Encoding ascii -Value (Limit-Text $selfTest.Stderr)
if ($selfTest.ExitCode -ne 0 -or ($selfTest.Stdout + "`n" + $selfTest.Stderr) -notmatch "self_test_bclone,D=1472,S=1,R=16,distinct_allocations=16,byte_identical=true,outputs_equal=true,checksums_equal=true" -or ($selfTest.Stdout + "`n" + $selfTest.Stderr) -notmatch "T0-M correction passed") {
  throw "Focused self-test failed"
}

$dumpbinArgs = @("/DISASM", "/SYMBOLS", $Executable)
Add-Content -LiteralPath $commandsPath -Encoding ascii -Value ("dumpbin_command `"$dumpbin`" " + ($dumpbinArgs -join ' '))
$dumpbinResult = Invoke-CapturedProcess $dumpbin $dumpbinArgs
$dumpbinText = $dumpbinResult.Stdout + "`n" + $dumpbinResult.Stderr
$dumpbinLines = @($dumpbinText -split '\r?\n')
$movMatches = @($dumpbinLines | Where-Object { $_ -match '(?i)\bvpmovsxbw\b' })
$maddMatches = @($dumpbinLines | Where-Object { $_ -match '(?i)\bvpmaddwd\b' })
$accumMatches = @($dumpbinLines | Where-Object { $_ -match '(?i)\bvpaddd\b' })
$persistentMatches = @($dumpbinLines | Where-Object { $_ -match '(?i)\bvpaddd\s+(ymm\d+|xmm\d+),(ymm\d+|xmm\d+),(ymm\d+)\b' })
$instructionEvidence = @($movMatches + $maddMatches + $persistentMatches | Select-Object -First 80)
Set-Content -LiteralPath $dumpbinPath -Encoding ascii -Value @(
  "command=`"$dumpbin`" /DISASM /SYMBOLS `"$Executable`""
  "exit_code=$($dumpbinResult.ExitCode)"
  "source_kernel=run_fused_impl<S_TILE=1> accepted t0m_int8_probe template; source arithmetic unchanged"
  "required_sequence=vpmovsxbw + vpmaddwd + persistent vpaddd accumulator"
  "vpmovsxbw_matches=$($movMatches.Count)"
  "vpmaddwd_matches=$($maddMatches.Count)"
  "vpaddd_matches=$($accumMatches.Count)"
  "persistent_vpaddd_matches=$($persistentMatches.Count)"
  "exact_instruction_evidence_begin"
  $instructionEvidence
  "exact_instruction_evidence_end"
)
if ($dumpbinResult.ExitCode -ne 0 -or $movMatches.Count -eq 0 -or $maddMatches.Count -eq 0 -or $persistentMatches.Count -eq 0) {
  throw "Dumpbin accepted fused AVX2 accumulator evidence failed"
}

$script:records = New-Object 'System.Collections.Generic.List[object]'
$script:invocations = 0
function Invoke-Speed([string]$Variant, [int]$Run, [int]$OrderIndex) {
  $args = @(
    "--D", "$D", "--S", "$S", "--R", "$R", "--mode", "fused", "--variant", $Variant,
    "--S-tile", "$tile", "--workers", "$workers", "--cpus", $cpuText,
    "--rows-per-worker", $rowText, "--iterations", "$iterations",
    "--timed-repetitions", "$timedRepetitions", "--warmup", "$warmup"
  )
  $script:invocations++
  $id = $script:invocations; $context = "variant=$Variant run=$Run order_index=$OrderIndex"
  Add-Content -LiteralPath $commandsPath -Encoding ascii -Value "[$id] speed $context `"$Executable`" $($args -join ' ')"
  $result = Invoke-CapturedProcess $Executable $args
  Add-Content -LiteralPath $stderrPath -Encoding ascii -Value @("[$id] $context exit=$($result.ExitCode)", "stdout:", $result.Stdout.TrimEnd(), "stderr:", $result.Stderr.TrimEnd())
  if ($result.ExitCode -ne 0) { throw "Nonzero speed exit $($result.ExitCode): $context" }
  if (($result.Stdout + "`n" + $result.Stderr) -match "(?i)(self[- ]?test|correction|autotest|test passed|test failed|recurrence|transition)") {
    throw "Forbidden preflight/test text in speed output: $context"
  }
  $lines = @($result.Stdout -split '\r?\n' | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
  if ($lines.Count -ne 2 -or $lines[0] -ne $header) { throw "Bad speed CSV schema: $context" }
  $row = $lines[1] | ConvertFrom-Csv -Header ($header -split ',')
  Require-Equal "D" ([string]$row.D) "$D" $context; Require-Equal "S" ([string]$row.S) "$S" $context
  Require-Equal "R" ([string]$row.R) "$R" $context; Require-Equal "variant" ([string]$row.variant) $Variant $context
  Require-Equal "rows_per_worker" ([string]$row.rows_per_worker) $rowText $context
  Require-Equal "bytes_per_worker" ([string]$row.bytes_per_worker) $(if ($Variant -eq "A") { "672704,437184,510784,546112" } else { "10763264,6994944,8172544,8737792" }) $context
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
  $order = if (($run % 2) -eq 0) { @("A", "Bclone") } else { @("Bclone", "A") }
  $first = Invoke-Speed $order[0] $run 1; [void]$script:records.Add($first)
  $second = Invoke-Speed $order[1] $run 2; [void]$script:records.Add($second)
  if ($first.checksum -ne $second.checksum) { throw "A/Bclone checksum mismatch at paired run $run" }
}
if ($script:records.Count -ne 20) { throw "Expected 20 raw rows, got $($script:records.Count)" }

$aggregates = New-Object 'System.Collections.Generic.List[object]'
foreach ($Variant in @("A", "Bclone")) {
  $group = @($script:records | Where-Object { $_.variant -eq $Variant })
  if ($group.Count -ne 10) { throw "Expected 10 rows for $Variant" }
  $checksums = @($group | ForEach-Object { $_.checksum } | Sort-Object -Unique)
  if ($checksums.Count -ne 1) { throw "Nondeterministic checksum for $Variant" }
  $values = [double[]]@($group | ForEach-Object { $_.mac_per_second })
  [void]$aggregates.Add([pscustomobject][ordered]@{
    variant = $Variant; D = $D; S = $S; R = $R; mode = "fused"; S_tile = $tile; rows_per_worker = $rowText; worker_count = $workers; worker_list = $cpuText
    n = 10; median_mac_per_second = Get-Median $values; min_mac_per_second = [double](($values | Measure-Object -Minimum).Minimum); max_mac_per_second = [double](($values | Measure-Object -Maximum).Maximum)
    population_sd_mac_per_second = Get-PopulationSd $values; checksum = [uint64]$checksums[0]; checksum_deterministic = $true; all_affinity_succeeded = $true; timed_repetitions_exact = $true
    timed_repetitions = $timedRepetitions; warmup = $warmup
  })
}
$a = @($aggregates | Where-Object { $_.variant -eq "A" })[0]; $bclone = @($aggregates | Where-Object { $_.variant -eq "Bclone" })[0]
$aOverBclone = [double]$a.median_mac_per_second / [double]$bclone.median_mac_per_second
$bcloneOverA = [double]$bclone.median_mac_per_second / [double]$a.median_mac_per_second
$comparison = [pscustomobject][ordered]@{
  record_type = "literal_fused_bclone"; D = $D; S = $S; R = $R; mode = "fused"; S_tile = $tile; rows_per_worker = $rowText
  A_median_mac_per_second = [double]$a.median_mac_per_second; Bclone_median_mac_per_second = [double]$bclone.median_mac_per_second
  A_over_Bclone = $aOverBclone; Bclone_over_A = $bcloneOverA; A_checksum = [uint64]$a.checksum; Bclone_checksum = [uint64]$bclone.checksum
  checksum_equality_all_pairs = $true; checksum_equality_deterministic = $true; speed_rows = 20; aggregate_rows = 2
}

Set-Content -LiteralPath $machinePath -Encoding ascii -Value @($script:records | ConvertTo-Csv -NoTypeInformation)
Set-Content -LiteralPath $aggregatePath -Encoding ascii -Value @($aggregates | ConvertTo-Csv -NoTypeInformation)
Set-Content -LiteralPath $comparisonPath -Encoding ascii -Value @($comparison | ConvertTo-Csv -NoTypeInformation)
Set-Content -LiteralPath $validationPath -Encoding ascii -Value @(
  "status=PASS"
  "raw_rows=20; aggregate_rows=2; expected_raw_rows=20; expected_aggregate_rows=2"
  "configuration=PASS; D=1472;S=1;R=16;mode=fused;S_tile=8;iterations=1;timed_repetitions=8;warmup=2"
  "affinity=PASS; workers=4;cpus=0,2,4,6;rows_per_worker=457,297,347,371;rows_fixed_no_recalibration=true"
  "avx2=PASS; every speed row avx2_supported=true;kernel_used=avx2"
  "repetitions=PASS; exact timed_repetitions=8; warmup=2; all rows timed_repetitions_exact=true"
  "checksums=PASS; nonzero; deterministic per variant; A==Bclone final checksum for all 10 pair-adjacent runs"
  "checksum_timing=PASS; checksum emitted after timed region by existing probe behavior"
  "order=PASS; even runs A then Bclone; odd runs Bclone then A; never concurrent"
  "preflight=PASS; exactly one t0m_int8_probe --self-test; speed invocations contain no self-test or test text"
  "dumpbin=PASS; rebuilt executable contains vpmovsxbw, vpmaddwd, and persistent vpaddd accumulator evidence"
  "A_median_mac_per_second=$(Format-Number $a.median_mac_per_second); Bclone_median_mac_per_second=$(Format-Number $bclone.median_mac_per_second)"
  "A_over_Bclone=$(Format-Number $aOverBclone); Bclone_over_A=$(Format-Number $bcloneOverA)"
)
Set-Content -LiteralPath $summaryPath -Encoding ascii -Value @(
  "status=PASS"
  "executive_summary=Literal accepted fused t0m_int8_probe kernel A versus byte-identical 16-allocation Bclone completed at D=1472,S=1,R=16 with frozen activations/state and no transition path."
  "A_median_mac_per_second=$(Format-Number $a.median_mac_per_second); Bclone_median_mac_per_second=$(Format-Number $bclone.median_mac_per_second)"
  "A_over_Bclone=$(Format-Number $aOverBclone); Bclone_over_A=$(Format-Number $bcloneOverA)"
  "checksum_proof=A_checksum=$($a.checksum); Bclone_checksum=$($bclone.checksum); exact equality for all 10 paired runs; deterministic and nonzero"
  "counts=20 raw rows; 2 aggregates; 10 independent runs per variant; pair-adjacent serial execution"
  "configuration=D=1472;S=1;R=16;mode=fused;S_tile=8;iterations=1;timed_repetitions=8;warmup=2;workers=4;cpus=0,2,4,6;rows_per_worker=457,297,347,371"
  "artifacts=machine.csv,aggregate.csv,comparison.csv,summary.txt,validation.txt,commands.log,stderr.log,preflight.build.stdout.log,preflight.build.stderr.log,preflight.self-test.stdout.log,preflight.self-test.stderr.log,dumpbin-evidence.txt"
)
Write-Output $summaryPath
