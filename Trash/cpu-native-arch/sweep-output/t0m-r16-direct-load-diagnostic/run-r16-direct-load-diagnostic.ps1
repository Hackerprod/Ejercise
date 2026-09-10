[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$outputDirectory = $PSScriptRoot
$projectRoot = Split-Path -Parent (Split-Path -Parent $outputDirectory)
$executable = Join-Path $projectRoot "build\t0m_int8_probe.exe"
$commandLog = Join-Path $outputDirectory "commands.log"
$outputLog = Join-Path $outputDirectory "outputs.log"
$machineCsv = Join-Path $outputDirectory "diagnostic.machine.csv"
$statsCsv = Join-Path $outputDirectory "diagnostic.stats.csv"
$pairsCsv = Join-Path $outputDirectory "diagnostic.pairs.csv"
$summaryPath = Join-Path $outputDirectory "diagnostic.summary.txt"
$metadataPath = Join-Path $outputDirectory "diagnostic.metadata.txt"

if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
  throw "Executable not found: $executable"
}
$artifactPaths = @($commandLog, $outputLog, $machineCsv, $statsCsv, $pairsCsv, $summaryPath, $metadataPath)
foreach ($artifact in $artifactPaths) {
  if (Test-Path -LiteralPath $artifact -PathType Leaf) {
    throw "Refusing to overwrite existing diagnostic artifact: $artifact"
  }
}

$cpus = "0,2,4,6"
$rowsBySize = @{
  512 = "1211,1136,683,1066"
  768 = "1816,1705,1023,1600"
}
$sizes = @(512, 768)
$repeatCount = 10
$depth = 16
$script:invocation = 0
$script:records = @()

function Write-ExactText([string]$Path, [string]$Text) {
  $utf8 = New-Object System.Text.UTF8Encoding($false)
  [System.IO.File]::WriteAllText($Path, $Text, $utf8)
}

$rowText = ($sizes | ForEach-Object {
    $size = $_
    $rows = $rowsBySize[$size]
    $sum = ($rows.Split(',') | ForEach-Object { [int64]$_ } | Measure-Object -Sum).Sum
    "target_kib=$size; rows_cpu_0_2_4_6=$rows; sum=$sum"
  }) -join "`r`n"
Write-ExactText $metadataPath (@(
  "T0-M isolated R=16 repeated diagnostic"
  "scope=current build t0m_int8_probe.exe only; no source edits; no full Phase 3/4; MRDL/Q4 untouched"
  "executable=$executable"
  "CPUs=$cpus; D=512; S=1; R=16; mode=fused; S_tile=8; iterations=1; timed_repetitions=8; warmup=2"
  "conditions=target 512 [1211,1136,683,1066] and target 768 [1816,1705,1023,1600], CPU order 0,2,4,6; variants A,B"
  "repeats_per_condition=$repeatCount; total_invocations=40; ordering=odd repeats A then B, even repeats B then A per size"
  "ratio_rule=A_over_B=MAC/s(A)/MAC/s(B); paired within size and repeat; no cross-executable comparison"
  "statistics=mean, median, min, max, population SD over ten mac_per_second values per target and variant"
  "near_rule=median A/B is near 2.5..2.9 when within 0.2 outside interval; report exact median and in-range status"
  $rowText
) -join "`r`n")
Write-ExactText $commandLog "T0-M isolated R=16 repeated diagnostic exact commands`r`n"
Write-ExactText $outputLog "T0-M isolated R=16 repeated diagnostic exact command outputs`r`n"

function Get-NonEmptyLines([string]$Text) {
  return @($Text -split '\r?\n' | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
}

function Assert-Field([pscustomobject]$Row, [string]$Name, [string]$Expected, [string]$Context) {
  if (-not ($Row.psobject.Properties.Name -contains $Name)) {
    throw "Missing field ${Name}: $Context"
  }
  if ([string]$Row.$Name -ne $Expected) {
    throw "Invalid ${Name}=$($Row.$Name), expected ${Expected}: $Context"
  }
}

function Invoke-Logged([int]$Size, [int]$Repeat, [string]$Variant, [string]$Rows) {
  $script:invocation++
  $id = $script:invocation
  $label = "target_kib=$Size repeat=$Repeat variant=$Variant"
  $arguments = @(
    "--D", 512, "--S", 1, "--R", 16, "--mode", "fused", "--variant", $Variant,
    "--S-tile", 8, "--workers", 4, "--cpus", $cpus, "--rows-per-worker", $Rows,
    "--iterations", 1, "--timed-repetitions", 8, "--warmup", 2
  )
  $command = '"' + $executable + '" ' + ($arguments -join ' ')
  Add-Content -LiteralPath $commandLog -Encoding ascii -Value "invocation[$id] $label`r`ncommand[$id] $command"

  $startInfo = New-Object System.Diagnostics.ProcessStartInfo
  $startInfo.FileName = $executable
  $startInfo.Arguments = ($arguments -join ' ')
  $startInfo.UseShellExecute = $false
  $startInfo.CreateNoWindow = $true
  $startInfo.RedirectStandardOutput = $true
  $startInfo.RedirectStandardError = $true
  $process = New-Object System.Diagnostics.Process
  $process.StartInfo = $startInfo
  try {
    if (-not $process.Start()) { throw "Could not start $label" }
    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()
    $process.WaitForExit()
    $stdout = $stdoutTask.Result
    $stderr = $stderrTask.Result
    $stdoutPath = Join-Path $outputDirectory ("invocation-{0:D3}.stdout.txt" -f $id)
    $stderrPath = Join-Path $outputDirectory ("invocation-{0:D3}.stderr.txt" -f $id)
    Write-ExactText $stdoutPath $stdout
    Write-ExactText $stderrPath $stderr
    Add-Content -LiteralPath $outputLog -Encoding ascii -Value @(
      "invocation[$id] $label"
      "command[$id] $command"
      "exit_code=$($process.ExitCode)"
      "stdout_file=$stdoutPath"
      "stderr_file=$stderrPath"
      "stdout[$id] $($stdout.TrimEnd())"
      "stderr[$id] $($stderr.TrimEnd())"
    )
    if ($process.ExitCode -ne 0) {
      throw "Probe failed with exit code $($process.ExitCode): $label"
    }
  } finally {
    $process.Dispose()
  }

  if ($stderr -notmatch 'T0-M correction passed: every Y\[S x O_i\] cell, fused/repeat/reference, A/B/C, S_tile=2/4/8, four shards, and one/four-worker accounting gate') {
    throw "Correction evidence missing: $label"
  }
  $lines = Get-NonEmptyLines $stdout
  if ($lines.Count -ne 2) {
    throw "Expected CSV header plus one row, got $($lines.Count): $label"
  }
  $row = $lines[1] | ConvertFrom-Csv -Header ($lines[0] -split ',')
  Assert-Field $row "D" "512" $label
  Assert-Field $row "S" "1" $label
  Assert-Field $row "R" "16" $label
  Assert-Field $row "mode" "fused" $label
  Assert-Field $row "variant" $Variant $label
  Assert-Field $row "S_tile" "8" $label
  Assert-Field $row "iterations" "1" $label
  Assert-Field $row "timed_repetitions" "8" $label
  Assert-Field $row "warmup" "2" $label
  Assert-Field $row "worker_count" "4" $label
  Assert-Field $row "worker_list" $cpus $label
  Assert-Field $row "rows_per_worker" $Rows $label
  Assert-Field $row "affinity_succeeded" "true" $label
  Assert-Field $row "affinity" "true,true,true,true" $label
  Assert-Field $row "affinity_error" "0,0,0,0" $label
  Assert-Field $row "timed_repetitions_exact" "true" $label
  Assert-Field $row "avx2_supported" "true" $label
  Assert-Field $row "kernel_used" "avx2" $label
  Assert-Field $row "eviction_bytes" "0" $label
  Assert-Field $row "eviction_checksum" "0" $label
  $expectedMacTotal = ([int64]($Rows.Split(',') | ForEach-Object { [int64]$_ } | Measure-Object -Sum).Sum) * 512 * 16 * 8
  Assert-Field $row "mac_total" "$expectedMacTotal" $label
  if ([uint64]$row.checksum -eq 0) { throw "Checksum unexpectedly zero: $label" }
  if ([double]$row.elapsed_seconds -le 0 -or [double]$row.mac_per_second -le 0) {
    throw "Timing/output rate invalid: $label"
  }
  $script:records += [pscustomobject][ordered]@{
    invocation = $id
    target_kib = $Size
    repeat = $Repeat
    variant = $Variant
    rows_per_worker = $Rows
    D = [string]$row.D
    S = [string]$row.S
    R = [string]$row.R
    mode = [string]$row.mode
    S_tile = [string]$row.S_tile
    iterations = [string]$row.iterations
    timed_repetitions = [string]$row.timed_repetitions
    warmup = [string]$row.warmup
    worker_count = [string]$row.worker_count
    worker_list = [string]$row.worker_list
    affinity_succeeded = [string]$row.affinity_succeeded
    affinity = [string]$row.affinity
    affinity_error = [string]$row.affinity_error
    timed_repetitions_exact = [string]$row.timed_repetitions_exact
    avx2_supported = [string]$row.avx2_supported
    kernel_used = [string]$row.kernel_used
    mac_total = [string]$row.mac_total
    elapsed_seconds = [string]$row.elapsed_seconds
    mac_per_second = [string]$row.mac_per_second
    checksum = [string]$row.checksum
    eviction_bytes = [string]$row.eviction_bytes
    eviction_checksum = [string]$row.eviction_checksum
    correction_evidence = "true"
    four_worker_affinity = "true"
    exact_repetitions = "true"
  }
}

foreach ($size in $sizes) {
  $rows = $rowsBySize[$size]
  foreach ($repeat in 1..$repeatCount) {
    $orderedVariants = if (($repeat % 2) -eq 1) { @("A", "B") } else { @("B", "A") }
    foreach ($variant in $orderedVariants) {
      Invoke-Logged $size $repeat $variant $rows
    }
  }
}

if ($script:records.Count -ne 40) {
  throw "Expected 40 validated invocations, got $($script:records.Count)"
}

$machineRows = $script:records | ConvertTo-Csv -NoTypeInformation
Write-ExactText $machineCsv (($machineRows -join "`r`n") + "`r`n")

function Get-Median([double[]]$Values) {
  $ordered = @($Values | Sort-Object)
  $middle = [int]($ordered.Count / 2)
  if (($ordered.Count % 2) -eq 0) {
    return ($ordered[$middle - 1] + $ordered[$middle]) / 2.0
  }
  return $ordered[$middle]
}

function Get-Stats([int]$Size, [string]$Variant) {
  $values = @($script:records | Where-Object {
      [int]$_.target_kib -eq $Size -and $_.variant -eq $Variant
    } | ForEach-Object { [double]$_.mac_per_second })
  if ($values.Count -ne $repeatCount) { throw "Expected $repeatCount values: target=$Size variant=$Variant" }
  $mean = [double](($values | Measure-Object -Average).Average)
  $sumSquares = 0.0
  foreach ($value in $values) { $sumSquares += ($value - $mean) * ($value - $mean) }
  [pscustomobject][ordered]@{
    target_kib = $Size
    variant = $Variant
    n = $values.Count
    mean_mac_per_second = $mean
    median_mac_per_second = [double](Get-Median ([double[]]$values))
    min_mac_per_second = [double](($values | Measure-Object -Minimum).Minimum)
    max_mac_per_second = [double](($values | Measure-Object -Maximum).Maximum)
    population_sd_mac_per_second = [math]::Sqrt($sumSquares / $values.Count)
  }
}

$stats = @()
foreach ($size in $sizes) {
  foreach ($variant in @("A", "B")) {
    $stats += Get-Stats $size $variant
  }
}
Write-ExactText $statsCsv ((($stats | ConvertTo-Csv -NoTypeInformation) -join "`r`n") + "`r`n")

$pairs = @()
foreach ($size in $sizes) {
  foreach ($repeat in 1..$repeatCount) {
    $a = @($script:records | Where-Object { [int]$_.target_kib -eq $size -and [int]$_.repeat -eq $repeat -and $_.variant -eq "A" })
    $b = @($script:records | Where-Object { [int]$_.target_kib -eq $size -and [int]$_.repeat -eq $repeat -and $_.variant -eq "B" })
    if ($a.Count -ne 1 -or $b.Count -ne 1) { throw "Expected A/B pair: target=$size repeat=$repeat" }
    $aRate = [double]$a[0].mac_per_second
    $bRate = [double]$b[0].mac_per_second
    $aChecksum = [uint64]$a[0].checksum
    $bChecksum = [uint64]$b[0].checksum
    $pairs += [pscustomobject][ordered]@{
      target_kib = $size
      repeat = $repeat
      A_mac_per_second = $aRate
      B_mac_per_second = $bRate
      A_over_B = $aRate / $bRate
      B_over_A = $bRate / $aRate
      A_checksum = $aChecksum
      B_checksum = $bChecksum
      checksum_equal = ($aChecksum -eq $bChecksum).ToString().ToLowerInvariant()
    }
  }
}
Write-ExactText $pairsCsv ((($pairs | ConvertTo-Csv -NoTypeInformation) -join "`r`n") + "`r`n")

function Format-Real([double]$Value) {
  return $Value.ToString("R", [Globalization.CultureInfo]::InvariantCulture)
}

$summaryLines = @(
  "T0-M isolated R=16 repeated diagnostic"
  "status=DIAGNOSTIC_COMPLETE"
  "scope=current build t0m_int8_probe.exe only; source unchanged; no full Phase 3/4; MRDL/Q4 untouched"
  "invocations=$($script:records.Count); expected=40; independent_processes=true; validation=correction evidence, AVX2/kernel_used, four-worker affinity, exact repetitions, D/S/R/mode/S_tile, MAC total, nonzero checksum"
  "CPUs=$cpus; D=512; S=1; R=16; mode=fused; S_tile=8; iterations=1; timed_repetitions=8; warmup=2"
  "ordering=odd repeats A then B, even repeats B then A per target size"
  $rowText
  "checksum_comparison=reported per within-size A/B pair; no cross-executable comparison"
  "statistics=population SD (sum squared deviations divided by n=10)"
  "near_rule=near means within 0.2 outside interval 2.5..2.9"
)
foreach ($stat in $stats) {
  $summaryLines += ("stats target_kib={0} variant={1} n={2} mean={3} median={4} min={5} max={6} population_sd={7}" -f `
    $stat.target_kib, $stat.variant, $stat.n, (Format-Real $stat.mean_mac_per_second),
    (Format-Real $stat.median_mac_per_second), (Format-Real $stat.min_mac_per_second),
    (Format-Real $stat.max_mac_per_second), (Format-Real $stat.population_sd_mac_per_second))
}
foreach ($size in $sizes) {
  $sizePairs = @($pairs | Where-Object { [int]$_.target_kib -eq $size })
  $ratios = @($sizePairs | ForEach-Object { [double]$_.A_over_B })
  $ratioMin = ($ratios | Measure-Object -Minimum).Minimum
  $ratioMax = ($ratios | Measure-Object -Maximum).Maximum
  $ratioMedian = Get-Median ([double[]]$ratios)
  $checksumEqualCount = @($sizePairs | Where-Object { $_.checksum_equal -eq "true" }).Count
  $aStat = @($stats | Where-Object { [int]$_.target_kib -eq $size -and $_.variant -eq "A" })[0]
  $bStat = @($stats | Where-Object { [int]$_.target_kib -eq $size -and $_.variant -eq "B" })[0]
  $medianAB = $aStat.median_mac_per_second / $bStat.median_mac_per_second
  $inRange = $medianAB -ge 2.5 -and $medianAB -le 2.9
  $near = ($medianAB -ge 2.3 -and $medianAB -lt 2.5) -or ($medianAB -gt 2.9 -and $medianAB -le 3.1) -or $inRange
  $summaryLines += ("ratios target_kib={0} pairwise_A_over_B_min={1} median={2} max={3} median_based_A_over_B={4} in_2.5_2.9={5} near_2.5_2.9={6} checksum_equal_pairs={7}/10" -f `
    $size, (Format-Real $ratioMin), (Format-Real $ratioMedian), (Format-Real $ratioMax),
    (Format-Real $medianAB), $inRange, $near, $checksumEqualCount)
}
$summaryLines += "paired_ratio_rows=$pairsCsv"
$summaryLines += "machine_rows=$machineCsv"
$summaryLines += "stats_rows=$statsCsv"
$summaryLines += "commands_log=$commandLog"
$summaryLines += "outputs_log=$outputLog"
$summaryLines += "metadata=$metadataPath"
Write-ExactText $summaryPath (($summaryLines -join "`r`n") + "`r`n")
Write-Output $summaryPath
