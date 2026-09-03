[CmdletBinding()]
param(
  [string]$Executable = "",
  [string]$OutputDirectory = "",
  [int]$RepeatCount = 5,
  [int]$Repetitions = 8,
  [int]$Warmup = 2,
  [string]$FixedRows = "",
  [string]$SizeList = "",
  [string]$DepthList = "",
  [string]$VariantList = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($Executable)) {
  $Executable = Join-Path $projectRoot "build\int8_probe.exe"
}
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
  $OutputDirectory = Join-Path $projectRoot "sweep-output\t0r-int8-sharded"
}
if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
  throw "Executable not found: $Executable"
}
if ($RepeatCount -le 0 -or $Repetitions -le 0 -or $Warmup -lt 0) {
  throw "RepeatCount and Repetitions must be positive; Warmup cannot be negative"
}
if (-not (Test-Path -LiteralPath $OutputDirectory -PathType Container)) {
  $null = New-Item -ItemType Directory -Path $OutputDirectory
}

$cpus = @(0, 2, 4, 6)
$Workers = $cpus.Count
$sizes = if ([string]::IsNullOrWhiteSpace($SizeList)) { @(384, 512, 640, 768) } else { @($SizeList -split ',' | ForEach-Object { [int]$_.Trim() }) }
$depths = if ([string]::IsNullOrWhiteSpace($DepthList)) { @(1, 4, 8, 16) } else { @($DepthList -split ',' | ForEach-Object { [int]$_.Trim() }) }
$variants = if ([string]::IsNullOrWhiteSpace($VariantList)) { @("A", "B", "C") } else { @($VariantList -split ',' | ForEach-Object { $_.Trim() }) }
if (@($sizes | Where-Object { $_ -le 0 }).Count -gt 0 -or @($depths | Where-Object { $_ -le 0 }).Count -gt 0) { throw "Sizes and depths must be positive" }
if (@($variants | Where-Object { $_ -notin @("A", "B", "Bclone", "C") }).Count -gt 0) { throw "Invalid variant list" }
$rawCsv = Join-Path $OutputDirectory "t0r_int8_sharded.csv"
$calibrationCsv = Join-Path $OutputDirectory "int8_calibration.csv"
$stderrLog = Join-Path $OutputDirectory "t0r_int8_sharded.stderr.log"
$summaryPath = Join-Path $OutputDirectory "t0r_int8_sharded.summary.txt"
$validationPath = Join-Path $OutputDirectory "validation.txt"
$commandsPath = Join-Path $OutputDirectory "commands.log"
if (Test-Path -LiteralPath $summaryPath -PathType Leaf) {
  throw "Refusing rerun: summary exists at $summaryPath"
}
$expectedInvocations = $sizes.Count * $depths.Count * $variants.Count * $RepeatCount
$dramGbps = 32.9295

function Invoke-Probe([string[]]$Arguments) {
  $startInfo = New-Object System.Diagnostics.ProcessStartInfo
  $startInfo.FileName = $Executable
  $startInfo.Arguments = ($Arguments -join ' ')
  $startInfo.UseShellExecute = $false
  $startInfo.CreateNoWindow = $true
  $startInfo.RedirectStandardOutput = $true
  $startInfo.RedirectStandardError = $true
  $process = New-Object System.Diagnostics.Process
  $process.StartInfo = $startInfo
  try {
    if (-not $process.Start()) {
      throw "Could not start probe"
    }
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

function Read-ProbeRow($Invocation, [string]$Context) {
  if ($Invocation.ExitCode -ne 0) {
    throw "Probe failed with exit code $($Invocation.ExitCode): $Context"
  }
  $lines = @($Invocation.Stdout -split '\r?\n' | Where-Object {
      -not [string]::IsNullOrWhiteSpace($_)
    })
  if ($lines.Count -ne 2) {
    throw "Probe did not produce one CSV row: $Context"
  }
  return $lines
}

Remove-Item -LiteralPath $rawCsv, $calibrationCsv, $stderrLog, $validationPath, $commandsPath -Force -ErrorAction SilentlyContinue
Set-Content -LiteralPath $stderrLog -Encoding ascii -Value @(
  "T0-R int8 sharded sweep stderr and command log"
  "executable=$Executable"
  "physical_cpus=$($cpus -join ','); dram_gbps_measured=$dramGbps"
  "target_kib=$($sizes -join ','); depths=$($depths -join ','); variants=$($variants -join ','); repeat_count=$RepeatCount; repetitions=$Repetitions; warmup=$Warmup"
)
Set-Content -LiteralPath $commandsPath -Encoding ascii -Value @(
  "script=$PSCommandPath"
  "executable=$Executable"
  "target_kib=$($sizes -join ','); depths=$($depths -join ','); variants=$($variants -join ','); repeat_count=$RepeatCount; repetitions=$Repetitions; warmup=$Warmup"
  "cpus=$($cpus -join ',')"
)

$calibration = @()
foreach ($cpu in $cpus) {
  $args = @("--cpu", $cpu, "--m", 64, "--K", 64, "--depth", 64,
            "--iterations", 4, "--repetitions", 20, "--warmup", 5,
            "--kernel", "avx2")
  $context = "calibration cpu=$cpu"
  $invocation = Invoke-Probe $args
  if ($invocation.Stderr) {
    Add-Content -LiteralPath $stderrLog -Encoding ascii -Value $invocation.Stderr.TrimEnd()
  }
  $lines = Read-ProbeRow $invocation $context
  $row = $lines[1] | ConvertFrom-Csv -Header ($lines[0] -split ',')
  $calibration += [pscustomobject]@{
    logical_cpu_index = $cpu
    mac_per_second = [double]$row.mac_per_second
    elapsed_seconds = [double]$row.elapsed_seconds
    affinity_succeeded = $row.affinity_succeeded
  }
}
if (@($calibration | Where-Object { $_.affinity_succeeded -ne "true" }).Count -gt 0) {
  throw "Calibration affinity failed"
}
Set-Content -LiteralPath $calibrationCsv -Encoding ascii -Value "logical_cpu_index,mac_per_second,elapsed_seconds,affinity_succeeded"
foreach ($row in $calibration) {
  Add-Content -LiteralPath $calibrationCsv -Encoding ascii -Value "$($row.logical_cpu_index),$($row.mac_per_second),$($row.elapsed_seconds),$($row.affinity_succeeded)"
}

$throughputTotal = ($calibration | Measure-Object -Property mac_per_second -Sum).Sum
$fixedRowValues = @()
if (-not [string]::IsNullOrWhiteSpace($FixedRows)) {
  $fixedRowValues = @($FixedRows -split ',' | ForEach-Object { [int]$_.Trim() })
  if ($fixedRowValues.Count -ne $Workers -or @($fixedRowValues | Where-Object { $_ -le 0 }).Count -gt 0) {
    throw "FixedRows must contain one positive row count per CPU"
  }
}
$planBySize = @{}
$planSummary = @()
foreach ($size in $sizes) {
  if ($fixedRowValues.Count -gt 0) {
    $rows = @($fixedRowValues)
    $totalRows = ($rows | Measure-Object -Sum).Sum
  } else {
    $rowsPerWorkerAtTarget = [math]::Floor(($size * 1024) / 512)
    $totalRows = $rowsPerWorkerAtTarget * $Workers
    $shares = @($calibration | ForEach-Object {
        $exact = $totalRows * $_.mac_per_second / $throughputTotal
        [pscustomobject]@{ cpu = $_.logical_cpu_index; exact = $exact; rows = [math]::Floor($exact); fraction = $exact - [math]::Floor($exact) }
      })
    $remaining = $totalRows - (($shares | Measure-Object -Property rows -Sum).Sum)
    foreach ($share in ($shares | Sort-Object fraction -Descending | Select-Object -First $remaining)) {
      $share.rows++
    }
    $rows = @($shares | Sort-Object cpu | ForEach-Object { [int]$_.rows })
  }
  $planBySize[$size] = $rows
  $planSummary += "target_kib_per_worker=$size total_rows=$totalRows logical_cpus=$($cpus -join ',') rows_per_worker=$($rows -join ',') fixed_rows=$($fixedRowValues.Count -gt 0)"
}

$headerWritten = $false
$probeHeader = $null
$invocationNumber = 0
for ($repeat = 1; $repeat -le $RepeatCount; $repeat++) {
  foreach ($size in $sizes) {
    foreach ($depth in $depths) {
      $orderedVariants = if ($repeat % 2 -eq 0) { @($variants[($variants.Count - 1)..0]) } else { $variants }
      $orderPosition = 0
      foreach ($variant in $orderedVariants) {
        $orderPosition++
        $invocationNumber++
        $rows = if ($fixedRowValues.Count -gt 0) { @($fixedRowValues) } else { $planBySize[[int]$size] }
        $args = @("--K", 512, "--target-kib", $size, "--depth", $depth,
                  "--variant", $variant, "--kernel", "avx2",
                  "--parallel-cpus", ($cpus -join ','),
                  "--parallel-rows", ($rows -join ','),
                  "--iterations", 1, "--repetitions", $Repetitions, "--warmup", $Warmup)
        $context = "batch=$invocationNumber/$expectedInvocations repeat=$repeat target_kib=$size depth=$depth variant=$variant"
        Add-Content -LiteralPath $stderrLog -Encoding ascii -Value "command[$invocationNumber/$expectedInvocations]=$(($Executable)) $($args -join ' ')"
        Add-Content -LiteralPath $commandsPath -Encoding ascii -Value "command[$invocationNumber/$expectedInvocations]=$(($Executable)) $($args -join ' ')"
        $invocation = Invoke-Probe $args
        if ($invocation.Stderr) {
          Add-Content -LiteralPath $stderrLog -Encoding ascii -Value $invocation.Stderr.TrimEnd()
        }
        $lines = Read-ProbeRow $invocation $context
        if (-not $headerWritten) {
          $probeHeader = $lines[0]
          Set-Content -LiteralPath $rawCsv -Encoding ascii -Value ($probeHeader + ",batch_repeat,variant_order")
          $headerWritten = $true
        } elseif ($lines[0] -ne $probeHeader) {
          throw "Probe CSV header changed: $context"
        }
        Add-Content -LiteralPath $rawCsv -Encoding ascii -Value ($lines[1] + ",$repeat,$orderPosition")
      }
    }
  }
}

$rows = @(Import-Csv -LiteralPath $rawCsv)
if ($rows.Count -ne $expectedInvocations) {
  throw "Expected $expectedInvocations rows, got $($rows.Count)"
}
$invalid = @($rows | Where-Object {
    $_.worker_count -ne "$Workers" -or $_.logical_cpu_indices -ne ($cpus -join ',') -or
    $_.kernel_used -ne "avx2" -or $_.avx2_supported -ne "true" -or
    $_.all_affinity_succeeded -ne "true"
  })
if ($invalid.Count -gt 0) {
  throw "Invalid parallel rows: $($invalid.Count)"
}
if ($variants -contains "Bclone") {
  foreach ($size in $sizes) {
    foreach ($depth in $depths) {
      foreach ($variant in $variants) {
        $variantChecksums = @($rows | Where-Object {
            $_.target_kib -eq "$size" -and $_.depth -eq "$depth" -and $_.variant -eq $variant
          } | ForEach-Object { $_.checksum } | Sort-Object -Unique)
        if ($variantChecksums.Count -ne 1 -or [int64]$variantChecksums[0] -eq 0) {
          throw "Checksum is nondeterministic or zero: size=$size depth=$depth variant=$variant"
        }
      }
      for ($repeat = 1; $repeat -le $RepeatCount; $repeat++) {
        $paired = @($rows | Where-Object {
            $_.target_kib -eq "$size" -and $_.depth -eq "$depth" -and $_.batch_repeat -eq "$repeat"
          })
        if ($paired.Count -ne $variants.Count) { throw "Incomplete paired run: size=$size depth=$depth repeat=$repeat" }
        $expectedOrder = if ($repeat % 2 -eq 0) { @($variants[($variants.Count - 1)..0]) } else { $variants }
        for ($orderIndex = 0; $orderIndex -lt $expectedOrder.Count; $orderIndex++) {
          $orderedRow = @($paired | Where-Object { $_.variant_order -eq "$($orderIndex + 1)" })
          if ($orderedRow.Count -ne 1 -or $orderedRow[0].variant -ne $expectedOrder[$orderIndex]) {
            throw "Unexpected alternating order: size=$size depth=$depth repeat=$repeat"
          }
        }
        $aChecksum = [int64](@($paired | Where-Object { $_.variant -eq "A" })[0].checksum)
        $bChecksum = [int64](@($paired | Where-Object { $_.variant -eq "B" })[0].checksum)
        $bcloneChecksum = [int64](@($paired | Where-Object { $_.variant -eq "Bclone" })[0].checksum)
        if ($aChecksum -eq 0 -or $bChecksum -eq 0 -or $bcloneChecksum -eq 0) { throw "Zero checksum: size=$size depth=$depth repeat=$repeat" }
        if ($aChecksum -ne $bcloneChecksum) { throw "A/Bclone checksum mismatch: size=$size depth=$depth repeat=$repeat" }
        if ($bChecksum -eq $aChecksum -or $bChecksum -eq $bcloneChecksum) { throw "B checksum is not distinct: size=$size depth=$depth repeat=$repeat" }
      }
    }
  }
}

$statistics = foreach ($size in $sizes) {
  foreach ($depth in $depths) {
    $lines = foreach ($variant in $variants) {
      $values = @($rows | Where-Object { $_.target_kib -eq "$size" -and $_.depth -eq "$depth" -and $_.variant -eq $variant } | ForEach-Object { [double]$_.mac_per_second })
      $mean = ($values | Measure-Object -Average).Average
      $sd = [math]::Sqrt((($values | ForEach-Object { ($_ - $mean) * ($_ - $mean) } | Measure-Object -Average).Average))
      $ordered = @($values | Sort-Object)
      $middle = [math]::Floor($ordered.Count / 2)
      $median = if (($ordered.Count % 2) -eq 0) { ([double]$ordered[$middle - 1] + [double]$ordered[$middle]) / 2.0 } else { [double]$ordered[$middle] }
      "$variant`_mean=$mean $variant`_median=$median $variant`_sd=$sd $variant`_min=$($values | Measure-Object -Minimum | Select-Object -ExpandProperty Minimum) $variant`_max=$($values | Measure-Object -Maximum | Select-Object -ExpandProperty Maximum)"
    }
    $aMean = (($rows | Where-Object { $_.target_kib -eq "$size" -and $_.depth -eq "$depth" -and $_.variant -eq "A" } | ForEach-Object { [double]$_.mac_per_second }) | Measure-Object -Average).Average
    $bMean = (($rows | Where-Object { $_.target_kib -eq "$size" -and $_.depth -eq "$depth" -and $_.variant -eq "B" } | ForEach-Object { [double]$_.mac_per_second }) | Measure-Object -Average).Average
     $line = $lines + "target_kib=$size depth=$depth B_over_A=$($bMean / $aMean) B_bandwidth_gbps=$($bMean / 1e9) B_over_measured_dram=$($bMean / 1e9 / $dramGbps)"
     if ($variants -contains "Bclone") {
       $bcloneMean = (($rows | Where-Object { $_.target_kib -eq "$size" -and $_.depth -eq "$depth" -and $_.variant -eq "Bclone" } | ForEach-Object { [double]$_.mac_per_second }) | Measure-Object -Average).Average
       $line += " A_over_Bclone=$($aMean / $bcloneMean) B_over_Bclone=$($bMean / $bcloneMean)"
     }
     $line
  }
}

$calibrationText = ($calibration | ForEach-Object {
  "cpu=$($_.logical_cpu_index):$($_.mac_per_second)"
}) -join '; '

$summary = @(
  "T0-R int8 sharded sweep summary"
  "raw_csv=$rawCsv"
  "calibration_csv=$calibrationCsv"
  "stderr_log=$stderrLog"
  "workers=$Workers; physical_logical_indices=$($cpus -join ','); per-depth barrier=true; batch metric=kernel-phase wall-clock sum (prep excluded)"
  "dram_gbps_measured=$dramGbps; int8 bytes_per_MAC=1"
  "expected_invocations=$expectedInvocations"
  "actual_rows=$($rows.Count)"
  "invalid_rows=$($invalid.Count)"
  "validation_path=$validationPath"
  "commands_path=$commandsPath"
  "calibration=$calibrationText"
  $planSummary
  "statistics=population SD and min/max over $RepeatCount alternating repetitions"
  $statistics
  "caveat=single-channel measured stream ceiling; int8 kernel uses independent per-worker shards; thermal state still matters"
)
Set-Content -LiteralPath $summaryPath -Encoding ascii -Value $summary
Set-Content -LiteralPath $validationPath -Encoding ascii -Value @(
  "status=PASS"
  "raw_rows=$($rows.Count); expected_raw_rows=$expectedInvocations; variants=$($variants -join ','); repeat_count=$RepeatCount"
  "configuration=target_kib=$($sizes -join ',');depths=$($depths -join ',');repetitions=$Repetitions;warmup=$Warmup"
  "affinity=PASS;workers=$Workers;cpus=$($cpus -join ',');fixed_rows=$($fixedRowValues.Count -gt 0)"
  "avx2=PASS; every row avx2_supported=true and kernel_used=avx2"
  "repetitions=PASS; exact repetitions=$Repetitions and warmup=$Warmup"
  "checksums=PASS; nonzero and deterministic per variant; A==Bclone and B distinct per paired run"
  "order=PASS; deterministic alternating variant order; serial process invocations"
)
Write-Output $summaryPath
