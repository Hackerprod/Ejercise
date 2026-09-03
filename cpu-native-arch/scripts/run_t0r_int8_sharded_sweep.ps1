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
  [string]$VariantList = "",
  [switch]$ReadyTimingDiagnostic,
  [switch]$WarmupAffinityTimingDiagnostic
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
if ($WarmupAffinityTimingDiagnostic -and -not $ReadyTimingDiagnostic) {
  throw "WarmupAffinityTimingDiagnostic requires ReadyTimingDiagnostic"
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
$readyTimingLog = Join-Path $OutputDirectory "ready-timing-diagnostic.log"
$blockTimingLog = Join-Path $OutputDirectory "block-timing-diagnostic.log"
$readyTimingSummaryPath = Join-Path $OutputDirectory "ready-timing-summary.txt"
$blockTimingSummaryPath = Join-Path $OutputDirectory "block-timing-summary.txt"
$warmupTimingLog = Join-Path $OutputDirectory "warmup-timing-diagnostic.log"
$affinityTimingLog = Join-Path $OutputDirectory "affinity-timing-diagnostic.log"
$warmupTimingSummaryPath = Join-Path $OutputDirectory "warmup-timing-summary.txt"
$affinityTimingSummaryPath = Join-Path $OutputDirectory "affinity-timing-summary.txt"
$summaryPath = Join-Path $OutputDirectory "t0r_int8_sharded.summary.txt"
$validationPath = Join-Path $OutputDirectory "validation.txt"
$commandsPath = Join-Path $OutputDirectory "commands.log"
if (Test-Path -LiteralPath $OutputDirectory -PathType Container) {
  throw "Refusing overwrite: output directory exists at $OutputDirectory"
}
$null = New-Item -ItemType Directory -Path $OutputDirectory
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
  "ready_timing_diagnostic=$ReadyTimingDiagnostic"
  "warmup_affinity_timing_diagnostic=$WarmupAffinityTimingDiagnostic"
)
Set-Content -LiteralPath $commandsPath -Encoding ascii -Value @(
  "script=$PSCommandPath"
  "executable=$Executable"
  "target_kib=$($sizes -join ','); depths=$($depths -join ','); variants=$($variants -join ','); repeat_count=$RepeatCount; repetitions=$Repetitions; warmup=$Warmup"
  "cpus=$($cpus -join ',')"
  "ready_timing_diagnostic=$ReadyTimingDiagnostic"
  "warmup_affinity_timing_diagnostic=$WarmupAffinityTimingDiagnostic"
)
Set-Content -LiteralPath $readyTimingLog -Encoding ascii -Value @(
  "T0-R ready timing diagnostic stderr"
  "boundaries=worker lambda entry through immediate QPC after existing ready.arrive_and_wait returns; includes make_data, output allocation, eviction allocation, AffinityGuard, warmup, and wait for coordinator ready arrival"
  "diagnostic_enabled=$ReadyTimingDiagnostic"
)
Set-Content -LiteralPath $blockTimingLog -Encoding ascii -Value @(
  "T0-R block timing diagnostic stderr"
  "boundaries=QPC immediately before and after make_data block-construction loop; input fill and all code outside loop excluded"
  "bclone_first=first WeightBlock copy plus push_back operation only; clone_copy_elapsed_seconds=sum of each clone copy operation; remaining=total-first"
  "diagnostic_enabled=$ReadyTimingDiagnostic"
)
Set-Content -LiteralPath $warmupTimingLog -Encoding ascii -Value @(
  "T0-R warmup timing diagnostic stderr"
  "boundaries=QPC immediately before first existing warmup-loop iteration through immediately after last existing warmup-loop iteration; includes only warmup loop pass_function calls and variant C preparation"
  "diagnostic_enabled=$WarmupAffinityTimingDiagnostic"
)
Set-Content -LiteralPath $affinityTimingLog -Encoding ascii -Value @(
  "T0-R AffinityGuard construction timing diagnostic stderr"
  "boundaries=QPC immediately before AffinityGuard declaration through immediately after constructor returns; excludes make_data, output allocation, eviction allocation, warmup, barriers, and result assignment"
  "diagnostic_enabled=$WarmupAffinityTimingDiagnostic"
)

$preflightLog = Join-Path $OutputDirectory "preflight.self-test.log"
$preflight = Invoke-Probe @("--self-test")
Add-Content -LiteralPath $commandsPath -Encoding ascii -Value "preflight_command=$Executable --self-test"
Set-Content -LiteralPath $preflightLog -Encoding ascii -Value @(
  "command=$Executable --self-test"
  "exit_code=$($preflight.ExitCode)"
  "stdout=$($preflight.Stdout.TrimEnd())"
  "stderr=$($preflight.Stderr.TrimEnd())"
)
if ($preflight.ExitCode -ne 0) {
  throw "Preflight self-test failed with exit code $($preflight.ExitCode)"
}

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
        if ($ReadyTimingDiagnostic) { $args += "--ready-timing-diagnostic" }
        if ($WarmupAffinityTimingDiagnostic) { $args += "--warmup-affinity-timing-diagnostic" }
        $context = "batch=$invocationNumber/$expectedInvocations repeat=$repeat target_kib=$size depth=$depth variant=$variant"
        Add-Content -LiteralPath $stderrLog -Encoding ascii -Value "command[$invocationNumber/$expectedInvocations]=$(($Executable)) $($args -join ' ')"
        Add-Content -LiteralPath $commandsPath -Encoding ascii -Value "command[$invocationNumber/$expectedInvocations]=$(($Executable)) $($args -join ' ')"
        $invocation = Invoke-Probe $args
        if ($invocation.Stderr) {
          Add-Content -LiteralPath $stderrLog -Encoding ascii -Value $invocation.Stderr.TrimEnd()
          if ($ReadyTimingDiagnostic) {
            $diagnosticLines = @($invocation.Stderr -split '\r?\n' | Where-Object {
                $_ -match '^ready_timing ' -or $_ -match '^ready_timing_aggregate ' -or
                $_ -match '^coordinator_kernel_elapsed '
              } | ForEach-Object { "invocation=$invocationNumber $_" })
            if ($diagnosticLines.Count -gt 0) {
              Add-Content -LiteralPath $readyTimingLog -Encoding ascii -Value $diagnosticLines
            }
            $blockDiagnosticLines = @($invocation.Stderr -split '\r?\n' | Where-Object {
                $_ -match '^block_timing '
              } | ForEach-Object { "invocation=$invocationNumber $_" })
            if ($blockDiagnosticLines.Count -gt 0) {
              Add-Content -LiteralPath $blockTimingLog -Encoding ascii -Value $blockDiagnosticLines
            }
            $warmupDiagnosticLines = @($invocation.Stderr -split '\r?\n' | Where-Object {
                $_ -match '^warmup_timing '
              } | ForEach-Object { "invocation=$invocationNumber $_" })
            if ($warmupDiagnosticLines.Count -gt 0) {
              Add-Content -LiteralPath $warmupTimingLog -Encoding ascii -Value $warmupDiagnosticLines
            }
            $affinityDiagnosticLines = @($invocation.Stderr -split '\r?\n' | Where-Object {
                $_ -match '^affinity_timing '
              } | ForEach-Object { "invocation=$invocationNumber $_" })
            if ($affinityDiagnosticLines.Count -gt 0) {
              Add-Content -LiteralPath $affinityTimingLog -Encoding ascii -Value $affinityDiagnosticLines
            }
          }
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
    $_.all_affinity_succeeded -ne "true" -or $_.K -ne "512" -or
    $_.target_kib -notin @($sizes | ForEach-Object { "$_" }) -or
    $_.depth -notin @($depths | ForEach-Object { "$_" }) -or
    [int]$_.m -ne [int](([int]$_.target_kib * 1024) / 512) -or
    $_.iterations -ne "1" -or $_.repetitions -ne "$Repetitions" -or
    $_.warmup -ne "$Warmup" -or ($fixedRowValues.Count -gt 0 -and
      $_.rows_per_worker -ne ($fixedRowValues -join ','))
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

$readyWorkerRecords = @()
$blockWorkerRecords = @()
$warmupWorkerRecords = @()
$affinityWorkerRecords = @()
$diagnosticInvalid = @()
if ($ReadyTimingDiagnostic) {
  foreach ($line in @(Get-Content -LiteralPath $readyTimingLog)) {
    if ($line -match '^invocation=(\d+) ready_timing variant=(\S+) worker=(\d+) cpu=(\d+) row=(\d+) ready_elapsed_seconds=([0-9.eE+-]+)$') {
      $readyWorkerRecords += [pscustomobject]@{
        invocation = [int]$matches[1]; variant = $matches[2]; worker = [int]$matches[3]
        cpu = [int]$matches[4]; row = [int]$matches[5]; elapsed = [double]$matches[6]
      }
    }
  }
  foreach ($line in @(Get-Content -LiteralPath $blockTimingLog)) {
    if ($line -match '^invocation=(\d+) block_timing variant=(\S+) worker=(\d+) cpu=(\d+) row=(\d+) block_loop_elapsed_seconds=([0-9.eE+-]+) first_clone_copy_elapsed_seconds=([0-9.eE+-]+) clone_copy_elapsed_seconds=([0-9.eE+-]+) remaining_clone_copy_elapsed_seconds=([0-9.eE+-]+)$') {
      $blockWorkerRecords += [pscustomobject]@{
        invocation = [int]$matches[1]; variant = $matches[2]; worker = [int]$matches[3]
        cpu = [int]$matches[4]; row = [int]$matches[5]; block_loop = [double]$matches[6]
        first_clone = [double]$matches[7]; clone_total = [double]$matches[8]; clone_remaining = [double]$matches[9]
      }
    }
  }
  foreach ($line in @(Get-Content -LiteralPath $warmupTimingLog)) {
    if ($line -match '^invocation=(\d+) warmup_timing variant=(\S+) worker=(\d+) cpu=(\d+) row=(\d+) warmup_elapsed_seconds=([0-9.eE+-]+)$') {
      $warmupWorkerRecords += [pscustomobject]@{
        invocation = [int]$matches[1]; variant = $matches[2]; worker = [int]$matches[3]
        cpu = [int]$matches[4]; row = [int]$matches[5]; elapsed = [double]$matches[6]
      }
    }
  }
  foreach ($line in @(Get-Content -LiteralPath $affinityTimingLog)) {
    if ($line -match '^invocation=(\d+) affinity_timing variant=(\S+) worker=(\d+) cpu=(\d+) row=(\d+) affinity_guard_elapsed_seconds=([0-9.eE+-]+)$') {
      $affinityWorkerRecords += [pscustomobject]@{
        invocation = [int]$matches[1]; variant = $matches[2]; worker = [int]$matches[3]
        cpu = [int]$matches[4]; row = [int]$matches[5]; elapsed = [double]$matches[6]
      }
    }
  }
  for ($invocation = 1; $invocation -le $expectedInvocations; $invocation++) {
    $csvRow = $rows[$invocation - 1]
    $readyForInvocation = @($readyWorkerRecords | Where-Object { $_.invocation -eq $invocation })
    $blockForInvocation = @($blockWorkerRecords | Where-Object { $_.invocation -eq $invocation })
    $warmupForInvocation = @($warmupWorkerRecords | Where-Object { $_.invocation -eq $invocation })
    $affinityForInvocation = @($affinityWorkerRecords | Where-Object { $_.invocation -eq $invocation })
    if ($readyForInvocation.Count -ne $Workers -or $blockForInvocation.Count -ne $Workers -or
        $warmupForInvocation.Count -ne $Workers -or $affinityForInvocation.Count -ne $Workers) {
      $diagnosticInvalid += "invocation=$invocation worker_line_count=$($readyForInvocation.Count)/$($blockForInvocation.Count)/$($warmupForInvocation.Count)/$($affinityForInvocation.Count)"
      continue
    }
    foreach ($record in @($readyForInvocation | Sort-Object worker)) {
      if ($record.variant -ne $csvRow.variant -or $record.worker -lt 0 -or $record.worker -ge $Workers -or
          $record.cpu -ne $cpus[$record.worker] -or $record.row -ne $fixedRowValues[$record.worker] -or
          $record.elapsed -le 0.0) {
        $diagnosticInvalid += "ready invocation=$invocation worker=$($record.worker)"
      }
    }
    foreach ($record in @($blockForInvocation | Sort-Object worker)) {
      if ($record.variant -ne $csvRow.variant -or $record.worker -lt 0 -or $record.worker -ge $Workers -or
          $record.cpu -ne $cpus[$record.worker] -or $record.row -ne $fixedRowValues[$record.worker] -or
          $record.block_loop -le 0.0 -or $record.first_clone -lt 0.0 -or
          $record.clone_total -lt $record.first_clone -or $record.clone_remaining -lt 0.0) {
        $diagnosticInvalid += "block invocation=$invocation worker=$($record.worker)"
      }
      if ([math]::Abs($record.clone_total - $record.first_clone - $record.clone_remaining) -gt 1e-9) {
        $diagnosticInvalid += "clone_remainder_mismatch invocation=$invocation worker=$($record.worker)"
      }
      if ($csvRow.variant -ne "Bclone" -and
          ($record.first_clone -ne 0.0 -or $record.clone_total -ne 0.0 -or $record.clone_remaining -ne 0.0)) {
        $diagnosticInvalid += "non_bclone_clone_timing invocation=$invocation worker=$($record.worker)"
      }
      if ($csvRow.variant -eq "Bclone" -and $record.first_clone -le 0.0) {
        $diagnosticInvalid += "bclone_first_clone_timing invocation=$invocation worker=$($record.worker)"
      }
    }
    foreach ($recordSet in @($warmupForInvocation, $affinityForInvocation)) {
      foreach ($record in @($recordSet | Sort-Object worker)) {
        if ($record.variant -ne $csvRow.variant -or $record.worker -lt 0 -or $record.worker -ge $Workers -or
            $record.cpu -ne $cpus[$record.worker] -or $record.row -ne $fixedRowValues[$record.worker] -or
            $record.elapsed -le 0.0) {
          $diagnosticInvalid += "warmup_or_affinity invocation=$invocation worker=$($record.worker)"
        }
      }
    }
  }
  if ($diagnosticInvalid.Count -gt 0) {
    throw "Invalid timing diagnostics: $($diagnosticInvalid -join '; ')"
  }
}

$readyTimingSummaryPath = Join-Path $OutputDirectory "ready-timing-summary.txt"
$blockTimingSummaryPath = Join-Path $OutputDirectory "block-timing-summary.txt"
$warmupTimingSummaryPath = Join-Path $OutputDirectory "warmup-timing-summary.txt"
$affinityTimingSummaryPath = Join-Path $OutputDirectory "affinity-timing-summary.txt"
$timingSummaryPath = Join-Path $OutputDirectory "timing-summary.txt"
$timingSummary = @(
  "T0-R ready and block timing summary"
  "ready_raw_worker_lines=$($readyWorkerRecords.Count)"
  "block_raw_worker_lines=$($blockWorkerRecords.Count)"
  "warmup_raw_worker_lines=$($warmupWorkerRecords.Count)"
  "affinity_raw_worker_lines=$($affinityWorkerRecords.Count)"
  "boundaries=block_loop is make_data block construction only; input fill and code outside make_data excluded"
)
$readySummary = @(
  "T0-R ready timing summary"
  "raw_worker_lines=$($readyWorkerRecords.Count)"
  "boundaries=existing ready timing boundary; retained unchanged"
)
$blockSummary = @(
  "T0-R block timing summary"
  "raw_worker_lines=$($blockWorkerRecords.Count)"
  "boundaries=QPC immediately before and after block-construction loop; Bclone copy/push operation only"
)
$warmupSummary = @(
  "T0-R warmup timing summary"
  "raw_worker_lines=$($warmupWorkerRecords.Count)"
  "boundaries=QPC immediately before and after existing warmup loop; no timing inside pass_function or barriers"
)
$affinitySummary = @(
  "T0-R AffinityGuard construction timing summary"
  "raw_worker_lines=$($affinityWorkerRecords.Count)"
  "boundaries=QPC immediately before AffinityGuard declaration through immediately after constructor returns"
)
foreach ($variant in $variants) {
  $readyValues = @($readyWorkerRecords | Where-Object { $_.variant -eq $variant } | ForEach-Object { $_.elapsed })
  $blockValues = @($blockWorkerRecords | Where-Object { $_.variant -eq $variant } | ForEach-Object { $_.block_loop })
  $firstCloneValues = @($blockWorkerRecords | Where-Object { $_.variant -eq $variant } | ForEach-Object { $_.first_clone })
  $remainingCloneValues = @($blockWorkerRecords | Where-Object { $_.variant -eq $variant } | ForEach-Object { $_.clone_remaining })
  $warmupValues = @($warmupWorkerRecords | Where-Object { $_.variant -eq $variant } | ForEach-Object { $_.elapsed })
  $affinityValues = @($affinityWorkerRecords | Where-Object { $_.variant -eq $variant } | ForEach-Object { $_.elapsed })
  $kernelValues = @($rows | Where-Object { $_.variant -eq $variant } | ForEach-Object { [double]$_.kernel_elapsed_seconds })
  $macValues = @($rows | Where-Object { $_.variant -eq $variant } | ForEach-Object { [double]$_.mac_per_second })
  $readyMean = ($readyValues | Measure-Object -Average).Average
  $blockMean = ($blockValues | Measure-Object -Average).Average
  $firstCloneMean = ($firstCloneValues | Measure-Object -Average).Average
  $remainingCloneMean = ($remainingCloneValues | Measure-Object -Average).Average
  $kernelMean = ($kernelValues | Measure-Object -Average).Average
  $macMean = ($macValues | Measure-Object -Average).Average
  $warmupMean = ($warmupValues | Measure-Object -Average).Average
  $affinityMean = ($affinityValues | Measure-Object -Average).Average
  $line = "$variant ready_min_seconds=$($readyValues | Measure-Object -Minimum | Select-Object -ExpandProperty Minimum) ready_max_seconds=$($readyValues | Measure-Object -Maximum | Select-Object -ExpandProperty Maximum) ready_mean_seconds=$readyMean block_loop_min_seconds=$($blockValues | Measure-Object -Minimum | Select-Object -ExpandProperty Minimum) block_loop_max_seconds=$($blockValues | Measure-Object -Maximum | Select-Object -ExpandProperty Maximum) block_loop_mean_seconds=$blockMean first_clone_min_seconds=$($firstCloneValues | Measure-Object -Minimum | Select-Object -ExpandProperty Minimum) first_clone_max_seconds=$($firstCloneValues | Measure-Object -Maximum | Select-Object -ExpandProperty Maximum) first_clone_mean_seconds=$firstCloneMean remaining_clone_min_seconds=$($remainingCloneValues | Measure-Object -Minimum | Select-Object -ExpandProperty Minimum) remaining_clone_max_seconds=$($remainingCloneValues | Measure-Object -Maximum | Select-Object -ExpandProperty Maximum) remaining_clone_mean_seconds=$remainingCloneMean kernel_elapsed_min_seconds=$($kernelValues | Measure-Object -Minimum | Select-Object -ExpandProperty Minimum) kernel_elapsed_max_seconds=$($kernelValues | Measure-Object -Maximum | Select-Object -ExpandProperty Maximum) kernel_elapsed_mean_seconds=$kernelMean mac_per_second_min=$($macValues | Measure-Object -Minimum | Select-Object -ExpandProperty Minimum) mac_per_second_max=$($macValues | Measure-Object -Maximum | Select-Object -ExpandProperty Maximum) mac_per_second_mean=$macMean"
  $timingSummary += $line
  $readySummary += "$variant ready_min_seconds=$($readyValues | Measure-Object -Minimum | Select-Object -ExpandProperty Minimum) ready_max_seconds=$($readyValues | Measure-Object -Maximum | Select-Object -ExpandProperty Maximum) ready_mean_seconds=$readyMean kernel_elapsed_min_seconds=$($kernelValues | Measure-Object -Minimum | Select-Object -ExpandProperty Minimum) kernel_elapsed_max_seconds=$($kernelValues | Measure-Object -Maximum | Select-Object -ExpandProperty Maximum) kernel_elapsed_mean_seconds=$kernelMean mac_per_second_min=$($macValues | Measure-Object -Minimum | Select-Object -ExpandProperty Minimum) mac_per_second_max=$($macValues | Measure-Object -Maximum | Select-Object -ExpandProperty Maximum) mac_per_second_mean=$macMean"
  $blockSummary += "$variant block_loop_min_seconds=$($blockValues | Measure-Object -Minimum | Select-Object -ExpandProperty Minimum) block_loop_max_seconds=$($blockValues | Measure-Object -Maximum | Select-Object -ExpandProperty Maximum) block_loop_mean_seconds=$blockMean first_clone_min_seconds=$($firstCloneValues | Measure-Object -Minimum | Select-Object -ExpandProperty Minimum) first_clone_max_seconds=$($firstCloneValues | Measure-Object -Maximum | Select-Object -ExpandProperty Maximum) first_clone_mean_seconds=$firstCloneMean remaining_clone_min_seconds=$($remainingCloneValues | Measure-Object -Minimum | Select-Object -ExpandProperty Minimum) remaining_clone_max_seconds=$($remainingCloneValues | Measure-Object -Maximum | Select-Object -ExpandProperty Maximum) remaining_clone_mean_seconds=$remainingCloneMean"
  $warmupSummary += "$variant warmup_min_seconds=$($warmupValues | Measure-Object -Minimum | Select-Object -ExpandProperty Minimum) warmup_max_seconds=$($warmupValues | Measure-Object -Maximum | Select-Object -ExpandProperty Maximum) warmup_mean_seconds=$warmupMean"
  $affinitySummary += "$variant affinity_guard_min_seconds=$($affinityValues | Measure-Object -Minimum | Select-Object -ExpandProperty Minimum) affinity_guard_max_seconds=$($affinityValues | Measure-Object -Maximum | Select-Object -ExpandProperty Maximum) affinity_guard_mean_seconds=$affinityMean"
}
Set-Content -LiteralPath $readyTimingSummaryPath -Encoding ascii -Value $readySummary
Set-Content -LiteralPath $blockTimingSummaryPath -Encoding ascii -Value $blockSummary
Set-Content -LiteralPath $warmupTimingSummaryPath -Encoding ascii -Value $warmupSummary
Set-Content -LiteralPath $affinityTimingSummaryPath -Encoding ascii -Value $affinitySummary
Set-Content -LiteralPath $timingSummaryPath -Encoding ascii -Value $timingSummary

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
  "ready_timing_log=$readyTimingLog"
  "ready_timing_summary_path=$readyTimingSummaryPath"
  "block_timing_log=$blockTimingLog"
  "block_timing_summary_path=$blockTimingSummaryPath"
  "warmup_timing_log=$warmupTimingLog"
  "warmup_timing_summary_path=$warmupTimingSummaryPath"
  "affinity_timing_log=$affinityTimingLog"
  "affinity_timing_summary_path=$affinityTimingSummaryPath"
  "timing_summary_path=$timingSummaryPath"
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
  "configuration=target_kib=$($sizes -join ',');K=512;m=target_kib*1024/K;depths=$($depths -join ',');iterations=1;repetitions=$Repetitions;warmup=$Warmup"
  "exact_fixed_rows=$($fixedRowValues -join ',')"
  "affinity=PASS;workers=$Workers;cpus=$($cpus -join ',');fixed_rows=$($fixedRowValues.Count -gt 0)"
  "avx2=PASS; every row avx2_supported=true and kernel_used=avx2"
  "repetitions=PASS; exact repetitions=$Repetitions and warmup=$Warmup"
  "checksums=PASS; nonzero and deterministic per variant; A==Bclone and B distinct per paired run"
  "order=PASS; deterministic alternating variant order; serial process invocations"
  "ready_timing=PASS; enabled=$ReadyTimingDiagnostic; stderr_log=$readyTimingLog"
  "block_timing=PASS; enabled=$ReadyTimingDiagnostic; worker_lines=$($blockWorkerRecords.Count); invalid=$($diagnosticInvalid.Count)"
  "warmup_timing=PASS; enabled=$WarmupAffinityTimingDiagnostic; worker_lines=$($warmupWorkerRecords.Count)"
  "affinity_timing=PASS; enabled=$WarmupAffinityTimingDiagnostic; worker_lines=$($affinityWorkerRecords.Count)"
  "diagnostic_counts=ready_worker_lines:$($readyWorkerRecords.Count);block_worker_lines:$($blockWorkerRecords.Count);warmup_worker_lines:$($warmupWorkerRecords.Count);affinity_worker_lines:$($affinityWorkerRecords.Count);expected_each:$($expectedInvocations * $Workers)"
  "preflight_self_test=PASS;log=$preflightLog"
)
Write-Output $summaryPath
