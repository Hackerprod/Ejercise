[CmdletBinding()]
param(
  [string]$Executable = "",
  [string]$OutputDirectory = "",
  [int]$Workers = 8,
  [int]$Repetitions = 8,
  [int]$Warmup = 2,
  [int]$RepeatCount = 1,
  [switch]$AlternateVariantOrder,
  [int[]]$LogicalCpuIndices = @()
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($Executable)) {
  $Executable = Join-Path $projectRoot "build\cpu_native_q4_probe.exe"
}
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
  $OutputDirectory = Join-Path $projectRoot "sweep-output\h0-parallel-avx2"
}
if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
  throw "Executable not found: $Executable"
}
if ($Workers -le 0) {
  throw "Workers must be positive"
}
if ($LogicalCpuIndices.Count -gt 0 -and $LogicalCpuIndices.Count -ne $Workers) {
  throw "LogicalCpuIndices count must match Workers"
}
if (-not (Test-Path -LiteralPath $OutputDirectory -PathType Container)) {
  $null = New-Item -ItemType Directory -Path $OutputDirectory
}

$rawCsv = Join-Path $OutputDirectory "h0_parallel.csv"
$stderrLog = Join-Path $OutputDirectory "h0_parallel.stderr.log"
$summaryPath = Join-Path $OutputDirectory "h0_parallel.summary.txt"
$sizes = @(128, 192, 512, 1024, 1280)
$depths = @(4, 8, 16)
$variants = @("A", "B")
$expectedInvocations = $sizes.Count * $depths.Count * $variants.Count * $RepeatCount
$parallelSelector = if ($LogicalCpuIndices.Count -gt 0) {
  "--parallel-cpus"
} else {
  "--parallel-workers"
}
$parallelValue = if ($LogicalCpuIndices.Count -gt 0) {
  $LogicalCpuIndices -join ','
} else {
  "$Workers"
}

Remove-Item -LiteralPath $rawCsv, $stderrLog, $summaryPath -Force -ErrorAction SilentlyContinue
Set-Content -LiteralPath $stderrLog -Encoding ascii -Value @(
  "H0 parallel sweep stderr and command log"
  "executable=$Executable"
  "kernel=avx2; workers=$Workers; K=512; repetitions=$Repetitions; warmup=$Warmup; repeat_count=$RepeatCount; alternate_variant_order=$AlternateVariantOrder"
  "target_kib=$($sizes -join ','); depths=$($depths -join ','); variants=$($variants -join ',')"
)

$headerWritten = $false
$probeHeader = $null
$invocation = 0
try {
  for ($repeat = 1; $repeat -le $RepeatCount; $repeat++) {
    foreach ($size in $sizes) {
      foreach ($depth in $depths) {
        $orderedVariants = $variants
        if ($AlternateVariantOrder -and ($repeat % 2 -eq 0)) {
          $orderedVariants = @($variants[($variants.Count - 1)..0])
        }
        $orderPosition = 0
        foreach ($variant in $orderedVariants) {
          $orderPosition++
        $invocation++
        $arguments = @(
          "--K", 512,
          "--target-kib", $size,
          "--depth", $depth,
          "--variant", $variant,
          "--kernel", "avx2",
          $parallelSelector, $parallelValue,
          "--iterations", 1,
          "--repetitions", $Repetitions,
          "--warmup", $Warmup
        )
        $commandText = '"{0}" {1}' -f $Executable, ($arguments -join ' ')
        Add-Content -LiteralPath $stderrLog -Encoding ascii -Value "command[$invocation/$expectedInvocations]=$commandText"

        $startInfo = New-Object System.Diagnostics.ProcessStartInfo
        $startInfo.FileName = $Executable
        $startInfo.Arguments = ($arguments -join ' ')
        $startInfo.UseShellExecute = $false
        $startInfo.CreateNoWindow = $true
        $startInfo.RedirectStandardOutput = $true
        $startInfo.RedirectStandardError = $true
        $process = New-Object System.Diagnostics.Process
        $process.StartInfo = $startInfo
        try {
          if (-not $process.Start()) {
            throw "Could not start probe: $commandText"
          }
          $stdoutTask = $process.StandardOutput.ReadToEndAsync()
          $stderrTask = $process.StandardError.ReadToEndAsync()
          $process.WaitForExit()
          $stdout = $stdoutTask.Result
          $diagnostics = $stderrTask.Result
          if (-not [string]::IsNullOrWhiteSpace($diagnostics)) {
            Add-Content -LiteralPath $stderrLog -Encoding ascii -Value ($diagnostics.TrimEnd())
          }
          if ($process.ExitCode -ne 0) {
            throw "Probe failed with exit code $($process.ExitCode): $commandText"
          }
          $lines = @($stdout -split '\r?\n' | Where-Object {
              -not [string]::IsNullOrWhiteSpace($_)
            })
          if ($lines.Count -ne 2) {
            throw "Probe did not produce exactly one batch CSV row: $commandText"
          }
          if (-not $headerWritten) {
            $probeHeader = $lines[0]
            Set-Content -LiteralPath $rawCsv -Encoding ascii -Value ($probeHeader + ",batch_repeat,variant_order")
            $headerWritten = $true
          } elseif ($lines[0] -ne $probeHeader) {
            throw "Probe CSV header changed: $commandText"
          }
          Add-Content -LiteralPath $rawCsv -Encoding ascii -Value ($lines[1] + ",$repeat,$orderPosition")
        } finally {
          $process.Dispose()
        }
        }
      }
    }
  }
} finally { }

$rows = @(Import-Csv -LiteralPath $rawCsv)
if ($rows.Count -ne $expectedInvocations) {
  throw "Expected $expectedInvocations rows, got $($rows.Count)"
}
$invalid = @($rows | Where-Object {
    $_.worker_count -ne "$Workers" -or $_.kernel_used -ne "avx2" -or
    $_.avx2_supported -ne "true" -or $_.fma_supported -ne "true" -or
    $_.all_affinity_succeeded -ne "true"
  })
if ($invalid.Count -gt 0) {
  throw "Parallel sweep contains $($invalid.Count) invalid rows"
}

$statistics = foreach ($size in $sizes) {
  foreach ($depth in $depths) {
    $a = @($rows | Where-Object { $_.target_kib -eq "$size" -and $_.depth -eq "$depth" -and $_.variant -eq "A" } | ForEach-Object { [double]$_.batch_mac_per_second })
    $b = @($rows | Where-Object { $_.target_kib -eq "$size" -and $_.depth -eq "$depth" -and $_.variant -eq "B" } | ForEach-Object { [double]$_.batch_mac_per_second })
    $aMean = ($a | Measure-Object -Average).Average
    $bMean = ($b | Measure-Object -Average).Average
    $aSd = [math]::Sqrt((($a | ForEach-Object { ($_ - $aMean) * ($_ - $aMean) } | Measure-Object -Average).Average))
    $bSd = [math]::Sqrt((($b | ForEach-Object { ($_ - $bMean) * ($_ - $bMean) } | Measure-Object -Average).Average))
    $bOverA = $bMean / $aMean
    "target_kib=$size depth=$depth A_mean=$aMean A_sd=$aSd A_min=$($a | Measure-Object -Minimum | Select-Object -ExpandProperty Minimum) A_max=$($a | Measure-Object -Maximum | Select-Object -ExpandProperty Maximum) B_mean=$bMean B_sd=$bSd B_min=$($b | Measure-Object -Minimum | Select-Object -ExpandProperty Minimum) B_max=$($b | Measure-Object -Maximum | Select-Object -ExpandProperty Maximum) B_over_A=$bOverA"
  }
}
$summary = @(
  "T0 parallel H0 AVX2 sweep summary"
  "raw_csv=$rawCsv"
  "stderr_log=$stderrLog"
  "executable=$Executable"
  "workers=$Workers; logical_cpu_indices=$(if ($LogicalCpuIndices.Count -gt 0) { $LogicalCpuIndices -join ',' } else { "0..$($Workers - 1)" }); measurements_are_wall_clock_batch_totals"
  "K=512; target_kib=$($sizes -join ','); depths=$($depths -join ','); variants=$($variants -join ','); kernel=avx2; repetitions=$Repetitions; warmup=$Warmup"
  "expected_invocations=$expectedInvocations"
  "actual_rows=$($rows.Count)"
  "invalid_rows=$($invalid.Count)"
  "statistics=population standard deviation and min/max across repeated batches per size/depth/variant"
  $statistics
  "caveat=$Workers logical workers are simultaneous; physical-core/SMT identity is caller-selected; thermal, scheduler, and DRAM state affect results"
)
Set-Content -LiteralPath $summaryPath -Encoding ascii -Value $summary
Write-Output $summaryPath
