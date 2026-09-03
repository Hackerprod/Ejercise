[CmdletBinding()]
param(
  [string]$Executable = "",
  [string]$OutputDirectory = "",
  [ValidateSet("scalar", "avx2", "auto")]
  [string]$Kernel = "scalar",
  [int[]]$Cpus = @(0..7),
  [int[]]$TargetSizes = @(256, 384, 512, 640, 768, 896, 1024, 1280),
  [int[]]$Depths = @(1, 2, 4, 8, 16),
  [ValidateSet("A", "B", "C")]
  [string[]]$Variants = @("A", "B", "C"),
  [int]$Warmup = 1,
  [int]$Repetitions = 2,
  [int]$Iterations = 1
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($Executable)) {
  $Executable = Join-Path $projectRoot "build\cpu_native_q4_probe.exe"
}
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
  $OutputDirectory = Join-Path $projectRoot "sweep-output"
}

if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
  throw "Executable not found: $Executable"
}
if (-not (Test-Path -LiteralPath $OutputDirectory -PathType Container)) {
  $null = New-Item -ItemType Directory -Path $OutputDirectory
}

$rawCsv = Join-Path $OutputDirectory "h0_sweep.csv"
$stderrLog = Join-Path $OutputDirectory "h0_sweep.stderr.log"
$summaryPath = Join-Path $OutputDirectory "h0_sweep.summary.txt"
$cpus = $Cpus
$targetSizes = $TargetSizes
$depths = $Depths
$variants = $Variants
$warmup = $Warmup
$repetitions = $Repetitions
$iterations = $Iterations
$expectedInvocations = $cpus.Count * $targetSizes.Count * $depths.Count * $variants.Count

Remove-Item -LiteralPath $rawCsv, $stderrLog, $summaryPath -Force -ErrorAction SilentlyContinue
Set-Content -LiteralPath $stderrLog -Encoding ascii -Value @(
  "H0 sweep stderr and command log"
  "executable=$Executable"
  "kernel=$Kernel"
  "K=512; iterations=$iterations; repetitions=$repetitions; warmup=$warmup"
  "cpus=$($cpus -join ','); target_kib=$($targetSizes -join ','); depths=$($depths -join ','); variants=$($variants -join ',')"
)

$headerWritten = $false
$invocation = 0
try {
  foreach ($cpu in $cpus) {
    foreach ($targetKib in $targetSizes) {
      foreach ($depth in $depths) {
        foreach ($variant in $variants) {
          $invocation++
          $arguments = @(
            "--cpu", $cpu,
            "--K", 512,
            "--target-kib", $targetKib,
            "--depth", $depth,
            "--variant", $variant,
            "--kernel", $Kernel,
            "--iterations", $iterations,
            "--repetitions", $repetitions,
            "--warmup", $warmup
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
            if ($lines.Count -ne 2 -or [string]::IsNullOrWhiteSpace($lines[0]) -or
                [string]::IsNullOrWhiteSpace($lines[1])) {
              throw "Probe did not produce exactly one CSV row: $commandText"
            }
            if (-not $headerWritten) {
              Set-Content -LiteralPath $rawCsv -Encoding ascii -Value $lines[0]
              $headerWritten = $true
            } elseif ($lines[0] -ne (Get-Content -LiteralPath $rawCsv -TotalCount 1)) {
              throw "Probe CSV header changed: $commandText"
            }
            Add-Content -LiteralPath $rawCsv -Encoding ascii -Value $lines[1]
          } finally {
            $process.Dispose()
          }
        }
      }
    }
  }
} finally { }

$rows = @(Import-Csv -LiteralPath $rawCsv)
$classificationTarget = ($targetSizes | Sort-Object | Select-Object -First 1)
$classicRows = @($rows | Where-Object {
    $_.target_kib -eq "$classificationTarget" -and $_.variant -eq "A"
  })
if ($classicRows.Count -eq 0) {
  throw "No 256 KiB variant A rows found for classification"
}
$winner = $classicRows | Sort-Object { [double]$_.mac_per_second } -Descending | Select-Object -First 1
$summary = @(
  "T0 H0 sweep summary"
  "raw_csv=$rawCsv"
  "stderr_log=$stderrLog"
  "executable=$Executable"
  ('command_template="{0}" --cpu <{1}> --K 512 --target-kib <{2}> --depth <{3}> --variant <{4}> --kernel {5} --iterations {6} --repetitions {7} --warmup {8}' -f $Executable, ($cpus -join ','), ($targetSizes -join ','), ($depths -join ','), ($variants -join ','), $Kernel, $iterations, $repetitions, $warmup)
  "mapping=K=512; actual bytes per block = ceil(m*K/2) + 4*ceil(m*K/32); m is largest positive value not exceeding target_kib*1024"
  "expected_invocations=$expectedInvocations"
  "actual_rows=$($rows.Count)"
  "classification=$classificationTarget KiB variant A; fastest logical CPU is empirical Classic candidate, not hard-coded"
  "classic_candidate.logical_cpu_index=$($winner.logical_cpu_index)"
  "classic_candidate.mac_per_second=$($winner.mac_per_second)"
  "classic_candidate.processor_group=$($winner.processor_group)"
  "classic_candidate.group_processor_index=$($winner.group_processor_index)"
  "caveat=$Kernel Q4 probe; approximate resident working set; SMT, scheduler, boost, thermal, and affinity state affect results"
)
Set-Content -LiteralPath $summaryPath -Encoding ascii -Value $summary
Write-Output $summaryPath
