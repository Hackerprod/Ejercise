[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$outputDirectory = $PSScriptRoot
$oldExecutable = Join-Path $projectRoot "archive\t0r\int8_probe.exe"
$newExecutable = Join-Path $projectRoot "build\t0m_int8_probe.exe"
$commandLog = Join-Path $outputDirectory "commands.log"
$outputLog = Join-Path $outputDirectory "outputs.log"
$machineCsv = Join-Path $outputDirectory "diagnostic.machine.csv"
$summaryPath = Join-Path $outputDirectory "diagnostic.summary.txt"
$metadataPath = Join-Path $outputDirectory "diagnostic.metadata.txt"

if (-not (Test-Path -LiteralPath $oldExecutable -PathType Leaf)) {
  throw "Executable not found: $oldExecutable"
}
if (-not (Test-Path -LiteralPath $newExecutable -PathType Leaf)) {
  throw "Executable not found: $newExecutable"
}
$existingArtifacts = @(
  (Test-Path -LiteralPath $commandLog -PathType Leaf),
  (Test-Path -LiteralPath $outputLog -PathType Leaf),
  (Test-Path -LiteralPath $machineCsv -PathType Leaf),
  (Test-Path -LiteralPath $summaryPath -PathType Leaf)
)
if ($existingArtifacts -contains $true) {
  throw "Refusing to overwrite existing diagnostic artifacts: $outputDirectory"
}

$cpus = "0,2,4,6"
$calibration = [ordered]@{ 0 = 19.3; 2 = 18.1; 4 = 10.9; 6 = 17.0 }
$rowsBySize = @{
  512 = @(1211, 1136, 683, 1066)
  768 = @(1816, 1705, 1023, 1600)
}
$sizes = @(512, 768)
$depths = @(1, 4, 8, 16)
$variants = @("A", "B")
$script:invocation = 0
$script:records = @()

$calibrationText = ($calibration.GetEnumerator() | ForEach-Object { "cpu$($_.Key)=$($_.Value)" }) -join ", "
$rowText = ($rowsBySize.GetEnumerator() | Sort-Object { [int]$_.Key } | ForEach-Object {
    "target_kib=$($_.Key); total_rows=$((4 * ([int]$_.Key * 1024 / 512))); rows_cpu_0_2_4_6=$($_.Value -join ','); sum=$((($_.Value | Measure-Object -Sum).Sum))"
  }) -join "`r`n"
Set-Content -LiteralPath $metadataPath -Encoding ascii -Value @(
  "T0-M proportional sharding bounded diagnostic"
  "scope=old archive t0r versus new t0m executable; no Phase 3 campaign rerun; no Phase 4; no source changes"
  "old_executable=$oldExecutable"
  "new_executable=$newExecutable"
  "CPUs=$cpus; D=512; S=1; mode=fused; S_tile=8; iterations=1; repetitions=8; warmup=2"
  "calibration_rule=proportional rows use exact requested prior calibration GMAC/s as weights in CPU order 0,2,4,6; integer rows use supplied expected allocations while preserving exact total"
  "calibration=$calibrationText"
  "total_rows_rule=4*(target_kib*1024/512)"
  $rowText
  "ratio_rule=B_over_A=MAC/s(B)/MAC/s(A); A_over_B=MAC/s(A)/MAC/s(B); checks within same executable, size, and R only"
  "checksum_rule=do not compare checksums across executables because data seeds differ"
)
Set-Content -LiteralPath $commandLog -Encoding ascii -Value "T0-M proportional sharding diagnostic exact commands"
Set-Content -LiteralPath $outputLog -Encoding ascii -Value "T0-M proportional sharding diagnostic command outputs"

function Get-NonEmptyLines([string]$Text) {
  return @($Text -split '\r?\n' | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
}

function Assert-Field([pscustomobject]$Row, [string]$Name, [string]$Expected, [string]$Context) {
  if (-not ($Row.psobject.Properties.Name -contains $Name)) {
    throw "Missing field $($Name): $Context"
  }
  if ([string]$Row.$Name -ne $Expected) {
    throw "Invalid $($Name)=$($Row.$Name), expected $($Expected): $Context"
  }
}

function Invoke-Logged([string]$Label, [string]$Executable, [string[]]$Arguments,
                       [int]$Size, [int]$Depth, [string]$Variant, [string]$Rows,
                       [string]$ExecutableKind) {
  $script:invocation++
  $id = $script:invocation
  $command = '"' + $Executable + '" ' + ($Arguments -join ' ')
  Add-Content -LiteralPath $commandLog -Encoding ascii -Value "command[$id] $command"

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
    if (-not $process.Start()) { throw "Could not start $Label" }
    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()
    $process.WaitForExit()
    $stdout = $stdoutTask.Result
    $stderr = $stderrTask.Result
    Add-Content -LiteralPath $outputLog -Encoding ascii -Value @(
      "invocation[$id] label=$Label"
      "command[$id] $command"
      "exit_code=$($process.ExitCode)"
      "stdout[$id] $($stdout.TrimEnd())"
      "stderr[$id] $($stderr.TrimEnd())"
    )
    if ($process.ExitCode -ne 0) {
      throw "Probe failed with exit code $($process.ExitCode): $Label"
    }
  } finally {
    $process.Dispose()
  }

  $lines = Get-NonEmptyLines $stdout
  if ($lines.Count -ne 2) {
    throw "Expected CSV header plus one row, got $($lines.Count): $Label"
  }
  $row = $lines[1] | ConvertFrom-Csv -Header ($lines[0] -split ',')
  if ($ExecutableKind -eq "old") {
    Assert-Field $row "kernel_requested" "avx2" $Label
    Assert-Field $row "kernel_used" "avx2" $Label
    Assert-Field $row "avx2_supported" "true" $Label
    Assert-Field $row "target_kib" "$Size" $Label
    Assert-Field $row "K" "512" $Label
    Assert-Field $row "depth" "$Depth" $Label
    Assert-Field $row "variant" $Variant $Label
    Assert-Field $row "iterations" "1" $Label
    Assert-Field $row "repetitions" "8" $Label
    Assert-Field $row "warmup" "2" $Label
    Assert-Field $row "worker_count" "4" $Label
    Assert-Field $row "logical_cpu_indices" $cpus $Label
    Assert-Field $row "rows_per_worker" $Rows $Label
    Assert-Field $row "all_affinity_succeeded" "true" $Label
    Assert-Field $row "affinity_error" "0" $Label
    $macField = "mac_count"
    $rateField = "mac_per_second"
  } else {
    Assert-Field $row "D" "512" $Label
    Assert-Field $row "S" "1" $Label
    Assert-Field $row "R" "$Depth" $Label
    Assert-Field $row "mode" "fused" $Label
    Assert-Field $row "variant" $Variant $Label
    Assert-Field $row "S_tile" "8" $Label
    Assert-Field $row "iterations" "1" $Label
    Assert-Field $row "timed_repetitions" "8" $Label
    Assert-Field $row "warmup" "2" $Label
    Assert-Field $row "worker_count" "4" $Label
    Assert-Field $row "worker_list" $cpus $Label
    Assert-Field $row "rows_per_worker" $Rows $Label
    Assert-Field $row "affinity_succeeded" "true" $Label
    Assert-Field $row "affinity" "true,true,true,true" $Label
    Assert-Field $row "affinity_error" "0,0,0,0" $Label
    Assert-Field $row "timed_repetitions_exact" "true" $Label
    Assert-Field $row "avx2_supported" "true" $Label
    Assert-Field $row "kernel_used" "avx2" $Label
    $macField = "mac_total"
    $rateField = "mac_per_second"
    if ($stderr -notmatch "T0-M correction passed") {
      throw "Correction gate evidence missing: $Label"
    }
  }

  $record = [ordered]@{
    executable = $ExecutableKind
    target_kib = $Size
    R = $Depth
    variant = $Variant
    rows_per_worker = $Rows
    mac_count = [string]$row.$macField
    mac_per_second = [string]$row.$rateField
    kernel_requested = if ($ExecutableKind -eq "old") { [string]$row.kernel_requested } else { "n/a" }
    kernel_used = [string]$row.kernel_used
    avx2_supported = [string]$row.avx2_supported
    affinity_validated = "true"
  }
  $script:records += [pscustomobject]$record
}

foreach ($size in $sizes) {
  $rows = $rowsBySize[$size] -join ','
  foreach ($depth in $depths) {
    foreach ($variant in $variants) {
      Invoke-Logged "old size=$size R=$depth variant=$variant" $oldExecutable @(
        "--K", 512, "--target-kib", $size, "--depth", $depth, "--variant", $variant,
        "--kernel", "avx2", "--parallel-cpus", $cpus, "--parallel-rows", $rows,
        "--iterations", 1, "--repetitions", 8, "--warmup", 2
      ) $size $depth $variant $rows "old"
      Invoke-Logged "new size=$size R=$depth variant=$variant" $newExecutable @(
        "--D", 512, "--S", 1, "--R", $depth, "--mode", "fused", "--variant", $variant,
        "--S-tile", 8, "--workers", 4, "--cpus", $cpus, "--rows-per-worker", $rows,
        "--iterations", 1, "--timed-repetitions", 8, "--warmup", 2
      ) $size $depth $variant $rows "new"
    }
  }
}

$ratioLines = @()
foreach ($kind in @("old", "new")) {
  foreach ($size in $sizes) {
    foreach ($depth in $depths) {
      $a = @($script:records | Where-Object { $_.executable -eq $kind -and [int]$_.target_kib -eq $size -and [int]$_.R -eq $depth -and $_.variant -eq "A" })
      $b = @($script:records | Where-Object { $_.executable -eq $kind -and [int]$_.target_kib -eq $size -and [int]$_.R -eq $depth -and $_.variant -eq "B" })
      if ($a.Count -ne 1 -or $b.Count -ne 1) { throw "Expected one A and B row: $kind size=$size R=$depth" }
      $bOverA = [double]$b[0].mac_per_second / [double]$a[0].mac_per_second
      $aOverB = [double]$a[0].mac_per_second / [double]$b[0].mac_per_second
      $ratioLines += "ratio executable=$kind target_kib=$size R=$depth B_over_A=$($bOverA.ToString('R',[Globalization.CultureInfo]::InvariantCulture)) A_over_B=$($aOverB.ToString('R',[Globalization.CultureInfo]::InvariantCulture))"
    }
  }
}

function Get-Ratio([string]$Kind, [int]$Size) {
  $a = @($script:records | Where-Object { $_.executable -eq $Kind -and [int]$_.target_kib -eq $Size -and [int]$_.R -eq 16 -and $_.variant -eq "A" })
  $b = @($script:records | Where-Object { $_.executable -eq $Kind -and [int]$_.target_kib -eq $Size -and [int]$_.R -eq 16 -and $_.variant -eq "B" })
  return [double]$a[0].mac_per_second / [double]$b[0].mac_per_second
}

$old512 = Get-Ratio "old" 512
$old768 = Get-Ratio "old" 768
$new512 = Get-Ratio "new" 512
$new768 = Get-Ratio "new" 768
$newRestores = $new512 -ge 2.5 -and $new512 -le 2.9 -and $new768 -ge 2.5 -and $new768 -le 2.9
$oldRestores = $old512 -ge 2.5 -and $old512 -le 2.9 -and $old768 -ge 2.5 -and $old768 -le 2.9

$machineRows = $script:records | ConvertTo-Csv -NoTypeInformation
Set-Content -LiteralPath $machineCsv -Encoding ascii -Value $machineRows
Set-Content -LiteralPath $summaryPath -Encoding ascii -Value @(
  "T0-M proportional sharding bounded diagnostic"
  "status=DIAGNOSTIC_COMPLETE"
  "scope=old archive t0r versus new t0m; Phase 3 campaign not rerun; Phase 4 not run; MRDL/Q4/later stages untouched; barrier implementation untouched"
  "invocations=$($script:invocation); expected=32; validation=all requested affinity, AVX2, explicit kernel, shape, repetition, and mode fields passed"
  "old_executable=$oldExecutable"
  "new_executable=$newExecutable"
  "CPUs=$cpus; D=512; S=1; mode=fused; S_tile=8; iterations=1; repetitions=8; warmup=2"
  "calibration_rule=proportional rows use exact requested prior calibration GMAC/s as weights in CPU order 0,2,4,6; supplied expected integer allocations preserve exact total"
  "calibration=$calibrationText"
  "total_rows_rule=4*(target_kib*1024/512)"
  $rowText
  "ratio_rule=B_over_A=MAC/s(B)/MAC/s(A); A_over_B=MAC/s(A)/MAC/s(B); same executable, target size, and R"
  "checksum_policy=checksums captured but never compared across executables because data seeds differ"
  $ratioLines
  "R16_expected_A_over_B_range=2.5..2.9"
  "R16_new_512_A_over_B=$($new512.ToString('R',[Globalization.CultureInfo]::InvariantCulture)); R16_new_768_A_over_B=$($new768.ToString('R',[Globalization.CultureInfo]::InvariantCulture)); new_restores_expected_range=$newRestores"
  "R16_old_512_A_over_B=$($old512.ToString('R',[Globalization.CultureInfo]::InvariantCulture)); R16_old_768_A_over_B=$($old768.ToString('R',[Globalization.CultureInfo]::InvariantCulture)); old_restores_expected_range=$oldRestores"
  "interpretation=diagnosis only; do not claim Phase 3 PASS"
  "machine_csv=$machineCsv"
  "commands_log=$commandLog"
  "outputs_log=$outputLog"
  "metadata=$metadataPath"
)
Write-Output $summaryPath
