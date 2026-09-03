[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$outputDirectory = $PSScriptRoot
$projectRoot = Split-Path -Parent (Split-Path -Parent $outputDirectory)
$executable = Join-Path $projectRoot "build\t0m_int8_probe.exe"
$commandLog = Join-Path $outputDirectory "commands.log"
$outputLog = Join-Path $outputDirectory "outputs.log"
$machineCsv = Join-Path $outputDirectory "diagnostic.machine.csv"
$summaryPath = Join-Path $outputDirectory "diagnostic.summary.txt"
$metadataPath = Join-Path $outputDirectory "diagnostic.metadata.txt"

if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
  throw "Executable not found: $executable"
}
$existingArtifacts = @(
  (Test-Path -LiteralPath $commandLog -PathType Leaf),
  (Test-Path -LiteralPath $outputLog -PathType Leaf),
  (Test-Path -LiteralPath $machineCsv -PathType Leaf),
  (Test-Path -LiteralPath $summaryPath -PathType Leaf),
  (Test-Path -LiteralPath $metadataPath -PathType Leaf)
)
if ($existingArtifacts -contains $true) {
  throw "Refusing to overwrite existing diagnostic artifacts: $outputDirectory"
}

$cpus = "0,2,4,6"
$rowsBySize = @{
  512 = @(1211, 1136, 683, 1066)
  768 = @(1816, 1705, 1023, 1600)
}
$sizes = @(512, 768)
$depths = @(1, 4, 8, 16)
$variants = @("A", "B")
$script:invocation = 0
$script:records = @()

$rowText = ($rowsBySize.GetEnumerator() | Sort-Object { [int]$_.Key } | ForEach-Object {
    "target_kib=$($_.Key); rows_cpu_0_2_4_6=$($_.Value -join ','); sum=$((($_.Value | Measure-Object -Sum).Sum))"
  }) -join "`r`n"
Set-Content -LiteralPath $metadataPath -Encoding ascii -Value @(
  "T0-M per-depth barrier bounded diagnostic"
  "scope=new t0m executable only; no old executable; no Phase 3 campaign; no Phase 4; MRDL/Q4/later stages untouched"
  "executable=$executable"
  "CPUs=$cpus; D=512; S=1; mode=fused; S_tile=8; iterations=1; timed_repetitions=8; warmup=2"
  "calibration_rows=target 512 [1211,1136,683,1066]; target 768 [1816,1705,1023,1600] in CPU order 0,2,4,6"
  $rowText
  "ratio_rule=B_over_A=MAC/s(B)/MAC/s(A); A_over_B=MAC/s(A)/MAC/s(B); same size and R"
  "gate_rule=R1 B_over_A in 0.97..1.03; R16 A_over_B in 2.5..2.9"
  "timing_rule=A/B non-C elapsed sums synchronized phase_ready/phase_done kernel phases only; no eviction prep"
)
Set-Content -LiteralPath $commandLog -Encoding ascii -Value "T0-M per-depth barrier diagnostic exact commands"
Set-Content -LiteralPath $outputLog -Encoding ascii -Value "T0-M per-depth barrier diagnostic command outputs"

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

function Invoke-Logged([string]$Label, [int]$Size, [int]$Depth, [string]$Variant, [string]$Rows) {
  $script:invocation++
  $id = $script:invocation
  $arguments = @(
    "--D", 512, "--S", 1, "--R", $Depth, "--mode", "fused", "--variant", $Variant,
    "--S-tile", 8, "--workers", 4, "--cpus", $cpus, "--rows-per-worker", $Rows,
    "--iterations", 1, "--timed-repetitions", 8, "--warmup", 2
  )
  $command = '"' + $executable + '" ' + ($arguments -join ' ')
  Add-Content -LiteralPath $commandLog -Encoding ascii -Value "command[$id] $command"

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
  Assert-Field $row "eviction_bytes" "0" $Label
  Assert-Field $row "eviction_checksum" "0" $Label
  $expectedMacTotal = ([int64]($Rows.Split(',') | ForEach-Object { [int64]$_ } | Measure-Object -Sum).Sum) * 512 * $Depth * 8
  Assert-Field $row "mac_total" "$expectedMacTotal" $Label
  if ([uint64]$row.checksum -eq 0) { throw "Checksum unexpectedly zero: $Label" }
  if ([double]$row.elapsed_seconds -le 0 -or [double]$row.mac_per_second -le 0) {
    throw "Timing/output rate invalid: $Label"
  }
  $script:records += [pscustomobject][ordered]@{
    target_kib = $Size
    R = $Depth
    variant = $Variant
    rows_per_worker = $Rows
    elapsed_seconds = [string]$row.elapsed_seconds
    mac_total = [string]$row.mac_total
    mac_per_second = [string]$row.mac_per_second
    checksum = [string]$row.checksum
    eviction_checksum = [string]$row.eviction_checksum
    affinity_validated = "true"
    avx2_validated = "true"
  }
}

foreach ($size in $sizes) {
  $rows = $rowsBySize[$size] -join ','
  foreach ($depth in $depths) {
    foreach ($variant in $variants) {
      Invoke-Logged "size=$size R=$depth variant=$variant" $size $depth $variant $rows
    }
  }
}

$ratioLines = @()
foreach ($size in $sizes) {
  foreach ($depth in $depths) {
    $a = @($script:records | Where-Object { [int]$_.target_kib -eq $size -and [int]$_.R -eq $depth -and $_.variant -eq "A" })
    $b = @($script:records | Where-Object { [int]$_.target_kib -eq $size -and [int]$_.R -eq $depth -and $_.variant -eq "B" })
    if ($a.Count -ne 1 -or $b.Count -ne 1) { throw "Expected one A and B row: size=$size R=$depth" }
    $bOverA = [double]$b[0].mac_per_second / [double]$a[0].mac_per_second
    $aOverB = [double]$a[0].mac_per_second / [double]$b[0].mac_per_second
    $ratioLines += "ratio target_kib=$size R=$depth B_over_A=$($bOverA.ToString('R',[Globalization.CultureInfo]::InvariantCulture)) A_over_B=$($aOverB.ToString('R',[Globalization.CultureInfo]::InvariantCulture))"
  }
}

function Get-Ratio([int]$Size, [int]$Depth, [string]$Numerator, [string]$Denominator) {
  $n = @($script:records | Where-Object { [int]$_.target_kib -eq $Size -and [int]$_.R -eq $Depth -and $_.variant -eq $Numerator })
  $d = @($script:records | Where-Object { [int]$_.target_kib -eq $Size -and [int]$_.R -eq $Depth -and $_.variant -eq $Denominator })
  return [double]$n[0].mac_per_second / [double]$d[0].mac_per_second
}

$r1_512 = Get-Ratio 512 1 "B" "A"
$r1_768 = Get-Ratio 768 1 "B" "A"
$r16_512 = Get-Ratio 512 16 "A" "B"
$r16_768 = Get-Ratio 768 16 "A" "B"
$r1Gate = $r1_512 -ge 0.97 -and $r1_512 -le 1.03 -and $r1_768 -ge 0.97 -and $r1_768 -le 1.03
$r16Gate = $r16_512 -ge 2.5 -and $r16_512 -le 2.9 -and $r16_768 -ge 2.5 -and $r16_768 -le 2.9
$interpretation = if (-not $r1Gate) {
  "STOPPED_R1_GATE_FAIL; no barrier-separation interpretation"
} elseif (-not $r16Gate) {
  "STOPPED_R16_GATE_FAIL; no barrier-separation interpretation"
} else {
  "BARRIER_RESTORES_SEPARATION; bounded diagnostic only; not Phase 3 PASS"
}

$machineRows = $script:records | ConvertTo-Csv -NoTypeInformation
Set-Content -LiteralPath $machineCsv -Encoding ascii -Value $machineRows
Set-Content -LiteralPath $summaryPath -Encoding ascii -Value @(
  "T0-M per-depth barrier bounded diagnostic"
  "status=DIAGNOSTIC_COMPLETE"
  "scope=new t0m executable only; Phase 3 campaign not run; Phase 4 not run; MRDL/Q4/later stages untouched"
  "invocations=$($script:invocation); expected=16; validation=affinity, AVX2, shape, mode, repetition, MAC total, output checksum, no-eviction fields"
  "CPUs=$cpus; D=512; S=1; mode=fused; S_tile=8; iterations=1; timed_repetitions=8; warmup=2"
  $rowText
  "ratio_rule=B_over_A=MAC/s(B)/MAC/s(A); A_over_B=MAC/s(A)/MAC/s(B); same size and R"
  $ratioLines
  "R1_B_over_A_expected_range=0.97..1.03; target_512=$($r1_512.ToString('R',[Globalization.CultureInfo]::InvariantCulture)); target_768=$($r1_768.ToString('R',[Globalization.CultureInfo]::InvariantCulture)); gate=$r1Gate"
  "R16_A_over_B_expected_range=2.5..2.9; target_512=$($r16_512.ToString('R',[Globalization.CultureInfo]::InvariantCulture)); target_768=$($r16_768.ToString('R',[Globalization.CultureInfo]::InvariantCulture)); gate=$r16Gate"
  "gradual_sequence_target_512_R1_R4_R8_R16=$((@(1,4,8,16) | ForEach-Object { $a = Get-Ratio 512 $_ 'A' 'B'; $a.ToString('R',[Globalization.CultureInfo]::InvariantCulture) }) -join ',')"
  "gradual_sequence_target_768_R1_R4_R8_R16=$((@(1,4,8,16) | ForEach-Object { $a = Get-Ratio 768 $_ 'A' 'B'; $a.ToString('R',[Globalization.CultureInfo]::InvariantCulture) }) -join ',')"
  "interpretation=$interpretation"
  "machine_csv=$machineCsv"
  "commands_log=$commandLog"
  "outputs_log=$outputLog"
  "metadata=$metadataPath"
)
Write-Output $summaryPath
