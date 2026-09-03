[CmdletBinding()]
param(
  [string]$Executable = "",
  [string]$OutputDirectory = "",
  [string]$CTestExecutable = "",
  [string]$DumpbinExecutable = "",
  [int]$TimedRepetitions = 8,
  [int]$Warmup = 2,
  [int]$IndependentRuns = 10
)

$ErrorActionPreference = "Stop"
[System.Threading.Thread]::CurrentThread.CurrentCulture = [Globalization.CultureInfo]::InvariantCulture
[System.Threading.Thread]::CurrentThread.CurrentUICulture = [Globalization.CultureInfo]::InvariantCulture
$projectRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($Executable)) { $Executable = Join-Path $projectRoot "build\t0m_recurrence_probe.exe" }
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) { $OutputDirectory = Join-Path $projectRoot "sweep-output\t0m-recurrence-component-breakdown" }
if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) { throw "Executable not found: $Executable" }
if ($TimedRepetitions -ne 8 -or $Warmup -ne 2 -or $IndependentRuns -ne 10) {
  throw "Approved breakdown requires --timed-repetitions 8, --warmup 2, and 10 independent runs"
}
$summaryPath = Join-Path $OutputDirectory "summary.txt"
if (Test-Path -LiteralPath $summaryPath -PathType Leaf) { throw "Refusing rerun: summary exists at $summaryPath" }
if (-not (Test-Path -LiteralPath $OutputDirectory -PathType Container)) { $null = New-Item -ItemType Directory -Path $OutputDirectory }
$logsDirectory = Join-Path $OutputDirectory "logs"
if (-not (Test-Path -LiteralPath $logsDirectory -PathType Container)) { $null = New-Item -ItemType Directory -Path $logsDirectory }

$D = 512
$SValues = @(1, 4, 8, 16)
$components = @("full", "gemv-only", "transition-only")
$cpus = @(0, 2, 4, 6)
$workers = 4
$rows = @(128, 128, 128, 128)
$rowText = $rows -join ','
$cpuText = $cpus -join ','
$machinePath = Join-Path $OutputDirectory "machine.csv"
$aggregatePath = Join-Path $OutputDirectory "aggregate.csv"
$commandsPath = Join-Path $OutputDirectory "commands.log"
$stderrPath = Join-Path $OutputDirectory "stderr.log"
$preflightCtestStdoutPath = Join-Path $OutputDirectory "preflight.ctest.stdout"
$preflightCtestStderrPath = Join-Path $OutputDirectory "preflight.ctest.stderr"
$preflightStdoutPath = Join-Path $OutputDirectory "preflight.self-test.stdout"
$preflightStderrPath = Join-Path $OutputDirectory "preflight.self-test.stderr"
$dumpbinPath = Join-Path $OutputDirectory "dumpbin-evidence.txt"
$validationPath = Join-Path $OutputDirectory "validation.txt"
$mapPath = Join-Path (Split-Path -Parent $Executable) "t0m_recurrence_probe.map"
$script:records = New-Object 'System.Collections.Generic.List[object]'
$script:invocations = 0

$nativeHeader = "D,S,R,rows_per_worker,component,mode,kernel,elapsed_seconds,elapsed_per_timed_step,qpc_ticks_per_timed_step,tsc_cycles_per_timed_step,tsc_supported,mac_total,mac_per_second,checksum_kind,validation_invariant,final_checksum,per_round_checksums,per_round_finite,per_round_overflow,per_round_clipped_cells,per_round_clipping_rates,clipped_cells,clipping_rate,all_rounds_valid,worker_count,cpus,affinity,affinity_errors,affinity_succeeded,timed_repetitions,warmup,timed_repetitions_exact"

Set-Content -LiteralPath $commandsPath -Encoding ascii -Value @(
  "T0-M recurrence component breakdown exact commands"
  "executable=$Executable"
  "preflight: CTest recurrence correction, one explicit --self-test, then dumpbin evidence"
  "measurement: D=512; workers=4; cpus=0,2,4,6; rows-per-worker=128,128,128,128; S=1,4,8,16; R=1"
  "components=full,gemv-only,transition-only; mode=fused; timed-repetitions=8; warmup=2; independent-runs=10"
  "semantics=full actual GEMV+transition; gemv-only worker/barrier GEMV with frozen state and no transition; transition-only main-thread apply_transition with deterministic synthetic output and worker/barrier no-op"
  "order=odd runs reverse component order; even runs forward component order; no concurrency; speed invocations never receive --self-test"
)
Set-Content -LiteralPath $stderrPath -Encoding ascii -Value @(
  "T0-M recurrence component breakdown stderr log"
  "speed stderr must contain no correction/autotest/self-test text"
)

function Limit-Text([string]$Text, [int]$Maximum = 8000) {
  if ($null -eq $Text) { return "" }
  $value = $Text.TrimEnd()
  if ($value.Length -le $Maximum) { return $value }
  return $value.Substring(0, $Maximum) + "...[truncated]"
}
function Invoke-CapturedProcess([string]$FileName, [string[]]$Arguments) {
  $startInfo = New-Object System.Diagnostics.ProcessStartInfo
  $startInfo.FileName = $FileName
  $startInfo.Arguments = (($Arguments | ForEach-Object { if ($_ -match '[\s"]') { '"' + ($_ -replace '"', '\"') + '"' } else { [string]$_ } }) -join ' ')
  $startInfo.UseShellExecute = $false
  $startInfo.CreateNoWindow = $true
  $startInfo.RedirectStandardOutput = $true
  $startInfo.RedirectStandardError = $true
  $process = New-Object System.Diagnostics.Process
  $process.StartInfo = $startInfo
  try {
    if (-not $process.Start()) { throw "Could not start process: $FileName" }
    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()
    $process.WaitForExit()
    return [pscustomobject]@{ ExitCode = $process.ExitCode; Stdout = $stdoutTask.Result; Stderr = $stderrTask.Result }
  } finally { $process.Dispose() }
}
function Invoke-Logged([string]$Label, [string]$FileName, [string[]]$Arguments, [bool]$SaveProcessLog) {
  $script:invocations++
  $id = $script:invocations
  $command = '"' + $FileName + '" ' + ($Arguments -join ' ')
  Add-Content -LiteralPath $commandsPath -Encoding ascii -Value "[$id] $Label $command"
  $result = Invoke-CapturedProcess $FileName $Arguments
  if ($SaveProcessLog) {
    Set-Content -LiteralPath (Join-Path $logsDirectory ("{0:D3}.stdout" -f $id)) -Encoding ascii -Value (Limit-Text $result.Stdout)
    Set-Content -LiteralPath (Join-Path $logsDirectory ("{0:D3}.stderr" -f $id)) -Encoding ascii -Value (Limit-Text $result.Stderr)
  }
  Add-Content -LiteralPath $stderrPath -Encoding ascii -Value "[$id] $Label exit=$($result.ExitCode) stderr=$(Limit-Text $result.Stderr 2000)"
  return [pscustomobject]@{ Id = $id; Command = $command; Result = $result }
}
function Require-Equal([string]$Name, [string]$Actual, [string]$Expected, [string]$Context) {
  if ($Actual -ne $Expected) { throw "Invalid ${Name}=$Actual, expected ${Expected}: $Context" }
}
function Require-Positive([string]$Name, [double]$Value, [string]$Context) {
  if ([double]::IsNaN($Value) -or [double]::IsInfinity($Value) -or $Value -le 0) { throw "Invalid ${Name}=${Value}: $Context" }
}
function Require-BoolList([string]$Text, [bool]$Expected, [int]$Count, [string]$Name, [string]$Context) {
  $values = @($Text -split ';')
  $expectedText = if ($Expected) { 'true' } else { 'false' }
  if ($values.Count -ne $Count -or @($values | Where-Object { $_ -ne $expectedText }).Count -gt 0) {
    throw "Invalid ${Name}=${Text}: expected $Count ${expectedText}: $Context"
  }
}
function Get-Median([double[]]$Values) {
  $ordered = @($Values | Sort-Object)
  if ($ordered.Count -ne 10) { throw "Median requires 10 values, got $($ordered.Count)" }
  return ([double]$ordered[4] + [double]$ordered[5]) / 2.0
}
function Get-PopulationSd([double[]]$Values) {
  $mean = [double](($Values | Measure-Object -Average).Average)
  $sum = 0.0
  foreach ($value in $Values) { $sum += ($value - $mean) * ($value - $mean) }
  return [math]::Sqrt($sum / $Values.Count)
}

if ([string]::IsNullOrWhiteSpace($CTestExecutable)) {
  $ctestCommand = Get-Command ctest.exe -ErrorAction SilentlyContinue
  if ($null -ne $ctestCommand) { $CTestExecutable = $ctestCommand.Path }
}
if ([string]::IsNullOrWhiteSpace($CTestExecutable)) {
  $knownCTestPaths = @(
    "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\ctest.exe",
    "C:\Program Files\CMake\bin\ctest.exe"
  )
  $CTestExecutable = @($knownCTestPaths | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf })[0]
}
if ([string]::IsNullOrWhiteSpace($CTestExecutable) -or -not (Test-Path -LiteralPath $CTestExecutable -PathType Leaf)) {
  throw "ctest.exe not found; provide -CTestExecutable"
}
if ([string]::IsNullOrWhiteSpace($DumpbinExecutable)) {
  $dumpbinCommand = Get-Command dumpbin.exe -ErrorAction SilentlyContinue
  if ($null -ne $dumpbinCommand) { $DumpbinExecutable = $dumpbinCommand.Path }
}
if ([string]::IsNullOrWhiteSpace($DumpbinExecutable)) {
  $knownDumpbinPaths = @(
    "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64\dumpbin.exe"
  )
  $DumpbinExecutable = @($knownDumpbinPaths | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf })[0]
}
if ([string]::IsNullOrWhiteSpace($DumpbinExecutable) -or -not (Test-Path -LiteralPath $DumpbinExecutable -PathType Leaf)) {
  throw "dumpbin.exe not found; run from VS developer prompt or provide -DumpbinExecutable"
}

$ctestArgs = @("--test-dir", (Join-Path $projectRoot "build"), "-R", "t0m_recurrence_correction", "--output-on-failure")
Add-Content -LiteralPath $commandsPath -Encoding ascii -Value ("ctest_command `"$CTestExecutable`" " + ($ctestArgs -join ' '))
$ctestResult = Invoke-CapturedProcess $CTestExecutable $ctestArgs
Set-Content -LiteralPath $preflightCtestStdoutPath -Encoding ascii -Value (Limit-Text $ctestResult.Stdout)
Set-Content -LiteralPath $preflightCtestStderrPath -Encoding ascii -Value (Limit-Text $ctestResult.Stderr)
if ($ctestResult.ExitCode -ne 0) { throw "CTest recurrence correction failed with exit code $($ctestResult.ExitCode)" }

$preflight = Invoke-Logged "preflight_self_test" $Executable @("--self-test") $true
Set-Content -LiteralPath $preflightStdoutPath -Encoding ascii -Value (Limit-Text $preflight.Result.Stdout)
Set-Content -LiteralPath $preflightStderrPath -Encoding ascii -Value (Limit-Text $preflight.Result.Stderr)
if ($preflight.Result.ExitCode -ne 0 -or (($preflight.Result.Stdout + "`n" + $preflight.Result.Stderr) -notmatch "(?i)T0-M recurrence correction passed")) {
  throw "Explicit recurrence self-test preflight failed or did not report correction pass"
}

$dumpbinArgs = @("/DISASM", $Executable)
Add-Content -LiteralPath $commandsPath -Encoding ascii -Value ("dumpbin_command `"$DumpbinExecutable`" " + ($dumpbinArgs -join ' '))
$dumpbinResult = Invoke-CapturedProcess $DumpbinExecutable $dumpbinArgs
$dumpbinText = $dumpbinResult.Stdout + "`n" + $dumpbinResult.Stderr
$disasmLines = @($dumpbinText -split '\r?\n')
if (-not (Test-Path -LiteralPath $mapPath -PathType Leaf)) { throw "Linker map not found: $mapPath" }
$mapLines = @(Get-Content -LiteralPath $mapPath)
$fastMapLine = @($mapLines | Where-Object { $_ -match '^\s*0001:' -and $_ -match '(?i)apply_transition_fast' })
$referenceMapLine = @($mapLines | Where-Object { $_ -match '^\s*0001:' -and $_ -match '(?i)apply_transition_reference' })
$checkedDotMapLine = @($mapLines | Where-Object { $_ -match '^\s*0001:' -and $_ -match '(?i)checked_dot_avx2' })
if ($fastMapLine.Count -ne 1 -or $referenceMapLine.Count -ne 1 -or $checkedDotMapLine.Count -ne 1) {
  throw "Transition map symbols missing or ambiguous"
}
function Get-MapAddress([string]$Line) {
  $match = [regex]::Match($Line, '000000014000([0-9A-Fa-f]+)\s+f\s')
  if (-not $match.Success) { throw "Could not parse map address: $Line" }
  return [uint64]::Parse($match.Groups[1].Value, [Globalization.NumberStyles]::HexNumber,
                          [Globalization.CultureInfo]::InvariantCulture) + 0x140000000
}
$fastStart = Get-MapAddress $fastMapLine[0]
$referenceStart = Get-MapAddress $referenceMapLine[0]
$referenceEnd = Get-MapAddress $checkedDotMapLine[0]
$fastDisasm = @($disasmLines | Where-Object {
  $match = [regex]::Match($_, '^\s+000000014000([0-9A-Fa-f]+):')
  if (-not $match.Success) { return $false }
  $address = [uint64]::Parse($match.Groups[1].Value, [Globalization.NumberStyles]::HexNumber,
                             [Globalization.CultureInfo]::InvariantCulture) + 0x140000000
  return $address -ge $fastStart -and $address -lt $referenceStart
})
$referenceDisasm = @($disasmLines | Where-Object {
  $match = [regex]::Match($_, '^\s+000000014000([0-9A-Fa-f]+):')
  if (-not $match.Success) { return $false }
  $address = [uint64]::Parse($match.Groups[1].Value, [Globalization.NumberStyles]::HexNumber,
                             [Globalization.CultureInfo]::InvariantCulture) + 0x140000000
  return $address -ge $referenceStart -and $address -lt $referenceEnd
})
$fastVectorMatches = @($fastDisasm | Where-Object { $_ -match '(?i)\bv(add|mul|div|sqrt|round)pd\b' })
$referencePackedMatches = @($referenceDisasm | Where-Object { $_ -match '(?i)\bv(add|mul|div|sqrt|round)pd\b' })
$referenceScalarMatches = @($referenceDisasm | Where-Object { $_ -match '(?i)\bv(add|mul|div|sqrt|round)sd\b' })
$requiredFastInstructions = @('vaddpd', 'vmulpd', 'vdivpd', 'vsqrtpd', 'vroundpd')
$missingFastInstructions = @($requiredFastInstructions | Where-Object {
  $instructionName = $_
  @($fastVectorMatches | Where-Object { $_ -match "(?i)\b$instructionName\b" }).Count -eq 0
})
Set-Content -LiteralPath $dumpbinPath -Encoding ascii -Value @(
  "command=`"$DumpbinExecutable`" /DISASM `"$Executable`""
  "exit_code=$($dumpbinResult.ExitCode)"
  "map=$mapPath"
  "fast_symbol=$($fastMapLine[0])"
  "reference_symbol=$($referenceMapLine[0])"
  "checked_dot_avx2_symbol=$($checkedDotMapLine[0])"
  "fast_address_range=0x$('{0:X}' -f $fastStart)..0x$('{0:X}' -f ($referenceStart - 1))"
  "reference_address_range=0x$('{0:X}' -f $referenceStart)..0x$('{0:X}' -f ($referenceEnd - 1))"
  "fast_transition_packed_math_matches=$($fastVectorMatches.Count)"
  $fastVectorMatches
  "reference_transition_packed_math_matches=$($referencePackedMatches.Count)"
  $referencePackedMatches
  "reference_transition_scalar_math_matches=$($referenceScalarMatches.Count)"
  $referenceScalarMatches
  "required_fast_instructions=$($requiredFastInstructions -join ',')"
  "missing_fast_instructions=$($missingFastInstructions -join ',')"
  "interpretation=map ranges bind dumpbin instruction evidence to named transition functions; fast path contains packed AVX2 math and reference range contains scalar-width math only"
)
if ($dumpbinResult.ExitCode -ne 0 -or $fastVectorMatches.Count -eq 0 -or $referenceScalarMatches.Count -eq 0 -or
    $referencePackedMatches.Count -ne 0 -or $missingFastInstructions.Count -ne 0) {
  throw "dumpbin evidence gate failed; see $dumpbinPath"
}

foreach ($S in $SValues) {
  for ($run = 1; $run -le $IndependentRuns; $run++) {
    $order = if (($run % 2) -eq 1) { @("transition-only", "gemv-only", "full") } else { $components }
    $orderIndex = 0
    foreach ($component in $order) {
      $orderIndex++
      $context = "S=$S component=$component run=$run order_index=$orderIndex"
      $arguments = @("--D", "$D", "--S", "$S", "--R", "1", "--mode", "fused", "--component", $component,
        "--workers", "$workers", "--cpus", $cpuText, "--rows-per-worker", $rowText,
        "--timed-repetitions", "$TimedRepetitions", "--warmup", "$Warmup")
      $logged = Invoke-Logged "speed $context" $Executable $arguments $true
      $result = $logged.Result
      if ($result.ExitCode -ne 0) { throw "Nonzero speed exit $($result.ExitCode): $context" }
      if ($result.Stderr -match "(?i)(correction|autotest|self.test|test passed|test failed)") {
        throw "Forbidden correction/autotest text in speed stderr: $context"
      }
      $lines = @($result.Stdout -split '\r?\n' | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
      if ($lines.Count -ne 2 -or $lines[0] -ne $nativeHeader) { throw "Bad CSV schema/row count: $context" }
      try { $row = $lines[1] | ConvertFrom-Csv -Header ($nativeHeader -split ',') } catch { throw "Bad CSV parse: $context" }
      Require-Equal "D" ([string]$row.D) "$D" $context
      Require-Equal "S" ([string]$row.S) "$S" $context
      Require-Equal "R" ([string]$row.R) "1" $context
      Require-Equal "rows_per_worker" ([string]$row.rows_per_worker) $rowText $context
      Require-Equal "component" ([string]$row.component) $component $context
      Require-Equal "mode" ([string]$row.mode) "fused" $context
      Require-Equal "worker_count" ([string]$row.worker_count) "$workers" $context
      Require-Equal "cpus" ([string]$row.cpus) $cpuText $context
      Require-Equal "affinity" ([string]$row.affinity) "1,1,1,1" $context
      Require-Equal "affinity_errors" ([string]$row.affinity_errors) "0,0,0,0" $context
      Require-Equal "affinity_succeeded" ([string]$row.affinity_succeeded) "true" $context
      Require-Equal "timed_repetitions" ([string]$row.timed_repetitions) "$TimedRepetitions" $context
      Require-Equal "warmup" ([string]$row.warmup) "$Warmup" $context
      Require-Equal "timed_repetitions_exact" ([string]$row.timed_repetitions_exact) "true" $context
      Require-Equal "all_rounds_valid" ([string]$row.all_rounds_valid) "true" $context
      Require-BoolList ([string]$row.per_round_finite) $true 1 "per_round_finite" $context
      Require-BoolList ([string]$row.per_round_overflow) $false 1 "per_round_overflow" $context
      Require-Equal "tsc_supported" ([string]$row.tsc_supported) "true" $context
      Require-Positive "elapsed_seconds" ([double]$row.elapsed_seconds) $context
      Require-Positive "elapsed_per_timed_step" ([double]$row.elapsed_per_timed_step) $context
      Require-Positive "qpc_ticks_per_timed_step" ([double]$row.qpc_ticks_per_timed_step) $context
      Require-Positive "tsc_cycles_per_timed_step" ([double]$row.tsc_cycles_per_timed_step) $context
      $checksum = [uint64]$row.final_checksum
      if ($checksum -eq 0) { throw "Checksum zero: $context" }
      $roundChecksums = @(([string]$row.per_round_checksums) -split ';')
      if ($roundChecksums.Count -ne 1 -or [uint64]$roundChecksums[0] -eq 0) { throw "Invalid per-round checksum: $context" }
      $expectedMac = if ($component -eq "transition-only") { [uint64]0 } else { [uint64]$D * $D * $S * $TimedRepetitions }
      Require-Equal "mac_total" ([string]$row.mac_total) ([string]$expectedMac) $context
      if ($component -eq "gemv-only") {
        Require-Equal "checksum_kind" ([string]$row.checksum_kind) "output" $context
        if (([string]$row.validation_invariant) -notmatch "output checksum") { throw "Missing GEMV checksum invariant: $context" }
      } elseif ($component -eq "transition-only") {
        Require-Equal "kernel" ([string]$row.kernel) "none" $context
        if (([string]$row.validation_invariant) -notmatch "synthetic output") { throw "Missing transition invariant: $context" }
      }
      [void]$script:records.Add([pscustomobject][ordered]@{
        run = $run; order_index = $orderIndex; D = [int]$row.D; S = [int]$row.S; R = [int]$row.R
        component = [string]$row.component; mode = [string]$row.mode; elapsed_seconds = [double]$row.elapsed_seconds
        elapsed_per_timed_step = [double]$row.elapsed_per_timed_step; qpc_ticks_per_timed_step = [double]$row.qpc_ticks_per_timed_step
        tsc_cycles_per_timed_step = [double]$row.tsc_cycles_per_timed_step; tsc_supported = [string]$row.tsc_supported
        mac_total = [uint64]$row.mac_total; mac_per_second = [double]$row.mac_per_second; checksum_kind = [string]$row.checksum_kind
        validation_invariant = [string]$row.validation_invariant; final_checksum = $checksum; per_round_checksums = [string]$row.per_round_checksums
        per_round_finite = [string]$row.per_round_finite; per_round_overflow = [string]$row.per_round_overflow
        all_rounds_valid = [string]$row.all_rounds_valid; worker_count = [int]$row.worker_count; cpus = [string]$row.cpus
        affinity = [string]$row.affinity; affinity_errors = [string]$row.affinity_errors; affinity_succeeded = [string]$row.affinity_succeeded
        timed_repetitions = [int]$row.timed_repetitions; warmup = [int]$row.warmup; timed_repetitions_exact = [string]$row.timed_repetitions_exact
      })
    }
  }
}

if ($script:records.Count -ne 120) { throw "Expected 120 raw rows, got $($script:records.Count)" }
$aggregates = New-Object 'System.Collections.Generic.List[object]'
foreach ($S in $SValues) {
  foreach ($component in $components) {
    $group = @($script:records | Where-Object { $_.S -eq $S -and $_.component -eq $component })
    if ($group.Count -ne 10) { throw "Expected 10 rows in S=$S component=$component, got $($group.Count)" }
    $checksums = @($group | ForEach-Object { [string]$_.final_checksum } | Sort-Object -Unique)
    if ($checksums.Count -ne 1) { throw "Nondeterministic checksum in S=$S component=$component" }
    [void]$aggregates.Add([pscustomobject][ordered]@{
      D = $D; S = $S; R = 1; component = $component; mode = "fused"; n = $group.Count
      elapsed_seconds_median = Get-Median ([double[]]@($group | ForEach-Object { $_.elapsed_seconds }))
      elapsed_per_timed_step_median = Get-Median ([double[]]@($group | ForEach-Object { $_.elapsed_per_timed_step }))
      qpc_ticks_per_timed_step_median = Get-Median ([double[]]@($group | ForEach-Object { $_.qpc_ticks_per_timed_step }))
      tsc_cycles_per_timed_step_median = Get-Median ([double[]]@($group | ForEach-Object { $_.tsc_cycles_per_timed_step }))
      elapsed_per_timed_step_min = ($group | Measure-Object -Property elapsed_per_timed_step -Minimum).Minimum
      elapsed_per_timed_step_max = ($group | Measure-Object -Property elapsed_per_timed_step -Maximum).Maximum
      elapsed_per_timed_step_population_sd = Get-PopulationSd ([double[]]@($group | ForEach-Object { $_.elapsed_per_timed_step }))
      checksum = $checksums[0]; checksum_deterministic = $true; all_rounds_valid = $true; affinity_succeeded = $true
      timed_repetitions_exact = $true; mac_total = [uint64]$group[0].mac_total; checksum_kind = [string]$group[0].checksum_kind
    })
  }
}
if ($aggregates.Count -ne 12) { throw "Expected 12 aggregate rows, got $($aggregates.Count)" }

$fullByS = @{}
$gemvByS = @{}
$transitionByS = @{}
foreach ($aggregate in $aggregates) {
  if ($aggregate.component -eq "full") { $fullByS[$aggregate.S] = $aggregate }
  elseif ($aggregate.component -eq "gemv-only") { $gemvByS[$aggregate.S] = $aggregate }
  else { $transitionByS[$aggregate.S] = $aggregate }
}
$bottleneckRows = @()
foreach ($S in $SValues) {
  $gemv = $gemvByS[$S].elapsed_per_timed_step_median
  $transition = $transitionByS[$S].elapsed_per_timed_step_median
  $bottleneckRows += [pscustomobject]@{ S = $S; full = $fullByS[$S].elapsed_per_timed_step_median; gemv = $gemv; transition = $transition; transition_over_gemv = $transition / $gemv }
}
$maxBottleneck = @($bottleneckRows | Sort-Object transition_over_gemv -Descending)[0]
$bottleneckName = if ($maxBottleneck.transition_over_gemv -ge 1.0) { "scalar transition" } else { "GEMV" }

Set-Content -LiteralPath $machinePath -Encoding ascii -Value @($script:records | ConvertTo-Csv -NoTypeInformation)
Set-Content -LiteralPath $aggregatePath -Encoding ascii -Value @($aggregates | ConvertTo-Csv -NoTypeInformation)
$validation = @(
  "status=PASS"
  "raw_rows=$($script:records.Count); expected_raw_rows=120"
  "aggregate_rows=$($aggregates.Count); expected_aggregate_rows=12"
  "cells=4; components_per_cell=3; runs_per_component=10; speed_processes=120"
  "configuration=D=512; R=1; workers=4; cpus=0,2,4,6; rows_per_worker=128,128,128,128; timed_repetitions=8; warmup=2"
  "checksum_determinism=PASS for every S,component cell; nonzero checksum=PASS"
  "all_rounds_valid=PASS; per_round_finite=true; per_round_overflow=false"
  "affinity=PASS for every raw row; affinity=1,1,1,1; affinity_errors=0,0,0,0"
  "repetition_validation=PASS; timed_repetitions_exact=true"
  "speed_stderr_correction_autotest_text=PASS absent"
  "component_invariants=full state checksum after actual GEMV+transition; gemv-only output checksum with frozen state; transition-only state checksum after deterministic synthetic output"
  "dumpbin=PASS; named map ranges prove fast packed AVX2 transition and scalar-only reference; see dumpbin-evidence.txt"
)
Set-Content -LiteralPath $validationPath -Encoding ascii -Value $validation
$summary = @(
  "status=PASS"
  "executive_summary=At D=512,R=1, $bottleneckName dominates isolated timing; maximum transition-over-GEMV median ratio occurs at S=$($maxBottleneck.S) and is $($maxBottleneck.transition_over_gemv.ToString('R',[Globalization.CultureInfo]::InvariantCulture))x. Full is actual GEMV+transition recurrence, not arithmetic sum claim."
  "configuration=D=$D; R=1 fixed for isolated per-step cost; workers=$workers; cpus=$cpuText; rows_per_worker=$rowText"
  "components=full,gemv-only,transition-only; mode=fused; S=$($SValues -join ','); timed_repetitions=$TimedRepetitions; warmup=$Warmup; independent_runs=$IndependentRuns"
  "semantics=full actual worker/barrier GEMV plus main-thread apply_transition; gemv-only same worker/barrier GEMV stage with frozen state and no transition; transition-only same main-thread barrier protocol with deterministic synthetic output/state and no GEMV"
  "counts=raw_rows=120; aggregate_rows=12; speed_processes=120; preflight_ctest=1; preflight_self_test=1"
  "aggregation=median of 10; min/max/population SD for elapsed per timed step; QPC ticks and TSC cycles per timed step"
  "bottleneck_rule=transition-only median elapsed_per_timed_step divided by gemv-only median; largest ratio S=$($maxBottleneck.S); ratio=$($maxBottleneck.transition_over_gemv.ToString('R',[Globalization.CultureInfo]::InvariantCulture))"
  "measurements=each aggregate row below reports exact median elapsed seconds, elapsed seconds per timed step, QPC ticks per step, TSC cycles per step"
  @($aggregates | ForEach-Object { "measurement,S=$($_.S),component=$($_.component),elapsed_seconds_median=$($_.elapsed_seconds_median.ToString('R',[Globalization.CultureInfo]::InvariantCulture)),elapsed_per_timed_step_median=$($_.elapsed_per_timed_step_median.ToString('R',[Globalization.CultureInfo]::InvariantCulture)),qpc_ticks_per_timed_step_median=$($_.qpc_ticks_per_timed_step_median.ToString('R',[Globalization.CultureInfo]::InvariantCulture)),tsc_cycles_per_timed_step_median=$($_.tsc_cycles_per_timed_step_median.ToString('R',[Globalization.CultureInfo]::InvariantCulture)),mac_total=$($_.mac_total),checksum=$($_.checksum)" })
  "dumpbin_evidence=$dumpbinPath; exact command, linker map ranges, and named transition instruction matches saved"
  "artifacts=machine.csv,aggregate.csv,commands.log,stderr.log,preflight.ctest.stdout,preflight.ctest.stderr,preflight.self-test.stdout,preflight.self-test.stderr,dumpbin-evidence.txt,logs/,summary.txt,validation.txt"
  "validation=PASS; see validation.txt"
)
Set-Content -LiteralPath $summaryPath -Encoding ascii -Value $summary
Write-Output $summaryPath
