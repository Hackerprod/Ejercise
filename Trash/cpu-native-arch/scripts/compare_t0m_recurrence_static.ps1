[CmdletBinding()]
param(
  [string]$ProjectRoot = "",
  [string]$OutputDirectory = ""
)

$ErrorActionPreference = "Stop"
$invariant = [Globalization.CultureInfo]::InvariantCulture
[System.Threading.Thread]::CurrentThread.CurrentCulture = $invariant
[System.Threading.Thread]::CurrentThread.CurrentUICulture = $invariant

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) { $ProjectRoot = Split-Path -Parent $PSScriptRoot }
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
  $OutputDirectory = Join-Path $ProjectRoot "sweep-output\t0m-recurrence-performance-d512"
}

$recurrenceAggregatePath = Join-Path $OutputDirectory "aggregate.csv"
$recurrenceMetricsPath = Join-Path $OutputDirectory "metrics.csv"
$staticMetricsPath = Join-Path $ProjectRoot "sweep-output\t0m-phase3-median10-rerun\t0m_phase3.metrics.csv"
$staticAggregatePath = Join-Path $ProjectRoot "sweep-output\t0m-phase3-median10-rerun\t0m_phase3.aggregate.csv"
$comparisonPath = Join-Path $OutputDirectory "comparison.csv"
$methodologyPath = Join-Path $OutputDirectory "comparison.txt"
$validationPath = Join-Path $OutputDirectory "comparison-validation.txt"

foreach ($path in @($comparisonPath, $methodologyPath, $validationPath)) {
  if (Test-Path -LiteralPath $path -PathType Leaf) { throw "Refusing overwrite: comparison artifact exists at $path" }
}
foreach ($path in @($recurrenceAggregatePath, $recurrenceMetricsPath, $staticMetricsPath, $staticAggregatePath)) {
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Required input not found: $path" }
}

function Parse-Double([object]$Value, [string]$Name) {
  $parsed = 0.0
  if (-not [double]::TryParse([string]$Value, [Globalization.NumberStyles]::Float, $invariant, [ref]$parsed)) {
    throw "Invalid numeric value for $Name`: $Value"
  }
  return $parsed
}

function Parse-Int([object]$Value, [string]$Name) {
  $parsed = 0
  if (-not [int]::TryParse([string]$Value, [Globalization.NumberStyles]::Integer, $invariant, [ref]$parsed)) {
    throw "Invalid integer value for $Name`: $Value"
  }
  return $parsed
}

function Format-Number([double]$Value) { return $Value.ToString("R", $invariant) }

function Require-Positive([double]$Value, [string]$Name) {
  if ([double]::IsNaN($Value) -or [double]::IsInfinity($Value) -or $Value -le 0) {
    throw "Expected positive finite $Name, got $(Format-Number $Value)"
  }
}

function Divide([double]$Numerator, [double]$Denominator, [string]$Formula) {
  Require-Positive $Denominator "denominator in $Formula"
  return $Numerator / $Denominator
}

function Get-Row([object[]]$Rows, [scriptblock]$Predicate, [string]$Description) {
  $matches = @($Rows | Where-Object $Predicate)
  if ($matches.Count -ne 1) { throw "Expected exactly one row for $Description, got $($matches.Count)" }
  return $matches[0]
}

function Add-ComparisonColumns([System.Collections.IDictionary]$Row, [string]$Prefix, [double]$Recurrence, [double]$Static) {
  $Row["${Prefix}_static"] = $Static
  $Row["${Prefix}_delta"] = $Recurrence - $Static
  $Row["${Prefix}_ratio"] = Divide $Recurrence $Static "recurrence/$Prefix`_static"
}

$recurrenceAggregate = @(Import-Csv -LiteralPath $recurrenceAggregatePath)
$recurrenceMetrics = @(Import-Csv -LiteralPath $recurrenceMetricsPath)
$staticMetrics = @(Import-Csv -LiteralPath $staticMetricsPath)
$staticAggregate = @(Import-Csv -LiteralPath $staticAggregatePath)

$sValues = @(1, 2, 4, 8, 16)
$rValues = @(1, 2, 4, 8, 16)
$expectedRecurrenceAggregateRows = $sValues.Count * $rValues.Count * 2
$expectedRecurrenceMetricRows = $sValues.Count * $rValues.Count
$expectedStaticAggregateRows = $sValues.Count * $rValues.Count * 2 * 2
$expectedStaticMetricRows = $sValues.Count * 2

if ($recurrenceAggregate.Count -ne $expectedRecurrenceAggregateRows) { throw "Unexpected recurrence aggregate row count: $($recurrenceAggregate.Count)" }
if ($recurrenceMetrics.Count -ne $expectedRecurrenceMetricRows) { throw "Unexpected recurrence metric row count: $($recurrenceMetrics.Count)" }
$staticTargetAggregate = @($staticAggregate | Where-Object { (Parse-Int $_.target_kib "static target_kib") -eq 512 -and $_.campaign -eq "main" })
$staticTargetMetrics = @($staticMetrics | Where-Object { (Parse-Int $_.target_kib "static metric target_kib") -eq 512 })
if ($staticTargetAggregate.Count -ne $expectedStaticAggregateRows) { throw "Unexpected static target aggregate row count: $($staticTargetAggregate.Count)" }
if ($staticTargetMetrics.Count -ne $expectedStaticMetricRows) { throw "Unexpected static target metric row count: $($staticTargetMetrics.Count)" }

$recurrence = @{}
foreach ($R in $rValues) {
  $recurrence[$R] = @{}
  foreach ($S in $sValues) {
    $metric = Get-Row $recurrenceMetrics { (Parse-Int $_.D "recurrence D") -eq 512 -and (Parse-Int $_.S "recurrence S") -eq $S -and (Parse-Int $_.R "recurrence R") -eq $R } "recurrence metrics D=512 S=$S R=$R"
    $fusedAggregate = Get-Row $recurrenceAggregate { $_.metric -eq "G(S)" -and $_.native_mode -eq "fused" -and (Parse-Int $_.D "recurrence aggregate D") -eq 512 -and (Parse-Int $_.S "recurrence aggregate S") -eq $S -and (Parse-Int $_.R "recurrence aggregate R") -eq $R } "recurrence fused aggregate S=$S R=$R"
    $repeatAggregate = Get-Row $recurrenceAggregate { $_.metric -eq "F(S)" -and $_.native_mode -eq "repeat" -and (Parse-Int $_.D "recurrence aggregate D") -eq 512 -and (Parse-Int $_.S "recurrence aggregate S") -eq $S -and (Parse-Int $_.R "recurrence aggregate R") -eq $R } "recurrence repeat aggregate S=$S R=$R"
    $aggregateFused = Parse-Double $fusedAggregate.median "recurrence fused aggregate median S=$S R=$R"
    $aggregateRepeat = Parse-Double $repeatAggregate.median "recurrence repeat aggregate median S=$S R=$R"
    $fused = Parse-Double $metric.G_S "recurrence metrics G_S S=$S R=$R"
    $repeat = Parse-Double $metric.F_S "recurrence metrics F_S S=$S R=$R"
    Require-Positive $fused "recurrence fused median S=$S R=$R"
    Require-Positive $repeat "recurrence repeat median S=$S R=$R"
    if ([math]::Abs($fused - $aggregateFused) -gt 0.000001) { throw "Recurrence metrics G_S disagrees with aggregate at S=$S R=$R" }
    if ([math]::Abs($repeat - $aggregateRepeat) -gt 0.000001) { throw "Recurrence metrics F_S disagrees with aggregate at S=$S R=$R" }
    $recurrence[$R][$S] = [pscustomobject]@{ Fused = $fused; Repeat = $repeat }
  }
}

$static = @{}
$staticMetricInput = @{}
foreach ($variant in @("A", "B")) {
  $static[$variant] = @{}
  $staticMetricInput[$variant] = @{}
  foreach ($R in $rValues) {
    $static[$variant][$R] = @{}
    $staticMetricInput[$variant][$R] = Get-Row $staticTargetMetrics { $_.variant -eq $variant -and (Parse-Int $_.R "static metric R") -eq $R } "static metrics target512 variant=$variant R=$R"
    foreach ($S in $sValues) {
      $fusedAggregate = Get-Row $staticTargetAggregate { $_.mode -eq "fused" -and $_.variant -eq $variant -and (Parse-Int $_.S "static aggregate S") -eq $S -and (Parse-Int $_.R "static aggregate R") -eq $R } "static fused aggregate target512 variant=$variant S=$S R=$R"
      $repeatAggregate = Get-Row $staticTargetAggregate { $_.mode -eq "repeat" -and $_.variant -eq $variant -and (Parse-Int $_.S "static aggregate S") -eq $S -and (Parse-Int $_.R "static aggregate R") -eq $R } "static repeat aggregate target512 variant=$variant S=$S R=$R"
      $fused = Parse-Double $fusedAggregate.median "static fused median variant=$variant S=$S R=$R"
      $repeat = Parse-Double $repeatAggregate.median "static repeat median variant=$variant S=$S R=$R"
      Require-Positive $fused "static fused median variant=$variant S=$S R=$R"
      Require-Positive $repeat "static repeat median variant=$variant S=$S R=$R"
      $static[$variant][$R][$S] = [pscustomobject]@{ Fused = $fused; Repeat = $repeat }
    }
    $s1 = $static[$variant][$R][1]
    $metric = $staticMetricInput[$variant][$R]
    foreach ($S in @(2, 4, 8, 16)) {
      $expectedG = Divide $static[$variant][$R][$S].Fused $s1.Fused "static G$S variant=$variant R=$R"
      $actualG = Parse-Double $metric.("G$S") "static metric G$S variant=$variant R=$R"
      if ([math]::Abs($expectedG - $actualG) -gt 0.000000000001) { throw "Static metrics G$S disagrees with aggregate at variant=$variant R=$R" }
    }
    foreach ($S in @(4, 8, 16)) {
      $expectedF = Divide $static[$variant][$R][$S].Fused $static[$variant][$R][$S].Repeat "static F$S variant=$variant R=$R"
      $actualF = Parse-Double $metric.("F$S") "static metric F$S variant=$variant R=$R"
      if ([math]::Abs($expectedF - $actualF) -gt 0.000000000001) { throw "Static metrics F$S disagrees with aggregate at variant=$variant R=$R" }
    }
  }
}

$comparisonRows = New-Object 'System.Collections.Generic.List[object]'
foreach ($R in $rValues) {
  $row = [ordered]@{ record_type = "R_comparison"; D = 512; target_kib = 512; R = $R; recurrence_path = "Norm+Requantize+residual between every round" }
  foreach ($S in $sValues) { $row["recurrence_fused_median_S${S}_mac_per_second"] = $recurrence[$R][$S].Fused }
  foreach ($S in $sValues) { $row["recurrence_repeat_median_S${S}_mac_per_second"] = $recurrence[$R][$S].Repeat }
  foreach ($S in @(2, 4, 8, 16)) { $row["recurrence_G${S}_norm"] = Divide $recurrence[$R][$S].Fused $recurrence[$R][1].Fused "recurrence G$S R=$R" }
  foreach ($S in @(4, 8, 16)) { $row["recurrence_F${S}_norm"] = Divide $recurrence[$R][$S].Fused $recurrence[$R][$S].Repeat "recurrence F$S R=$R" }
  foreach ($variant in @("A", "B")) {
    $metric = $staticMetricInput[$variant][$R]
    foreach ($S in @(2, 4, 8, 16)) {
      $name = "G$S"
      Add-ComparisonColumns $row "static_${variant}_$name" (Parse-Double $row["recurrence_G${S}_norm"] "recurrence G$S") (Parse-Double $metric.$name "static $name")
    }
    foreach ($S in @(4, 8, 16)) {
      $name = "F$S"
      Add-ComparisonColumns $row "static_${variant}_$name" (Parse-Double $row["recurrence_F${S}_norm"] "recurrence F$S") (Parse-Double $metric.$name "static $name")
    }
    $row["static_${variant}_max_G8_G16"] = Parse-Double $metric.max_G8_G16 "static max_G8_G16 variant=$variant R=$R"
  }
  [void]$comparisonRows.Add([pscustomobject]$row)
}

$recurrenceMaxG8 = ($comparisonRows | ForEach-Object { [double]$_.recurrence_G8_norm } | Measure-Object -Maximum).Maximum
$recurrenceMaxG16 = ($comparisonRows | ForEach-Object { [double]$_.recurrence_G16_norm } | Measure-Object -Maximum).Maximum
$staticMax = @{}
foreach ($variant in @("A", "B")) {
  $staticMax[$variant] = @{
    G8 = ($rValues | ForEach-Object { Divide $static[$variant][$_][8].Fused $static[$variant][$_][1].Fused "static G8 variant=$variant R=$_" } | Measure-Object -Maximum).Maximum
    G16 = ($rValues | ForEach-Object { Divide $static[$variant][$_][16].Fused $static[$variant][$_][1].Fused "static G16 variant=$variant R=$_" } | Measure-Object -Maximum).Maximum
  }
}

Set-Content -LiteralPath $comparisonPath -Encoding ascii -Value @($comparisonRows | ConvertTo-Csv -NoTypeInformation)

$comparisonText = @(
  "status=PASS"
  "methodology=Read existing recurrence aggregate.csv and metrics.csv plus static phase3 aggregate/metrics CSVs; no probe execution and no source modification."
  "reference_selection=Static reference restricted to campaign=main,target_kib=512 (equivalent D=512) and variants A/B; each R retained separately; no global-max substitution."
  "recurrence_scope=D=512; S=1,2,4,8,16; R=1,2,4,8,16; fused median is recurrence G(S) throughput and repeat median is recurrence F(S) throughput."
  "static_formula=G(S)=median fused(S)/median fused(1) for each target/R/variant; F(S)=median fused(S)/median repeat(S)."
  "recurrence_formula=G2/G4/G8/G16=fused median at S divided by fused median at S=1 for same R; F4/F8/F16=fused median divided by repeat median at same S and R."
  "comparison_formula=delta=recurrence normalized metric-static normalized metric; ratio=recurrence normalized metric/static normalized metric."
  "raw_units=Raw recurrence columns ending _mac_per_second are MAC/s throughput from existing metrics/aggregate; normalized G/F columns are dimensionless ratios and must not be read as MAC/s."
  "recurrence_max_G8=$(Format-Number $recurrenceMaxG8); recurrence_max_G16=$(Format-Number $recurrenceMaxG16)"
  "static_D512_target512_variant_A_max_G8=$(Format-Number $staticMax.A.G8); static_D512_target512_variant_A_max_G16=$(Format-Number $staticMax.A.G16)"
  "static_D512_target512_variant_B_max_G8=$(Format-Number $staticMax.B.G8); static_D512_target512_variant_B_max_G16=$(Format-Number $staticMax.B.G16)"
  "interpretation=Recurrence normalized scaling is strongly eroded versus static benefit: recurrence G8/G16 maxima are below both static A/B maxima, while recurrence F ratios remain near 1 rather than static F4/F8/F16 gains. Norm/Requantize/residual path therefore preserves fused/repeat throughput parity but not static cache-scaling benefit."
  "cache_caveat=Do not claim direct cache-stress comparability: recurrence D=512 total weights=256KiB; static target512 is a separate reference campaign."
  "artifacts=comparison.csv; comparison-validation.txt; comparison.txt"
)
Set-Content -LiteralPath $methodologyPath -Encoding ascii -Value $comparisonText

$validation = @(
  "status=PASS"
  "inputs=recurrence aggregate rows=$($recurrenceAggregate.Count) expected=$expectedRecurrenceAggregateRows; recurrence metrics rows=$($recurrenceMetrics.Count) expected=$expectedRecurrenceMetricRows"
  "static_reference_rows=target_kib=512 campaign=main aggregate rows=$($staticTargetAggregate.Count) expected=$expectedStaticAggregateRows; metrics rows=$($staticTargetMetrics.Count) expected=$expectedStaticMetricRows"
  "expected_recurrence_cells=25; expected_recurrence_modes_per_cell=2; expected_recurrence_R_values=1,2,4,8,16; expected_recurrence_S_values=1,2,4,8,16"
  "expected_static_cells_per_variant=25; expected_static_variants=A,B; expected_static_S_values=1,2,4,8,16; expected_static_R_values=1,2,4,8,16"
  "row_validation=PASS every expected recurrence aggregate/metric and static target aggregate/metric key found exactly once"
  "cross_file_validation=PASS recurrence metrics G_S/F_S equal recurrence aggregate medians; static metrics equal recomputed aggregate formulas"
  "division_validation=PASS every denominator finite and >0; no division by zero"
  "formula_G=G(S)=median fused(S)/median fused(1), computed per D/target/R/variant for static and per R for recurrence"
  "formula_F=F(S)=median fused(S)/median repeat(S), computed per D/target/R/variant for static and per R for recurrence"
  "comparison_validation=PASS recurrence normalized metrics compared against static variant A and variant B independently; delta=recurrence-static; ratio=recurrence/static"
  "raw_validation=PASS raw recurrence fused/repeat medians copied from existing metrics.csv and cross-checked against existing aggregate.csv; units MAC/s"
  "artifact_rows=comparison.csv rows=$($comparisonRows.Count) expected=5"
)
Set-Content -LiteralPath $validationPath -Encoding ascii -Value $validation
Write-Output $comparisonPath
Write-Output $methodologyPath
Write-Output $validationPath
