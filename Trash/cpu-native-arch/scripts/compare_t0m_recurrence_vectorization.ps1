[CmdletBinding()]
param(
  [string]$ProjectRoot = "",
  [string]$PreOutputDirectory = "",
  [string]$PostOutputDirectory = "",
  [string]$OutputDirectory = ""
)

$ErrorActionPreference = "Stop"
$invariant = [Globalization.CultureInfo]::InvariantCulture
[System.Threading.Thread]::CurrentThread.CurrentCulture = $invariant
[System.Threading.Thread]::CurrentThread.CurrentUICulture = $invariant
if ([string]::IsNullOrWhiteSpace($ProjectRoot)) { $ProjectRoot = Split-Path -Parent $PSScriptRoot }
if ([string]::IsNullOrWhiteSpace($PreOutputDirectory)) {
  $PreOutputDirectory = Join-Path $ProjectRoot "sweep-output\t0m-recurrence-performance-d512"
}
if ([string]::IsNullOrWhiteSpace($PostOutputDirectory)) {
  $PostOutputDirectory = Join-Path $ProjectRoot "sweep-output\t0m-recurrence-performance-d512-avx2"
}
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
  $OutputDirectory = Join-Path $ProjectRoot "sweep-output\t0m-recurrence-vectorization-comparison"
}
if (-not (Test-Path -LiteralPath $OutputDirectory -PathType Container)) {
  $null = New-Item -ItemType Directory -Path $OutputDirectory
}
$comparisonPath = Join-Path $OutputDirectory "comparison-pre-post.csv"
$maximaPath = Join-Path $OutputDirectory "maxima.csv"
$summaryPath = Join-Path $OutputDirectory "summary.txt"
$validationPath = Join-Path $OutputDirectory "validation.txt"
foreach ($path in @($comparisonPath, $maximaPath, $summaryPath, $validationPath)) {
  if (Test-Path -LiteralPath $path -PathType Leaf) { throw "Refusing overwrite: $path" }
}

function Parse-Double([object]$Value, [string]$Name) {
  $parsed = 0.0
  if (-not [double]::TryParse([string]$Value, [Globalization.NumberStyles]::Float, $invariant, [ref]$parsed) -or
      [double]::IsNaN($parsed) -or [double]::IsInfinity($parsed)) {
    throw "Invalid finite numeric value for $Name`: $Value"
  }
  return $parsed
}
function Format-Number([double]$Value) { return $Value.ToString("R", $invariant) }
function Get-Row([object[]]$Rows, [int]$R, [string]$Name) {
  $matches = @($Rows | Where-Object { [int]$_.R -eq $R })
  if ($matches.Count -ne 1) { throw "Expected one $Name row for R=$R, got $($matches.Count)" }
  return $matches[0]
}
function Get-MetricRow([object[]]$Rows, [int]$R, [int]$S, [string]$Name) {
  $matches = @($Rows | Where-Object { [int]$_.R -eq $R -and [int]$_.S -eq $S })
  if ($matches.Count -ne 1) { throw "Expected one $Name metric row for R=$R S=$S, got $($matches.Count)" }
  return $matches[0]
}

$metricNames = @("G8", "G16", "F4", "F8", "F16")
$rValues = @(1, 2, 4, 8, 16)
$phaseInputs = [ordered]@{
  "pre-vectorization" = [pscustomobject]@{
    Comparison = Join-Path $PreOutputDirectory "comparison.csv"
    Metrics = Join-Path $PreOutputDirectory "metrics.csv"
  }
  "post-vectorization" = [pscustomobject]@{
    Comparison = Join-Path $PostOutputDirectory "comparison.csv"
    Metrics = Join-Path $PostOutputDirectory "metrics.csv"
  }
}
$phaseRows = [ordered]@{}
$outputRows = New-Object 'System.Collections.Generic.List[object]'
foreach ($phase in $phaseInputs.Keys) {
  $inputPath = $phaseInputs[$phase].Comparison
  $metricsPath = $phaseInputs[$phase].Metrics
  if (-not (Test-Path -LiteralPath $inputPath -PathType Leaf)) { throw "Required input not found: $inputPath" }
  if (-not (Test-Path -LiteralPath $metricsPath -PathType Leaf)) { throw "Required input not found: $metricsPath" }
  $rows = @(Import-Csv -LiteralPath $inputPath)
  $metrics = @(Import-Csv -LiteralPath $metricsPath)
  if ($rows.Count -ne 5) { throw "Expected 5 comparison rows for $phase, got $($rows.Count)" }
  if ($metrics.Count -ne 25) { throw "Expected 25 metric rows for $phase, got $($metrics.Count)" }
  $phaseRows[$phase] = @{}
  foreach ($R in $rValues) {
    $source = Get-Row $rows $R $phase
    $baseMetric = Get-MetricRow $metrics $R 1 $phase
    $row = [ordered]@{ record_type = "R_comparison"; phase = $phase; D = 512; target_kib = 512; R = $R }
    foreach ($metric in $metricNames) {
      $metricSlot = [int]$metric.Substring(1)
      if ($metric.StartsWith("G")) {
        $metricInput = Get-MetricRow $metrics $R $metricSlot $phase
        $value = (Parse-Double $metricInput.G_S "$phase G_S S=$metricSlot R=$R") /
                 (Parse-Double $baseMetric.G_S "$phase G_S S=1 R=$R")
      } else {
        $metricInput = Get-MetricRow $metrics $R $metricSlot $phase
        $value = (Parse-Double $metricInput.G_S "$phase G_S S=$metricSlot R=$R") /
                 (Parse-Double $metricInput.F_S "$phase F_S S=$metricSlot R=$R")
      }
      $staticA = Parse-Double $source.("static_A_${metric}_static") "$phase static A $metric R=$R"
      $staticB = Parse-Double $source.("static_B_${metric}_static") "$phase static B $metric R=$R"
      $row["recurrence_${metric}_norm"] = $value
      $row["static_A_${metric}"] = $staticA
      $row["static_B_${metric}"] = $staticB
      $phaseRows[$phase][$R] = if ($phaseRows[$phase].Contains($R)) { $phaseRows[$phase][$R] } else { @{} }
      $phaseRows[$phase][$R]["recurrence"] = if ($phaseRows[$phase][$R].Contains("recurrence")) { $phaseRows[$phase][$R]["recurrence"] } else { @{} }
      $phaseRows[$phase][$R]["static_A"] = if ($phaseRows[$phase][$R].Contains("static_A")) { $phaseRows[$phase][$R]["static_A"] } else { @{} }
      $phaseRows[$phase][$R]["static_B"] = if ($phaseRows[$phase][$R].Contains("static_B")) { $phaseRows[$phase][$R]["static_B"] } else { @{} }
      $phaseRows[$phase][$R]["recurrence"][$metric] = $value
      $phaseRows[$phase][$R]["static_A"][$metric] = $staticA
      $phaseRows[$phase][$R]["static_B"][$metric] = $staticB
    }
    [void]$outputRows.Add([pscustomobject]$row)
  }
}

$maximaRows = New-Object 'System.Collections.Generic.List[object]'
foreach ($phase in $phaseInputs.Keys) {
  foreach ($sourceName in @("recurrence", "static_A", "static_B")) {
    $row = [ordered]@{ record_type = "maximum"; phase = $phase; source = $sourceName }
    foreach ($metric in $metricNames) {
      $values = @($rValues | ForEach-Object { [double]$phaseRows[$phase][$_][$sourceName][$metric] })
      $row[$metric] = ($values | Measure-Object -Maximum).Maximum
    }
    [void]$maximaRows.Add([pscustomobject]$row)
  }
}

$changeRow = [ordered]@{ record_type = "post_minus_pre"; phase = "post-vectorization minus pre-vectorization"; source = "recurrence" }
$ratioRow = [ordered]@{ record_type = "post_over_pre"; phase = "post-vectorization divided by pre-vectorization"; source = "recurrence" }
foreach ($metric in $metricNames) {
  $pre = [double]$maximaRows[0].$metric
  $post = [double]$maximaRows[3].$metric
  $changeRow[$metric] = $post - $pre
  $ratioRow[$metric] = $post / $pre
}
[void]$maximaRows.Add([pscustomobject]$changeRow)
[void]$maximaRows.Add([pscustomobject]$ratioRow)

Set-Content -LiteralPath $comparisonPath -Encoding ascii -Value @($outputRows | ConvertTo-Csv -NoTypeInformation)
Set-Content -LiteralPath $maximaPath -Encoding ascii -Value @($maximaRows | ConvertTo-Csv -NoTypeInformation)

$validation = @(
  "status=PASS"
  "inputs=pre=$($phaseInputs['pre-vectorization'].Comparison),$($phaseInputs['pre-vectorization'].Metrics); post=$($phaseInputs['post-vectorization'].Comparison),$($phaseInputs['post-vectorization'].Metrics)"
  "input_rows=pre=comparison=5,metrics=25; post=comparison=5,metrics=25; expected_each=comparison=5,metrics=25"
  "output_rows=comparison-pre-post.csv=$($outputRows.Count); expected=10; maxima.csv=$($maximaRows.Count); expected=8"
  "scope=D=512; R=1,2,4,8,16; metrics=G8,G16,F4,F8,F16"
  "static_reference=carried independently from each validated comparison.csv; target_kib=512; variants=A,B"
  "finite_validation=PASS all pre/post recurrence and static maxima inputs finite"
  "phase_labels=PASS explicit pre-vectorization and post-vectorization rows"
)
Set-Content -LiteralPath $validationPath -Encoding ascii -Value $validation

$preMax = @($maximaRows | Where-Object { $_.phase -eq "pre-vectorization" -and $_.source -eq "recurrence" })[0]
$postMax = @($maximaRows | Where-Object { $_.phase -eq "post-vectorization" -and $_.source -eq "recurrence" })[0]
$staticAMax = @($maximaRows | Where-Object { $_.phase -eq "post-vectorization" -and $_.source -eq "static_A" })[0]
$staticBMax = @($maximaRows | Where-Object { $_.phase -eq "post-vectorization" -and $_.source -eq "static_B" })[0]
$summary = @(
  "status=PASS"
  "methodology=Compared validated pre/post normalized recurrence comparison.csv artifacts; maxima are across R=1,2,4,8,16 for each metric."
  "pre_input=$($phaseInputs['pre-vectorization'].Comparison),$($phaseInputs['pre-vectorization'].Metrics)"
  "post_input=$($phaseInputs['post-vectorization'].Comparison),$($phaseInputs['post-vectorization'].Metrics)"
  "pre-vectorization recurrence_max_G8=$(Format-Number $preMax.G8); recurrence_max_G16=$(Format-Number $preMax.G16); recurrence_max_F4=$(Format-Number $preMax.F4); recurrence_max_F8=$(Format-Number $preMax.F8); recurrence_max_F16=$(Format-Number $preMax.F16)"
  "post-vectorization recurrence_max_G8=$(Format-Number $postMax.G8); recurrence_max_G16=$(Format-Number $postMax.G16); recurrence_max_F4=$(Format-Number $postMax.F4); recurrence_max_F8=$(Format-Number $postMax.F8); recurrence_max_F16=$(Format-Number $postMax.F16)"
  "post_minus_pre=G8=$(Format-Number ($postMax.G8 - $preMax.G8)); G16=$(Format-Number ($postMax.G16 - $preMax.G16)); F4=$(Format-Number ($postMax.F4 - $preMax.F4)); F8=$(Format-Number ($postMax.F8 - $preMax.F8)); F16=$(Format-Number ($postMax.F16 - $preMax.F16))"
  "post_over_pre=G8=$(Format-Number ($postMax.G8 / $preMax.G8)); G16=$(Format-Number ($postMax.G16 / $preMax.G16)); F4=$(Format-Number ($postMax.F4 / $preMax.F4)); F8=$(Format-Number ($postMax.F8 / $preMax.F8)); F16=$(Format-Number ($postMax.F16 / $preMax.F16))"
  "static_D512_target512_variant_A_max_G8=$(Format-Number $staticAMax.G8); static_D512_target512_variant_A_max_G16=$(Format-Number $staticAMax.G16); static_D512_target512_variant_A_max_F4=$(Format-Number $staticAMax.F4); static_D512_target512_variant_A_max_F8=$(Format-Number $staticAMax.F8); static_D512_target512_variant_A_max_F16=$(Format-Number $staticAMax.F16)"
  "static_D512_target512_variant_B_max_G8=$(Format-Number $staticBMax.G8); static_D512_target512_variant_B_max_G16=$(Format-Number $staticBMax.G16); static_D512_target512_variant_B_max_F4=$(Format-Number $staticBMax.F4); static_D512_target512_variant_B_max_F8=$(Format-Number $staticBMax.F8); static_D512_target512_variant_B_max_F16=$(Format-Number $staticBMax.F16)"
  "artifacts=comparison-pre-post.csv,maxima.csv,summary.txt,validation.txt"
  "validation=PASS; see validation.txt"
)
Set-Content -LiteralPath $summaryPath -Encoding ascii -Value $summary
Write-Output $comparisonPath
Write-Output $maximaPath
Write-Output $summaryPath
Write-Output $validationPath
