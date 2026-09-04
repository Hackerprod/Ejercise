[CmdletBinding()]
param(
  [string]$ObjectFile = "",
  [string]$BuildDirectory = "build-windows",
  [string]$Configuration = "Release",
  [string]$OutputFile = "results/kernels.dumpbin.txt"
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot

function Resolve-Dumpbin {
  $command = Get-Command dumpbin.exe -ErrorAction SilentlyContinue
  if ($null -ne $command) { return $command.Source }
  $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio/Installer/vswhere.exe"
  if (Test-Path $vswhere -PathType Leaf) {
    $installation = (& $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath | Select-Object -First 1)
    if (-not [string]::IsNullOrWhiteSpace($installation)) {
      $vcRoot = Join-Path $installation "VC/Tools/MSVC"
      $matches = @(Get-ChildItem -Path $vcRoot -Directory -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending | ForEach-Object {
          Join-Path $_.FullName "bin/Hostx64/x64/dumpbin.exe"
        } | Where-Object { Test-Path $_ -PathType Leaf })
      if ($matches.Count -gt 0) { return $matches[0] }
    }
  }
  throw "dumpbin.exe was not found. Install MSVC C++ tools or run from a Developer PowerShell."
}


function Resolve-CnrlObject([string]$ExplicitPath, [string]$Stem) {
  if (-not [string]::IsNullOrWhiteSpace($ExplicitPath)) {
    $candidate = if ([IO.Path]::IsPathRooted($ExplicitPath)) { $ExplicitPath } else { Join-Path $Root $ExplicitPath }
    if (-not (Test-Path $candidate -PathType Leaf)) { throw "Object file not found: $candidate" }
    return (Resolve-Path $candidate).Path
  }
  $build = if ([IO.Path]::IsPathRooted($BuildDirectory)) { $BuildDirectory } else { Join-Path $Root $BuildDirectory }
  if (-not (Test-Path $build -PathType Container)) { throw "Build directory not found: $build" }
  $matches = @(Get-ChildItem -Path $build -Recurse -File -Filter "*.obj" | Where-Object {
    $_.FullName -match [regex]::Escape("cnrl_core.dir") -and
    $_.BaseName -match "^$([regex]::Escape($Stem))(\.cpp)?$" -and
    ($_.FullName -match "[\\/]$([regex]::Escape($Configuration))[\\/]" -or
     $_.DirectoryName -notmatch "[\\/](Debug|RelWithDebInfo|MinSizeRel)[\\/]")
  })
  if ($matches.Count -eq 0) {
    throw "Could not locate $Stem object below $build. Build configuration=$Configuration first."
  }
  if ($matches.Count -gt 1) {
    $releaseMatches = @($matches | Where-Object { $_.FullName -match "[\\/]$([regex]::Escape($Configuration))[\\/]" })
    if ($releaseMatches.Count -eq 1) { return $releaseMatches[0].FullName }
    throw "Ambiguous $Stem object candidates:`n$($matches.FullName -join "`n")"
  }
  return $matches[0].FullName
}

$Object = Resolve-CnrlObject $ObjectFile "kernels"
$Output = if ([IO.Path]::IsPathRooted($OutputFile)) { $OutputFile } else { Join-Path $Root $OutputFile }
$Directory = Split-Path -Parent $Output
if (-not (Test-Path $Directory)) { $null = New-Item -ItemType Directory -Path $Directory }
$Dumpbin = Resolve-Dumpbin
& $Dumpbin /DISASM:nobytes $Object | Set-Content -Encoding ascii $Output
if ($LASTEXITCODE -ne 0) { throw "dumpbin failed" }
$text = (Get-Content -Raw $Output).ToLowerInvariant()

function Get-FunctionBody([string]$Name,[string]$NextName) {
  $start = $text.IndexOf($Name.ToLowerInvariant())
  if ($start -lt 0) { throw "Function marker not found in dumpbin output: $Name" }
  $end = if ([string]::IsNullOrWhiteSpace($NextName)) { $text.Length } else { $text.IndexOf($NextName.ToLowerInvariant(), $start + $Name.Length) }
  if ($end -lt 0) { $end = $text.Length }
  return $text.Substring($start, $end - $start)
}

$fused4 = Get-FunctionBody "fused4" "fused8"
foreach ($opcode in @("vpmaddwd","vpmovsxbw")) {
  if ($fused4 -notmatch $opcode) { throw "Required opcode not found inside fused4: $opcode" }
}
$spillPattern = '(?im)^.*\b(?:xmm|ymm)\d*\b.*\[(?:r|e)(?:sp|bp)[^\]]*\].*$'
if ($fused4 -match $spillPattern) {
  throw "Vector stack access found inside fused4; audited baseline is no longer spill-safe"
}
Write-Output "PASS fused4: AVX2 signed expansion + pairwise dot product, no XMM/YMM stack access."

try {
  $fused8 = Get-FunctionBody "fused8" "validate_kernel_call"
  if ($fused8 -match $spillPattern) {
    Write-Warning "fused8 contains vector stack access. Keep --slot-tile 4 as the release baseline."
  } else {
    Write-Output "INFO fused8 contains no detected XMM/YMM stack access in this MSVC build."
  }
} catch {
  Write-Warning "Could not isolate fused8: $($_.Exception.Message)"
}
Write-Output "Object audited: $Object"
Write-Output "Disassembly archived at $Output"
