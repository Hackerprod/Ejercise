[CmdletBinding()]
param(
  [string]$ObjectFile = "",
  [string]$BuildDirectory = "build-windows",
  [string]$Configuration = "Release",
  [string]$OutputFile = "results/transitions.dumpbin.txt"
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

$Object = Resolve-CnrlObject $ObjectFile "transitions"
$Output = if ([IO.Path]::IsPathRooted($OutputFile)) { $OutputFile } else { Join-Path $Root $OutputFile }
$Directory = Split-Path -Parent $Output
if (-not (Test-Path $Directory)) { $null = New-Item -ItemType Directory -Path $Directory }
$Dumpbin = Resolve-Dumpbin
& $Dumpbin /DISASM:nobytes $Object | Set-Content -Encoding ascii $Output
if ($LASTEXITCODE -ne 0) { throw "dumpbin failed" }
$text = (Get-Content -Raw $Output).ToLowerInvariant()
foreach ($opcode in @(
  "vpmovsxbd", "vpmulld", "vpsrlvd", "vpackssdw", "vpacksswb",
  "vcvtdq2pd", "vroundpd", "vcvtpd2dq"
)) {
  if ($text -notmatch "\b$opcode\b") { throw "Required transition opcode not found: $opcode" }
}
if ($text -notmatch "\bvsqrtsd\b" -and $text -notmatch "\bvsqrtpd\b") {
  throw "Required transition square-root opcode not found: vsqrtsd or vsqrtpd"
}
Write-Output "PASS transitions: vector scale/shift, saturated packing, and RMS path confirmed."
Write-Output "Object audited: $Object"
Write-Output "Disassembly archived at $Output"
