Set-StrictMode -Version Latest

function Get-CnrlRoot {
  return (Split-Path -Parent $PSScriptRoot)
}

function Resolve-CnrlBuildDirectory {
  param(
    [Parameter(Mandatory=$true)][string]$Root,
    [Parameter(Mandatory=$true)][string]$BuildDirectory
  )
  if ([IO.Path]::IsPathRooted($BuildDirectory)) { return $BuildDirectory }
  return (Join-Path $Root $BuildDirectory)
}

function Resolve-CnrlBinary {
  param(
    [Parameter(Mandatory=$true)][string]$Root,
    [Parameter(Mandatory=$true)][string]$BuildDirectory,
    [Parameter(Mandatory=$true)][string]$Configuration,
    [Parameter(Mandatory=$true)][string]$Name
  )
  $Build = Resolve-CnrlBuildDirectory -Root $Root -BuildDirectory $BuildDirectory
  $FileName = if ($Name.EndsWith('.exe', [StringComparison]::OrdinalIgnoreCase)) { $Name } else { "$Name.exe" }
  $Candidates = @(
    (Join-Path (Join-Path $Build $Configuration) $FileName),
    (Join-Path $Build $FileName),
    (Join-Path (Join-Path (Join-Path $Build 'bin') $Configuration) $FileName),
    (Join-Path (Join-Path $Build 'bin') $FileName)
  )
  foreach ($Candidate in $Candidates) {
    if (Test-Path -LiteralPath $Candidate -PathType Leaf) {
      return (Resolve-Path -LiteralPath $Candidate).Path
    }
  }
  if (Test-Path -LiteralPath $Build -PathType Container) {
    $Matches = @(Get-ChildItem -LiteralPath $Build -Recurse -File -Filter $FileName -ErrorAction SilentlyContinue)
    if ($Matches.Count -eq 1) { return $Matches[0].FullName }
    if ($Matches.Count -gt 1) {
      $Configured = @($Matches | Where-Object { $_.FullName -match "[\\/]$([regex]::Escape($Configuration))[\\/]" })
      if ($Configured.Count -eq 1) { return $Configured[0].FullName }
      throw "Ambiguous binary '$FileName' below '$Build':`n$($Matches.FullName -join "`n")"
    }
  }
  throw "Binary not found: $FileName below $Build"
}

function Resolve-CnrlDefaultExecutable {
  param(
    [Parameter(Mandatory=$true)][string]$Root,
    [Parameter(Mandatory=$true)][string]$Name,
    [string]$BuildDirectory = 'build-windows',
    [string]$Configuration = 'Release'
  )
  return Resolve-CnrlBinary -Root $Root -BuildDirectory $BuildDirectory -Configuration $Configuration -Name $Name
}

function Find-CnrlVsWhere {
  $Candidates = @()
  if (${env:ProgramFiles(x86)}) {
    $Candidates += (Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe')
  }
  if ($env:ProgramFiles) {
    $Candidates += (Join-Path $env:ProgramFiles 'Microsoft Visual Studio/Installer/vswhere.exe')
  }
  foreach ($Candidate in $Candidates) {
    if (Test-Path -LiteralPath $Candidate -PathType Leaf) { return $Candidate }
  }
  return $null
}

function Find-CnrlVisualStudioInstallation {
  $VsWhere = Find-CnrlVsWhere
  if ($null -eq $VsWhere) { return $null }
  $Installation = (& $VsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath | Select-Object -First 1)
  if ([string]::IsNullOrWhiteSpace($Installation)) { return $null }
  return $Installation.Trim()
}

function Find-CnrlVisualStudioInstallationVersion {
  $VsWhere = Find-CnrlVsWhere
  if ($null -eq $VsWhere) { return $null }
  $Version = (& $VsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationVersion | Select-Object -First 1)
  if ([string]::IsNullOrWhiteSpace($Version)) { return $null }
  return $Version.Trim()
}

function Import-CnrlVcVars {
  param([string]$Architecture = 'x64')
  if (Get-Command cl.exe -ErrorAction SilentlyContinue) { return }
  $Installation = Find-CnrlVisualStudioInstallation
  if ([string]::IsNullOrWhiteSpace($Installation)) {
    throw 'MSVC cl.exe is not in PATH and no Visual Studio/Build Tools C++ installation was found.'
  }
  $VcVars = Join-Path $Installation 'VC/Auxiliary/Build/vcvarsall.bat'
  if (-not (Test-Path -LiteralPath $VcVars -PathType Leaf)) {
    throw "vcvarsall.bat was not found: $VcVars"
  }
  $Command = "`"$VcVars`" $Architecture >nul && set"
  $Lines = @(& $env:ComSpec /d /s /c $Command)
  if ($LASTEXITCODE -ne 0) { throw "vcvarsall.bat failed with exit code $LASTEXITCODE" }
  foreach ($Line in $Lines) {
    $Separator = $Line.IndexOf('=')
    if ($Separator -le 0) { continue }
    $Name = $Line.Substring(0, $Separator)
    $Value = $Line.Substring($Separator + 1)
    Set-Item -Path "Env:$Name" -Value $Value
  }
  if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    throw 'vcvarsall.bat completed but cl.exe is still unavailable.'
  }
}

function Find-CnrlNinja {
  $Command = Get-Command ninja.exe -ErrorAction SilentlyContinue
  if ($null -eq $Command) { $Command = Get-Command ninja -ErrorAction SilentlyContinue }
  if ($null -ne $Command) { return $Command.Source }
  $Installation = Find-CnrlVisualStudioInstallation
  if (-not [string]::IsNullOrWhiteSpace($Installation)) {
    $Candidates = @(
      (Join-Path $Installation 'Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe'),
      (Join-Path $Installation 'Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja')
    )
    foreach ($Candidate in $Candidates) {
      if (Test-Path -LiteralPath $Candidate -PathType Leaf) { return $Candidate }
    }
  }
  return $null
}


function Find-CnrlLatestVisualStudioGenerator {
  $CapabilitiesText = (& cmake -E capabilities) -join "`n"
  if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($CapabilitiesText)) {
    throw 'cmake -E capabilities failed while locating a Visual Studio generator.'
  }
  $Capabilities = $CapabilitiesText | ConvertFrom-Json
  $Candidates = @($Capabilities.generators | Where-Object {
      $_.name -match '^Visual Studio [0-9]+ '
    } | Sort-Object {
      if ($_.name -match '^Visual Studio ([0-9]+) ') { [int]$Matches[1] } else { 0 }
    } -Descending)
  if ($Candidates.Count -eq 0) {
    throw 'CMake reports no Visual Studio generator with C++ support.'
  }
  $InstallationVersion = Find-CnrlVisualStudioInstallationVersion
  if (-not [string]::IsNullOrWhiteSpace($InstallationVersion)) {
    $MajorText = ($InstallationVersion -split '\.')[0]
    $Major = 0
    if ([int]::TryParse($MajorText, [ref]$Major)) {
      $InstalledMatch = @($Candidates | Where-Object {
          $_.name -match ("^Visual Studio " + $Major + " ")
        } | Select-Object -First 1)
      if ($InstalledMatch.Count -eq 1) { return [string]$InstalledMatch[0].name }
    }
  }
  return [string]$Candidates[0].name
}

function Get-CnrlCacheValue {
  param(
    [Parameter(Mandatory=$true)][string]$CachePath,
    [Parameter(Mandatory=$true)][string]$Name
  )
  if (-not (Test-Path -LiteralPath $CachePath -PathType Leaf)) { return $null }
  $Prefix = "$Name="
  $TypedPrefix = "$Name`:"
  foreach ($Line in Get-Content -LiteralPath $CachePath) {
    if ($Line.StartsWith($Prefix, [StringComparison]::Ordinal)) {
      return $Line.Substring($Prefix.Length)
    }
    if ($Line.StartsWith($TypedPrefix, [StringComparison]::Ordinal)) {
      $Equals = $Line.IndexOf('=')
      if ($Equals -ge 0) { return $Line.Substring($Equals + 1) }
    }
  }
  return $null
}
