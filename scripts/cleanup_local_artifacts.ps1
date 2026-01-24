param(
  [switch]$DryRun = $false,
  [switch]$RemoveRootChromaDb = $true,
  [switch]$RemovePythonCaches = $true,
  [switch]$RemoveFrontendBuildArtifacts = $true
)

$ErrorActionPreference = "Stop"

function Remove-Matches([string]$Pattern) {
  $items = Get-ChildItem -Force -Path $Pattern -ErrorAction SilentlyContinue
  if (-not $items) { return }

  foreach ($item in $items) {
    if ($DryRun) {
      Write-Host "[DRYRUN] Would remove: $($item.FullName)"
    } else {
      Write-Host "[OK] Removing: $($item.FullName)"
      Remove-Item -Force -Recurse -LiteralPath $item.FullName
    }
  }
}

function Remove-PathIfExists([string]$Path) {
  if (-not (Test-Path -LiteralPath $Path)) { return }

  if ($DryRun) {
    Write-Host "[DRYRUN] Would remove: $Path"
  } else {
    Write-Host "[OK] Removing: $Path"
    Remove-Item -Force -Recurse -LiteralPath $Path
  }
}

function Remove-PythonCaches() {
  # __pycache__ directories
  $dirs = Get-ChildItem -Force -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue
  foreach ($d in $dirs) {
    if ($DryRun) {
      Write-Host "[DRYRUN] Would remove: $($d.FullName)"
    } else {
      Write-Host "[OK] Removing: $($d.FullName)"
      Remove-Item -Force -Recurse -LiteralPath $d.FullName
    }
  }

  # Compiled python artifacts
  $pyc = Get-ChildItem -Force -Recurse -File -Include "*.pyc","*.pyo" -ErrorAction SilentlyContinue
  foreach ($f in $pyc) {
    if ($DryRun) {
      Write-Host "[DRYRUN] Would remove: $($f.FullName)"
    } else {
      Write-Host "[OK] Removing: $($f.FullName)"
      Remove-Item -Force -LiteralPath $f.FullName
    }
  }
}

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# Data artifacts that should never be committed and are safe to delete locally.
Remove-Matches "DATA\*.bak_*"
Remove-Matches "DATA\*_migrated.parquet"
Remove-Matches "DATA\*_v2.parquet"
Remove-Matches "DATA\_test_*"
Remove-Matches "DATA\_test*.txt"

if ($RemoveRootChromaDb) {
  # Remove the directory itself (not just contents).
  Remove-PathIfExists "chroma_db"
}

if ($RemovePythonCaches) {
  Remove-PythonCaches
}

if ($RemoveFrontendBuildArtifacts) {
  Remove-PathIfExists "frontend\\node_modules"
  Remove-PathIfExists "frontend\\dist"
  Remove-PathIfExists "frontend\\.vite"
}

Write-Host "[DONE] Cleanup complete. (DryRun=$DryRun)"
