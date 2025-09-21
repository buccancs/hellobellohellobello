<#!
.SYNOPSIS
  Run all unit tests for the Multi-Modal Physiological Sensing Platform (Android + Python).

.DESCRIPTION
  Convenience PowerShell script to execute both Android JVM unit tests and Python pytest
  with a single command, or selectively per platform. Designed for Windows/IntelliJ users.

.PARAMETER AndroidOnly
  Run only Android JVM unit tests (:android_sensor_node:app:testDebugUnitTest).

.PARAMETER PythonOnly
  Run only Python tests (pytest using repo-level pytest.ini).

.PARAMETER NoDaemon
  Pass --no-daemon to Gradle (default: enabled). Use -NoDaemon:$false to allow daemon.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\scripts\run_all_tests.ps1

.EXAMPLE
  .\scripts\run_all_tests.ps1 -AndroidOnly

.EXAMPLE
  .\scripts\run_all_tests.ps1 -PythonOnly -NoDaemon:$false

.NOTES
  - Requires Java 17+ and Android SDK for Android tests.
  - Requires Python 3.11+ and pytest for Python tests. Use Gradle task :pc_controller:installRequirements to prep venv.
#>

[CmdletBinding()]
param(
  [switch] $AndroidOnly,
  [switch] $PythonOnly,
  [switch] $NoDaemon = $true
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $repoRoot

function Write-Header($text) {
  Write-Host ''
  Write-Host '==================================================' -ForegroundColor Cyan
  Write-Host "${text}" -ForegroundColor Cyan
  Write-Host '==================================================' -ForegroundColor Cyan
}

function Run-GradleTask([string[]] $gradleTasks) {
  $gradleArgs = @()
  if ($NoDaemon) { $gradleArgs += '--no-daemon' }
  $gradleArgs += '--console=plain'
  $gradleArgs += $gradleTasks

  $stdoutFile = [System.IO.Path]::GetTempFileName()
  $stderrFile = [System.IO.Path]::GetTempFileName()
  try {
    $argList = $gradleArgs -join ' '
    $proc = Start-Process -FilePath .\gradlew.bat -ArgumentList $argList -NoNewWindow -Wait -PassThru -RedirectStandardOutput $stdoutFile -RedirectStandardError $stderrFile
    $code = $proc.ExitCode

    $ansi = [regex]'\x1B\[[0-9;]*[A-Za-z]'
    $lines = @()
    if (Test-Path $stdoutFile) { $lines += Get-Content -LiteralPath $stdoutFile -Raw -ErrorAction SilentlyContinue }
    if (Test-Path $stderrFile) { $lines += Get-Content -LiteralPath $stderrFile -Raw -ErrorAction SilentlyContinue }
    $textAll = [string]::Join("`n", $lines)
    $textAll -split "(`r`n|`n|`r)" | ForEach-Object {
      $text = $_
      if ($null -eq $text) { return }
      $clean = $ansi.Replace($text, '')
      $trim = $clean.Trim()
      $drop = $false
      if ($trim -match '(?i)sun\.misc\.unsafe' -or $trim -match '(?i)terminally deprecated method' -or $trim -match '(?i)Unsafe has been called' -or $trim -match '(?i)^It will be removed in a future release' -or $trim -match '(?i)^Please consider reporting this') {
        $drop = $true
      }
      if (-not $drop -and $trim.Length -gt 0) { Write-Host $text }
    }
    return [int]$code
  }
  finally {
    if (Test-Path $stdoutFile) { Remove-Item -LiteralPath $stdoutFile -ErrorAction SilentlyContinue }
    if (Test-Path $stderrFile) { Remove-Item -LiteralPath $stderrFile -ErrorAction SilentlyContinue }
  }
}

function Run-Pytest() {
  # Use repo-level pytest.ini (pythonpath and testpaths already configured)
  & pytest 2>&1 | Out-Host
  $code = $LASTEXITCODE
  return [int]$code
}

$androidExit = 0
$pythonExit = 0

try {
  if ($AndroidOnly -and $PythonOnly) {
    throw 'Use only one of -AndroidOnly or -PythonOnly, or neither to run both.'
  }

  if ($AndroidOnly) {
    Write-Header 'Running Android JVM unit tests'
    Write-Host 'Note: Forcing Gradle to re-run tasks (--rerun-tasks) to display per-test logs.' -ForegroundColor Yellow
    $androidExit = Run-GradleTask @('--rerun-tasks', ':android_sensor_node:app:testDebugUnitTest')
  }
  elseif ($PythonOnly) {
    Write-Header 'Running Python pytest'
    $pythonExit = Run-Pytest
  }
  else {
    Write-Header 'Running ALL tests (Android + Python) via Gradle aggregate task checkAll'
    Write-Host 'Note: Forcing Gradle to re-run tasks (--rerun-tasks) to display per-test logs.' -ForegroundColor Yellow
    $gradleExitCode = Run-GradleTask @('--rerun-tasks', 'checkAll')
    if ($gradleExitCode -ne 0) { throw "Gradle checkAll failed with exit code $gradleExitCode" }
  }
}
catch {
  Write-Host "[ERROR] $($_.Exception.Message)" -ForegroundColor Red
  if ($androidExit -ne 0 -or $pythonExit -ne 0) { exit 1 } else { exit 1 }
}

if ($AndroidOnly -or $PythonOnly) {
  Write-Header 'Summary'
  if ($AndroidOnly) { Write-Host ("Android tests exit code: {0}" -f $androidExit) }
  if ($PythonOnly) { Write-Host ("Python tests exit code: {0}" -f $pythonExit) }
  if ($androidExit -ne 0 -or $pythonExit -ne 0) { exit 1 } else { exit 0 }
} else {
  # Aggregate task case
  Write-Header 'Summary'
  Write-Host 'All tests finished via Gradle checkAll.' -ForegroundColor Green
  exit 0
}
