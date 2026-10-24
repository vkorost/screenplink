# Install the pre-commit hook, and optionally the SessionStart hook.
#
#     powershell -NoProfile -File tools\install-hooks.ps1
#     powershell -NoProfile -File tools\install-hooks.ps1 -SessionStart
#
# The second form also writes .claude\settings.json so every new Claude
# session in this project starts by printing tools\status.py. That changes
# how your sessions behave rather than how the app behaves, so it is opt-in
# and never installed on the assistant's own initiative.
param([switch]$SessionStart)
#
# The hook lives in tools/ so it is in the repo and reviewable; .git/hooks
# is not version-controlled, so a hook that lives only there is invisible
# and un-reviewable, and nobody can tell whether it is installed.
#
# Run from the project root:
#     powershell -NoProfile -File tools\install-hooks.ps1

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$src = Join-Path $root 'tools\pre-commit'
# $root is app\; the repository (and its .git) is the folder above it.
$dstDir = Join-Path (Split-Path -Parent $root) '.git\hooks'
$dst = Join-Path $dstDir 'pre-commit'

if (-not (Test-Path $src)) { throw "missing $src" }
if (-not (Test-Path $dstDir)) { throw "not a git repo: $dstDir" }

# Write LF bytes, not Copy-Item: this repo has core.autocrlf=true, so the
# working copy of tools/pre-commit can be CRLF, and `#!/bin/sh` + CR does not
# launch under Git for Windows. Then read it back and compare, because
# tests/test_rules.py section N fails the build if what git runs differs from
# what the repo shows -- and "installed" must mean that, not "a copy ran".
$bytes = [System.IO.File]::ReadAllBytes($src)
$text = [System.Text.Encoding]::UTF8.GetString($bytes) -replace "`r`n", "`n"
[System.IO.File]::WriteAllBytes($dst, [System.Text.Encoding]::UTF8.GetBytes($text))
$back = [System.Text.Encoding]::UTF8.GetString([System.IO.File]::ReadAllBytes($dst))
if ($back -ne $text) { throw "installed hook does not match $src after writing" }
Write-Host "installed: $dst"
Write-Host ""
Write-Host "It runs, cheapest first:"
Write-Host "  the executable rules, the Director's properties, the mutation"
Write-Host "  check that proves those can fail, then the synthetic suite"
Write-Host "  (~12s, no camera, no screen)."
Write-Host ""
Write-Host "And when the startup chain is among the staged files, the"
Write-Host "cold-start harness as well -- several minutes, no monitor. That"
Write-Host "is the one that finds the races; RULES C13 asked for it to be run"
Write-Host "before committing, and then eight changes went in without it."
Write-Host ""
Write-Host "Deliberate bypass: git commit --no-verify"

if ($SessionStart) {
    $cfgDir = Join-Path $root '.claude'
    $cfg = Join-Path $cfgDir 'settings.json'
    if (-not (Test-Path $cfgDir)) {
        New-Item -ItemType Directory -Path $cfgDir | Out-Null
    }
    if (Test-Path $cfg) {
        Write-Host ""
        Write-Host "NOT overwriting the existing $cfg."
        Write-Host "Merge the 'hooks' block from tools\session-start-hook.json"
        Write-Host "into it by hand -- a settings file you already edited is"
        Write-Host "yours, and clobbering it is exactly the class of mistake"
        Write-Host "that put auto_connect:false into settings.local.json and"
        Write-Host "stopped the app connecting on launch."
    } else {
        $src = Join-Path $root 'tools\session-start-hook.json'
        $j = Get-Content $src -Raw | ConvertFrom-Json
        # Only the hooks block: the _what_this_is prose is documentation.
        @{ hooks = $j.hooks } | ConvertTo-Json -Depth 8 |
            Out-File $cfg -Encoding utf8
        Write-Host ""
        Write-Host "installed: $cfg"
        Write-Host "Every new session in this project now opens with"
        Write-Host "tools\status.py -- the honest state, read from the logs,"
        Write-Host "the git tags and the files rather than from anyone's"
        Write-Host "memory. Delete that file to undo it."
    }
}
