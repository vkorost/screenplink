# Finds the places a PowerShell script DESTROYS his evidence (the live log and
# the verdict images), judged by WHAT is being destroyed rather than by which
# cmdlet spelling or variable name happens to be on the line.
#
# The first version of this check was a regex: `Remove-Item <target>` with the
# target matched against a list of variable names. Measured by review, it went
# red for the exact line it was written for and green for `rm`/`del`/`ri`,
# `[IO.File]::Delete($log)`, `Clear-Content $log`, and the same delete with
# the variable renamed -- and it fired on an unrelated `Remove-Item $p` in a
# script that merely mentioned the log in a comment.
#
# So this uses the PowerShell AST: comments are not nodes; a command is
# destructive by what it does, whatever it is called; and an argument is
# evidence if it names the files OR is a variable assigned (transitively, and
# through foreach) from something that does.
#
# usage: ps_evidence_probe.ps1 -Files "a.ps1|b.ps1" -Out result.json
# $Files is paths joined by "|": powershell -File cannot pass an array.
param([string]$Files, [string]$Out)
$ErrorActionPreference = 'Stop'
$EVIDENCE = 'camera_lab|verdict_(screen|camera)'
$DESTROY_CMDS = 'remove-item', 'ri', 'rm', 'del', 'erase', 'rmdir', 'rd',
                'clear-content', 'clc', 'set-content', 'sc', 'out-file'
$DESTROY_METHODS = 'Delete', 'WriteAllText', 'WriteAllBytes', 'WriteAllLines',
                   'Create', 'CreateText', 'OpenWrite', 'SetLength'

function Get-VarNames($node) {
    $node.FindAll({ param($n) $n -is [System.Management.Automation.Language.VariableExpressionAst] }, $true) |
        ForEach-Object { $_.VariablePath.UserPath.ToLowerInvariant() }
}

$found = @()
foreach ($file in ($Files -split "[|]" | Where-Object { $_ })) {
    $tokens = $null; $errors = $null
    $ast = [System.Management.Automation.Language.Parser]::ParseFile($file, [ref]$tokens, [ref]$errors)
    if ($errors.Count) { throw "$file does not parse: $($errors[0].Message)" }

    # -- which variables hold (a path derived from) his evidence ----------
    $tainted = New-Object 'System.Collections.Generic.HashSet[string]'
    $binds = @()      # (name, source text) for assignments and foreach
    $ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.AssignmentStatementAst] }, $true) |
        ForEach-Object {
            $l = $_.Left
            if ($l -is [System.Management.Automation.Language.VariableExpressionAst]) {
                $binds += [pscustomobject]@{ Name = $l.VariablePath.UserPath.ToLowerInvariant(); Node = $_.Right }
            }
        }
    $ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.ForEachStatementAst] }, $true) |
        ForEach-Object {
            $binds += [pscustomobject]@{ Name = $_.Variable.VariablePath.UserPath.ToLowerInvariant(); Node = $_.Condition }
        }
    do {
        $before = $tainted.Count
        foreach ($b in $binds) {
            if ($tainted.Contains($b.Name)) { continue }
            $hit = ($b.Node.Extent.Text -match $EVIDENCE)
            if (-not $hit) { $hit = [bool](Get-VarNames $b.Node | Where-Object { $tainted.Contains($_) }) }
            if ($hit) { [void]$tainted.Add($b.Name) }
        }
    } while ($tainted.Count -ne $before)

    function Test-Evidence($node) {
        if ($null -eq $node) { return $false }
        if ($node.Extent.Text -match $EVIDENCE) { return $true }
        return [bool](Get-VarNames $node | Where-Object { $tainted.Contains($_) })
    }

    # -- commands: destructive by what they do, under any name ------------
    $ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.CommandAst] }, $true) |
        ForEach-Object {
            $cmd = $_
            $name = $cmd.GetCommandName()
            if (-not $name -or $DESTROY_CMDS -notcontains $name.ToLowerInvariant()) { return }
            $argText = ($cmd.CommandElements | Select-Object -Skip 1 | ForEach-Object { $_.Extent.Text }) -join ' '
            # Appending destroys nothing.
            if ($argText -match '(?i)(^|\s)-Append\b') { return }
            $targets = @($cmd.CommandElements | Select-Object -Skip 1)
            # `... | Remove-Item`: the thing being removed arrives on the pipe.
            $pipe = $cmd.Parent
            if ($pipe -is [System.Management.Automation.Language.PipelineAst]) {
                foreach ($e in $pipe.PipelineElements) { if ($e -ne $cmd) { $targets += $e } }
            }
            if ($targets | Where-Object { Test-Evidence $_ }) {
                $found += [pscustomobject]@{ file = $file; line = $cmd.Extent.StartLineNumber
                                             what = $name; text = $cmd.Extent.Text }
            }
        }

    # -- .NET methods: [IO.File]::Delete($x), (Get-Item $x).Delete() ------
    $ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.InvokeMemberExpressionAst] }, $true) |
        ForEach-Object {
            $m = $_
            $mname = $m.Member.Extent.Text
            if ($DESTROY_METHODS -notcontains $mname) { return }
            if ((Test-Evidence $m.Expression) -or ($m.Arguments | Where-Object { Test-Evidence $_ })) {
                $found += [pscustomobject]@{ file = $file; line = $m.Extent.StartLineNumber
                                             what = $mname; text = $m.Extent.Text }
            }
        }

    # -- `> $log` (a redirect that truncates) -----------------------------
    $ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.FileRedirectionAst] }, $true) |
        ForEach-Object {
            if (-not $_.Append -and (Test-Evidence $_.Location)) {
                $found += [pscustomobject]@{ file = $file; line = $_.Extent.StartLineNumber
                                             what = 'redirect'; text = $_.Parent.Extent.Text }
            }
        }
}
ConvertTo-Json -InputObject @($found) -Depth 3 | Set-Content -Path $Out -Encoding UTF8
