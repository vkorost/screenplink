# Runs the REAL gate judgement (repeat_gate.ps1's Get-LaunchJudgement) on
# crafted launches, without launching anything.
#
# The function is pulled out of the script by AST, together with the
# constants it reads, so nothing here is a paraphrase of it (RULES C12) and
# the script is never executed -- running it would start the app.
#
# usage: gate_probe.ps1 -Cases cases.json -Out results.json
param([string]$Gate, [string]$Cases, [string]$Out)
$ErrorActionPreference = 'Stop'
$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($Gate, [ref]$tokens, [ref]$errors)
if ($errors.Count) { throw "repeat_gate.ps1 does not parse: $($errors[0].Message)" }
$wanted = 'HELD_OUT_MM_MAX', 'FPS_MIN', 'WRITE_BUDGET', 'GATE_LINE'
$nodes = $ast.FindAll({
    param($n)
    ($n -is [System.Management.Automation.Language.FunctionDefinitionAst]) -or
    ($n -is [System.Management.Automation.Language.AssignmentStatementAst] -and
     $n.Left -is [System.Management.Automation.Language.VariableExpressionAst] -and
     $wanted -contains $n.Left.VariablePath.UserPath)
}, $true)
foreach ($n in $nodes) { Invoke-Expression $n.Extent.Text }
if (-not (Get-Command Get-LaunchJudgement -ErrorAction SilentlyContinue)) {
    throw 'Get-LaunchJudgement is not defined in the gate'
}
$results = @()
foreach ($c in (Get-Content $Cases -Raw | ConvertFrom-Json)) {
    $j = Get-LaunchJudgement -Text ([string]$c.text) -Done ([bool]$c.done) `
                             -CleanClose ([bool]$c.clean)
    $results += [pscustomobject]@{ pass = [bool]$j.Pass; reasons = @($j.Reasons) }
}
ConvertTo-Json -InputObject @($results) -Depth 4 | Set-Content -Path $Out -Encoding UTF8
