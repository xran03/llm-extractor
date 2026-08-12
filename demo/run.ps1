<#
.SYNOPSIS
  Run the demo against a live gateway. Configure credentials first (see ..\.env.example).
.EXAMPLE
  .\demo\run.ps1 --api llmhub --model gpt-4.1
#>
$ErrorActionPreference = 'Stop'
$here = $PSScriptRoot
$root = Split-Path -Parent $here

& (Join-Path $root 'bin\llm-extract.ps1') `
    --input $here `
    --output (Join-Path $here 'out') `
    --ocr always `
    --format both `
    @args
exit $LASTEXITCODE
