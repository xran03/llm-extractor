<#
.SYNOPSIS
  Convenience wrapper: llm-extract with sensible defaults.
.EXAMPLE
  .\bin\llm-extract.ps1 -i .\docs -o .\out --api llmhub
#>
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot

$venv = Join-Path $root '.venv\Scripts\llm-extract.exe'
if (Test-Path $venv) {
    & $venv @args
} elseif (Get-Command llm-extract -ErrorAction SilentlyContinue) {
    & llm-extract @args
} else {
    & python -m llm_extractor @args
}
exit $LASTEXITCODE
