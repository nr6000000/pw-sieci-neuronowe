$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$Epochs = 300
$Only = "all"
$Out = "experiment_results/$(Get-Date -Format yyyyMMdd_HHmmss)"

foreach ($Dataset in @("cube", "activation", "simple")) {
    Write-Host "`nRunning: $Dataset"

    uv run python proj1/main.py `
        --dataset $Dataset `
        --only $Only `
        --epochs $Epochs `
        --batch-size 32 `
        --learning-rate 0.02 `
        --seed 42 `
        --count 100 `
        --out "$Out/$Dataset"

    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

Write-Host "`nResults: $Out"
