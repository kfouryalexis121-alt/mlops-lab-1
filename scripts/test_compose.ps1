$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)

function Invoke-Docker {
    & docker @args
    if ($LASTEXITCODE -ne 0) { throw "Docker command failed: $args" }
}

$sample = Get-ChildItem -LiteralPath 'data/food11_processed_mini/validation/Bread' -Filter '*.jpg' | Select-Object -First 1
if (-not $sample) { throw 'Restore the mini dataset before running these checks.' }

Invoke-Docker compose exec -T inference mkdir -p /tmp/lab4-check/data/food11_processed_mini/validation/Bread /tmp/lab4-check/tests
Invoke-Docker compose cp tests/. inference:/tmp/lab4-check/tests
Invoke-Docker compose cp $sample.FullName inference:/tmp/lab4-check/data/food11_processed_mini/validation/Bread/sample.jpg
Invoke-Docker compose exec -T -e PYTHONPATH=/app inference python -m unittest discover -s /tmp/lab4-check/tests -v

Invoke-Docker compose exec -T frontend mkdir -p /app/tests
Invoke-Docker compose cp frontend/tests/. frontend:/app/tests
Invoke-Docker compose cp $sample.FullName frontend:/tmp/food11-test.jpg
Invoke-Docker compose exec -T -e FOOD11_LIVE_TEST_IMAGE=/tmp/food11-test.jpg frontend python -m unittest discover -s /app/tests -v
