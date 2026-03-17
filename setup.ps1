Write-Host "Checking for uv..." -ForegroundColor Cyan

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv not found. Installing uv..." -ForegroundColor Yellow
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    Write-Host "uv installed. Please close and reopen this terminal, then run .\setup.ps1 again." -ForegroundColor Green
    exit 0
}

Write-Host "Creating virtual environment with uv..." -ForegroundColor Cyan
uv venv .venv

Write-Host "Installing dependencies..." -ForegroundColor Cyan
uv sync --extra dev

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host ".env file created from .env.example" -ForegroundColor Green
}

Write-Host ""
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host "You can now run commands using:" -ForegroundColor Cyan
Write-Host "  .\run.ps1 validate"
Write-Host "  .\run.ps1 train"
Write-Host "  .\run.ps1 serve"
Write-Host ""
