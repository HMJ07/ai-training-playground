# One-command setup for Windows PowerShell: creates the venv, installs
# dependencies, and prepares your .env file.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "==> Creando entorno virtual (.venv)..."
python -m venv .venv

Write-Host "==> Instalando dependencias (esto puede tardar unos minutos por PyTorch)..."
& .\.venv\Scripts\python -m pip install --upgrade pip --quiet
& .\.venv\Scripts\python -m pip install -r requirements.txt

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "==> Creado .env a partir de .env.example. Abrelo y pon tu clave de API."
} else {
    Write-Host "==> Ya existe un .env, no lo toco."
}

Write-Host ""
Write-Host "Listo. Para arrancar la plataforma:"
Write-Host "  .\.venv\Scripts\python run.py"
