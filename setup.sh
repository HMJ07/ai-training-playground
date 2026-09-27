#!/usr/bin/env bash
# One-command setup for Linux/macOS/Git-Bash: creates the venv, installs
# dependencies, and prepares your .env file.
set -e

cd "$(dirname "$0")"

echo "==> Creando entorno virtual (.venv)..."
python3 -m venv .venv 2>/dev/null || python -m venv .venv

if [ -f ".venv/bin/python" ]; then
  PY=".venv/bin/python"
else
  PY=".venv/Scripts/python"
fi

echo "==> Instalando dependencias (esto puede tardar unos minutos por PyTorch)..."
"$PY" -m pip install --upgrade pip --quiet
"$PY" -m pip install -r requirements.txt

if [ ! -f ".env" ]; then
  cp .env.example .env
  echo "==> Creado .env a partir de .env.example. Abrelo y pon tu clave de API."
else
  echo "==> Ya existe un .env, no lo toco."
fi

echo ""
echo "Listo. Para arrancar la plataforma:"
echo "  $PY run.py"
