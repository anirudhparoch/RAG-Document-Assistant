#!/bin/bash
# One-time setup on macOS (Apple Silicon). Run from this folder:  bash setup_mac.sh
set -e
cd "$(dirname "$0")"

PY=""
for v in python3.12 python3.11 python3.13 python3; do
  if command -v $v >/dev/null 2>&1; then PY=$v; break; fi
done
if [ -z "$PY" ] || ! $PY -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
  echo "❌ Python 3.10+ is required (3.12 recommended). Install it with:  brew install python@3.12"
  exit 1
fi
echo "Using $($PY --version)"

$PY -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name "$(basename "$PWD")" --display-name "Python ($(basename "$PWD"))"

# VS Code: use the .venv interpreter and add an F5 launch config for the Streamlit app
mkdir -p .vscode
cat > .vscode/settings.json <<'JSON'
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "python.terminal.activateEnvironment": true,
  "jupyter.notebookFileRoot": "${workspaceFolder}",
  "python.envFile": "${workspaceFolder}/.env"
}
JSON
cat > .vscode/launch.json <<'JSON'
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Run Streamlit app",
      "type": "debugpy",
      "request": "launch",
      "module": "streamlit",
      "args": ["run", "app.py"],
      "cwd": "${workspaceFolder}",
      "envFile": "${workspaceFolder}/.env"
    }
  ]
}
JSON

if [ ! -f .env ]; then cp .env.example .env; echo "Created .env - add your GROQ_API_KEY to it."; fi
echo ""
echo "✅ Setup complete.  Next:  source .venv/bin/activate && streamlit run app.py"
