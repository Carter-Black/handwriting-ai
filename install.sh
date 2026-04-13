#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# HandwritingAI — macOS / Linux installer
# ─────────────────────────────────────────────────────────────────────────────
set -e

PYTHON_MIN="3.10"
VENV_DIR=".venv"

echo ""
echo "🖊️  HandwritingAI Installer"
echo "────────────────────────────"
echo ""

# ── Check Python ──────────────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
  echo "❌  Python 3 is not installed."
  echo "    Please install Python $PYTHON_MIN+ from https://python.org and re-run this script."
  exit 1
fi

PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✅  Python $PY_VERSION found."

# ── Install potrace ───────────────────────────────────────────────────────────
echo ""
echo "Checking for potrace (bitmap → vector tracer)..."

if command -v potrace &>/dev/null; then
  echo "✅  potrace already installed."
else
  echo "Installing potrace..."
  if command -v brew &>/dev/null; then
    brew install potrace
  elif command -v apt-get &>/dev/null; then
    sudo apt-get update -qq && sudo apt-get install -y potrace
  elif command -v dnf &>/dev/null; then
    sudo dnf install -y potrace
  else
    echo "⚠️   Could not auto-install potrace. Please install it manually:"
    echo "    macOS:  brew install potrace"
    echo "    Ubuntu: sudo apt install potrace"
    echo "    Then re-run this script."
    exit 1
  fi
  echo "✅  potrace installed."
fi

# ── Create virtual environment ────────────────────────────────────────────────
echo ""
echo "Creating Python virtual environment..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
echo "✅  Virtual environment ready."

# ── Install Python packages ───────────────────────────────────────────────────
echo ""
echo "Installing Python packages (this may take a few minutes on first run)..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo "✅  Python packages installed."

# ── Download TrOCR model ──────────────────────────────────────────────────────
echo ""
echo "Pre-downloading TrOCR model (~300 MB, one-time download)..."
python3 -c "
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
print('  Downloading TrOCR processor...')
TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
print('  Downloading TrOCR model...')
VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
print('  Done.')
"
echo "✅  TrOCR model cached."

# ── Create run script ─────────────────────────────────────────────────────────
cat > run.sh <<'RUN'
#!/usr/bin/env bash
cd "$(dirname "$0")"
source .venv/bin/activate
echo ""
echo "🖊️  Starting HandwritingAI..."
echo "   Open your browser at: http://localhost:8000"
echo "   Press Ctrl+C to stop."
echo ""
cd backend
python main.py
RUN
chmod +x run.sh

echo ""
echo "────────────────────────────────────"
echo "✅  Installation complete!"
echo ""
echo "To start the app, run:"
echo "   ./run.sh"
echo ""
echo "Then open your browser at: http://localhost:8000"
echo ""
