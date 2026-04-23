#!/usr/bin/env bash
# HandwritingAI installer — macOS and Linux
set -e

VENV_DIR=".venv"

echo ""
echo " HandwritingAI Installer"
echo " ------------------------"
echo ""

# python
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] Python 3 is not installed."
    echo "        Install from https://python.org and re-run this script."
    exit 1
fi

PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
MAJOR=$(python3 -c "import sys; print(sys.version_info.major)")
MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")

if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 10 ]); then
    echo "[ERROR] Python 3.10+ required. Found $PY_VERSION."
    echo "        Update Python at https://python.org and re-run."
    exit 1
fi
echo "[OK] Python $PY_VERSION found."

# potrace
echo ""
echo "Checking for potrace..."

if command -v potrace &>/dev/null; then
    echo "[OK] potrace already installed."
else
    echo "potrace not found — attempting auto-install..."
    INSTALLED=false

    if command -v brew &>/dev/null; then
        echo "Trying Homebrew..."
        brew install potrace && INSTALLED=true
    fi

    if [ "$INSTALLED" = false ] && command -v apt-get &>/dev/null; then
        echo "Trying apt..."
        sudo apt-get update -qq && sudo apt-get install -y potrace && INSTALLED=true
    fi

    if [ "$INSTALLED" = false ] && command -v dnf &>/dev/null; then
        echo "Trying dnf..."
        sudo dnf install -y potrace && INSTALLED=true
    fi

    if [ "$INSTALLED" = false ] && command -v pacman &>/dev/null; then
        echo "Trying pacman..."
        sudo pacman -S --noconfirm potrace && INSTALLED=true
    fi

    if [ "$INSTALLED" = false ]; then
        echo ""
        echo "[ERROR] Could not auto-install potrace."
        echo ""
        echo "  Install it manually, then re-run this script:"
        echo "    macOS:  brew install potrace"
        echo "    Ubuntu: sudo apt install potrace"
        echo "    Fedora: sudo dnf install potrace"
        exit 1
    fi

    echo "[OK] potrace installed."
fi

# virtual environment
echo ""
echo "Setting up Python environment..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
echo "[OK] Virtual environment ready."

# packages
echo ""
echo "Installing Python packages (this may take a few minutes)..."
pip install --upgrade pip --quiet
pip install -r requirements.txt
echo "[OK] Packages installed."

# download model
echo ""
echo "Downloading TrOCR model (~300 MB, one-time only)..."
python3 -c "
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
print('[OK] Model downloaded.')
" || echo "[WARN] Model download failed. Re-run this script when you have internet access."

# create run script
cat > run.sh <<'EOF'
#!/usr/bin/env bash
cd "$(dirname "$0")"
source .venv/bin/activate
echo ""
echo " HandwritingAI is starting..."
echo " Open your browser at: http://localhost:8000"
echo " Press Ctrl+C to stop."
echo ""
cd backend
python main.py
EOF
chmod +x run.sh

echo ""
echo " --------------------------------"
echo " [OK] Installation complete!"
echo ""
echo " Start the app:  ./run.sh"
echo " Then open:      http://localhost:8000"
echo ""
