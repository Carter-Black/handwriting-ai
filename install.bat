@echo off
:: ─────────────────────────────────────────────────────────────────────────────
:: HandwritingAI — Windows installer
:: ─────────────────────────────────────────────────────────────────────────────
setlocal EnableDelayedExpansion

echo.
echo  🖊️  HandwritingAI Installer
echo  ────────────────────────────
echo.

:: ── Check Python ──────────────────────────────────────────────────────────────
where python >nul 2>&1
if errorlevel 1 (
    echo  ❌  Python is not installed or not in PATH.
    echo      Please install Python 3.10+ from https://python.org
    echo      Make sure to tick "Add Python to PATH" during install.
    pause
    exit /b 1
)

for /f "tokens=*" %%v in ('python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do set PY_VER=%%v
echo  ✅  Python %PY_VER% found.

:: ── Check for potrace ─────────────────────────────────────────────────────────
echo.
echo  Checking for potrace...
where potrace >nul 2>&1
if errorlevel 1 (
    echo  ⚠️   potrace not found in PATH.
    echo.
    echo  Please install potrace for Windows:
    echo    1. Download from: http://potrace.sourceforge.net/#downloading
    echo    2. Extract and copy potrace.exe to C:\Windows\System32\
    echo       OR add its folder to your PATH environment variable.
    echo    3. Re-run this installer.
    echo.
    echo  Press any key to open the download page...
    pause >nul
    start http://potrace.sourceforge.net/#downloading
    exit /b 1
)
echo  ✅  potrace found.

:: ── Create virtual environment ────────────────────────────────────────────────
echo.
echo  Creating Python virtual environment...
if not exist ".venv" (
    python -m venv .venv
)
call .venv\Scripts\activate.bat
echo  ✅  Virtual environment ready.

:: ── Install packages ──────────────────────────────────────────────────────────
echo.
echo  Installing Python packages (may take a few minutes)...
python -m pip install --upgrade pip --quiet
python -m pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo  ❌  Package installation failed. Check the error above.
    pause
    exit /b 1
)
echo  ✅  Python packages installed.

:: ── Download model ────────────────────────────────────────────────────────────
echo.
echo  Pre-downloading TrOCR model (~300 MB, one-time)...
python -c "from transformers import TrOCRProcessor, VisionEncoderDecoderModel; TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten'); VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')"
if errorlevel 1 (
    echo  ⚠️   Model download failed. Check your internet connection.
    echo      You can try again by re-running this installer.
) else (
    echo  ✅  TrOCR model cached.
)

:: ── Create run.bat ────────────────────────────────────────────────────────────
(
echo @echo off
echo cd /d "%%~dp0"
echo call .venv\Scripts\activate.bat
echo echo.
echo echo  🖊️  Starting HandwritingAI...
echo echo     Open your browser at: http://localhost:8000
echo echo     Press Ctrl+C to stop.
echo echo.
echo cd backend
echo python main.py
echo pause
) > run.bat

echo.
echo  ────────────────────────────────────
echo  ✅  Installation complete!
echo.
echo  To start the app, double-click:  run.bat
echo  Then open your browser at:       http://localhost:8000
echo.
pause
