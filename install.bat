@echo off
chcp 65001 >nul 2>&1
setlocal EnableDelayedExpansion

echo.
echo  HandwritingAI Installer
echo  ------------------------
echo.

:: ── Python ────────────────────────────────────────────────────────────────────
:: Try the Python Launcher first (py.exe), then plain python
:: We also verify it actually runs — avoids the Windows Store stub issue

set PY_CMD=
for %%c in (py python python3) do (
    if "!PY_CMD!"=="" (
        %%c --version >nul 2>&1
        if !errorlevel! == 0 (
            :: Make sure it's real Python, not the Store stub
            %%c -c "import sys; assert sys.version_info >= (3,10)" >nul 2>&1
            if !errorlevel! == 0 (
                set PY_CMD=%%c
            )
        )
    )
)

if "!PY_CMD!"=="" (
    echo [ERROR] Python 3.10 or higher was not found.
    echo.
    echo   Please install Python from https://python.org
    echo   Make sure to check "Add Python to PATH" during setup.
    echo   Then re-run this installer.
    echo.
    pause
    exit /b 1
)

for /f "tokens=*" %%v in ('!PY_CMD! -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do set PY_VER=%%v
echo [OK] Python %PY_VER% found (using: %PY_CMD%)

:: ── potrace ───────────────────────────────────────────────────────────────────
echo.
echo Checking for potrace...

where potrace >nul 2>&1
if !errorlevel! == 0 (
    echo [OK] potrace found.
    goto :potrace_done
)

echo potrace not found — attempting automatic install...
echo.

:: try winget (built into Windows 10 1709+ and Windows 11)
winget --version >nul 2>&1
if !errorlevel! == 0 (
    echo Trying winget...
    winget install --id=potrace.potrace -e --silent --accept-package-agreements --accept-source-agreements
    if !errorlevel! == 0 (
        :: refresh PATH so potrace is visible in this session
        for /f "tokens=*" %%p in ('where potrace 2^>nul') do set POTRACE_PATH=%%p
        if not "!POTRACE_PATH!"=="" (
            echo [OK] potrace installed via winget.
            goto :potrace_done
        )
    )
    echo winget install did not succeed, trying next option...
)

:: try chocolatey
choco --version >nul 2>&1
if !errorlevel! == 0 (
    echo Trying chocolatey...
    choco install potrace -y
    where potrace >nul 2>&1
    if !errorlevel! == 0 (
        echo [OK] potrace installed via chocolatey.
        goto :potrace_done
    )
    echo chocolatey install did not succeed, trying next option...
)

:: try scoop
scoop --version >nul 2>&1
if !errorlevel! == 0 (
    echo Trying scoop...
    scoop install potrace
    where potrace >nul 2>&1
    if !errorlevel! == 0 (
        echo [OK] potrace installed via scoop.
        goto :potrace_done
    )
)

:: all auto-install attempts failed — give clear manual instructions
echo.
echo [WARN] Could not auto-install potrace.
echo.
echo  Please install it manually:
echo    1. Go to: http://potrace.sourceforge.net/#downloading
echo    2. Download the Windows binary (potrace-X.X.win64.zip)
echo    3. Unzip it and copy potrace.exe to C:\Windows\System32\
echo    4. Re-run this installer.
echo.
echo  Press any key to open the download page in your browser...
pause >nul
start http://potrace.sourceforge.net/#downloading
exit /b 1

:potrace_done

:: ── Virtual environment ───────────────────────────────────────────────────────
echo.
echo Setting up Python environment...

if not exist ".venv" (
    !PY_CMD! -m venv .venv
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
)

call .venv\Scripts\activate.bat
echo [OK] Virtual environment ready.

:: ── Python packages ───────────────────────────────────────────────────────────
echo.
echo Installing Python packages (this may take a few minutes)...

python -m pip install --upgrade pip --quiet
python -m pip install -r requirements.txt
if !errorlevel! neq 0 (
    echo [ERROR] Package installation failed. See errors above.
    pause
    exit /b 1
)
echo [OK] Packages installed.

:: ── Download TrOCR model ──────────────────────────────────────────────────────
echo.
echo Downloading TrOCR model (~300 MB, one-time only)...

python -c "from transformers import TrOCRProcessor, VisionEncoderDecoderModel; TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten'); VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten'); print('[OK] Model downloaded.')"
if !errorlevel! neq 0 (
    echo [WARN] Model download failed. Check your internet connection.
    echo        You can retry by running install.bat again.
)

:: ── Create run.bat ────────────────────────────────────────────────────────────
(
echo @echo off
echo cd /d "%%~dp0"
echo call .venv\Scripts\activate.bat
echo echo.
echo echo  HandwritingAI is starting...
echo echo  Open your browser at: http://localhost:8000
echo echo  Press Ctrl+C to stop.
echo echo.
echo cd backend
echo python main.py
echo pause
) > run.bat

echo.
echo  --------------------------------
echo  [OK] Installation complete!
echo.
echo  To start the app: double-click run.bat
echo  Then open:        http://localhost:8000
echo.
pause
