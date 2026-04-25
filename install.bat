@echo off
setlocal EnableDelayedExpansion

:: if a prior install dropped potrace into tools\potrace\, pick it up
if exist "%~dp0tools\potrace\potrace.exe" set "PATH=%~dp0tools\potrace;%PATH%"

echo.
echo  HandwritingAI Installer
echo  ------------------------
echo.

:: Python
:: try the Python Launcher first (py.exe), then plain python
:: also verify it actually runs to avoid the Windows Store stub issue

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

:: potrace
echo.
echo Checking for potrace...

where potrace >nul 2>&1
if !errorlevel! == 0 (
    echo [OK] potrace found.
    goto :potrace_done
)

echo potrace not found. Attempting automatic install...
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

:: try direct download from sourceforge (no admin required, drops into tools\potrace)
echo Trying direct download from sourceforge...

set "POTRACE_VERSION=1.16"
set "POTRACE_URL=https://sourceforge.net/projects/potrace/files/%POTRACE_VERSION%/potrace-%POTRACE_VERSION%.win64.zip/download"
set "POTRACE_ZIP=%TEMP%\potrace-%POTRACE_VERSION%.win64.zip"
set "POTRACE_EXTRACT=%TEMP%\potrace-extract"
set "POTRACE_LOCAL=%~dp0tools\potrace"

if not exist "%~dp0tools" mkdir "%~dp0tools" >nul 2>&1
if not exist "%POTRACE_LOCAL%" mkdir "%POTRACE_LOCAL%" >nul 2>&1
if exist "%POTRACE_EXTRACT%" rmdir /s /q "%POTRACE_EXTRACT%" >nul 2>&1

powershell -NoProfile -ExecutionPolicy Bypass -Command "try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; $ProgressPreference = 'SilentlyContinue'; Invoke-WebRequest -Uri '%POTRACE_URL%' -OutFile '%POTRACE_ZIP%' -UseBasicParsing -ErrorAction Stop } catch { exit 1 }"
if !errorlevel! == 0 (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "try { Expand-Archive -LiteralPath '%POTRACE_ZIP%' -DestinationPath '%POTRACE_EXTRACT%' -Force -ErrorAction Stop; $exe = Get-ChildItem -Path '%POTRACE_EXTRACT%' -Recurse -Filter potrace.exe -ErrorAction Stop | Select-Object -First 1; if (-not $exe) { exit 1 }; Copy-Item -LiteralPath $exe.FullName -Destination '%POTRACE_LOCAL%\potrace.exe' -Force -ErrorAction Stop } catch { exit 1 }"
    if !errorlevel! == 0 (
        if exist "%POTRACE_LOCAL%\potrace.exe" (
            set "PATH=%POTRACE_LOCAL%;%PATH%"
            echo [OK] potrace downloaded to tools\potrace\.
            if exist "%POTRACE_ZIP%" del /q "%POTRACE_ZIP%" >nul 2>&1
            if exist "%POTRACE_EXTRACT%" rmdir /s /q "%POTRACE_EXTRACT%" >nul 2>&1
            goto :potrace_done
        )
    )
)
echo Direct download did not succeed.
if exist "%POTRACE_ZIP%" del /q "%POTRACE_ZIP%" >nul 2>&1
if exist "%POTRACE_EXTRACT%" rmdir /s /q "%POTRACE_EXTRACT%" >nul 2>&1

:: all auto-install attempts failed; give clear manual instructions
echo.
echo [WARN] Could not auto-install potrace.
echo.
echo  Please install it manually (no admin required):
echo    1. Go to: http://potrace.sourceforge.net/#downloading
echo    2. Download the Windows binary (potrace-X.X.win64.zip)
echo    3. Unzip it and copy potrace.exe into:
echo         %~dp0tools\potrace\
echo       (create that folder if it doesn't exist)
echo    4. Re-run this installer.
echo.
echo  Press any key to open the download page in your browser...
pause >nul
start http://potrace.sourceforge.net/#downloading
exit /b 1

:potrace_done

:: virtual environment
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

:: python packages
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

:: download TrOCR model
echo.
echo Downloading TrOCR model (~300 MB, one-time only)...

python -c "from transformers import TrOCRProcessor, VisionEncoderDecoderModel; TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten'); VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten'); print('[OK] Model downloaded.')"
if !errorlevel! neq 0 (
    echo [WARN] Model download failed. Check your internet connection.
    echo        You can retry by running install.bat again.
)

:: create run.bat
:: use the venv's python.exe directly via absolute path; avoids the Windows
:: Store python stub and the "cd backend" working-directory guessing game.
(
echo @echo off
echo set "ROOT=%%~dp0"
echo set "ROOT=%%ROOT:~0,-1%%"
echo set "PYTHON=%%ROOT%%\.venv\Scripts\python.exe"
echo if exist "%%ROOT%%\tools\potrace\potrace.exe" set "PATH=%%ROOT%%\tools\potrace;%%PATH%%"
echo if not exist "%%PYTHON%%" ^(
echo     echo [ERROR] Virtual environment not found. Please re-run install.bat.
echo     pause
echo     exit /b 1
echo ^)
echo echo.
echo echo  HandwritingAI is starting...
echo echo  Open your browser at: http://localhost:8000
echo echo  Press Ctrl+C to stop.
echo echo.
echo "%%PYTHON%%" "%%ROOT%%\backend\main.py"
echo pause
) > run.bat

echo.
echo  --------------------------------
echo  [OK] Installation complete!
echo.

set "RUN_NOW="
set /p RUN_NOW="Start HandwritingAI now? [Y/n]: "
if /i "%RUN_NOW%"=="" set "RUN_NOW=Y"

if /i "%RUN_NOW%"=="Y" (
    echo.
    call "%~dp0run.bat"
) else (
    echo.
    echo  Run later by double-clicking run.bat
    echo.
    pause
)
