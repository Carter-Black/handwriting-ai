@echo off
set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"
set "PYTHON=%ROOT%\.venv\Scripts\python.exe"

if exist "%ROOT%\tools\potrace\potrace.exe" set "PATH=%ROOT%\tools\potrace;%PATH%"

if not exist "%PYTHON%" (
    echo [ERROR] Virtual environment not found at %PYTHON%
    echo         Please re-run install.bat first.
    pause
    exit /b 1
)

echo.
echo  HandwritingAI is starting...
echo  Open your browser at: http://localhost:8000
echo  Press Ctrl+C to stop.
echo.

"%PYTHON%" "%ROOT%\backend\main.py"
pause
