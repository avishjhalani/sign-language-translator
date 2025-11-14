@echo off
echo Checking Python version...
python --version
echo.
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Python version: %PYTHON_VERSION%
echo.
if "%PYTHON_VERSION:~0,4%"=="3.13" (
    echo ERROR: Python 3.13 is not supported by MediaPipe!
    echo.
    echo Please install Python 3.11 from:
    echo https://www.python.org/downloads/release/python-3110/
    echo.
    echo Then create a virtual environment:
    echo python3.11 -m venv venv
    echo venv\Scripts\activate
    echo pip install -r requirements.txt
) else if "%PYTHON_VERSION:~0,4%"=="3.12" (
    echo WARNING: Python 3.12 may have compatibility issues!
    echo MediaPipe officially supports Python 3.8-3.11
    echo.
    echo Try installing with: pip install -r requirements.txt
) else if "%PYTHON_VERSION:~0,4%"=="3.11" (
    echo ✓ Python 3.11 is compatible!
    echo.
    echo You can now install requirements:
    echo pip install -r requirements.txt
) else if "%PYTHON_VERSION:~0,4%"=="3.10" (
    echo ✓ Python 3.10 is compatible!
    echo.
    echo You can now install requirements:
    echo pip install -r requirements.txt
) else if "%PYTHON_VERSION:~0,4%"=="3.9" (
    echo ✓ Python 3.9 is compatible!
    echo.
    echo You can now install requirements:
    echo pip install -r requirements.txt
) else if "%PYTHON_VERSION:~0,4%"=="3.8" (
    echo ✓ Python 3.8 is compatible!
    echo.
    echo You can now install requirements:
    echo pip install -r requirements.txt
) else (
    echo Unknown Python version. Please use Python 3.8-3.11
)
echo.
pause

