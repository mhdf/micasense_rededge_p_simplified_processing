@echo off
rem Opens the radiometric correction window keeping this console open, to see messages and errors
rem Edit PYTHON_DIR if the Python 3.10 environment with Tkinter, OpenCV, pyzbar and scikit-image is elsewhere.
set "PYTHON_DIR=C:\path\to\your\Python310"
if not exist "%PYTHON_DIR%\python.exe" (
    echo Python 3.10 executable was not found: "%PYTHON_DIR%\python.exe"
    echo Edit PYTHON_DIR in this launcher.
    pause
    exit /b 1
)
cd /d "%~dp0"
"%PYTHON_DIR%\python.exe" -X faulthandler "%~dp0radiometric_window.py" --root "%~dp0."
echo.
echo The window was closed (exit code %ERRORLEVEL%).
pause
