@echo off
rem Opens the radiometric correction window (no console; use the _debug launcher to see errors)
rem Edit PYTHON_DIR if the Python 3.10 environment with Tkinter, OpenCV, pyzbar and scikit-image is elsewhere.
set "PYTHON_DIR=C:\path\to\your\Python310"
if not exist "%PYTHON_DIR%\pythonw.exe" (
    echo Python 3.10 window executable was not found: "%PYTHON_DIR%\pythonw.exe"
    echo Edit PYTHON_DIR in this launcher.
    pause
    exit /b 1
)
start "Radiometric correction" "%PYTHON_DIR%\pythonw.exe" "%~dp0radiometric_window.py" --root "%~dp0."
