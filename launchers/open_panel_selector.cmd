@echo off
rem Draws the reference panel regions band by band (uses config\radiometric_inputs.json if it exists, else the defaults), keeping this console open
rem Edit PYTHON_DIR if the Python 3.10 environment with Tkinter, OpenCV, pyzbar and scikit-image is elsewhere.
set "PYTHON_DIR=C:\path\to\your\Python310"
if not exist "%PYTHON_DIR%\python.exe" (
    echo Python 3.10 executable was not found: "%PYTHON_DIR%\python.exe"
    echo Edit PYTHON_DIR in this launcher.
    pause
    exit /b 1
)
cd /d "%~dp0.."
"%PYTHON_DIR%\python.exe" -X faulthandler -m radiometric_app.panel_selector --root . %*
echo.
echo Exit code %ERRORLEVEL%.
pause
