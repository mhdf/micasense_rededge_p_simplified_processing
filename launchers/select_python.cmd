@echo off
rem Chooses the Python that the launchers use (saved in config\python_environment.json) and then checks it: the version,
rem Tkinter, the packages and ExifTool, with what to install if something is missing.
cd /d "%~dp0.."
set "PYTHON_EXE="
for /f "usebackq delims=" %%P in (`powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0select_python.ps1" -Root "%CD%" -Dialog`) do set "PYTHON_EXE=%%P"
if not defined PYTHON_EXE (
    echo No Python was chosen.
    pause
    exit /b 1
)
echo Python: "%PYTHON_EXE%"
echo.
"%PYTHON_EXE%" -m radiometric_app check --root .
echo.
pause
