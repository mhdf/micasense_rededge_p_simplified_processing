@echo off
rem Lists the Python, Tkinter, the Python packages (with their versions) and ExifTool of this computer, and what to install
rem if something is missing. It uses the Python of config\python_environment.json (select_python.cmd chooses it).
cd /d "%~dp0.."
call "%~dp0_python.cmd"
if errorlevel 1 exit /b 1
"%PYTHON_EXE%" -m radiometric_app check --root . --deep
echo.
pause
