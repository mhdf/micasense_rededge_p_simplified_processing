@echo off
rem Finds the Python that the launchers use and sets PYTHON_EXE and PYTHONW_EXE (called by the other launchers, from the project folder).
rem It is the python.exe saved in config\python_environment.json (select_python.cmd writes it); if there is none, or it is no
rem longer there, the window to choose it opens. Returns 1 if there is no Python that can be used.
set "PYTHON_EXE="
for /f "usebackq delims=" %%P in (`powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0select_python.ps1" -Root "%CD%"`) do set "PYTHON_EXE=%%P"
if not defined PYTHON_EXE (
    echo No Python was chosen. Run launchers\select_python.cmd to choose the Python 3.10 or newer that has Tkinter and the packages.
    pause
    exit /b 1
)
"%PYTHON_EXE%" -c "import sys, tkinter; sys.exit(0 if sys.version_info >= (3, 10) else 1)" 2>nul
if errorlevel 1 (
    echo This Python cannot be used, it needs to be 3.10 or newer and to have Tkinter: "%PYTHON_EXE%"
    echo Run launchers\select_python.cmd to choose another one.
    pause
    exit /b 1
)
for %%I in ("%PYTHON_EXE%") do set "PYTHONW_EXE=%%~dpIpythonw.exe"
if not exist "%PYTHONW_EXE%" set "PYTHONW_EXE=%PYTHON_EXE%"
exit /b 0
