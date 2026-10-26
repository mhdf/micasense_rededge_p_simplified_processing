@echo off
rem Opens the radiometric correction window (no console; use the _debug launcher to see errors).
rem The Python is the one saved in config\python_environment.json: select_python.cmd chooses it, and this launcher asks for it
rem the first time. Before the window opens the computer is checked, and a message says what to install if something is missing.
rem The project folder (the one above this launchers folder) is where the program runs and where the relative paths start
cd /d "%~dp0.."
call "%~dp0_python.cmd"
if errorlevel 1 exit /b 1
start "Radiometric correction" "%PYTHONW_EXE%" -m radiometric_app window --root .
