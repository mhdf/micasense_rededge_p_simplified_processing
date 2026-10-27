@echo off
rem Draws the reference panel regions band by band (uses config\radiometric_inputs.json if it exists, else the defaults), keeping this console open
rem (the Python is the one of config\python_environment.json, see select_python.cmd)
cd /d "%~dp0.."
call "%~dp0_python.cmd"
if errorlevel 1 exit /b 1
"%PYTHON_EXE%" -X faulthandler -m radiometric_app panels --root . %*
echo.
echo Exit code %ERRORLEVEL%.
pause
