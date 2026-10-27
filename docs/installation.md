# Installing and launching

[Back to the README](../README.md)

What the program needs, how to choose the Python it runs with, how the computer is checked and the launchers (Windows).

## What the program needs

1. **Python 3.10 or newer, with Tkinter** (the window). The Python of python.org and conda have it (python.org: keep "tcl/tk and IDLE" selected; Linux: ```sudo apt install python3-tk```). Windows 10 or 11 for the launchers; the program itself is plain Python.
2. **The Python packages** of ```requirements.txt```: numpy, opencv-python, pillow, tifffile, matplotlib, scikit-image, shapely, pyzbar and PyExifTool (the versions in the file are the ones it was tested with: Python 3.10.20 on Windows, numpy 2.2.6, opencv 4.12.0, pillow 12.0.0, tifffile 2025.5.10, matplotlib 3.10.8, scikit-image 0.25.2, shapely 2.1.2, pyzbar 0.1.9, PyExifTool 0.5.6). Install them in a **dedicated environment**, not in the system Python (the program never installs anything: when a package is missing it says which and how, and stops). In a terminal, in the project folder:
   - conda: ```conda env create -f environment.yml``` (the environment is called ```radiometric```; the file was checked by solving it with conda-forge, but the environment was not created and used from scratch on another computer);
   - or a virtual environment: ```python -m venv .venv```, then ```.venv\Scripts\python.exe -m pip install -r requirements.txt```;
   - then choose the ```python.exe``` of that environment (see [Choosing the Python](#choosing-the-python)). If the Python that was chosen already is a dedicated environment, ```python -m pip install -r requirements.txt``` with it is enough.
   - pyzbar reads the QR code of the panel; the Windows package brings the zbar library it needs (and asks for the Microsoft Visual C++ Redistributable of 2013), macOS needs ```brew install zbar``` and Linux ```sudo apt install libzbar0```.
3. **ExifTool**, the program (not a Python package): see [ExifTool](#exiftool). Without it the window opens but nothing can be read from the images.

## ExifTool

The metadata (bands, panel QR code, DLS irradiance) is read with ExifTool, a separate program, through the Python package ```PyExifTool``` (```pip install PyExifTool```; both are needed). On Windows, download the "Windows Executable" from [exiftool.org](https://exiftool.org/), rename ```exiftool(-k).exe``` to ```exiftool.exe``` and keep it next to its ```exiftool_files``` folder. The program is found in this order: the ```INPUTS.exiftool_path``` field of the configuration (the "ExifTool executable" field of the window), the environment variable ```exiftoolpath```, the system PATH (it is enough that ```exiftool -ver``` works in a terminal). The field is only needed when it is not found by itself. The window looks for ExifTool and pyexiftool when it opens and whenever the field changes, and says in the help column what it found (the program, how it was found, its version, whether the package is installed); ```Check inputs``` and the runner report them as a problem when they are missing, and ```radiometric_workflow.find_exiftool``` gives the same answer from Python.

## Choosing the Python

The program runs with the Python that you choose, which should be the dedicated environment above. The choice is kept in ```config/python_environment.json``` (plain json, ```{"python": "C:\\...\\python.exe"}```, not kept in git). There are three ways to make or change it:

- ```launchers/select_python.cmd``` (Windows): double-click it. A window asks for the ```python.exe``` (Browse...), checks that it is Python 3.10 or newer with Tkinter, saves it and then lists what else is missing. The other launchers use that file; the first time, with no file, they open the same window by themselves, and when the saved Python no longer exists they ask again.
- **The window asks every time it starts.** The dialog "Select Python Environment" (```radiometric_app/python_selection.py```) proposes the saved Python (or the one that is running) and you confirm it, paste another path or browse. "Confirm and open window" tries the Python first, in a subprocess (Python 3.10 or newer, and a Tk window can be created), saves it, and, if it is not the Python that is running, starts the window again with it (with its own folder and, for a conda environment, ```Library\bin``` first in the PATH, and without the ```PYTHONHOME``` and ```PYTHONPATH``` of the other one) and closes this one. A Python that does not work is explained and not saved, so the one that works is never replaced. Cancel closes everything. The packages and ExifTool of the Python that is confirmed are checked next (see [The check of the computer](#the-check-of-the-computer)). ```--skip-python-dialog```, or the environment variable ```RADIOMETRIC_PYTHON_CONFIRMED=1``` (which the window that was started again has), skip the dialog, for example to run ```python -m radiometric_app window``` from a script; the panel tool and the check never ask.
- **The "Python environment" section of the Inputs page** of the window (see [the window](window.md#page-1-inputs)): it checks a Python and saves it for the next launches.

![The dialog that confirms the Python environment before the window opens](screenshots/python_dialog.png)

## The check of the computer

The check (```radiometric_app/environment.py```, only the standard library, so it works before anything is installed) lists the Python and its path, Tkinter, each package with its version and ExifTool with its version and how it was found (the ```INPUTS.exiftool_path``` of the configuration, ```exiftoolpath```, the PATH), and what to install for each thing that is missing, in one ```pip install``` command for the packages. It is used in four places:

- Before the window or the panel tool opens (```python -m radiometric_app window```, which is what the launchers run): when something is missing a message box says what and what to install, and nothing is opened; when only ExifTool is missing the message asks whether to open the program anyway. A package that is installed but cannot be loaded (a damaged installation, a missing DLL) is named in the message instead of a traceback that nobody sees without a console.
- The button "Check environment" on the "Run and results" page lists the same in the log (with every package imported, so a broken one shows).
- ```launchers/check_environment.cmd``` or ```python -m radiometric_app check [--deep] [--json] [--quiet] [--root folder]```: the same list in a terminal, exit code 0 when all is in place and 1 when not (```--deep``` imports every package, ```--json``` gives the result for other programs).
- From Python: ```environment.check_environment()``` returns one dictionary per thing (```name```, ```status``` ok, missing, broken or old, ```version```, ```detail```, ```fix```, ```level``` required or needed), and ```environment.install_text()``` and ```environment.message_text()``` write the instructions.

## Launchers (Windows)

Double-click ```open_radiometric_window.cmd```, in the project folder, to open the window; the other files are in the ```launchers``` folder. They use the Python chosen as described in [Choosing the Python](#choosing-the-python) and the project folder as ```--root```, so the relative paths of the configuration (for example ```images/raw```) are resolved from there. Each one checks that the Python is 3.10 or newer with Tkinter; the packages and ExifTool are checked by the program before it opens (see [The check of the computer](#the-check-of-the-computer)).

- ```open_radiometric_window.cmd``` (in the project folder, next to ```radiometric_correction_tools.py```): the window, without a console; it is ```pythonw -m radiometric_app window --root .```.
- ```launchers/open_radiometric_window_debug.cmd```: the same window with the console kept open, to see messages and error traces while testing.
- ```launchers/open_panel_selector.cmd```: only the dialog to draw the panel regions, using ```config/radiometric_inputs.json``` in the project folder if it exists, otherwise the defaults. Extra arguments are passed on (for example ```--capture IMG_0005```).
- ```launchers/select_python.cmd```: choose the ```python.exe``` (window written in PowerShell, so that it works before any Python is chosen; ```select_python.ps1```) and check it.
- ```launchers/check_environment.cmd```: list the Python, Tkinter, the packages with their versions and ExifTool, and what to install.
- ```launchers/_python.cmd```: used by the others to find the Python; it is not run by itself.
