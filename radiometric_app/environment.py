"""
Check of the computer: the Python that runs the program, Tkinter, the Python packages (with their versions) and the ExifTool
program, and what to install when something is missing. It uses only the standard library, so that it works before
any package is installed, and it is used by the launchers, by "python -m radiometric_app" (before the window opens), by the
"Check environment" button of the window and from the command line:

    python -m radiometric_app.environment [--root folder] [--config file] [--deep] [--json]

The exit code is 0 when everything the program needs is in place and 1 otherwise. The Python that the launchers use is
saved in config/python_environment.json (see save_python_environment).
"""
import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

MINIMUM_PYTHON = (3, 10)
CONFIG_FOLDER = 'config'                                   # in the project folder: the configuration, the profiles, the own presets
PYTHON_ENVIRONMENT_FILE = 'python_environment.json'
DEFAULT_CONFIG_NAME = 'radiometric_inputs.json'

# (module, names of the distributions that provide it, name for pip install, what it is for)
PACKAGES = (
    ('numpy', ('numpy',), 'numpy', 'the images and all the calculations'),
    ('cv2', ('opencv-python', 'opencv-python-headless', 'opencv-contrib-python', 'opencv-contrib-python-headless'),
     'opencv-python', 'finding the panel and processing the images'),
    ('PIL', ('pillow',), 'pillow', 'the pictures shown in the window'),
    ('tifffile', ('tifffile',), 'tifffile', 'reading and writing the TIFF images'),
    ('matplotlib', ('matplotlib',), 'matplotlib', 'the plots of the library'),
    ('skimage', ('scikit-image',), 'scikit-image', 'the regions of the panel'),
    ('shapely', ('shapely',), 'shapely', 'the polygons of the panel regions'),
    ('pyzbar', ('pyzbar',), 'pyzbar', 'reading the QR code of the panel'),
    ('exiftool', ('PyExifTool', 'pyexiftool'), 'PyExifTool', 'reading the metadata of the images through ExifTool'),
)

EXIFTOOL_INSTALL = ("Install ExifTool (https://exiftool.org). Windows: download the Windows executable, rename "
                    "exiftool(-k).exe to exiftool.exe and put it, with its exiftool_files folder, in a folder of the PATH, or "
                    "choose it in the window (Inputs, ExifTool). macOS: brew install exiftool. Linux: sudo apt install "
                    "libimage-exiftool-perl.")
TKINTER_INSTALL = ("This Python has no Tkinter (Tcl/Tk). Windows and macOS: install Python from python.org with 'tcl/tk and "
                   "IDLE' selected; conda: conda install tk; Linux: sudo apt install python3-tk.")
PYZBAR_INSTALL = ("pyzbar needs the zbar library. Windows: the package brings it, but the library needs the Microsoft "
                  "Visual C++ 2013 Redistributable (x64), which a new Windows often lacks (the error says that libzbar-64.dll "
                  "or one of its dependencies was not found): winget install Microsoft.VCRedist.2013.x64, or download "
                  "vcredist_x64.exe from Microsoft, and then check again. macOS: brew install zbar. Linux: sudo apt install "
                  "libzbar0.")
EXIFTOOL_PACKAGE_INSTALL = ("The module 'exiftool' must be the package PyExifTool (it has ExifToolHelper): pip uninstall "
                            "exiftool, then pip install PyExifTool.")


def _item(name, status, version=None, detail='', fix=None, level='required'):
    """
    One thing that was checked: status 'ok', 'missing', 'broken' (it is installed but cannot be used) or 'old'; level
    'required' (the window cannot open without it) or 'needed' (the window opens, but the correction cannot run)
    """
    return {'name': name, 'status': status, 'version': version, 'detail': detail, 'fix': fix, 'level': level}


def python_item():
    version = platform.python_version()
    if sys.version_info[:2] < MINIMUM_PYTHON:
        return _item('Python', 'old', version, f"{sys.executable}; Python {MINIMUM_PYTHON[0]}.{MINIMUM_PYTHON[1]} or newer is "
                     f"needed", f"Install Python {MINIMUM_PYTHON[0]}.{MINIMUM_PYTHON[1]} or newer and choose it in the launcher.")
    return _item('Python', 'ok', version, sys.executable)


def tkinter_item():
    try:
        import tkinter
    except Exception as error:                              # ImportError, or the Tcl/Tk libraries are not found
        return _item('Tkinter', 'missing', None, str(error), TKINTER_INSTALL)
    return _item('Tkinter', 'ok', str(tkinter.TkVersion), 'the window of the program')


def package_item(module, distributions, install, purpose, deep=False):
    """
    A Python package: found without importing it (and its version from the installation); with deep=True, or for exiftool
    (a package of another project has the same module name), the module is imported to see that it works
    """
    name = distributions[0]
    if importlib.util.find_spec(module) is None:
        return _item(name, 'missing', None, purpose, f"pip install {install}")
    version = None
    for distribution in distributions:
        try:
            version = importlib.metadata.version(distribution)
            name = distribution
            break
        except importlib.metadata.PackageNotFoundError:
            continue
    if deep or module == 'exiftool':
        try:
            imported = importlib.import_module(module)
        except Exception as error:                          # also native libraries that cannot be loaded
            fix = PYZBAR_INSTALL if module == 'pyzbar' else f"Reinstall it: pip install --force-reinstall {install}"
            return _item(name, 'broken', version, f"{type(error).__name__}: {error}", fix)
        version = version or getattr(imported, '__version__', None)     # a conda package may have no record of its version
        if module == 'exiftool' and not hasattr(imported, 'ExifToolHelper'):
            return _item(name, 'broken', version, "the module exiftool is not PyExifTool", EXIFTOOL_PACKAGE_INSTALL)
        if module == 'pyzbar':
            try:
                importlib.import_module('pyzbar.pyzbar')    # this one loads the zbar library
            except Exception as error:
                return _item(name, 'broken', version, f"{type(error).__name__}: {error}", PYZBAR_INSTALL)
    return _item(name, 'ok', version, purpose)


def locate_exiftool(configured='', root='.'):
    """
    The ExifTool that will be used and where it was looked for, in this order: the configured path (INPUTS.exiftool_path), the
    environment variable exiftoolpath, the system PATH. Returns (path, source); the path is None if it is not there
    """
    configured = str(configured or '').strip()
    if configured:
        path = Path(configured)
        path = path if path.is_absolute() else Path(root) / path
        return (str(path) if path.is_file() else None), 'INPUTS.exiftool_path'
    from_environment = os.environ.get('exiftoolpath', '').strip()
    if from_environment:
        return (from_environment if Path(from_environment).is_file() else shutil.which(from_environment)), \
            'the environment variable exiftoolpath'
    return shutil.which('exiftool'), 'the system PATH'


def exiftool_version(path, timeout=30):
    """ The version that 'exiftool -ver' prints, or None if the program cannot be run """
    try:
        done = subprocess.run([str(path), '-ver'], capture_output=True, text=True, timeout=timeout,
                              creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except (OSError, subprocess.SubprocessError):
        return None
    return (done.stdout.strip() or None) if done.returncode == 0 else None


def pyexiftool_installed():
    """ Whether the Python package pyexiftool (import exiftool) is installed """
    return importlib.util.find_spec('exiftool') is not None


def exiftool_item(configured='', root='.', version=True):
    path, source = locate_exiftool(configured, root)
    if path is None:
        return _item('ExifTool', 'missing', None, f"not found in {source}", EXIFTOOL_INSTALL, level='needed')
    found = exiftool_version(path) if version else None
    if version and not found:
        return _item('ExifTool', 'broken', None, f"{path} cannot be run", EXIFTOOL_INSTALL, level='needed')
    return _item('ExifTool', 'ok', found, f"{path} (found through {source})", level='needed')


def configured_exiftool_path(root='.', config=None):
    """ INPUTS.exiftool_path of the configuration file that the window would use, read as plain json ('' if there is none) """
    folder = Path(root)
    candidates = [Path(config)] if config else [folder / CONFIG_FOLDER / DEFAULT_CONFIG_NAME, folder / DEFAULT_CONFIG_NAME]
    for candidate in candidates:
        try:
            return str(json.loads(candidate.read_text(encoding='utf-8')).get('INPUTS.exiftool_path', '') or '')
        except (OSError, ValueError, AttributeError):
            continue
    return ''


def check_environment(exiftool_path='', root='.', deep=False, exiftool_version_too=True):
    """
    Check the computer. Inputs: the ExifTool path of the configuration (see configured_exiftool_path), the folder that
    relative paths start from, deep (import every package, which finds the ones that are installed but broken) and whether to
    run ExifTool to get its version. Outputs: a list of dictionaries {'name', 'status', 'version', 'detail', 'fix', 'level'}
    """
    items = [python_item(), tkinter_item()]
    items += [package_item(module, distributions, install, purpose, deep) for module, distributions, install, purpose in PACKAGES]
    items.append(exiftool_item(exiftool_path, root, exiftool_version_too))
    return items


def problems(items, level=None):
    """ The items that are not ok (only those of one level, 'required' or 'needed', if it is given) """
    return [item for item in items if item['status'] != 'ok' and (level is None or item['level'] == level)]


def report_lines(items):
    """ One line for each thing checked, to read in a log or in a terminal """
    width = max(len(item['name']) for item in items)
    lines = []
    for item in items:
        mark = 'ok' if item['status'] == 'ok' else item['status'].upper()
        text = f"{mark:<8} {item['name']:<{width}}  {item['version'] or '-':<12} {item['detail']}"
        lines.append(text.rstrip())
    return lines


def install_text(items, python=None):
    """
    What to install, as text for a message: the packages that are missing in one pip command (for the Python that was
    checked: python, default the one that is running) and the instructions for the rest. '' when there is nothing to do
    """
    lines = []
    missing = [item['fix'][len('pip install '):] for item in items
               if item['status'] == 'missing' and (item['fix'] or '').startswith('pip install ')]
    if missing:
        executable = str(normalized_python(python or sys.executable))     # python.exe: pythonw would write nowhere
        executable = f'"{executable}"' if ' ' in executable else executable
        lines.append("Use a dedicated environment for the packages, not the system Python: in the project folder, "
                     "conda env create -f environment.yml, or python -m venv .venv and then "
                     ".venv\\Scripts\\python.exe -m pip install -r requirements.txt; then choose its python.exe "
                     "(launchers\\select_python.cmd, or the Python environment section of the window).")
        lines.append(f"If the Python that was checked already is such an environment, the missing packages are installed "
                     f"with: {executable} -m pip install {' '.join(missing)}")
    for item in problems(items):
        if item['fix'] and not (item['status'] == 'missing' and item['fix'].startswith('pip install ')):
            lines.append(f"{item['name']}: {item['fix']}")
    return '\n'.join(lines)


def message_text(items):
    """ The problems and what to install, as the text of the message that is shown before the window opens ('' if none) """
    found = problems(items)
    if not found:
        return ''
    lines = [f"{item['name']}: {item['status']}" + (f" ({item['detail']})" if item['detail'] else '') for item in found]
    return '\n'.join(lines) + '\n\n' + install_text(items) + f"\n\nPython used: {sys.executable}"


def _tk_message(title, text, question):
    import tkinter
    from tkinter import messagebox
    root = tkinter.Tk()
    root.withdraw()
    try:
        root.attributes('-topmost', True)
        if question:
            return bool(messagebox.askyesno(title, text, parent=root, icon='warning'))
        messagebox.showerror(title, text, parent=root)
        return False
    finally:
        root.destroy()


def _windows_message(title, text, question):
    import ctypes
    flags = (0x04 | 0x30) if question else 0x10              # yes/no with a warning icon, or an error icon
    return ctypes.windll.user32.MessageBoxW(None, text, title, flags) == 6


def show_message(title, text, question=False):
    """
    Show a message and, with question=True, ask whether to go on (returns True if so). It uses Tkinter and, without it,
    the message box of Windows; without a window system the text goes to the error output (and the answer is False
    for a question)
    """
    for show in (_tk_message, _windows_message):
        if show is _windows_message and os.name != 'nt':
            continue
        try:
            return show(title, text, question)
        except Exception:
            continue
    print(f"{title}\n{text}", file=sys.stderr)
    return False


# The Python that the launchers use
def python_environment_path(root='.'):
    return Path(root) / CONFIG_FOLDER / PYTHON_ENVIRONMENT_FILE


def read_python_environment(root='.'):
    """ What config/python_environment.json says ({'python': path to python.exe}); {} if there is no usable file """
    try:
        data = json.loads(python_environment_path(root).read_text(encoding='utf-8-sig'))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save_python_environment(root, python):
    """ Save the python.exe that the launchers use (written whole or not at all); returns the file written """
    path = python_environment_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    try:
        temporary.write_text(json.dumps({'python': str(python)}, indent=2) + '\n', encoding='utf-8')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


# Another Python: how to start it, what it can do, and the choice of the one that the program uses
CONFIRMED_VARIABLE = 'RADIOMETRIC_PYTHON_CONFIRMED'          # set for the Python that was started after the confirmation


def normalized_python(executable):
    """ The python.exe of a pythonw.exe (it can print what it finds), as a Path """
    executable = Path(str(executable).strip().strip('"'))
    if executable.name.lower() == 'pythonw.exe':
        executable = executable.with_name('python.exe')
    return executable


def same_python(first, second):
    """ Whether two paths are the same Python (python.exe and pythonw.exe of a folder count as one) """
    def key(path):
        try:
            return os.path.normcase(str(normalized_python(path).resolve()))
        except OSError:
            return os.path.normcase(str(normalized_python(path)))
    return key(first) == key(second)


def process_environment(executable):
    """
    The environment variables to start another Python: its folder (and, for a conda environment, Library\\bin, where its
    DLLs are) first in the PATH, and without the PYTHONHOME and PYTHONPATH of this one, which would mix the two
    """
    environment = os.environ.copy()
    environment.pop('PYTHONHOME', None)
    environment.pop('PYTHONPATH', None)
    folder = Path(executable).parent
    paths = [folder]
    if (folder / 'Library' / 'bin').is_dir():
        paths.insert(0, folder / 'Library' / 'bin')
    environment['PATH'] = os.pathsep.join(map(str, paths)) + os.pathsep + environment.get('PATH', '')
    return environment


_PROBE = """
import json, sys
report = {'executable': sys.executable, 'version': sys.version.split()[0],
          'python_compatible': sys.version_info >= (3, 10), 'tkinter': {'ok': False, 'detail': ''}}
try:
    import tkinter
    root = tkinter.Tk()
    root.withdraw()
    root.update_idletasks()
    root.destroy()
    report['tkinter'] = {'ok': True, 'detail': 'Tk ' + str(tkinter.TkVersion)}
except Exception as error:
    report['tkinter'] = {'ok': False, 'detail': str(error)}
print(json.dumps(report))
"""


def inspect_python(executable, timeout=25):
    """
    Run another Python to see its version and whether it can open a window (a Tk root is created, not only imported)
    Outputs: {'executable', 'version', 'python_compatible' (3.10 or newer), 'tkinter': {'ok', 'detail'}}; FileNotFoundError
    if the file does not exist, RuntimeError if it does not run or does not answer
    """
    executable = normalized_python(executable)
    if not executable.is_file():
        raise FileNotFoundError(f"Python executable does not exist: {executable}")
    try:
        done = subprocess.run([str(executable), '-c', _PROBE], env=process_environment(executable), capture_output=True,
                              text=True, timeout=timeout, check=False,
                              creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"{executable} could not be run: {error}") from error
    if done.returncode:
        raise RuntimeError((done.stderr or done.stdout or "Python could not start").strip())
    try:
        report = json.loads(done.stdout.strip().splitlines()[-1])        # libraries may print a notice before the report
        report['tkinter']['ok']
    except (ValueError, IndexError, KeyError, TypeError) as error:
        raise RuntimeError("Could not read the answer of the Python: " + done.stdout.strip()) from error
    return report


def remember_python(root, executable):
    """
    Save a Python as the one to use, only after it was run and found able to open a window (Python 3.10 or newer, working
    Tkinter): a choice that works is never replaced by one that does not. Outputs: the report of inspect_python; ValueError
    that says what is wrong, or what inspect_python raises
    """
    report = inspect_python(executable)
    if not report['python_compatible']:
        raise ValueError(f"The program needs Python {MINIMUM_PYTHON[0]}.{MINIMUM_PYTHON[1]} or newer; this is "
                         f"Python {report['version']}.")
    if not report['tkinter']['ok']:
        raise ValueError("The program needs a Python with working Tkinter. " + report['tkinter']['detail'])
    save_python_environment(root, report['executable'])
    return report


def check_python_environment(executable, root='.', deep=True, timeout=180):
    """
    The check of this module (see check_environment) made by another Python: the list of its items. RuntimeError if that
    Python cannot run it (it is run from the folder of this program, with python -m radiometric_app check --json)
    """
    executable = normalized_python(executable)
    command = [str(executable), '-m', 'radiometric_app', 'check', '--json', '--root', str(root)] + (['--deep'] if deep else [])
    try:
        done = subprocess.run(command, cwd=str(Path(__file__).resolve().parent.parent), env=process_environment(executable),
                              capture_output=True, text=True, timeout=timeout, check=False,
                              creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"{executable} could not be run: {error}") from error
    try:
        return json.loads(done.stdout)                                  # the exit code is 1 when something is missing
    except ValueError as error:
        raise RuntimeError((done.stderr or done.stdout or "The check did not answer").strip()) from error


def relaunch(executable, command, argv):
    """
    Start this program again with another Python, without the console of python.exe and without asking again, and
    leave it running: python -m radiometric_app <command> <argv> from the current folder. Returns the process
    """
    executable = normalized_python(executable)
    windowed = executable.with_name('pythonw.exe')
    if os.name == 'nt' and windowed.is_file():
        executable = windowed
    environment = process_environment(executable)
    environment[CONFIRMED_VARIABLE] = '1'
    return subprocess.Popen([str(executable), '-m', 'radiometric_app', command, *argv], cwd=os.getcwd(), env=environment,
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check the Python, Tkinter, the packages and ExifTool that the program needs")
    parser.add_argument('--root', default='.', help="project folder (where config/ is)")
    parser.add_argument('--config', help="configuration file, to read the ExifTool path from (default: config/radiometric_inputs.json)")
    parser.add_argument('--deep', action='store_true', help="import every package to find the ones that are installed but broken")
    parser.add_argument('--json', action='store_true', help="print the result as json")
    parser.add_argument('--quiet', action='store_true', help="print nothing when everything is in place")
    arguments = parser.parse_args(argv)
    items = check_environment(configured_exiftool_path(arguments.root, arguments.config), arguments.root, arguments.deep)
    found = problems(items)
    if arguments.json:
        print(json.dumps(items, indent=2))
    elif found or not arguments.quiet:
        print('\n'.join(report_lines(items)))
        if found:
            print('\n' + install_text(items))
        else:
            print('\nEverything the program needs is in place.')
    return 1 if found else 0


if __name__ == '__main__':
    sys.exit(main())
