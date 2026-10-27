"""
python -m radiometric_app [window | panels | check] [options]

Opens the window (default), the panel regions tool or the check of the computer. The window starts by asking which
Python to use (see python_selection.py): a different one than the one that is running starts the window again with it.
Before the window or the tool opens, the computer is checked (see environment.py) and a message tells what is missing
and what to install; the packages are not imported until the check is done, so that a missing one is reported instead
of ending in a traceback that nobody sees when the program is started without a console (pythonw). The options after
the command are those of the window (--config, --root) or of the tool, and --skip-python-dialog, which does not ask
for the Python (as does the environment variable RADIOMETRIC_PYTHON_CONFIRMED=1, which is what the window that was
started again by the dialog has).
"""
import argparse
import importlib
import os
import sys
import traceback

from . import environment

COMMANDS = {'window': 'radiometric_app.radiometric_window', 'panels': 'radiometric_app.panel_selector'}
SKIP_OPTION = '--skip-python-dialog'


def _location(argv):
    """ --root and --config of the options, without failing on the others """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--root', default='.')
    parser.add_argument('--config')
    arguments, _ = parser.parse_known_args(argv)
    return arguments.root, arguments.config


def _confirm_python(root, command, argv):
    """
    Ask which Python to use (the one saved, or else the one that is running). Returns None to go on in this process, or the
    exit code when there is nothing more to do here: 0 if the dialog was cancelled or if the window was started again
    with the Python that was chosen
    """
    try:
        import tkinter as tk
        from . import python_selection
    except Exception:                      # without Tkinter there is no dialog; the check below says what is wrong
        return None
    parent = tk.Tk()
    parent.withdraw()
    try:
        saved = environment.read_python_environment(root).get('python') or ''
        proposed = saved if saved and os.path.isfile(saved) else sys.executable
        chosen = python_selection.choose_python_environment(parent, proposed, root)
    finally:
        parent.destroy()
    if chosen is None:
        return 0
    if environment.same_python(chosen, sys.executable):
        return None
    environment.relaunch(chosen, command, argv)
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    command = argv.pop(0) if argv and argv[0] in (*COMMANDS, 'check') else 'window'
    if command == 'check':
        return environment.main(argv)
    skip = SKIP_OPTION in argv
    argv = [argument for argument in argv if argument != SKIP_OPTION]
    confirmed = os.environ.pop(environment.CONFIRMED_VARIABLE, None) == '1'
    root, config = _location(argv)
    if command == 'window' and not skip and not confirmed:
        outcome = _confirm_python(root, command, argv)
        if outcome is not None:
            return outcome
    items = environment.check_environment(environment.configured_exiftool_path(root, config), root, exiftool_version_too=False)
    title = "Radiometric correction: the computer is not ready"
    if environment.problems(items, 'required'):
        environment.show_message(title, environment.message_text(items))
        return 1
    if environment.problems(items, 'needed'):
        text = environment.message_text(items) + "\n\nOpen the program anyway (it cannot correct images without ExifTool)?"
        if not environment.show_message(title, text, question=True):
            return 1
    try:
        module = importlib.import_module(COMMANDS[command])
    except Exception as error:                      # a package that is installed but cannot be loaded
        deep = environment.check_environment(environment.configured_exiftool_path(root, config), root, deep=True,
                                             exiftool_version_too=False)
        text = (environment.message_text(deep) or
                f"The program could not be loaded: {type(error).__name__}: {error}\n\n{traceback.format_exc(limit=3)}")
        environment.show_message(title, text)
        return 1
    return module.main(argv)


if __name__ == '__main__':
    sys.exit(main())
