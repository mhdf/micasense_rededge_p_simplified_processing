"""
python -m radiometric_app [window | panels | check] [options]

Opens the window (default), the panel regions tool or the check of the computer. Before the window or the tool opens, the
computer is checked (see environment.py) and a message tells what is missing and what to install; the packages are not
imported until the check is done, so that a missing one is reported instead of ending in a traceback that nobody sees when
the program is started without a console (pythonw). The options after the command are those of the window
(--config, --root) or of the tool.
"""
import argparse
import importlib
import sys
import traceback

from . import environment

COMMANDS = {'window': 'radiometric_app.radiometric_window', 'panels': 'radiometric_app.panel_selector'}


def _location(argv):
    """ --root and --config of the options, without failing on the others """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--root', default='.')
    parser.add_argument('--config')
    arguments, _ = parser.parse_known_args(argv)
    return arguments.root, arguments.config


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    command = argv.pop(0) if argv and argv[0] in (*COMMANDS, 'check') else 'window'
    if command == 'check':
        return environment.main(argv)
    root, config = _location(argv)
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
