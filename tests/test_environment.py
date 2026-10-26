"""
Tests for environment.py (the check of the computer) and for __main__.py (the check before the window opens)

Run from the project folder with:  python -m unittest tests.test_environment -v
"""

import contextlib
import importlib.machinery
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from radiometric_app import __main__ as launch
from radiometric_app import environment as env


def item(name, status='ok', level='required', fix=None, detail='', version='1.0'):
    return {'name': name, 'status': status, 'version': version, 'detail': detail, 'fix': fix, 'level': level}


class PythonAndPackagesTests(unittest.TestCase):

    def test_this_python_is_accepted_and_an_old_one_is_not(self):
        self.assertEqual(env.python_item()['status'], 'ok')
        with mock.patch.object(env.sys, 'version_info', (3, 9, 0, 'final', 0)):
            old = env.python_item()
        self.assertEqual(old['status'], 'old')
        self.assertIn('3.10 or newer', old['detail'])
        self.assertIsNotNone(old['fix'])

    def test_tkinter_is_reported_with_its_version(self):
        found = env.tkinter_item()
        self.assertIn(found['status'], ('ok', 'missing'))
        with mock.patch.dict(sys.modules, {'tkinter': None}):                      # the import fails
            self.assertEqual(env.tkinter_item()['status'], 'missing')
            self.assertIn('Tcl/Tk', env.tkinter_item()['fix'])

    def test_a_missing_package_says_how_to_install_it(self):
        found = env.package_item('no_such_module_xyz', ('no-such-package',), 'no-such-package', 'testing')
        self.assertEqual((found['name'], found['status'], found['version']), ('no-such-package', 'missing', None))
        self.assertEqual(found['fix'], 'pip install no-such-package')

    def test_an_installed_package_has_its_version(self):
        found = env.package_item('numpy', ('numpy',), 'numpy', 'the images')
        self.assertEqual(found['status'], 'ok')
        self.assertRegex(found['version'], r'^\d+\.\d+')
        self.assertEqual(env.package_item('numpy', ('numpy',), 'numpy', 'the images', deep=True)['status'], 'ok')

    def test_the_version_comes_from_the_module_when_the_installation_has_no_record_of_it(self):
        found = env.package_item('json', ('no-record-of-json',), 'json', 'testing', deep=True)       # json.__version__
        self.assertEqual((found['status'], found['version']), ('ok', json.__version__))

    def test_a_package_that_is_installed_but_cannot_be_loaded_is_broken(self):
        with mock.patch.object(env.importlib, 'import_module', side_effect=OSError('the DLL was not found')):
            found = env.package_item('numpy', ('numpy',), 'numpy', 'the images', deep=True)
        self.assertEqual(found['status'], 'broken')
        self.assertIn('OSError: the DLL was not found', found['detail'])
        self.assertIn('force-reinstall numpy', found['fix'])

    def test_pyzbar_that_cannot_load_zbar_says_what_to_install(self):
        def import_module(name):
            if name == 'pyzbar.pyzbar':
                raise ImportError('Unable to find zbar shared library')
            return types.ModuleType(name)
        with mock.patch.object(env.importlib, 'import_module', side_effect=import_module):
            found = env.package_item('pyzbar', ('pyzbar',), 'pyzbar', 'the QR code', deep=True)
        self.assertEqual(found['status'], 'broken')
        self.assertIn('zbar', found['fix'])

    def test_the_exiftool_module_of_another_package_is_recognised(self):
        other = types.ModuleType('exiftool')
        other.__spec__ = importlib.machinery.ModuleSpec('exiftool', None)
        with mock.patch.dict(sys.modules, {'exiftool': other}):
            found = env.package_item('exiftool', ('PyExifTool', 'pyexiftool'), 'PyExifTool', 'the metadata')
        self.assertEqual(found['status'], 'broken')
        self.assertIn('pip uninstall exiftool', found['fix'])

    def test_the_packages_are_the_ones_the_program_imports(self):
        modules = {module for module, _, _, _ in env.PACKAGES}
        self.assertEqual(modules, {'numpy', 'cv2', 'PIL', 'tifffile', 'matplotlib', 'skimage', 'shapely', 'pyzbar', 'exiftool'})
        root = Path(__file__).resolve().parent.parent
        listed = {line.split('>')[0].split('=')[0].strip().lower() for line in (root / 'requirements.txt').read_text(
            encoding='utf-8').splitlines() if line.strip() and not line.startswith('#')}
        self.assertEqual(listed, {install.lower() for _, _, install, _ in env.PACKAGES})
        conda = (root / 'environment.yml').read_text(encoding='utf-8')
        for name in ('numpy', 'opencv', 'pillow', 'tifffile', 'matplotlib', 'scikit-image', 'shapely', 'pyzbar', 'PyExifTool', 'tk'):
            self.assertIn(name, conda)


class ExifToolTests(unittest.TestCase):

    def test_the_configured_file_comes_first_and_relative_paths_start_at_the_root(self):
        with tempfile.TemporaryDirectory() as folder:
            program = Path(folder) / 'tools' / 'exiftool.exe'
            program.parent.mkdir()
            program.write_text('x')
            self.assertEqual(env.locate_exiftool(str(program)), (str(program), 'INPUTS.exiftool_path'))
            self.assertEqual(env.locate_exiftool('tools/exiftool.exe', folder), (str(Path(folder) / 'tools/exiftool.exe'),
                                                                                'INPUTS.exiftool_path'))
            self.assertEqual(env.locate_exiftool(str(program) + '.no'), (None, 'INPUTS.exiftool_path'))

    def test_then_the_environment_variable_and_then_the_path(self):
        with mock.patch.dict(env.os.environ, {'exiftoolpath': ''}), mock.patch.object(env.shutil, 'which', return_value=None):
            self.assertEqual(env.locate_exiftool(''), (None, 'the system PATH'))
        with mock.patch.dict(env.os.environ, {'exiftoolpath': 'exiftool'}), \
                mock.patch.object(env.shutil, 'which', return_value='C:/x/exiftool'):
            self.assertEqual(env.locate_exiftool(''), ('C:/x/exiftool', 'the environment variable exiftoolpath'))

    def test_the_item_for_the_program(self):
        with mock.patch.object(env, 'locate_exiftool', return_value=(None, 'the system PATH')):
            missing = env.exiftool_item()
        self.assertEqual((missing['status'], missing['level']), ('missing', 'needed'))
        self.assertIn('https://exiftool.org', missing['fix'])
        with mock.patch.object(env, 'locate_exiftool', return_value=('C:/x/exiftool.exe', 'the system PATH')):
            with mock.patch.object(env, 'exiftool_version', return_value='12.70'):
                found = env.exiftool_item()
            self.assertEqual((found['status'], found['version']), ('ok', '12.70'))
            with mock.patch.object(env, 'exiftool_version', return_value=None):
                self.assertEqual(env.exiftool_item()['status'], 'broken')
            with mock.patch.object(env, 'exiftool_version') as asked:
                self.assertEqual(env.exiftool_item(version=False)['status'], 'ok')       # the program is not run
                asked.assert_not_called()

    def test_the_configured_path_is_read_from_the_configuration_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.assertEqual(env.configured_exiftool_path(root), '')
            (root / env.CONFIG_FOLDER).mkdir()
            (root / env.CONFIG_FOLDER / env.DEFAULT_CONFIG_NAME).write_text(
                json.dumps({'INPUTS.exiftool_path': 'C:/tools/exiftool.exe'}), encoding='utf-8')
            self.assertEqual(env.configured_exiftool_path(root), 'C:/tools/exiftool.exe')
            other = root / 'other.json'
            other.write_text(json.dumps({'INPUTS.exiftool_path': 'D:/e.exe'}), encoding='utf-8')
            self.assertEqual(env.configured_exiftool_path(root, other), 'D:/e.exe')
            other.write_text('{not json', encoding='utf-8')
            self.assertEqual(env.configured_exiftool_path(root, other), '')

    def test_the_configuration_of_earlier_versions_is_read_too(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / env.DEFAULT_CONFIG_NAME).write_text(json.dumps({'INPUTS.exiftool_path': 'E:/x.exe'}), encoding='utf-8')
            self.assertEqual(env.configured_exiftool_path(folder), 'E:/x.exe')


class ReportTests(unittest.TestCase):

    def setUp(self):
        self.items = [item('Python'), item('numpy'),
                      item('pyzbar', 'missing', fix='pip install pyzbar', detail='the QR code', version=None),
                      item('shapely', 'missing', fix='pip install shapely', version=None),
                      item('ExifTool', 'missing', 'needed', fix='Install ExifTool (https://exiftool.org).', version=None)]

    def test_the_levels(self):
        self.assertEqual([found['name'] for found in env.problems(self.items)], ['pyzbar', 'shapely', 'ExifTool'])
        self.assertEqual([found['name'] for found in env.problems(self.items, 'required')], ['pyzbar', 'shapely'])
        self.assertEqual([found['name'] for found in env.problems(self.items, 'needed')], ['ExifTool'])
        self.assertEqual(env.problems(self.items[:2]), [])

    def test_one_line_for_each_thing(self):
        lines = env.report_lines(self.items)
        self.assertEqual(len(lines), 5)
        self.assertTrue(lines[0].startswith('ok       Python'))
        self.assertTrue(lines[2].startswith('MISSING  pyzbar'))
        self.assertTrue(all(line == line.rstrip() for line in lines))

    def test_the_missing_packages_are_in_one_install_command(self):
        text = env.install_text(self.items)
        self.assertIn('-m pip install pyzbar shapely', text)
        self.assertIn('requirements.txt', text)
        self.assertIn('environment.yml', text)
        self.assertIn('ExifTool: Install ExifTool', text)
        self.assertNotIn('pyzbar: pip install', text)                          # not repeated one by one
        self.assertEqual(env.install_text(self.items[:2]), '')

    def test_the_pip_command_uses_the_python_with_a_console(self):
        with mock.patch.object(env.sys, 'executable', r'C:\Program Files\Python310\pythonw.exe'):
            text = env.install_text(self.items)
        self.assertIn(r'"C:\Program Files\Python310\python.exe" -m pip install pyzbar shapely', text)

    def test_the_message_names_the_problems_and_the_python(self):
        text = env.message_text(self.items)
        self.assertIn('pyzbar: missing (the QR code)', text)
        self.assertIn('ExifTool: missing', text)
        self.assertIn(sys.executable, text)
        self.assertEqual(env.message_text(self.items[:2]), '')


class MessageTests(unittest.TestCase):

    def test_a_question_returns_the_answer_of_the_box(self):
        with mock.patch.object(env, '_tk_message', return_value=True) as shown:
            self.assertTrue(env.show_message('t', 'text', question=True))
        shown.assert_called_once_with('t', 'text', True)

    def test_without_tkinter_windows_uses_its_own_box_and_without_both_the_text_goes_to_the_error_output(self):
        with mock.patch.object(env, '_tk_message', side_effect=RuntimeError), \
                mock.patch.object(env.os, 'name', 'nt'), \
                mock.patch.object(env, '_windows_message', return_value=True) as windows:
            self.assertTrue(env.show_message('t', 'text', question=True))
        windows.assert_called_once()
        errors = io.StringIO()
        with mock.patch.object(env, '_tk_message', side_effect=RuntimeError), \
                mock.patch.object(env, '_windows_message', side_effect=RuntimeError), \
                contextlib.redirect_stderr(errors):
            self.assertFalse(env.show_message('title', 'the text', question=True))
        self.assertEqual(errors.getvalue(), 'title\nthe text\n')


class PythonEnvironmentFileTests(unittest.TestCase):

    def test_the_python_is_saved_in_the_config_folder_and_read_back(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(env.read_python_environment(folder), {})
            path = env.save_python_environment(folder, r'C:\envs\py310\python.exe')
            self.assertEqual(path, Path(folder) / 'config' / 'python_environment.json')
            self.assertEqual(env.read_python_environment(folder), {'python': r'C:\envs\py310\python.exe'})
            self.assertEqual(json.loads(path.read_text(encoding='utf-8')), {'python': r'C:\envs\py310\python.exe'})

    def test_the_file_written_by_powershell_has_a_byte_order_mark_and_is_read(self):
        with tempfile.TemporaryDirectory() as folder:
            path = env.python_environment_path(folder)
            path.parent.mkdir()
            path.write_bytes(b'\xef\xbb\xbf' + json.dumps({'python': 'D:/p/python.exe'}).encode('utf-8'))
            self.assertEqual(env.read_python_environment(folder), {'python': 'D:/p/python.exe'})

    def test_a_file_that_cannot_be_used_is_no_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = env.python_environment_path(folder)
            path.parent.mkdir()
            path.write_text('{not json', encoding='utf-8')
            self.assertEqual(env.read_python_environment(folder), {})
            path.write_text('["a list"]', encoding='utf-8')
            self.assertEqual(env.read_python_environment(folder), {})

    def test_the_launchers_know_the_file(self):
        launchers = Path(__file__).resolve().parent.parent / 'launchers'
        self.assertIn('config\\python_environment.json', (launchers / 'select_python.ps1').read_text(encoding='utf-8'))
        self.assertIn('select_python.ps1', (launchers / '_python.cmd').read_text(encoding='utf-8'))
        for name in ('open_radiometric_window.cmd', 'open_radiometric_window_debug.cmd', 'open_panel_selector.cmd',
                     'check_environment.cmd'):
            text = (launchers / name).read_bytes()
            self.assertIn(b'_python.cmd', text, name)
            self.assertIn(b'\r\n', text, name)                                 # cmd files keep their Windows line endings


class CommandLineTests(unittest.TestCase):

    def run_main(self, items, *arguments):
        output = io.StringIO()
        with mock.patch.object(env, 'check_environment', return_value=items) as check, contextlib.redirect_stdout(output):
            code = env.main(list(arguments))
        return code, output.getvalue(), check

    def test_everything_in_place_returns_0(self):
        code, text, _ = self.run_main([item('Python'), item('numpy')])
        self.assertEqual(code, 0)
        self.assertIn('Everything the program needs is in place.', text)
        self.assertEqual(self.run_main([item('Python')], '--quiet')[:2], (0, ''))

    def test_a_problem_returns_1_and_says_what_to_install(self):
        code, text, _ = self.run_main([item('Python'), item('shapely', 'missing', fix='pip install shapely')], '--quiet')
        self.assertEqual(code, 1)
        self.assertIn('MISSING  shapely', text)
        self.assertIn('pip install shapely', text)

    def test_deep_and_json(self):
        code, text, check = self.run_main([item('Python')], '--deep', '--json', '--root', 'somewhere')
        self.assertEqual(json.loads(text)[0]['name'], 'Python')
        self.assertTrue(check.call_args.args[2])                              # deep
        self.assertEqual(check.call_args.args[1], 'somewhere')

    def test_this_computer_passes_when_exiftool_is_there(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = env.main(['--root', '.'])
        self.assertIn(code, (0, 1))
        self.assertIn('Python', output.getvalue())
        self.assertIn('Tkinter', output.getvalue())


class LaunchTests(unittest.TestCase):
    """ python -m radiometric_app: the check, the message and what is opened """

    def run_launch(self, argv, items, answer=False):
        window = mock.Mock(main=mock.Mock(return_value=0))
        with mock.patch.object(env, 'check_environment', return_value=items), \
                mock.patch.object(env, 'show_message', return_value=answer) as shown, \
                mock.patch.object(launch.importlib, 'import_module', return_value=window) as opened:
            code = launch.main(argv)
        return code, shown, opened, window

    def test_what_is_missing_is_shown_and_nothing_is_opened(self):
        code, shown, opened, _ = self.run_launch(['window', '--root', '.'], [item('Python'), item('shapely', 'missing')])
        self.assertEqual(code, 1)
        shown.assert_called_once()
        self.assertIn('shapely', shown.call_args.args[1])
        self.assertFalse(shown.call_args.kwargs.get('question'))
        opened.assert_not_called()

    def test_without_exiftool_the_user_decides(self):
        items = [item('Python'), item('ExifTool', 'missing', 'needed', fix='Install ExifTool')]
        code, shown, opened, _ = self.run_launch([], items, answer=False)
        self.assertEqual(code, 1)
        self.assertTrue(shown.call_args.kwargs['question'])
        opened.assert_not_called()
        code, _, opened, window = self.run_launch(['--root', 'x'], items, answer=True)
        self.assertEqual(code, 0)
        opened.assert_called_once_with('radiometric_app.radiometric_window')
        window.main.assert_called_once_with(['--root', 'x'])

    def test_when_all_is_in_place_the_window_opens_without_a_message(self):
        code, shown, opened, window = self.run_launch(['--config', 'c.json'], [item('Python')])
        self.assertEqual(code, 0)
        shown.assert_not_called()
        opened.assert_called_once_with('radiometric_app.radiometric_window')
        window.main.assert_called_once_with(['--config', 'c.json'])

    def test_the_panel_tool_has_its_own_command(self):
        _, _, opened, window = self.run_launch(['panels', '--root', '.', '--capture', 'IMG_0005'], [item('Python')])
        opened.assert_called_once_with('radiometric_app.panel_selector')
        window.main.assert_called_once_with(['--root', '.', '--capture', 'IMG_0005'])

    def test_check_is_the_command_line_check(self):
        with mock.patch.object(env, 'main', return_value=1) as check:
            self.assertEqual(launch.main(['check', '--deep']), 1)
        check.assert_called_once_with(['--deep'])

    def test_a_package_that_cannot_be_loaded_is_explained_in_the_message(self):
        broken = [item('Python'), item('opencv-python', 'broken', detail='OSError: the DLL was not found')]
        with mock.patch.object(env, 'check_environment', side_effect=[[item('Python')], broken]), \
                mock.patch.object(env, 'show_message') as shown, \
                mock.patch.object(launch.importlib, 'import_module', side_effect=ImportError('DLL load failed')):
            self.assertEqual(launch.main([]), 1)
        self.assertIn('opencv-python: broken (OSError: the DLL was not found)', shown.call_args.args[1])

    def test_the_exiftool_path_of_the_configuration_is_used_and_the_program_is_not_run(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / 'config').mkdir()
            (Path(folder) / 'config' / 'radiometric_inputs.json').write_text(
                json.dumps({'INPUTS.exiftool_path': 'C:/tools/exiftool.exe'}), encoding='utf-8')
            with mock.patch.object(env, 'check_environment', return_value=[item('Python')]) as check, \
                    mock.patch.object(launch.importlib, 'import_module', return_value=mock.Mock(main=lambda argv: 0)):
                launch.main(['--root', folder])
        self.assertEqual(check.call_args.args[0], 'C:/tools/exiftool.exe')
        self.assertIs(check.call_args.kwargs['exiftool_version_too'], False)


if __name__ == '__main__':
    unittest.main()
