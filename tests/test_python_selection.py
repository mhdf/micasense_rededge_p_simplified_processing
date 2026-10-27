"""
Tests for python_selection.py: the dialog that confirms the Python before the window opens (with a hidden Tk root; skipped
if Tk cannot open one)

Run from the project folder with:  python -m unittest tests.test_python_selection -v
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

from radiometric_app import python_selection as ps


class DialogTests(unittest.TestCase):

    def setUp(self):
        import tkinter as tk
        try:
            self.root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"no display for Tk: {error}")
        self.root.withdraw()
        self.remembered = []

    def tearDown(self):
        try:
            self.root.destroy()
        except Exception:
            pass

    def open(self, remember=None, executable=r'C:\envs\a\python.exe'):
        def accept(root, path):
            self.remembered.append((root, path))
            return {'executable': path, 'version': '3.10.1'}
        dialog = ps.PythonSelectionDialog(self.root, executable, 'project', remember or accept)
        self.root.update()
        return dialog

    def test_the_dialog_shows_the_python_to_confirm_with_the_texts_of_the_other_workflow(self):
        dialog = self.open()
        self.assertEqual(dialog.window.title(), 'Select Python Environment')
        self.assertEqual(dialog.path.get(), r'C:\envs\a\python.exe')
        self.assertEqual(dialog.confirm_button.cget('text'), 'Confirm and open window')
        self.assertIn('Python 3.10+ with Tkinter is required', dialog.status.get())
        dialog.window.destroy()

    def test_confirming_tries_the_path_without_quotes_and_returns_the_python(self):
        dialog = self.open()
        dialog.path.set('  "C:\\envs\\b\\python.exe" ')
        dialog.confirm()
        self.assertEqual(self.remembered, [('project', 'C:\\envs\\b\\python.exe')])
        self.assertEqual(dialog.result, Path('C:\\envs\\b\\python.exe'))
        self.assertFalse(dialog.window.winfo_exists())

    def test_a_python_that_does_not_work_is_explained_and_the_dialog_stays_open(self):
        def refuse(root, path):
            raise ValueError("The program needs a Python with working Tkinter. no display")
        dialog = self.open(refuse)
        with mock.patch('tkinter.messagebox.showerror') as error:
            dialog.confirm()
        self.assertIn('working Tkinter', error.call_args[0][1])
        self.assertIsNone(dialog.result)
        self.assertTrue(dialog.window.winfo_exists())
        self.assertIn('Choose a Python 3.10+', dialog.status.get())
        dialog.window.destroy()

    def test_browse_puts_the_chosen_file_in_the_field_and_cancelling_leaves_it(self):
        dialog = self.open()
        with mock.patch('tkinter.filedialog.askopenfilename', return_value=r'D:\other\python.exe'):
            dialog.browse()
        self.assertEqual(dialog.path.get(), r'D:\other\python.exe')
        with mock.patch('tkinter.filedialog.askopenfilename', return_value=''):
            dialog.browse()
        self.assertEqual(dialog.path.get(), r'D:\other\python.exe')
        dialog.window.destroy()

    def test_escape_and_cancel_give_no_python(self):
        dialog = self.open()
        dialog.window.event_generate('<Escape>')
        self.root.update()
        self.assertIsNone(dialog.result)
        self.assertFalse(dialog.window.winfo_exists())

    def test_the_real_python_can_be_confirmed(self):
        dialog = ps.PythonSelectionDialog(self.root, sys.executable, 'project', lambda root, path: {'executable': path})
        dialog.confirm()
        self.assertEqual(dialog.result, Path(sys.executable))


if __name__ == '__main__':
    unittest.main()
