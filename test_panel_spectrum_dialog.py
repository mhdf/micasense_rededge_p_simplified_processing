"""
Tests for panel_spectrum_dialog.py: the session (what was read, the bands, the messages) without a display and the
dialog with a hidden Tk root (skipped if Tk cannot open one)

Run from this folder with:  python -m unittest test_panel_spectrum_dialog -v
"""

import tempfile
import time
import unittest
from pathlib import Path

import panel_spectrum as ps
import panel_spectrum_dialog as psd


def write_curve(folder, name='panel.csv', level=0.5, start=400, stop=900, header=False, separator=',', decimal='.'):
    path = Path(folder) / name
    text = ('Wavelength;Reflectance\n' if header else '') + ''.join(
        f"{w}{separator}{str(level).replace('.', decimal)}\n" for w in range(start, stop + 1))
    path.write_text(text, encoding='utf-8')
    return path


class SessionTests(unittest.TestCase):

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.folder.cleanup()

    def test_without_a_file_there_is_a_message_and_no_result(self):
        session = psd.ImportSession()
        self.assertEqual(session.messages(), [('Choose a file.', 'error')])
        with self.assertRaises(ValueError):
            session.result('use_file')

    def test_a_missing_file_is_reported(self):
        session = psd.ImportSession(Path(self.folder.name) / 'missing.csv')
        self.assertIn('File not found', session.error)

    def test_a_valid_file_gives_rows_curve_bands_and_a_result(self):
        session = psd.ImportSession(write_curve(self.folder.name, header=True), dict(skip_rows='auto', delimiter='semicolon'))
        self.assertIsNotNone(session.error)                         # the file uses commas, not semicolons
        session = psd.ImportSession(write_curve(self.folder.name, header=True))
        self.assertIsNone(session.error)
        self.assertEqual(session.preview_rows()[0][4], 'header')
        self.assertEqual(session.preview_rows()[1][:4], (2, '400,0.5', '400', '0.5'))
        wavelengths, reflectances = session.curve()
        self.assertEqual((wavelengths[0], wavelengths[-1], len(wavelengths)), (400.0, 900.0, 501))
        rows = session.band_rows()
        self.assertEqual([row[0] for row in rows], list(ps.REDEDGE_P_BANDS))
        self.assertEqual(rows[0], ('Blue', '459 - 491', 33, '100 %', '0.5000'))
        result = session.result('use_file')
        self.assertEqual(result['action'], 'use_file')
        self.assertEqual(result['centers']['Blue'], 475.0)
        self.assertEqual(result['bandwidths']['NIR'], 57.0)
        self.assertEqual(result['levels']['Red'], 0.5)
        self.assertEqual(result['format'], ps.DEFAULT_FORMAT)

    def test_changing_the_format_changes_what_is_read(self):
        path = write_curve(self.folder.name, separator=';', decimal=',')
        session = psd.ImportSession(path)
        self.assertIsNotNone(session.error)                         # the delimiter is detected, the decimal mark is not
        self.assertEqual(session.band_rows(), [])
        session.set_format(decimal=',')
        self.assertIsNone(session.error)
        self.assertEqual(session.band_rows()[0][4], '0.5000')
        session.set_format(delimiter='comma')
        self.assertIsNotNone(session.error)                         # a comma cannot be both the delimiter and the mark
        session.set_format(delimiter='semicolon')
        self.assertIsNone(session.error)

    def test_detect_fills_the_format_and_reports_what_was_guessed(self):
        path = write_curve(self.folder.name, header=True, separator=';', decimal=',')
        session = psd.ImportSession(path, dict(delimiter='comma'))
        notes = session.detect()
        self.assertEqual((session.fmt['delimiter'], session.fmt['decimal']), ('semicolon', ','))
        self.assertTrue(any('delimiter: semicolon' in note for note in notes))
        self.assertIsNone(session.error)

    def test_messages_report_partial_coverage_unusable_samples_and_uncovered_bands(self):
        partial = psd.ImportSession(write_curve(self.folder.name, 'a.csv', start=400, stop=848))
        texts = [text for text, kind in partial.messages() if kind == 'warning']
        self.assertTrue(any('NIR' in text and 'partly covered' in text for text in texts))
        zeros = Path(self.folder.name) / 'z.csv'
        zeros.write_text(''.join(f"{w},{0.0 if w < 420 else 0.5}\n" for w in range(400, 901)), encoding='utf-8')
        self.assertTrue(any('not used' in text for text, kind in psd.ImportSession(zeros).messages() if kind == 'warning'))
        statuses = [row[4] for row in psd.ImportSession(zeros).preview_rows()]
        self.assertEqual(statuses[0], 'not used (outside 0-1)')
        self.assertEqual(statuses[-1], 'ok')
        short = psd.ImportSession(write_curve(self.folder.name, 'b.csv', start=400, stop=700))
        self.assertIsNone(short.band_table)
        self.assertIn("Band 'NIR'", short.band_error)
        self.assertTrue(any(kind == 'error' for _, kind in short.messages()))
        with self.assertRaisesRegex(ValueError, "Band 'NIR'"):
            short.result('use_file')
        self.assertEqual(len(short.curve()[0]), 301)                # the curve can still be inspected

    def test_changing_the_windows_and_the_coverage_recomputes_the_bands(self):
        session = psd.ImportSession(write_curve(self.folder.name, start=400, stop=900))
        windows = dict(ps.REDEDGE_P_BANDS, NIR=(600.0, 40.0))
        session.set_windows(windows, 0.9)
        self.assertEqual(session.band_rows()[-1][0], 'NIR')
        self.assertEqual(session.band_rows()[-1][1], '580 - 620')

    def test_the_preview_is_limited(self):
        session = psd.ImportSession(write_curve(self.folder.name))
        self.assertEqual(len(session.preview_rows()), 300)
        self.assertEqual(len(session.preview_rows(10)), 10)

    def test_the_hash_and_the_file_are_in_the_loaded_spectrum(self):
        session = psd.ImportSession(write_curve(self.folder.name))
        self.assertEqual(len(session.spectrum['sha256']), 64)


class DialogTests(unittest.TestCase):

    def setUp(self):
        import tkinter as tk
        try:
            self.root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"no display for Tk: {error}")
        self.root.withdraw()
        from ui_theme import apply_dark_theme
        apply_dark_theme(self.root)
        self.folder = tempfile.TemporaryDirectory()
        self.window = None

    def tearDown(self):
        try:
            self.root.destroy()
        except Exception:
            pass
        self.folder.cleanup()

    def open(self, path='', **kwargs):
        import tkinter as tk
        self.window = tk.Toplevel(self.root)
        self.dialog = psd.ImportDialog(self.window, psd.ImportSession(path, **kwargs))
        self.window.update()
        return self.dialog

    def pump(self, seconds=0.6):
        end = time.time() + seconds
        while time.time() < end:
            self.root.update()
            time.sleep(0.02)

    def test_the_rows_the_bands_and_the_plot_are_shown(self):
        dialog = self.open(write_curve(self.folder.name, header=True))
        self.assertEqual(len(dialog.rows_tree.get_children()), 300)
        first = dialog.rows_tree.item(dialog.rows_tree.get_children()[0])
        self.assertEqual(first['values'][4], 'header')
        self.assertEqual(len(dialog.bands_tree.get_children()), 6)
        self.assertGreater(len(dialog.plot.find_all()), 10)
        self.assertEqual(dialog.use_button.instate(['disabled']), False)
        self.assertTrue(any('501 samples' in line for line in dialog.message_lines()))

    def test_without_a_file_the_buttons_are_disabled_and_the_plot_says_so(self):
        dialog = self.open('')
        self.assertEqual(dialog.use_button.instate(['disabled']), True)
        self.assertIn('Choose a file.', dialog.message_lines())
        self.assertEqual(len(dialog.rows_tree.get_children()), 0)

    def test_typing_a_format_option_updates_the_preview(self):
        dialog = self.open(write_curve(self.folder.name))
        dialog.format_vars['delimiter'].set('semicolon')
        self.pump()
        self.assertEqual(dialog.use_button.instate(['disabled']), True)
        self.assertTrue(any('data row' in line for line in dialog.message_lines()))
        dialog.format_vars['delimiter'].set('comma')
        self.pump()
        self.assertEqual(dialog.use_button.instate(['disabled']), False)

    def test_text_that_is_not_a_number_is_reported_without_a_crash(self):
        dialog = self.open(write_curve(self.folder.name))
        dialog.format_vars['wavelength_column'].set('x')
        self.pump()
        self.assertTrue(any('whole numbers' in line for line in dialog.message_lines()))
        dialog.format_vars['wavelength_column'].set('1')
        dialog.center_vars['Blue'].set('abc')
        self.pump()
        self.assertTrue(any('windows need numbers' in line for line in dialog.message_lines()))

    def test_detect_updates_the_fields(self):
        dialog = self.open(write_curve(self.folder.name, header=True, separator=';', decimal=','),
                           fmt=dict(delimiter='comma'))
        dialog.detect()
        self.assertEqual(dialog.format_vars['delimiter'].get(), 'semicolon')
        self.assertEqual(dialog.format_vars['decimal'].get(), ',')
        self.assertTrue(any(line.startswith('Detected:') for line in dialog.message_lines()))
        self.assertEqual(dialog.use_button.instate(['disabled']), False)

    def test_use_file_returns_the_format_the_windows_and_the_levels(self):
        dialog = self.open(write_curve(self.folder.name, level=0.48))
        dialog.width_vars['Blue'].set('40')
        dialog.accept('use_file')
        result = dialog.result
        self.assertEqual(result['action'], 'use_file')
        self.assertEqual(result['bandwidths']['Blue'], 40.0)
        self.assertAlmostEqual(result['levels']['Blue'], 0.48)
        self.assertFalse(self.window.winfo_exists())

    def test_an_invalid_state_cannot_be_accepted(self):
        dialog = self.open(write_curve(self.folder.name, start=400, stop=700))
        dialog.accept('use_file')
        self.assertIsNone(dialog.result)
        self.assertTrue(self.window.winfo_exists())
        self.assertTrue(any("Band 'NIR'" in line for line in dialog.message_lines()))

    def test_cancel_returns_nothing(self):
        dialog = self.open(write_curve(self.folder.name))
        dialog.cancel()
        self.assertIsNone(dialog.result)
        self.assertFalse(self.window.winfo_exists())


if __name__ == '__main__':
    unittest.main()
