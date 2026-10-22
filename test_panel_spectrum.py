"""
Tests for panel_spectrum.py (spectral response of the reference panel, no display needed)

Run from this folder with:  python -m unittest test_panel_spectrum -v
The file of the example panel (outside this repository) is used only if it exists.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

import panel_spectrum as ps

REAL_FILE = Path(r'C:\path\to\your\files'
                 r'\reference_panel_spectral_response\RP06-2201497-OB.csv')


def write(folder, name, text):
    path = Path(folder) / name
    path.write_text(text, encoding='utf-8')
    return path


def flat_curve(start=365, stop=848, level=0.477, zeros_below=400):
    return ''.join(f"{w},{0.0 if w < zeros_below else level}\n" for w in range(start, stop + 1))


class FormatTests(unittest.TestCase):

    def test_the_default_format_is_valid(self):
        self.assertEqual(ps.validate_format(ps.DEFAULT_FORMAT), [])

    def test_problems_are_listed(self):
        bad = dict(ps.DEFAULT_FORMAT, delimiter='pipe', skip_rows=-1, wavelength_column=0, reflectance_column=0,
                   wavelength_unit='m', reflectance_unit='ratio', decimal=';')
        text = '; '.join(ps.validate_format(bad))
        for expected in ('delimiter must be', 'skip_rows', 'wavelength_column', 'must be different',
                         'wavelength_unit', 'reflectance_unit', 'decimal must be'):
            self.assertIn(expected, text)
        self.assertIn('both be a comma', '; '.join(ps.validate_format(
            dict(ps.DEFAULT_FORMAT, decimal=',', delimiter='comma'))))


class ParseTests(unittest.TestCase):

    def parse(self, text, **changes):
        return ps.parse_rows(text.splitlines(), dict(ps.DEFAULT_FORMAT, **changes))

    def test_header_rows_are_skipped_automatically_and_blank_lines_are_ignored(self):
        rows = self.parse('Wavelength,Reflectance\n(nm),(fraction)\n\n400,0.4\n401,0.41\n')
        self.assertEqual([row['status'] for row in rows], ['header', 'header', 'blank', 'ok', 'ok'])
        self.assertEqual(rows[3]['wavelength'], 400.0)

    def test_a_bad_row_after_the_data_started_is_an_error_with_its_line(self):
        rows = self.parse('400,0.4\n401,abc\n402,0.42\n403\n')
        self.assertEqual(rows[1]['status'][:6], 'error:')
        self.assertIn('only 1 column', rows[3]['status'])

    def test_an_integer_skip_rows_skips_exactly_those_lines(self):
        rows = self.parse('1,2\n400,0.4\n', skip_rows=1)
        self.assertEqual(rows[0]['status'], 'header')
        self.assertEqual(rows[1]['status'], 'ok')
        rows = self.parse('abc,def\n400,0.4\n', skip_rows=0)           # a header that is not skipped is an error
        self.assertTrue(rows[0]['status'].startswith('error'))

    def test_semicolon_and_decimal_comma(self):
        rows = self.parse('400;0,4\n401;0,41\n', delimiter='semicolon', decimal=',')
        self.assertEqual([(r['wavelength'], r['reflectance']) for r in rows], [(400.0, 0.4), (401.0, 0.41)])

    def test_tab_and_runs_of_spaces(self):
        self.assertEqual(self.parse('400\t0.4\n', delimiter='tab')[0]['reflectance'], 0.4)
        self.assertEqual(self.parse('400    0.4\n', delimiter='space')[0]['reflectance'], 0.4)

    def test_auto_delimiter_picks_the_one_with_a_constant_number_of_columns(self):
        self.assertEqual(ps.resolve_delimiter(['400;0,4', '401;0,41']), ';')
        self.assertEqual(ps.resolve_delimiter(['400\t0.4', '401\t0.41']), '\t')
        self.assertEqual(ps.resolve_delimiter(['400 0.4', '401 0.41']), ' ')

    def test_columns_and_units(self):
        rows = self.parse('id,0.475,47.7\nid,0.476,47.8\n', wavelength_column=2, reflectance_column=3,
                          wavelength_unit='um', reflectance_unit='percent')
        self.assertAlmostEqual(rows[0]['wavelength'], 475.0)
        self.assertAlmostEqual(rows[0]['reflectance'], 0.477)

    def test_an_invalid_format_raises(self):
        with self.assertRaises(ps.SpectrumError):
            self.parse('400,0.4', wavelength_column=1, reflectance_column=1)


class LoadTests(unittest.TestCase):

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.folder.cleanup()

    def test_a_file_is_loaded_with_its_hash_and_warnings_about_unusable_samples(self):
        path = write(self.folder.name, 'curve.csv', flat_curve())
        spectrum = ps.load_spectrum(path, ps.DEFAULT_FORMAT)
        self.assertEqual(len(spectrum['wavelength']), 484)
        self.assertEqual(len(spectrum['sha256']), 64)
        self.assertTrue(any('35 sample(s)' in warning for warning in spectrum['warnings']))

    def test_unsorted_wavelengths_are_sorted_with_a_warning(self):
        path = write(self.folder.name, 'curve.csv', '402,0.42\n400,0.4\n401,0.41\n')
        spectrum = ps.load_spectrum(path, ps.DEFAULT_FORMAT)
        self.assertEqual(list(spectrum['wavelength']), [400.0, 401.0, 402.0])
        self.assertEqual(list(spectrum['reflectance']), [0.4, 0.41, 0.42])
        self.assertTrue(any('sorted' in warning for warning in spectrum['warnings']))

    def test_errors_name_the_lines(self):
        path = write(self.folder.name, 'curve.csv', '400,0.4\n401,x\n402,y\n')
        with self.assertRaisesRegex(ps.SpectrumError, 'curve.csv: line 2.*line 3'):
            ps.load_spectrum(path, ps.DEFAULT_FORMAT)

    def test_duplicates_and_too_few_rows_are_rejected(self):
        with self.assertRaisesRegex(ps.SpectrumError, 'more than once'):
            ps.load_spectrum(write(self.folder.name, 'a.csv', '400,0.4\n400,0.41\n401,0.4\n'), ps.DEFAULT_FORMAT)
        with self.assertRaisesRegex(ps.SpectrumError, '1 data row'):
            ps.load_spectrum(write(self.folder.name, 'b.csv', 'wavelength,reflectance\n400,0.4\n'), ps.DEFAULT_FORMAT)

    def test_a_byte_order_mark_is_ignored(self):
        path = Path(self.folder.name) / 'bom.csv'
        path.write_bytes(b'\xef\xbb\xbf400,0.4\n401,0.41\n')
        self.assertEqual(ps.load_spectrum(path, ps.DEFAULT_FORMAT)['wavelength'][0], 400.0)


class BandTests(unittest.TestCase):

    def spectrum(self, wavelength, reflectance):
        return {'wavelength': np.array(wavelength, float), 'reflectance': np.array(reflectance, float)}

    def test_a_flat_curve_gives_its_level_in_every_band(self):
        w = np.arange(400, 901)
        result = ps.band_reflectances(self.spectrum(w, np.full(len(w), 0.48)), ps.REDEDGE_P_BANDS)
        for band, values in result.items():
            self.assertAlmostEqual(values['reflectance'], 0.48)
            self.assertEqual(values['coverage'], 1.0, band)

    def test_the_mean_over_the_window_of_a_ramp_is_its_middle_value(self):
        w = np.arange(400, 901)
        result = ps.band_reflectances(self.spectrum(w, 0.0004 * w), {'Red': (668.0, 16.0)})
        self.assertAlmostEqual(result['Red']['reflectance'], 0.0004 * 668.0)
        self.assertEqual(result['Red']['samples'], 17)
        self.assertEqual(result['Red']['window_nm'], [660.0, 676.0])

    def test_a_window_beyond_the_end_of_the_file_is_averaged_over_the_part_that_exists(self):
        w = np.arange(400, 849)
        result = ps.band_reflectances(self.spectrum(w, np.full(len(w), 0.477)), {'NIR': (842.0, 57.0)})
        self.assertAlmostEqual(result['NIR']['reflectance'], 0.477)
        self.assertAlmostEqual(result['NIR']['coverage'], 35 / 57)

    def test_too_little_coverage_is_an_error_naming_the_band(self):
        w = np.arange(400, 849)
        with self.assertRaisesRegex(ps.SpectrumError, "Band 'NIR'.*61%"):
            ps.band_reflectances(self.spectrum(w, np.full(len(w), 0.477)), {'NIR': (842.0, 57.0)}, min_coverage=0.7)
        with self.assertRaisesRegex(ps.SpectrumError, "Band 'Red'"):
            ps.band_reflectances(self.spectrum(w, np.full(len(w), 0.477)), {'Red': (1200.0, 16.0)})

    def test_zero_samples_are_not_used(self):
        w = np.arange(400, 500)
        r = np.full(len(w), 0.5)
        r[:30] = 0.0                                                    # no valid data below 430 nm
        result = ps.band_reflectances(self.spectrum(w, r), {'Blue': (440.0, 40.0)})       # window 420-460
        self.assertAlmostEqual(result['Blue']['reflectance'], 0.5)
        self.assertAlmostEqual(result['Blue']['coverage'], 31 / 41)

    def test_one_sample_in_the_window_is_used_as_it_is(self):
        result = ps.band_reflectances(self.spectrum([500, 600, 700], [0.4, 0.5, 0.6]), {'X': (600.0, 20.0)}, 0.0)
        self.assertEqual(result['X']['reflectance'], 0.5)


class DetectFormatTests(unittest.TestCase):

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.folder.cleanup()

    def test_plain_csv(self):
        fmt, notes = ps.detect_format(write(self.folder.name, 'a.csv', flat_curve(400, 420)))
        self.assertEqual(fmt, dict(ps.DEFAULT_FORMAT, delimiter='comma'))

    def test_semicolon_decimal_comma_and_header(self):
        text = 'Wavelength;Reflectance\n' + ''.join(f"{w};{str(0.477).replace('.', ',')}\n" for w in range(400, 420))
        fmt, notes = ps.detect_format(write(self.folder.name, 'b.csv', text))
        self.assertEqual((fmt['delimiter'], fmt['decimal']), ('semicolon', ','))
        self.assertTrue(any('header row' in note for note in notes))
        rows = ps.parse_rows(text.splitlines(), fmt)
        self.assertEqual(rows[1]['reflectance'], 0.477)

    def test_micrometers_and_percent(self):
        text = ''.join(f"{w / 1000:.3f}\t{47.7}\n" for w in range(400, 420))
        fmt, notes = ps.detect_format(write(self.folder.name, 'c.csv', text))
        self.assertEqual((fmt['delimiter'], fmt['wavelength_unit'], fmt['reflectance_unit']), ('tab', 'um', 'percent'))
        rows = ps.parse_rows(text.splitlines(), fmt)
        self.assertAlmostEqual(rows[0]['wavelength'], 400.0)
        self.assertAlmostEqual(rows[0]['reflectance'], 0.477)

    def test_a_file_without_numbers_is_reported(self):
        fmt, notes = ps.detect_format(write(self.folder.name, 'd.csv', 'a,b\nc,d\n'))
        self.assertTrue(any('no numeric rows' in note for note in notes))


@unittest.skipUnless(REAL_FILE.is_file(), 'the file of the example panel is not available')
class ExamplePanelTests(unittest.TestCase):

    def test_the_example_panel_reproduces_the_values_printed_on_it(self):
        spectrum = ps.load_spectrum(REAL_FILE, ps.DEFAULT_FORMAT)
        result = ps.band_reflectances(spectrum, ps.REDEDGE_P_BANDS)
        printed = {'Blue': 0.477, 'Green': 0.478, 'Panchro': 0.479, 'Red': 0.478, 'Red edge': 0.478, 'NIR': 0.477}
        for band, value in printed.items():
            self.assertAlmostEqual(result[band]['reflectance'], value, delta=0.002, msg=band)
        self.assertAlmostEqual(result['NIR']['coverage'], 35 / 57)
        self.assertEqual(ps.detect_format(REAL_FILE)[0], dict(ps.DEFAULT_FORMAT, delimiter='comma'))


if __name__ == '__main__':
    unittest.main()
