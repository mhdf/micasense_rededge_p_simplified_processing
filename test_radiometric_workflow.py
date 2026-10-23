"""
Tests for radiometric_workflow.py (configuration, validation and runner), using the example images

Run from this folder with:  python -m unittest test_radiometric_workflow -v
"""

import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import tifffile

import radiometric_correction_tools as rct
import radiometric_workflow as wf

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'images' / 'raw'


def config_for(results_folder, **changes):
    values = {'INPUTS.images_folder': str(RAW), 'INPUTS.results_folder': str(results_folder)}
    values.update(changes)
    return values


class ConfigurationTests(unittest.TestCase):

    def test_defaults_cover_every_field_and_survive_json(self):
        config = wf.default_config()
        self.assertEqual(list(config), wf.FIELD_KEYS)
        self.assertEqual(json.loads(json.dumps(config)), config)
        self.assertEqual(len(set(wf.FIELD_KEYS)), len(wf.FIELD_KEYS))

    def test_default_config_is_a_copy(self):
        config = wf.default_config()
        config['PANEL.reflectance_by_band']['Blue'] = 0.1
        self.assertEqual(wf.default_config()['PANEL.reflectance_by_band']['Blue'], 0.477)

    def test_unknown_key_is_an_error(self):
        with self.assertRaisesRegex(wf.ConfigError, 'INPUTS.typo'):
            wf.merge_config({'INPUTS.typo': 1})

    def test_the_defaults_are_valid_for_the_example_folder(self):
        self.assertEqual(wf.validate_config(wf.default_config(), ROOT), [])

    def test_problems_are_listed_with_their_key(self):
        # wrong types are reported first
        config = wf.merge_config({'PANEL.saturation_threshold': 'high', 'CORRECTION.use_dls': 'yes',
                                  'PANEL.reflectance_by_band': {'Blue': 2.0}})
        text = '\n'.join(wf.validate_config(config, ROOT))
        for expected in ('PANEL.saturation_threshold must be an integer', 'CORRECTION.use_dls must be true or false',
                         "PANEL.reflectance_by_band['Blue']"):
            self.assertIn(expected, text)
        # then the values and the captures
        config = wf.merge_config({'PANEL.percent_inner_area': 1.5, 'CHECKS.ratio_range_min': 3.0,
                                  'INPUTS.reference_panel_capture': 'IMG_9999'})
        text = '\n'.join(wf.validate_config(config, ROOT))
        self.assertIn('PANEL.percent_inner_area must be in (0, 1]', text)
        self.assertIn('CHECKS.ratio_range_min must be smaller', text)
        self.assertIn("Capture 'IMG_9999' not found", text)

    def test_missing_folder_and_exiftool_are_reported(self):
        config = wf.merge_config({'INPUTS.images_folder': 'does/not/exist', 'INPUTS.exiftool_path': 'nope.exe'})
        text = '\n'.join(wf.validate_config(config, ROOT))
        self.assertIn('Images folder not found', text)
        self.assertIn('ExifTool not found', text)

    def test_captures_are_listed_from_the_file_names(self):
        self.assertEqual(rct.list_capture_names(RAW), ['IMG_0005', 'IMG_0010'])
        self.assertEqual(wf.capture_names(' A , B,,C '), ['A', 'B', 'C'])

    def test_panel_regions_are_saved_and_loaded_keeping_other_captures(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'regions.json'
            wf.save_panel_regions(path, 'IMG_0005', {'Blue': [[1, 2], [30, 2], [30, 40]], 'Red': None})
            wf.save_panel_regions(path, 'IMG_0100', {'Green': [[5, 5], [20, 5], [20, 20]]})
            self.assertEqual(wf.load_panel_regions(path), {'IMG_0005': {'Blue': [[1, 2], [30, 2], [30, 40]]},
                                                           'IMG_0100': {'Green': [[5, 5], [20, 5], [20, 20]]}})
            wf.save_panel_regions(path, 'IMG_0005', {'Blue': None})      # nothing left for this capture
            self.assertEqual(list(wf.load_panel_regions(path)), ['IMG_0100'])
            self.assertFalse(Path(str(path) + '.tmp').exists())

    def test_the_origin_of_each_region_is_saved_next_to_it(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'regions.json'
            square = [[1, 2], [30, 2], [30, 40]]
            wf.save_panel_regions(path, 'IMG_0005', {'Blue': square, 'Red': square},
                                  {'Blue': 'accepted_automatic', 'Red': 'edited', 'NIR': 'edited'})
            wf.save_panel_regions(path, 'IMG_0100', {'Green': square}, {'Green': 'edited'})
            self.assertEqual(wf.load_panel_origins(path), {'IMG_0005': {'Blue': 'accepted_automatic', 'Red': 'edited'},
                                                           'IMG_0100': {'Green': 'edited'}})      # only for saved bands
            wf.save_panel_regions(path, 'IMG_0005', {'Blue': square})                              # no origins given
            self.assertEqual(wf.load_panel_origins(path), {'IMG_0100': {'Green': 'edited'}})
            wf.save_panel_regions(path, 'IMG_0100', {})
            self.assertEqual(wf.load_panel_origins(path), {})
            self.assertNotIn('origins', json.loads(path.read_text(encoding='utf-8')))

    def test_a_file_without_origins_is_still_read(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'old.json'
            path.write_text(json.dumps({'version': 1, 'geometry': 'undistorted',
                                        'captures': {'IMG_0005': {'Blue': [[1, 2], [30, 2], [30, 40]]}}}), encoding='utf-8')
            self.assertEqual(wf.load_panel_origins(path), {})
            self.assertEqual(list(wf.load_panel_regions(path)['IMG_0005']), ['Blue'])

    def test_a_file_that_is_not_a_regions_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'regions.json'
            for content in ({'version': 9, 'geometry': 'undistorted', 'captures': {}},
                            {'version': 1, 'geometry': 'original', 'captures': {}},
                            {'version': 1, 'geometry': 'undistorted', 'captures': {'A': [1]}}):
                path.write_text(json.dumps(content), encoding='utf-8')
                with self.assertRaises(wf.ConfigError):
                    wf.load_panel_regions(path)

    def test_regions_file_problems_are_reported_by_the_validation(self):
        config = wf.merge_config({'PANEL.manual_regions_file': 'missing.json'})
        self.assertIn('Panel regions file not found', '\n'.join(wf.validate_config(config, ROOT)))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'regions.json'
            wf.save_panel_regions(path, 'IMG_0005', {'Infrared': [[1, 2], [30, 2], [30, 40]]})
            config = wf.merge_config({'PANEL.manual_regions_file': str(path)})
            self.assertIn("band 'Infrared' of IMG_0005 is not one of the panel bands",
                          '\n'.join(wf.validate_config(config, ROOT)))

    def write_spectrum(self, folder, level=0.5, start=400, stop=900, name='panel.csv', header=False):
        path = Path(folder) / name
        text = ('Wavelength (nm),Reflectance\n' if header else '') + \
               ''.join(f"{w},{level}\n" for w in range(start, stop + 1))
        path.write_text(text, encoding='utf-8')
        return path

    def spectral_config(self, path, **changes):
        values = {'PANEL.reflectance_source': 'spectral_file', 'PANEL.spectral_file': str(path)}
        values.update(changes)
        return wf.merge_config(values)

    def test_levels_come_from_the_typed_values_or_from_the_spectral_file(self):
        config = wf.default_config()
        levels, info = wf.panel_reflectance_levels(config)
        self.assertEqual(levels, config['PANEL.reflectance_by_band'])
        self.assertEqual(info, {'source': 'values'})
        with tempfile.TemporaryDirectory() as folder:
            config = self.spectral_config(self.write_spectrum(folder, 0.5, header=True))
            levels, info = wf.panel_reflectance_levels(config)
        self.assertEqual(list(levels), list(wf.BAND_NAMES))
        self.assertTrue(all(abs(value - 0.5) < 1e-12 for value in levels.values()))
        self.assertEqual(info['source'], 'spectral_file')
        self.assertEqual(len(info['sha256']), 64)
        self.assertEqual(info['bands']['Blue']['window_nm'], [459.0, 491.0])
        self.assertEqual(wf.panel_bands(config), list(wf.BAND_NAMES))

    def test_the_format_options_are_used(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'panel.csv'
            path.write_text(''.join(f"{w / 1000:.3f};{'50,0'}\n" for w in range(400, 901)), encoding='utf-8')
            config = self.spectral_config(path, **{'PANEL.spectral_delimiter': 'semicolon', 'PANEL.spectral_decimal': ',',
                                                  'PANEL.spectral_wavelength_unit': 'um',
                                                  'PANEL.spectral_reflectance_unit': 'percent'})
            levels, _ = wf.panel_reflectance_levels(config)
        self.assertAlmostEqual(levels['Red'], 0.5)

    def test_spectral_file_problems_are_reported(self):
        def problems(config):
            return '\n'.join(wf.validate_config(config, ROOT))
        self.assertIn('PANEL.spectral_file is empty',
                      problems(wf.merge_config({'PANEL.reflectance_source': 'spectral_file'})))
        self.assertIn('Spectral response file not found', problems(self.spectral_config('missing.csv')))
        with tempfile.TemporaryDirectory() as folder:
            path = self.write_spectrum(folder, 0.5, 400, 700)           # ends at 700 nm: the NIR window is not covered
            self.assertIn("Band 'Red edge'", problems(self.spectral_config(path)))
            self.assertIn("Band 'NIR'", problems(self.spectral_config(path)))      # all the bands are listed
            self.assertIn('Spectral response format', problems(self.spectral_config(
                path, **{'PANEL.spectral_reflectance_column': 1})))
            self.assertIn("PANEL.spectral_skip_rows must be 'auto'", problems(self.spectral_config(
                path, **{'PANEL.spectral_skip_rows': 'two'})))
            self.assertIn('same bands', problems(self.spectral_config(
                path, **{'PANEL.band_bandwidths': {'Blue': 32.0}})))
            self.assertIn('PANEL.spectral_min_coverage', problems(self.spectral_config(
                path, **{'PANEL.spectral_min_coverage': 0})))
            self.assertIn('PANEL.reflectance_source must be one of', problems(
                wf.merge_config({'PANEL.reflectance_source': 'both'})))
            self.assertIn("PANEL.band_bandwidths['Blue'] must be a number in (0, 5000]", problems(
                wf.merge_config({'PANEL.band_bandwidths': dict(wf.default_config()['PANEL.band_bandwidths'], Blue=-1)})))

    def test_the_reading_options_of_the_configuration_reach_the_format(self):
        config = wf.merge_config({'PANEL.spectral_delimiter': 'custom', 'PANEL.spectral_delimiter_char': '|',
                                  'PANEL.spectral_skip_footer': 2, 'PANEL.spectral_comment_prefix': '#',
                                  'PANEL.spectral_exclude_rows': '5-6', 'PANEL.spectral_wavelength_min': '400',
                                  'PANEL.spectral_wavelength_max': ' ', 'PANEL.spectral_reflectance_scale': 0.01})
        fmt = wf.spectral_format(config)
        self.assertEqual((fmt['delimiter'], fmt['delimiter_char'], fmt['skip_footer']), ('custom', '|', 2))
        self.assertEqual((fmt['comment_prefix'], fmt['exclude_rows']), ('#', '5-6'))
        self.assertEqual((fmt['wavelength_min'], fmt['wavelength_max'], fmt['reflectance_scale']), (400.0, None, 0.01))
        self.assertEqual(wf.spectral_format(wf.default_config())['wavelength_min'], None)

    def test_bad_reading_options_are_reported(self):
        for key, value, message in (('PANEL.spectral_wavelength_min', 'low', 'must be a number or empty'),
                                    ('PANEL.spectral_delimiter', 'custom', 'custom delimiter needs'),
                                    ('PANEL.spectral_exclude_rows', 'x', 'exclude_rows'),
                                    ('PANEL.spectral_skip_footer', -1, 'skip_footer'),
                                    ('PANEL.spectral_reflectance_scale', 0.0, 'reflectance_scale')):
            with tempfile.TemporaryDirectory() as folder:
                path = self.write_spectrum(folder)
                config = self.spectral_config(path, **{key: value})
                self.assertIn(message, '\n'.join(wf.validate_config(config, ROOT)), key)

    def test_levels_from_a_file_that_needs_the_options(self):
        with tempfile.TemporaryDirectory() as folder:
            lines = ['# panel', 'wavelength|reflectance'] + [f"{w}|{500 if w < 700 else 700}" for w in range(400, 901)] + ['end|x']
            path = Path(folder) / 'panel.txt'
            path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
            config = self.spectral_config(path, **{
                'PANEL.spectral_delimiter': 'custom', 'PANEL.spectral_delimiter_char': '|',
                'PANEL.spectral_comment_prefix': '#', 'PANEL.spectral_skip_footer': 1,
                'PANEL.spectral_reflectance_scale': 0.001, 'PANEL.spectral_wavelength_max': '880'})
            self.assertEqual(wf.validate_config(config, ROOT), [])
            levels, info = wf.panel_reflectance_levels(config)
        self.assertAlmostEqual(levels['Blue'], 0.5)
        self.assertAlmostEqual(levels['Red edge'], 0.7)

    def test_a_low_coverage_limit_accepts_a_file_that_ends_early(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.write_spectrum(folder, 0.5, 400, 848)           # like the example: NIR window 61 % covered
            self.assertEqual(wf.validate_config(self.spectral_config(path), ROOT), [])
            strict = self.spectral_config(path, **{'PANEL.spectral_min_coverage': 0.9})
            self.assertIn("Band 'NIR'", '\n'.join(wf.validate_config(strict, ROOT)))

    def test_irrelevant_values_are_not_checked_for_the_unused_source(self):
        # the typed values are not used when the spectral file is the source, and the file is not read for 'values'
        config = wf.merge_config({'PANEL.spectral_file': 'missing.csv'})
        self.assertEqual(wf.validate_config(config, ROOT), [])

    def test_the_band_lists_must_describe_the_same_bands(self):
        defaults = wf.default_config()
        config = wf.merge_config({'PANEL.band_center_wavelengths': {'Blue': 475.0}})
        self.assertIn('must have the same bands', '\n'.join(wf.validate_config(config, ROOT)))
        config = wf.merge_config({'PANEL.reflectance_by_band': {'Blue': 0.477, 'Coastal': 0.5}})
        self.assertIn('must have the same bands', '\n'.join(wf.validate_config(config, ROOT)))
        self.assertEqual(wf.validate_config(defaults, ROOT), [])

    def test_ignored_bands_cannot_also_be_processed(self):
        config = wf.merge_config({'INPUTS.ignored_bands': 'NIR, LWIR'})
        self.assertIn("INPUTS.ignored_bands has bands that are also processed: ['NIR']",
                      '\n'.join(wf.validate_config(config, ROOT)))

    def test_a_bad_file_pattern_is_reported_with_its_key(self):
        for pattern, message in (('(unclosed', 'not a valid regular expression'), (r'(.+)_(\d+)\.tif$', "group named 'capture'")):
            config = wf.merge_config({'INPUTS.file_pattern': pattern})
            problems = '\n'.join(wf.validate_config(config, ROOT))
            self.assertIn('INPUTS.file_pattern', problems)
            self.assertIn(message, problems)

    def test_a_pattern_that_matches_nothing_means_that_the_capture_is_not_found(self):
        config = wf.merge_config({'INPUTS.file_pattern': r'(?P<capture>.+)_B(?P<index>\d+)\.tif$'})
        self.assertIn("Capture 'IMG_0005' not found", '\n'.join(wf.validate_config(config, ROOT)))

    def test_the_bands_of_a_capture_can_be_detected_from_the_images(self):
        result = wf.detect_bands(wf.default_config(), ROOT)
        self.assertEqual([band['name'] for band in result['bands']],
                         ['Blue', 'Green', 'Red', 'NIR', 'Red edge', 'Panchro'])
        self.assertEqual(wf.detect_bands(wf.default_config(), ROOT, 'IMG_0010')['camera'], 'MicaSense RedEdge-P')
        with self.assertRaises(FileNotFoundError):
            wf.detect_bands(wf.default_config(), ROOT, 'IMG_9999')

    def test_a_second_panel_is_needed_to_use_two_panels(self):
        for mode in ('mean', 'interpolate_time', 'nearest_time'):
            config = wf.merge_config({'PANEL.panels_used': mode})
            self.assertIn('there is no second panel capture', '\n'.join(wf.validate_config(config, ROOT)))
            config = wf.merge_config({'PANEL.panels_used': mode, 'INPUTS.reference_panel_capture_after': 'IMG_0005'})
            self.assertEqual(wf.validate_config(config, ROOT), [])
        self.assertEqual(wf.validate_config(wf.merge_config({'PANEL.panels_used': 'first'}), ROOT), [])
        self.assertIn('PANEL.panels_used must be one of', '\n'.join(wf.validate_config(
            wf.merge_config({'PANEL.panels_used': 'last'}), ROOT)))

    def test_irradiance_series_changes_names_the_capture(self):
        series = {'Blue': [('c1', 100.0), ('c2', 101.0), ('c3', 150.0), ('c4', 100.0), ('c5', 100.0)],
                  'Green': [('c1', 100.0), ('c2', 100.0)],
                  'Red': [('c1', 100.0), ('c2', -1.0), ('c3', 100.0)]}
        changes = wf.irradiance_series_changes(series, 0.1)
        self.assertEqual(list(changes), ['Blue'])
        self.assertEqual([(e['capture'], e['kind']) for e in changes['Blue']], [('c3', 'spike')])


class RunTests(unittest.TestCase):
    """ The runs are done once and shared: each processes the six bands of the example capture """

    @classmethod
    def setUpClass(cls):
        cls.folder = Path(tempfile.mkdtemp())
        cls.panel_only = wf.run(config_for(cls.folder / 'panel_only'), ROOT)
        cls.with_dls = wf.run(config_for(cls.folder / 'dls', **{'CORRECTION.use_dls': True}), ROOT)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.folder, ignore_errors=True)

    def test_panel_only_run_completes_and_names_outputs_like_the_sources(self):
        result = self.panel_only
        self.assertEqual(result['status'], 'completed', result['errors'])
        names = sorted(Path(path).name for path in result['outputs'])
        self.assertEqual(names, [f'IMG_0010_{number}.tif' for number in range(1, 7)])
        self.assertEqual(list(result['captures']), ['IMG_0010'])
        self.assertEqual(result['stages'], list(wf.STAGES))

    def test_panel_only_output_equals_the_library_result(self):
        folder = self.folder / 'panel_only'
        panel_files = rct.read_band_files(rct.find_capture_paths(RAW, 'IMG_0005'), list(wf.BAND_NAMES))
        panel = rct.Image(*panel_files['Blue'])
        factor = panel.detect_reference_panel(wf.default_config()['PANEL.reflectance_by_band'], 0.7, 65000)[1]
        target = rct.Image(*rct.read_band_files(rct.find_capture_paths(RAW, 'IMG_0010'),
                                                list(wf.BAND_NAMES))['Blue'])
        expected = target.digital_numbers_to_reflectance(factor)
        saved = tifffile.imread(folder / 'IMG_0010_1.tif')
        self.assertEqual(saved.dtype, np.float64)
        self.assertTrue(np.array_equal(saved, expected))

    def test_result_and_provenance_files_are_saved(self):
        folder = self.folder / 'panel_only'
        saved = json.loads((folder / wf.RESULT_FILE_NAME).read_text(encoding='utf-8'))
        self.assertEqual(saved['status'], 'completed')
        self.assertEqual(Path(saved['result_file']), folder / wf.RESULT_FILE_NAME)
        self.assertEqual(set(saved['panel']['before']['bands']), set(wf.BAND_NAMES))
        provenance = json.loads((folder / 'IMG_0010_provenance.json').read_text(encoding='utf-8'))
        self.assertEqual(provenance['method'], 'panel_only')
        self.assertEqual(provenance['parameters']['config']['PANEL.percent_inner_area'], 0.7)
        self.assertFalse((folder / (wf.RESULT_FILE_NAME + '.tmp')).exists())

    def test_dls_run_corrects_every_band_including_panchro(self):
        folder = self.folder / 'dls'
        provenance = json.loads((folder / 'IMG_0010_provenance.json').read_text(encoding='utf-8'))
        self.assertEqual(provenance['method'], 'panel_and_dls')
        ratios = {band['band']: band['dls']['ratio_panel_over_image'] for band in provenance['bands']}
        self.assertEqual(set(ratios), set(wf.BAND_NAMES))
        self.assertAlmostEqual(ratios['Blue'], 138.05218450296033 / 156.04327208007237)
        self.assertAlmostEqual(ratios['Panchro'], 99.211212701455025 / 112.20932502581493)
        ratio = tifffile.imread(folder / 'IMG_0010_6.tif').mean() / \
                tifffile.imread(self.folder / 'panel_only' / 'IMG_0010_6.tif').mean()
        self.assertAlmostEqual(ratio, ratios['Panchro'])

    def test_dls_run_reports_the_tilt_flag_of_the_example(self):
        flags = self.with_dls['flags']
        self.assertEqual(len(flags), 1)
        self.assertIn('IMG_0010', flags[0])
        self.assertIn('sun angle', flags[0])

    def test_thresholds_of_the_configuration_are_used_and_then_restored(self):
        limits_before = dict(rct.DLS_CHECKS)
        result = wf.run(config_for(self.folder / 'loose', **{'CHECKS.max_sun_angle_discrepancy_deg': 30.0}), ROOT)
        self.assertEqual(result['flags'], [])
        self.assertEqual(rct.DLS_CHECKS, limits_before)

    def test_two_panel_captures_are_compared(self):
        result = wf.run(config_for(self.folder / 'two_panels', **{'INPUTS.reference_panel_capture_after': 'IMG_0005'}), ROOT)
        comparison = result['panel']['comparison']
        self.assertEqual(comparison['flags'], [])
        self.assertTrue(all(entry['factor_ratio'] == 1.0 for entry in comparison['bands'].values()))

    def test_a_region_drawn_by_hand_replaces_the_automatic_one_for_that_band_only(self):
        automatic = self.panel_only['panel']['before']['bands']
        regions = self.folder / 'regions.json'
        wf.save_panel_regions(regions, 'IMG_0005', {'Blue': automatic['Blue']['panel_bounds']})
        result = wf.run(config_for(self.folder / 'manual_blue', **{'PANEL.manual_regions_file': str(regions)}), ROOT)
        self.assertEqual(result['status'], 'completed', result['errors'])
        bands = result['panel']['before']['bands']
        self.assertEqual(bands['Blue']['region_source'], 'manual')
        self.assertEqual({band for band, audit in bands.items() if audit['region_source'] == 'automatic'},
                         set(wf.BAND_NAMES) - {'Blue'})
        self.assertEqual(bands['Blue']['radiance_to_reflectance_factor'], automatic['Blue']['radiance_to_reflectance_factor'])

    def test_regions_drawn_by_hand_allow_a_panel_without_a_readable_qr_code(self):
        regions = self.folder / 'regions_no_qr.json'
        square = [[600, 300], [700, 300], [700, 400], [600, 400]]
        wf.save_panel_regions(regions, 'IMG_0010', {band: square for band in wf.BAND_NAMES})
        result = wf.run(config_for(self.folder / 'no_qr', **{'INPUTS.reference_panel_capture': 'IMG_0010',
                                                             'INPUTS.image_captures': 'IMG_0005',
                                                             'PANEL.manual_regions_file': str(regions)}), ROOT)
        self.assertEqual(result['status'], 'completed', result['errors'])
        self.assertTrue(all(audit['region_source'] == 'manual' for audit in result['panel']['before']['bands'].values()))

    def test_the_origin_of_the_regions_reaches_the_audit_of_the_panel(self):
        automatic = self.panel_only['panel']['before']['bands']
        regions = self.folder / 'regions_origin.json'
        wf.save_panel_regions(regions, 'IMG_0005', {band: audit['panel_bounds'] for band, audit in automatic.items()},
                              {'Blue': 'accepted_automatic', 'Green': 'edited'})
        result = wf.run(config_for(self.folder / 'origin', **{'PANEL.manual_regions_file': str(regions)}), ROOT)
        self.assertEqual(result['status'], 'completed', result['errors'])
        bands = result['panel']['before']['bands']
        self.assertEqual((bands['Blue']['region_source'], bands['Blue']['region_origin']), ('manual', 'accepted_automatic'))
        self.assertEqual(bands['Green']['region_origin'], 'edited')
        self.assertIsNone(bands['Red']['region_origin'])                              # a file that does not say it
        self.assertEqual(bands['Blue']['radiance_to_reflectance_factor'], automatic['Blue']['radiance_to_reflectance_factor'])
        self.assertIsNone(automatic['Blue'].get('region_origin'))                     # the automatic run has none

    def test_an_invalid_region_names_the_band(self):
        regions = self.folder / 'regions_bad.json'
        wf.save_panel_regions(regions, 'IMG_0005', {'Green': [[1, 1], [9000, 1], [9000, 50]]})
        result = wf.run(config_for(self.folder / 'bad_region', **{'PANEL.manual_regions_file': str(regions)}), ROOT)
        self.assertEqual(result['status'], 'failed')
        self.assertIn("Panel of band 'Green' in IMG_0005", result['errors'][0])
        self.assertIn('not inside', result['errors'][0])

    def test_a_run_with_the_spectral_file_uses_its_reflectance_and_records_it(self):
        folder = self.folder / 'spectral_inputs'
        folder.mkdir()
        path = folder / 'panel.csv'
        path.write_text(''.join(f"{w},0.5\n" for w in range(400, 901)), encoding='utf-8')
        result = wf.run(config_for(self.folder / 'spectral', **{'PANEL.reflectance_source': 'spectral_file',
                                                                'PANEL.spectral_file': str(path)}), ROOT)
        self.assertEqual(result['status'], 'completed', result['errors'])
        reflectance = result['panel']['reflectance']
        self.assertEqual(reflectance['source'], 'spectral_file')
        self.assertTrue(all(abs(level - 0.5) < 1e-12 for level in reflectance['levels'].values()))
        values_run = self.panel_only['panel']['before']['bands']
        spectral_run = result['panel']['before']['bands']
        for band in wf.BAND_NAMES:
            self.assertAlmostEqual(spectral_run[band]['radiance_to_reflectance_factor'] /
                                   values_run[band]['radiance_to_reflectance_factor'],
                                   0.5 / wf.default_config()['PANEL.reflectance_by_band'][band])
        provenance = json.loads((self.folder / 'spectral' / 'IMG_0010_provenance.json').read_text(encoding='utf-8'))
        self.assertEqual(provenance['parameters']['panel_reflectance_source']['sha256'], reflectance['sha256'])
        self.assertEqual(provenance['parameters']['panel_reflectance_levels']['Blue'], 0.5)

    def test_a_spectral_file_that_does_not_cover_a_band_stops_the_run_before_processing(self):
        folder = self.folder / 'short_inputs'
        folder.mkdir()
        path = folder / 'panel.csv'
        path.write_text(''.join(f"{w},0.5\n" for w in range(400, 700)), encoding='utf-8')
        result = wf.run(config_for(self.folder / 'short', **{'PANEL.reflectance_source': 'spectral_file',
                                                             'PANEL.spectral_file': str(path)}), ROOT)
        self.assertEqual(result['status'], 'failed')
        self.assertIn("Band 'Red edge'", result['errors'][0])
        self.assertIn("Band 'NIR'", result['errors'][0])
        self.assertEqual(result['outputs'], [])

    def test_fewer_bands_can_be_processed_if_the_others_are_ignored(self):
        three = ('Blue', 'Green', 'Red')
        defaults = wf.default_config()
        subset = {'PANEL.reflectance_by_band': {band: defaults['PANEL.reflectance_by_band'][band] for band in three},
                  'PANEL.band_center_wavelengths': {band: defaults['PANEL.band_center_wavelengths'][band] for band in three},
                  'PANEL.band_bandwidths': {band: defaults['PANEL.band_bandwidths'][band] for band in three}}
        result = wf.run(config_for(self.folder / 'three_bands', **subset,
                                   **{'INPUTS.ignored_bands': 'Red edge, NIR, Panchro'}), ROOT)
        self.assertEqual(result['status'], 'completed', result['errors'])
        self.assertEqual(sorted(Path(path).name for path in result['outputs']),
                         ['IMG_0010_1.tif', 'IMG_0010_2.tif', 'IMG_0010_3.tif'])
        self.assertEqual(set(result['panel']['before']['bands']), set(three))
        provenance = json.loads((self.folder / 'three_bands' / 'IMG_0010_provenance.json').read_text(encoding='utf-8'))
        self.assertEqual([band['band'] for band in provenance['bands']], list(three))
        # a band gives the same result whether the others are processed or not
        for name in ('IMG_0010_1.tif', 'IMG_0010_3.tif'):
            self.assertTrue(np.array_equal(tifffile.imread(self.folder / 'three_bands' / name),
                                           tifffile.imread(self.folder / 'panel_only' / name)))

    def test_extra_bands_in_the_files_stop_the_capture_unless_they_are_ignored(self):
        defaults = wf.default_config()
        subset = {key: {'Blue': defaults[key]['Blue']} for key in ('PANEL.reflectance_by_band',
                  'PANEL.band_center_wavelengths', 'PANEL.band_bandwidths')}
        result = wf.run(config_for(self.folder / 'one_band', **subset), ROOT)
        self.assertEqual(result['status'], 'failed')
        self.assertIn('unexpected', result['errors'][0])
        self.assertIn('NIR', result['errors'][0])

    def test_a_band_that_is_not_in_the_files_is_a_missing_band(self):
        coastal = {'PANEL.reflectance_by_band': {'Coastal': 0.5}, 'PANEL.band_center_wavelengths': {'Coastal': 444.0},
                   'PANEL.band_bandwidths': {'Coastal': 28.0}, 'INPUTS.ignored_bands': 'Blue, Green, Red, NIR, Red edge, Panchro'}
        result = wf.run(config_for(self.folder / 'coastal', **coastal), ROOT)
        self.assertEqual(result['status'], 'failed')
        self.assertIn("missing: ['Coastal']", result['errors'][0])

    def test_files_with_another_naming_scheme_give_the_same_result(self):
        images = self.folder / 'renamed_images'
        images.mkdir()
        for path in RAW.glob('*.tif'):
            number = int(path.stem.rsplit('_', 1)[1])
            new_name = path.stem.rsplit('_', 1)[0].replace('IMG_', 'FLIGHT-') + f'-band{number + 4}.TIF'    # 5..10
            shutil.copy(path, images / new_name)
        pattern = r'(?P<capture>FLIGHT-\d+)-band(?P<index>\d+)\.tif$'
        result = wf.run({'INPUTS.images_folder': str(images), 'INPUTS.results_folder': str(self.folder / 'renamed'),
                         'INPUTS.reference_panel_capture': 'FLIGHT-0005', 'INPUTS.file_pattern': pattern}, ROOT)
        self.assertEqual(result['status'], 'completed', result['errors'])
        names = sorted(Path(path).name for path in result['outputs'])
        self.assertEqual(names, sorted(f'FLIGHT-0010-band{number}.TIF' for number in range(5, 11)))
        # the band comes from the metadata: file band5 is the Blue band, as IMG_0010_1.tif was
        self.assertTrue(np.array_equal(tifffile.imread(self.folder / 'renamed' / 'FLIGHT-0010-band5.TIF'),
                                       tifffile.imread(self.folder / 'panel_only' / 'IMG_0010_1.tif')))

    def run_with_two_panels(self, name, mode, **changes):
        values = {'INPUTS.reference_panel_capture_after': 'IMG_0005', 'PANEL.panels_used': mode}
        values.update(changes)
        return wf.run(config_for(self.folder / name, **values), ROOT)

    def test_the_mean_of_two_equal_panels_gives_the_result_of_one(self):
        result = self.run_with_two_panels('mean_panels', 'mean')
        self.assertEqual(result['status'], 'completed', result['errors'])
        self.assertEqual(result['captures']['IMG_0010']['panels'], {'used': 'mean', 'weight_of_the_second_panel': 0.5})
        for number in range(1, 7):
            self.assertTrue(np.array_equal(tifffile.imread(self.folder / 'mean_panels' / f'IMG_0010_{number}.tif'),
                                           tifffile.imread(self.folder / 'panel_only' / f'IMG_0010_{number}.tif')))
        provenance = json.loads((self.folder / 'mean_panels' / 'IMG_0010_provenance.json').read_text(encoding='utf-8'))
        blue = provenance['bands'][0]
        self.assertEqual(blue['second_panel_weight'], 0.5)
        self.assertTrue(blue['second_panel_file'].endswith('IMG_0005_1.tif'))
        self.assertEqual(blue['effective_factor'], blue['panel']['radiance_to_reflectance_factor'])

    def test_the_mean_with_the_dls_uses_the_irradiance_of_each_panel(self):
        result = self.run_with_two_panels('mean_dls', 'mean', **{'CORRECTION.use_dls': True})
        self.assertEqual(result['status'], 'completed', result['errors'])
        for number in range(1, 7):
            mean = tifffile.imread(self.folder / 'mean_dls' / f'IMG_0010_{number}.tif')
            single = tifffile.imread(self.folder / 'dls' / f'IMG_0010_{number}.tif')
            self.assertTrue(np.allclose(mean, single, rtol=1e-12, atol=0), number)

    def test_the_first_panel_alone_is_still_the_default_and_leaves_the_second_for_the_comparison(self):
        result = self.run_with_two_panels('first_panel', 'first')
        self.assertEqual(result['captures']['IMG_0010']['panels'], {'used': 'first', 'weight_of_the_second_panel': 0.0})
        self.assertIn('comparison', result['panel'])
        provenance = json.loads((self.folder / 'first_panel' / 'IMG_0010_provenance.json').read_text(encoding='utf-8'))
        self.assertIsNone(provenance['bands'][0]['second_panel_file'])

    def test_a_second_panel_that_is_not_used_when_there_is_only_one(self):
        entry = self.panel_only['captures']['IMG_0010']
        self.assertEqual(entry['panels'], {'used': 'first', 'weight_of_the_second_panel': 0.0})

    def test_the_weight_follows_the_time_of_the_image_between_the_panels(self):
        import exiftool
        images = self.folder / 'later_panel_images'
        images.mkdir()
        for path in RAW.glob('*.tif'):
            shutil.copy(path, images / path.name)
            if path.name.startswith('IMG_0005'):
                shutil.copy(path, images / path.name.replace('IMG_0005', 'IMG_0007'))   # the panel again, "after the flight"
        with exiftool.ExifToolHelper() as et:
            for path in images.glob('IMG_0007_*.tif'):
                et.execute('-overwrite_original', '-EXIF:DateTimeOriginal=2023:02:02 13:37:44', str(path))
        result = wf.run({'INPUTS.images_folder': str(images), 'INPUTS.results_folder': str(self.folder / 'timed'),
                         'INPUTS.reference_panel_capture': 'IMG_0005', 'INPUTS.reference_panel_capture_after': 'IMG_0007',
                         'INPUTS.image_captures': 'IMG_0010', 'PANEL.panels_used': 'interpolate_time'}, ROOT)
        self.assertEqual(result['status'], 'completed', result['errors'])
        # IMG_0005 at 13:26:38, IMG_0010 at 13:30:11, IMG_0007 at 13:37:44: 213 s of 666 s
        self.assertAlmostEqual(result['captures']['IMG_0010']['panels']['weight_of_the_second_panel'], 213 / 666)

    def run_with_a_later_panel(self, name, mode, time):
        import exiftool
        images = self.folder / (name + '_images')
        images.mkdir()
        for path in RAW.glob('*.tif'):
            shutil.copy(path, images / path.name)
            if path.name.startswith('IMG_0005'):
                shutil.copy(path, images / path.name.replace('IMG_0005', 'IMG_0007'))
        with exiftool.ExifToolHelper() as et:
            for path in images.glob('IMG_0007_*.tif'):
                et.execute('-overwrite_original', '-EXIF:DateTimeOriginal=' + time, str(path))
        return wf.run({'INPUTS.images_folder': str(images), 'INPUTS.results_folder': str(self.folder / name),
                       'INPUTS.reference_panel_capture': 'IMG_0005', 'INPUTS.reference_panel_capture_after': 'IMG_0007',
                       'INPUTS.image_captures': 'IMG_0010', 'PANEL.panels_used': mode}, ROOT)

    def test_the_panel_closest_in_time_is_the_one_that_is_used(self):
        # IMG_0005 at 13:26:38 and IMG_0010 at 13:30:11: the second panel at 13:37:44 is farther, at 13:31:00 it is nearer
        far = self.run_with_a_later_panel('nearest_far', 'nearest_time', '2023:02:02 13:37:44')
        self.assertEqual(far['status'], 'completed', far['errors'])
        self.assertEqual(far['captures']['IMG_0010']['panels'], {'used': 'nearest_time', 'weight_of_the_second_panel': 0.0})
        near = self.run_with_a_later_panel('nearest_near', 'nearest_time', '2023:02:02 13:31:00')
        self.assertEqual(near['status'], 'completed', near['errors'])
        self.assertEqual(near['captures']['IMG_0010']['panels'], {'used': 'nearest_time', 'weight_of_the_second_panel': 1.0})
        provenance = json.loads((self.folder / 'nearest_near' / 'IMG_0010_provenance.json').read_text(encoding='utf-8'))
        self.assertTrue(provenance['bands'][0]['second_panel_file'].endswith('IMG_0007_1.tif'))
        self.assertEqual(provenance['bands'][0]['second_panel_weight'], 1.0)

    def test_the_time_of_the_images_is_needed_to_interpolate(self):
        original = wf._read_capture

        def without_time(images_folder, name, bands, exiftool_path, *arguments):
            objects = original(images_folder, name, bands, exiftool_path, *arguments)
            if name == 'IMG_0010':
                for image in objects.values():
                    image.metadata['EXIF:DateTimeOriginal'] = None
            return objects

        with mock.patch.object(wf, '_read_capture', without_time):
            result = self.run_with_two_panels('no_time', 'interpolate_time')
        self.assertEqual(result['status'], 'failed')
        self.assertIn('IMG_0010: EXIF:DateTimeOriginal', result['captures']['IMG_0010']['errors'][0])

    def test_progress_reports_the_stages_in_order(self):
        events = []
        wf.run(config_for(self.folder / 'progress'), ROOT, events.append)
        stages = [event['stage'] for event in events]
        self.assertEqual([stage for i, stage in enumerate(stages) if i == 0 or stage != stages[i - 1]], list(wf.STAGES))

    def test_invalid_configuration_fails_without_processing(self):
        result = wf.run(config_for(self.folder / 'invalid', **{'INPUTS.reference_panel_capture': 'IMG_9999'}), ROOT)
        self.assertEqual(result['status'], 'failed')
        self.assertIn("Capture 'IMG_9999' not found", result['errors'][0])
        self.assertEqual(result['outputs'], [])

    def test_a_capture_without_a_panel_fails_with_a_clear_message(self):
        result = wf.run(config_for(self.folder / 'no_panel', **{'INPUTS.reference_panel_capture': 'IMG_0010',
                                                                'INPUTS.image_captures': 'IMG_0005'}), ROOT)
        self.assertEqual(result['status'], 'failed')
        self.assertIn('no QR code found', result['errors'][0])
        self.assertIn("Panel of band", result['errors'][0])

    def test_a_missing_band_file_fails_that_capture_only(self):
        images = self.folder / 'incomplete_images'
        images.mkdir()
        for path in RAW.glob('*.tif'):
            if path.name != 'IMG_0010_3.tif':
                shutil.copy(path, images / path.name)
        result = wf.run({'INPUTS.images_folder': str(images), 'INPUTS.results_folder': str(self.folder / 'incomplete')}, ROOT)
        self.assertEqual(result['status'], 'failed')
        self.assertIn("missing: ['Red']", result['captures']['IMG_0010']['errors'][0])
        self.assertEqual(result['outputs'], [])

    def test_missing_irradiance_in_one_band_fails_the_dls_correction_of_the_capture(self):
        original = wf._read_capture

        def without_panchro_irradiance(images_folder, name, bands, exiftool_path, *arguments):
            objects = original(images_folder, name, bands, exiftool_path, *arguments)
            if name == 'IMG_0010':
                objects['Panchro'].metadata['XMP:HorizontalIrradiance'] = None
            return objects

        with mock.patch.object(wf, '_read_capture', without_panchro_irradiance):
            result = wf.run(config_for(self.folder / 'no_irradiance', **{'CORRECTION.use_dls': True}), ROOT)
            panel_only = wf.run(config_for(self.folder / 'no_irr_panel'), ROOT)
        self.assertEqual(result['status'], 'failed')
        self.assertIn('HorizontalIrradiance', result['captures']['IMG_0010']['errors'][0])
        self.assertEqual(result['outputs'], [])      # nothing is saved for a capture that cannot be corrected
        self.assertEqual(panel_only['status'], 'completed', panel_only['errors'])


class CommandLineTests(unittest.TestCase):

    def run_main(self, arguments):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = wf.main(arguments)
        return code, out.getvalue(), err.getvalue()

    def test_default_configuration_can_be_written_and_loaded(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'inputs.json'
            code, _, _ = self.run_main(['--write-default-config', str(path)])
            self.assertEqual(code, 0)
            self.assertEqual(wf.load_config(path), wf.default_config())

    def test_request_json_runs_and_prints_the_result_path(self):
        with tempfile.TemporaryDirectory() as folder:
            request = Path(folder) / 'request.json'
            request.write_text(json.dumps({'version': 1, 'root': str(ROOT),
                                           'config': config_for(Path(folder) / 'out')}), encoding='utf-8')
            code, out, err = self.run_main(['--request-json', str(request)])
            self.assertEqual(code, 0, err)
            self.assertIn('Status: completed', out)
            line = [line for line in out.splitlines() if line.startswith('RADIOMETRIC_RESULT_JSON=')][0]
            self.assertTrue(Path(line.split('=', 1)[1]).is_file())

    def test_wrong_request_version_and_missing_file_return_2(self):
        with tempfile.TemporaryDirectory() as folder:
            request = Path(folder) / 'request.json'
            request.write_text(json.dumps({'version': 99}), encoding='utf-8')
            self.assertEqual(self.run_main(['--request-json', str(request)])[0], 2)
            self.assertEqual(self.run_main(['--config', str(Path(folder) / 'missing.json')])[0], 2)

    def test_the_bands_of_a_capture_can_be_printed(self):
        code, out, err = self.run_main(['--detect-bands', 'IMG_0010', '--root', str(ROOT)])
        self.assertEqual(code, 0, err)
        detected = json.loads(out)
        self.assertEqual(detected['camera'], 'MicaSense RedEdge-P')
        self.assertEqual([band['name'] for band in detected['bands']], ['Blue', 'Green', 'Red', 'NIR', 'Red edge', 'Panchro'])
        code, out, err = self.run_main(['--detect-bands', '--root', str(ROOT)])          # the panel capture by default
        self.assertEqual(code, 0, err)
        self.assertEqual(len(json.loads(out)['bands']), 6)
        code, out, err = self.run_main(['--detect-bands', 'IMG_9999', '--root', str(ROOT)])
        self.assertEqual(code, 1)
        self.assertIn('FileNotFoundError', err)

    def test_invalid_configuration_returns_1_and_prints_the_errors(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'inputs.json'
            path.write_text(json.dumps({'PANEL.percent_inner_area': 5}), encoding='utf-8')
            code, _, err = self.run_main(['--config', str(path), '--root', str(ROOT)])
            self.assertEqual(code, 1)
            self.assertIn('PANEL.percent_inner_area must be in (0, 1]', err)


if __name__ == '__main__':
    unittest.main()
