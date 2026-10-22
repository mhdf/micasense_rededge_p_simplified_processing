"""
Tests for the metadata validation and band-file mapping in radiometric_correction_tools.py

Run from this folder with:  python -m unittest test_radiometric_correction_tools -v
The tests that read the example images need ExifTool; set the environment variable
'exiftoolpath' if it is not on the PATH.
"""

import os
import tempfile
import unittest
import warnings

import numpy as np
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import radiometric_correction_tools as rct

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'images' / 'raw'
BANDS = ['Blue', 'Green', 'Panchro', 'Red', 'Red edge', 'NIR']
REFLECTANCE_LEVELS = { "Blue": 0.477, "Green": 0.478, "Panchro": 0.479,
                       "Red": 0.478, "Red edge": 0.478, "NIR": 0.477 }


def fake_metadata(band, capture='A'):
    return {'XMP:BandName': band, 'XMP:CaptureId': capture}


class MapBandFilesTests(unittest.TestCase):

    def test_maps_by_band_name_not_by_file_order(self):
        paths = ['a_1.tif', 'a_2.tif', 'a_3.tif']
        metadata = [fake_metadata('Red'), fake_metadata('Blue'), fake_metadata('Green')]
        mapping = rct.map_band_files(paths, metadata, ['Blue', 'Green', 'Red'])
        self.assertEqual(mapping['Red'][0], 'a_1.tif')
        self.assertEqual(mapping['Blue'][0], 'a_2.tif')
        self.assertEqual(mapping['Green'][0], 'a_3.tif')

    def test_duplicate_band_is_rejected(self):
        with self.assertRaisesRegex(rct.MetadataError, "Band 'Blue' appears twice"):
            rct.map_band_files(['a_1.tif', 'a_2.tif'], [fake_metadata('Blue'), fake_metadata('Blue')])

    def test_missing_band_is_reported(self):
        with self.assertRaisesRegex(rct.MetadataError, "missing: \\['Green'\\]"):
            rct.map_band_files(['a_1.tif'], [fake_metadata('Blue')], ['Blue', 'Green'])

    def test_unexpected_band_is_reported(self):
        with self.assertRaisesRegex(rct.MetadataError, "unexpected: \\['Red'\\]"):
            rct.map_band_files(['a_1.tif', 'a_2.tif'], [fake_metadata('Blue'), fake_metadata('Red')], ['Blue'])

    def test_mixed_captures_are_rejected(self):
        with self.assertRaisesRegex(rct.MetadataError, 'different captures'):
            rct.map_band_files(['a_1.tif', 'b_1.tif'],
                               [fake_metadata('Blue', 'A'), fake_metadata('Green', 'B')])

    def test_missing_band_name_names_the_file(self):
        with self.assertRaisesRegex(rct.MetadataError, "a_1.tif: missing metadata field 'XMP:BandName'"):
            rct.map_band_files(['a_1.tif'], [{}])


class FilePatternTests(unittest.TestCase):

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.folder.cleanup()

    def touch(self, *names):
        for name in names:
            (Path(self.folder.name) / name).write_bytes(b'')

    def test_the_default_pattern_groups_the_files_of_a_capture_in_numeric_order(self):
        self.touch('IMG_0001_1.tif', 'IMG_0001_2.tif', 'IMG_0001_10.tif', 'IMG_0002_1.tif', 'notes.txt', 'IMG_x.tif')
        self.assertEqual(rct.list_capture_names(self.folder.name), ['IMG_0001', 'IMG_0002'])
        names = [Path(path).name for path in rct.find_capture_paths(self.folder.name, 'IMG_0001')]
        self.assertEqual(names, ['IMG_0001_1.tif', 'IMG_0001_2.tif', 'IMG_0001_10.tif'])    # 10 after 2, not after 1

    def test_the_extension_is_not_case_sensitive(self):
        self.touch('DJI_0001_1.TIF', 'DJI_0001_2.Tif')
        self.assertEqual(rct.list_capture_names(self.folder.name), ['DJI_0001'])

    def test_another_naming_scheme(self):
        self.touch('DJI_0001_MS_G.TIF', 'DJI_0001_MS_R.TIF', 'DJI_0002_MS_G.TIF', 'DJI_0001.JPG')
        pattern = r'(?P<capture>DJI_\d+)_MS_(?P<band>[A-Z]+)\.tif$'
        self.assertEqual(rct.list_capture_names(self.folder.name, pattern), ['DJI_0001', 'DJI_0002'])
        names = [Path(path).name for path in rct.find_capture_paths(self.folder.name, 'DJI_0001', pattern)]
        self.assertEqual(names, ['DJI_0001_MS_G.TIF', 'DJI_0001_MS_R.TIF'])             # no index: by name

    def test_a_missing_capture_names_the_pattern(self):
        self.touch('IMG_0001_1.tif')
        with self.assertRaisesRegex(FileNotFoundError, "'IMG_9'.*pattern"):
            rct.find_capture_paths(self.folder.name, 'IMG_9')

    def test_invalid_patterns_are_reported(self):
        with self.assertRaisesRegex(ValueError, 'not a valid regular expression'):
            rct.compile_file_pattern('(unclosed')
        with self.assertRaisesRegex(ValueError, "group named 'capture'"):
            rct.compile_file_pattern(r'(.+)_(\d+)\.tif$')
        self.assertIsNotNone(rct.compile_file_pattern(None))


class IgnoredBandsTests(unittest.TestCase):

    def test_ignored_bands_are_left_out_of_the_mapping_and_the_checks(self):
        metadata = [fake_metadata('Blue'), fake_metadata('LWIR', 'OTHER-CAPTURE-ID'), fake_metadata('Green')]
        mapping = rct.map_band_files(['a_1.tif', 'a_2.tif', 'a_3.tif'], metadata, ['Blue', 'Green'], ignore_bands=['LWIR'])
        self.assertEqual(list(mapping), ['Blue', 'Green'])

    def test_an_extra_band_that_is_not_ignored_is_still_an_error(self):
        metadata = [fake_metadata('Blue'), fake_metadata('LWIR')]
        with self.assertRaisesRegex(rct.MetadataError, "unexpected: \\['LWIR'\\]"):
            rct.map_band_files(['a_1.tif', 'a_2.tif'], metadata, ['Blue'])


class BandDefinitionTests(unittest.TestCase):

    @staticmethod
    def metadata(name, center, width, rig, **extra):
        return dict({'XMP:BandName': name, 'XMP:CentralWavelength': center, 'XMP:WavelengthFWHM': width,
                     'XMP:RigCameraIndex': rig, 'EXIF:Make': 'MicaSense', 'EXIF:Model': 'Altum',
                     'EXIF:SerialNumber': 'AL06-1'}, **extra)

    def read(self, metadata_list):
        paths = [f'f_{i}.tif' for i in range(len(metadata_list))]
        with mock.patch.object(rct, '_read_metadata', return_value=metadata_list):
            return rct.read_band_definitions(paths)

    def test_bands_come_in_the_order_of_the_rig_with_their_windows(self):
        result = self.read([self.metadata('NIR', 842, 57, 3), self.metadata('Blue', 475, 32, 0),
                            self.metadata('Green', 560, 27, 1)])
        self.assertEqual([band['name'] for band in result['bands']], ['Blue', 'Green', 'NIR'])
        self.assertEqual(result['bands'][2]['center_nm'], 842.0)
        self.assertEqual(result['bands'][2]['bandwidth_nm'], 57.0)
        self.assertEqual((result['camera'], result['serial']), ('MicaSense Altum', 'AL06-1'))

    def test_a_thermal_band_is_suggested_to_be_left_out(self):
        result = self.read([self.metadata('Blue', 475, 32, 0), self.metadata('LWIR', 11000, 6000, 5)])
        self.assertEqual([band['thermal'] for band in result['bands']], [False, True])
        result = self.read([self.metadata('Thermal', 10, 1, 0)])
        self.assertTrue(result['bands'][0]['thermal'])

    def test_missing_fields_and_duplicate_names_are_reported(self):
        broken = self.metadata('Blue', 475, 32, 0)
        del broken['XMP:WavelengthFWHM']
        with self.assertRaisesRegex(rct.MetadataError, "XMP:WavelengthFWHM"):
            self.read([broken])
        with self.assertRaisesRegex(rct.MetadataError, 'more than one file'):
            self.read([self.metadata('Blue', 475, 32, 0), self.metadata('Blue', 475, 32, 1)])

    def test_files_without_a_rig_index_come_after_the_others_by_name(self):
        result = self.read([self.metadata('B', 500, 10, None), self.metadata('A', 400, 10, None),
                            self.metadata('C', 600, 10, 0)])
        self.assertEqual([band['name'] for band in result['bands']], ['C', 'A', 'B'])


class ValidatePolygonTests(unittest.TestCase):

    SIZE = (100, 80)

    def test_a_simple_polygon_is_accepted(self):
        rct.validate_polygon([[10, 10], [60, 10], [60, 50], [10, 50]], self.SIZE)
        rct.validate_polygon([[10, 10], [60, 10], [30, 50]], self.SIZE)

    def test_reasons_for_rejection(self):
        cases = [([[10, 10], [20, 20]], 'at least 3 vertices'),
                 ([[10, 10], [20, 20], [30, 30]], 'no area'),
                 ([[10, 10], [60, 10], [60, 90]], 'not inside'),
                 ([[-1, 10], [60, 10], [60, 50]], 'not inside'),
                 ([[10, 10], [60, 50], [60, 10], [10, 50]], 'cross'),
                 ([[10, 10], [float('nan'), 10], [60, 50]], 'non-finite')]
        for points, message in cases:
            with self.assertRaisesRegex(ValueError, message):
                rct.validate_polygon(points, self.SIZE, 'region')

    def test_the_name_is_in_the_message(self):
        with self.assertRaisesRegex(ValueError, 'IMG_0005_1.tif: a region needs'):
            rct.validate_polygon([[1, 1]], self.SIZE, 'IMG_0005_1.tif')


class IrradianceChangesTests(unittest.TestCase):

    def test_stable_irradiance_has_no_events(self):
        self.assertEqual(rct.classify_irradiance_changes([100, 101, 99, 100, 102]), [])

    def test_single_capture_jump_is_a_spike(self):
        events = rct.classify_irradiance_changes([100, 101, 140, 100, 99])
        self.assertEqual([(e['index'], e['kind']) for e in events], [(2, 'spike')])
        self.assertAlmostEqual(events[0]['change'], 140 / 101 - 1)

    def test_persistent_change_is_a_step(self):
        events = rct.classify_irradiance_changes([100, 101, 60, 59, 58])
        self.assertEqual([(e['index'], e['kind']) for e in events], [(2, 'step')])

    def test_spike_and_step_together_and_custom_threshold(self):
        events = rct.classify_irradiance_changes([100, 101, 100, 140, 100, 99, 60, 59, 58])
        self.assertEqual([(e['index'], e['kind']) for e in events], [(3, 'spike'), (6, 'step')])
        self.assertEqual(rct.classify_irradiance_changes([100, 105, 100], threshold=0.03)[0]['kind'], 'spike')

    def test_change_at_the_last_capture_is_a_step(self):
        self.assertEqual([e['kind'] for e in rct.classify_irradiance_changes([100, 100, 50])], ['step'])

    def test_invalid_irradiance_is_rejected(self):
        for bad in ([100, 0, 100], [100, -1], [100, float('nan')]):
            with self.assertRaises(ValueError):
                rct.classify_irradiance_changes(bad)


class ComparePanelCapturesTests(unittest.TestCase):

    @staticmethod
    def fake_panel(factor, irradiance, estimate):
        return SimpleNamespace(panel_audit={'radiance_to_reflectance_factor': factor, 'panel_irradiance_estimate': estimate},
                               dls_irradiance=lambda: irradiance)

    def test_identical_captures_have_no_flags(self):
        before = {'Blue': self.fake_panel(2.0, 100.0, 1.0)}
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            result = rct.compare_panel_captures(before, before)
        self.assertEqual(result['flags'], [])
        self.assertEqual(result['bands']['Blue']['factor_ratio'], 1.0)

    def test_drift_of_the_factor_is_flagged(self):
        before = {'Blue': self.fake_panel(2.0, 100.0, 1.0)}
        after = {'Blue': self.fake_panel(2.2, 90.0, 1 / 1.1)}   # 10 % less light: the factor rises, the camera estimate and the DLS fall
        with self.assertWarns(RuntimeWarning):
            result = rct.compare_panel_captures(before, after)
        self.assertEqual(len(result['flags']), 1)
        self.assertIn('factor changed 10.0 %', result['flags'][0])
        self.assertAlmostEqual(result['bands']['Blue']['dls_over_panel_estimate_ratio'], 0.9 * 1.1)

    def test_dls_not_following_the_camera_is_flagged(self):
        before = {'Blue': self.fake_panel(2.0, 100.0, 1.0)}
        after = {'Blue': self.fake_panel(2.0, 130.0, 1.0)}  # camera saw the same light, the DLS did not
        with self.assertWarns(RuntimeWarning):
            result = rct.compare_panel_captures(before, after)
        self.assertEqual(len(result['flags']), 1)
        self.assertIn('DLS irradiance / panel irradiance estimate changed 30.0 %', result['flags'][0])

    def test_missing_band_is_flagged(self):
        before = {'Blue': self.fake_panel(2.0, 100.0, 1.0)}
        with self.assertWarns(RuntimeWarning):
            result = rct.compare_panel_captures(before, {})
        self.assertIn("'Blue' is missing", result['flags'][0])


class ExampleImageTests(unittest.TestCase):
    """ Use the example images to check the real metadata """

    @classmethod
    def setUpClass(cls):
        try:
            import exiftool
        except ImportError:
            raise unittest.SkipTest('pyexiftool is not installed')
        cls.exiftool_path = os.environ.get('exiftoolpath')
        cls.paths = rct.find_capture_paths(RAW, 'IMG_0010')
        cls.band_files = rct.read_band_files(cls.paths, BANDS, cls.exiftool_path)

    def test_example_bands_do_not_follow_the_readme_dictionary_order(self):
        # file _3 is Red and _6 is Panchro, so the file index cannot be used as band index
        self.assertTrue(self.band_files['Red'][0].endswith('IMG_0010_3.tif'))
        self.assertTrue(self.band_files['Panchro'][0].endswith('IMG_0010_6.tif'))
        self.assertEqual(list(self.band_files), ['Blue', 'Green', 'Red', 'NIR', 'Red edge', 'Panchro'])

    def test_valid_metadata_builds_an_image(self):
        path, metadata = self.band_files['Blue']
        image = rct.Image(path, metadata)
        self.assertEqual(image.band_name, 'Blue')
        self.assertEqual(len(image.distortion_parameters), 5)

    def test_missing_field_names_file_and_field(self):
        path, metadata = self.band_files['Blue']
        metadata = {key: val for key, val in metadata.items() if key != 'EXIF:BlackLevel'}
        with self.assertRaisesRegex(rct.MetadataError, "IMG_0010_1.tif.*'EXIF:BlackLevel'"):
            rct.Image(path, metadata)

    def test_nonpositive_exposure_is_rejected(self):
        path, metadata = self.band_files['Blue']
        with self.assertRaisesRegex(rct.MetadataError, "'EXIF:ExposureTime' must be positive"):
            rct.Image(path, dict(metadata, **{'EXIF:ExposureTime': 0}))

    def test_wrong_number_of_calibration_coefficients_is_rejected(self):
        path, metadata = self.band_files['Blue']
        with self.assertRaisesRegex(rct.MetadataError, "'XMP:RadiometricCalibration' has 2 values"):
            rct.Image(path, dict(metadata, **{'XMP:RadiometricCalibration': [1.0, 2.0]}))

    def test_non_numeric_field_is_rejected(self):
        path, metadata = self.band_files['Blue']
        with self.assertRaisesRegex(rct.MetadataError, "'EXIF:ISOSpeed' is not numeric"):
            rct.Image(path, dict(metadata, **{'EXIF:ISOSpeed': 'abc'}))

    def test_panel_detection_records_an_audit(self):
        panel_files = rct.read_band_files(rct.find_capture_paths(RAW, 'IMG_0005'), BANDS, self.exiftool_path)
        path, metadata = panel_files['Blue']
        panel = rct.Image(path, metadata)
        _, factor, _, _ = panel.detect_reference_panel(REFLECTANCE_LEVELS, percent_inner_area=0.7, sat_threshold=65000)
        audit = panel.panel_audit
        self.assertEqual(audit['band_name'], 'Blue')
        self.assertEqual(audit['radiance_to_reflectance_factor'], factor)
        self.assertAlmostEqual(audit['reflectance_level'] / audit['radiance_mean'], factor)
        self.assertEqual(audit['n_saturated'], 0)
        self.assertGreater(audit['n_pixels'], 0)
        # the original code (panel region measured on the distorted image) gave 2.03712; sampling the region on the
        # undistorted image, where the QR code is found, changes it by about -0.3 %
        self.assertAlmostEqual(factor, 2.0371197142446085, delta=0.01 * 2.0371197142446085)
        self.assertLess(factor, 2.0371197142446085)
        self.assertLess(audit['radiance_std'] / audit['radiance_mean'], 0.03)

    def test_panel_detection_without_qr_code_is_a_clear_error(self):
        path, metadata = self.band_files['Blue']        # target capture, no reference panel
        with self.assertRaisesRegex(ValueError, 'IMG_0010_1.tif: no QR code found'):
            rct.Image(path, metadata).detect_reference_panel(REFLECTANCE_LEVELS, 0.7)

    def test_panel_band_without_reflectance_level_is_a_clear_error(self):
        panel_files = rct.read_band_files(rct.find_capture_paths(RAW, 'IMG_0005'), BANDS, self.exiftool_path)
        path, metadata = panel_files['Blue']
        levels = {band: val for band, val in REFLECTANCE_LEVELS.items() if band != 'Blue'}
        with self.assertRaisesRegex(KeyError, "band 'Blue' not in the reference panel"):
            rct.Image(path, metadata).detect_reference_panel(levels, 0.7)

    # --- DLS correction ---------------------------------------------------------------------------------------

    def panel_and_target(self, band='Blue', **target_changes):
        panel_files = rct.read_band_files(rct.find_capture_paths(RAW, 'IMG_0005'), BANDS, self.exiftool_path)
        panel = rct.Image(*panel_files[band])
        factor = panel.detect_reference_panel(REFLECTANCE_LEVELS, 0.7)[1]
        path, metadata = self.band_files[band]
        target = rct.Image(path, dict(metadata, **target_changes))
        return panel, target, factor

    def test_dls_ratio_is_irradiance_at_panel_over_irradiance_at_image(self):
        panel, target, _ = self.panel_and_target('Blue')
        self.assertAlmostEqual(rct.dls_irradiance_ratio(panel, target), 138.05218450296033 / 156.04327208007237)

    def test_dls_reproduces_panel_only_result_at_the_panel_capture(self):
        panel, _, factor = self.panel_and_target('Blue')
        panel_only = panel.digital_numbers_to_reflectance(factor)
        with_dls = panel.digital_numbers_to_reflectance(factor, panel_image=panel)
        self.assertEqual(panel.dls_ratio, 1.0)
        self.assertTrue((panel_only == with_dls).all())

    def test_panel_only_mode_is_the_original_conversion(self):
        _, target, factor = self.panel_and_target('Blue')
        self.assertTrue((target.digital_numbers_to_reflectance(factor) ==
                         target.digital_numbers_to_radiance() * factor).all())
        self.assertIsNone(target.dls_ratio)

    def test_doubling_the_image_irradiance_halves_the_reflectance(self):
        panel, target, factor = self.panel_and_target('Blue')
        _, doubled, _ = self.panel_and_target('Blue', **{'XMP:HorizontalIrradiance': 2 * target.horizontal_irradiance})
        stable = target.digital_numbers_to_reflectance(factor, panel_image=panel)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)   # ratio 0.44 is outside the plausible range
            brighter = doubled.digital_numbers_to_reflectance(factor, panel_image=panel)
        self.assertTrue(abs(brighter - stable / 2).max() < 1e-12)

    def test_common_irradiance_scale_cancels_in_the_ratio(self):
        panel, target, _ = self.panel_and_target('Blue')
        scale = {'XMP:IrradianceScaleToSIUnits': 0.01}
        panel_scaled = rct.Image(panel.image_path, dict(panel.metadata, **scale))
        target_scaled = rct.Image(target.image_path, dict(target.metadata, **scale))
        self.assertAlmostEqual(rct.dls_irradiance_ratio(panel_scaled, target_scaled),
                               rct.dls_irradiance_ratio(panel, target))

    def test_scale_present_in_only_one_image_is_rejected(self):
        panel, target, _ = self.panel_and_target('Blue', **{'XMP:IrradianceScaleToSIUnits': 0.01})
        with self.assertRaisesRegex(rct.MetadataError, 'exists in only one'):
            rct.dls_irradiance_ratio(panel, target)

    def test_missing_nonpositive_or_nonnumeric_irradiance_is_rejected(self):
        panel, target, _ = self.panel_and_target('Blue')
        for value, message in [(None, 'missing metadata field'), (0, 'must be positive'),
                               (-3.0, 'must be positive'), (float('nan'), 'no finite values'),
                               ('abc', 'not numeric')]:
            bad = rct.Image(target.image_path, dict(target.metadata, **{'XMP:HorizontalIrradiance': value}))
            with self.assertRaisesRegex(rct.MetadataError, message):
                rct.dls_irradiance_ratio(panel, bad)

    def test_dls_without_irradiance_does_not_affect_panel_only_mode(self):
        metadata = {key: val for key, val in self.band_files['Blue'][1].items() if 'Irradiance' not in key}
        image = rct.Image(self.band_files['Blue'][0], metadata)
        self.assertIsNotNone(image.digital_numbers_to_reflectance(2.0))

    def test_different_bands_are_rejected(self):
        panel, _, _ = self.panel_and_target('Blue')
        green = rct.Image(*self.band_files['Green'])
        with self.assertRaisesRegex(rct.MetadataError, 'needs the same band'):
            rct.dls_irradiance_ratio(panel, green)

    # --- Panchro, checks of the DLS data ----------------------------------------------------------------------

    @classmethod
    def capture_dicts(cls):
        """ Images of both captures for all the bands, with the panel detected (done once) """
        if not hasattr(cls, '_P'):
            panel_files = rct.read_band_files(rct.find_capture_paths(RAW, 'IMG_0005'), BANDS, cls.exiftool_path)
            cls._P = {band: rct.Image(*panel_files[band]) for band in BANDS}
            for panel in cls._P.values():
                panel.detect_reference_panel(REFLECTANCE_LEVELS, 0.7)
            cls._T = {band: rct.Image(*cls.band_files[band]) for band in BANDS}
        return cls._P, cls._T

    @staticmethod
    def changed(image, **changes):
        return rct.Image(image.image_path, dict(image.metadata, **changes))

    def test_panchro_uses_its_own_irradiance_like_the_other_bands(self):
        panel, target = self.capture_dicts()
        expected = 99.211212701455025 / 112.20932502581493
        self.assertAlmostEqual(rct.dls_irradiance_ratio(panel['Panchro'], target['Panchro']), expected)
        target['Panchro'].digital_numbers_to_reflectance(
            panel['Panchro'].panel_audit['radiance_to_reflectance_factor'], panel_image=panel['Panchro'])
        self.assertAlmostEqual(target['Panchro'].dls_ratio, expected)
        self.assertEqual(target['Panchro'].dls_ratio_source, 'metadata')

    def test_capture_ratios_use_the_metadata_of_every_band(self):
        panel, target = self.capture_dicts()
        ratios = rct.capture_dls_ratios(panel, target)
        self.assertEqual(set(ratios), set(BANDS))
        self.assertTrue(all(source == 'metadata' for _, source in ratios.values()))

    def test_missing_irradiance_is_an_error_in_every_band_including_panchro(self):
        panel, target = self.capture_dicts()
        for band in BANDS:
            broken = dict(target, **{band: self.changed(target[band], **{'XMP:HorizontalIrradiance': None})})
            with self.assertRaisesRegex(rct.MetadataError, "XMP:HorizontalIrradiance"):
                rct.capture_dls_ratios(panel, broken)

    def test_invalid_irradiance_is_an_error_in_panchro(self):
        panel, target = self.capture_dicts()
        for value in (0, -5.0, float('nan'), 'abc'):
            broken = dict(target, Panchro=self.changed(target['Panchro'], **{'XMP:HorizontalIrradiance': value}))
            with self.assertRaises(rct.MetadataError):
                rct.capture_dls_ratios(panel, broken)

    def test_ratio_outside_the_plausible_range_gives_a_warning(self):
        panel, target = self.capture_dicts()
        bright = self.changed(target['Blue'], **{'XMP:HorizontalIrradiance': 10 * target['Blue'].horizontal_irradiance})
        with self.assertWarnsRegex(RuntimeWarning, 'outside the plausible range'):
            rct.dls_irradiance_ratio(panel['Blue'], bright)

    def test_example_capture_bands_agree_and_the_image_dls_is_flagged_for_its_sun_angle(self):
        panel, target = self.capture_dicts()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            checks = rct.check_dls_capture(panel, target)
        self.assertEqual(checks['unavailable'], [])
        self.assertTrue(all(abs(dep) < 0.01 for dep in checks['band_ratio_departures'].values()))
        self.assertLess(checks['sun_angle_discrepancy_deg']['panel'], 1)
        self.assertGreater(checks['sun_angle_discrepancy_deg']['image'], 10)
        self.assertEqual(len(checks['flags']), 1)
        self.assertIn('sun angle', checks['flags'][0])
        self.assertEqual(len(caught), 1)
        # the irradiance estimated from the panel is reported, not judged
        self.assertEqual(set(checks['dls_over_panel_estimate']), set(BANDS))

    def test_a_band_that_departs_from_the_others_is_flagged(self):
        panel, target = self.capture_dicts()
        odd = self.changed(target['Green'], **{'XMP:HorizontalIrradiance': 1.1 * target['Green'].horizontal_irradiance})
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            checks = rct.check_dls_capture(panel, dict(target, Green=odd))
        self.assertTrue(any("Band 'Green'" in flag and 'median' in flag for flag in checks['flags']))

    def test_sun_angle_discrepancy_is_none_without_the_fields(self):
        _, target = self.capture_dicts()
        metadata = {key: val for key, val in target['Blue'].metadata.items() if key != 'XMP:SolarElevation'}
        self.assertIsNone(rct.Image(target['Blue'].image_path, metadata).dls_sun_angle_discrepancy_deg())

    def test_checks_are_recorded_in_the_provenance(self):
        import json, tempfile
        panel, target = self.capture_dicts()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            checks = rct.check_dls_capture(panel, target)
        record = rct.band_record('out/x.tif', panel['Blue'], target['Blue'], target['Blue'].digital_numbers_to_radiance())
        self.assertIn('sun_angle_discrepancy_deg', record['dls'])
        self.assertEqual(set(record['dls']['attitude_as_written']), {'Roll', 'Pitch', 'Yaw'})
        with tempfile.TemporaryDirectory() as folder:
            saved = json.loads(rct.write_provenance(Path(folder) / 'p.json', [record], checks=checks).read_text(encoding='utf-8'))
        self.assertEqual(saved['checks']['flags'], checks['flags'])
        self.assertEqual(saved['dls_checks_limits']['ratio_range'], [0.5, 2.0])

    # --- Provenance record ------------------------------------------------------------------------------------

    def test_provenance_of_a_panel_only_run(self):
        panel, target, factor = self.panel_and_target('Blue')
        reflectance = target.digital_numbers_to_reflectance(factor)
        record = rct.band_record('out/IMG_0010_1.tif', panel, target, reflectance)
        self.assertEqual(record['band'], 'Blue')
        self.assertTrue(record['panel_file'].endswith('IMG_0005_1.tif'))
        self.assertTrue(record['image_file'].endswith('IMG_0010_1.tif'))
        self.assertEqual(record['output_dtype'], 'float64')
        self.assertFalse(record['dls']['used'])
        self.assertIsNone(record['dls']['ratio_panel_over_image'])
        # the irradiances are recorded even when they are not used
        self.assertAlmostEqual(record['dls']['image_horizontal_irradiance'], 156.04327208007237)
        self.assertEqual(record['panel']['radiance_to_reflectance_factor'], factor)

    def test_provenance_of_a_dls_run_is_saved_as_json(self):
        import json, tempfile
        panel, target, factor = self.panel_and_target('Blue')
        reflectance = target.digital_numbers_to_reflectance(factor, panel_image=panel)
        record = rct.band_record('out/IMG_0010_1.tif', panel, target, reflectance)
        self.assertTrue(record['dls']['used'])
        self.assertAlmostEqual(record['dls']['ratio_panel_over_image'], 138.05218450296033 / 156.04327208007237)
        with tempfile.TemporaryDirectory() as folder:
            path = rct.write_provenance(Path(folder) / 'provenance.json', [record],
                                        parameters={'percent_inner_area': 0.7, 'levels': REFLECTANCE_LEVELS})
            saved = json.loads(path.read_text(encoding='utf-8'))
        self.assertEqual(saved['method'], 'panel_and_dls')
        self.assertEqual(saved['parameters']['percent_inner_area'], 0.7)
        self.assertEqual(len(saved['software']['radiometric_correction_tools_sha256']), 64)
        self.assertEqual(saved['bands'][0]['panel']['panel_bounds'], record['panel']['panel_bounds'])

    def test_method_is_panel_only_without_dls(self):
        import json, tempfile
        panel, target, factor = self.panel_and_target('Blue')
        reflectance = target.digital_numbers_to_reflectance(factor)
        record = rct.band_record('out/IMG_0010_1.tif', panel, target, reflectance)
        with tempfile.TemporaryDirectory() as folder:
            saved = json.loads(rct.write_provenance(Path(folder) / 'p.json', [record]).read_text(encoding='utf-8'))
        self.assertEqual(saved['method'], 'panel_only')

    # --- Reference panel region drawn by hand -----------------------------------------------------------------

    def panel_image(self, band='Blue'):
        panel_files = rct.read_band_files(rct.find_capture_paths(RAW, 'IMG_0005'), BANDS, self.exiftool_path)
        return rct.Image(*panel_files[band])

    def test_a_manual_region_equal_to_the_automatic_one_gives_the_same_factor(self):
        automatic = self.panel_image()
        bounds, factor, barcode_data, source = automatic.detect_reference_panel(REFLECTANCE_LEVELS, 0.7)
        self.assertEqual(automatic.panel_audit['region_source'], 'automatic')
        manual = self.panel_image()
        m_bounds, m_factor, m_data, m_source = manual.detect_reference_panel(REFLECTANCE_LEVELS, 0.7, manual_bounds=bounds.tolist())
        self.assertEqual(m_factor, factor)
        self.assertEqual((m_data, m_source), (None, 'manual'))
        self.assertEqual(manual.panel_audit['region_source'], 'manual')
        self.assertEqual(manual.panel_audit['geometry'], 'undistorted')
        self.assertEqual(manual.panel_audit['n_pixels'], automatic.panel_audit['n_pixels'])

    def test_a_manual_region_needs_no_qr_code(self):
        _, target = self.capture_dicts()
        image = rct.Image(target['Blue'].image_path, target['Blue'].metadata)
        self.assertIsNone(image.locate_reference_panel(0.7))
        _, factor, _, source = image.detect_reference_panel(REFLECTANCE_LEVELS, 0.7,
                                                              manual_bounds=[[600, 300], [700, 300], [700, 400], [600, 400]])
        self.assertEqual(source, 'manual')
        self.assertGreater(factor, 0)

    def test_an_invalid_manual_region_is_rejected(self):
        for bounds, message in (([[10, 10], [20, 20]], 'at least 3 vertices'),
                                ([[10, 10], [5000, 10], [5000, 50]], 'not inside'),
                                ([[10, 10], [60, 50], [60, 10], [10, 50]], 'cross')):
            with self.assertRaisesRegex(ValueError, message):
                self.panel_image().detect_reference_panel(REFLECTANCE_LEVELS, 0.7, manual_bounds=bounds)

    def test_locate_reference_panel_returns_the_region_without_calibrating(self):
        image = self.panel_image()
        bounds, barcode_data, geometry = image.locate_reference_panel(0.7)
        self.assertEqual(geometry, 'undistorted')
        self.assertEqual(bounds.shape, (4, 2))
        self.assertFalse(hasattr(image, 'panel_audit'))

    def test_region_statistics_and_display_image(self):
        image = self.panel_image()
        bounds, _, _ = image.locate_reference_panel(0.7)
        stats = image.panel_region_stats(bounds)
        self.assertEqual(stats['n_saturated'], 0)
        self.assertLess(stats['variation_percent'], 5)
        outside = image.panel_region_stats([[0, 0], [3, 0], [3, 3]])   # corner of the undistorted image: black border
        self.assertTrue(outside is None or outside['variation_percent'] >= 0)
        shown = image.display_image()
        self.assertEqual(shown.dtype, np.uint8)
        self.assertEqual(shown.shape, image.image.shape)
        self.assertGreater(shown.max(), 200)

    def test_the_bands_of_the_example_capture_are_read_from_its_metadata(self):
        result = rct.read_band_definitions(rct.find_capture_paths(RAW, 'IMG_0010'), self.exiftool_path)
        self.assertEqual(result['camera'], 'MicaSense RedEdge-P')
        table = [(band['name'], band['center_nm'], band['bandwidth_nm']) for band in result['bands']]
        self.assertEqual(table, [('Blue', 475.0, 32.0), ('Green', 560.0, 27.0), ('Red', 668.0, 14.0),
                                 ('NIR', 842.0, 57.0), ('Red edge', 717.0, 12.0), ('Panchro', 634.5, 463.0)])
        self.assertFalse(any(band['thermal'] for band in result['bands']))
        self.assertTrue(result['bands'][0]['file'].endswith('IMG_0010_1.tif'))

    def test_missing_capture_is_reported(self):
        with self.assertRaises(FileNotFoundError):
            rct.find_capture_paths(RAW, 'IMG_9999')


if __name__ == '__main__':
    unittest.main()
