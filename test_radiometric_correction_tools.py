"""
Tests for the metadata validation and band-file mapping in radiometric_correction_tools.py

Run from this folder with:  python -m unittest test_radiometric_correction_tools -v
The tests that read the example images need ExifTool; set the environment variable
'exiftoolpath' if it is not on the PATH.
"""

import os
import unittest
from pathlib import Path

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

    def test_panchro_is_not_dls_corrected(self):
        panel_files = rct.read_band_files(rct.find_capture_paths(RAW, 'IMG_0005'), BANDS, self.exiftool_path)
        with self.assertRaisesRegex(ValueError, "not defined for the 'Panchro' band"):
            rct.dls_irradiance_ratio(rct.Image(*panel_files['Panchro']), rct.Image(*self.band_files['Panchro']))

    def test_missing_capture_is_reported(self):
        with self.assertRaises(FileNotFoundError):
            rct.find_capture_paths(RAW, 'IMG_9999')


if __name__ == '__main__':
    unittest.main()
