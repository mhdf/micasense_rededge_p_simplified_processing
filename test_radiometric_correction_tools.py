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
        # factor obtained with the original code on the same images
        self.assertAlmostEqual(factor, 2.0371197142446085, places=9)

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

    def test_missing_capture_is_reported(self):
        with self.assertRaises(FileNotFoundError):
            rct.find_capture_paths(RAW, 'IMG_9999')


if __name__ == '__main__':
    unittest.main()
