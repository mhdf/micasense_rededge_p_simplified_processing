"""
Tests for camera_presets.py (no display needed)

Run from this folder with:  python -m unittest test_camera_presets -v
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

import camera_presets as cp
import radiometric_correction_tools as rct

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'images' / 'raw'


def valid_preset(**changes):
    data = {'label': 'Test camera', 'bands': [{'name': 'Blue', 'center_nm': 475, 'bandwidth_nm': 32},
                                              {'name': 'LWIR', 'center_nm': 11000, 'bandwidth_nm': 6000, 'process': False}]}
    data.update(changes)
    return data


class ValidatePresetTests(unittest.TestCase):

    def problems(self, data):
        return '\n'.join(cp.validate_preset(data, 'test.json'))

    def test_a_valid_preset_has_no_problems(self):
        self.assertEqual(cp.validate_preset(valid_preset(file_pattern=rct.CAPTURE_FILE_PATTERN, notes='x')), [])

    def test_problems_are_reported_with_the_band(self):
        self.assertIn('JSON object', self.problems([]))
        self.assertIn("'label' is missing", self.problems(valid_preset(label='')))
        self.assertIn("'bands' must be a list", self.problems(valid_preset(bands=[])))
        self.assertIn('needs a name', self.problems(valid_preset(bands=[{'center_nm': 1, 'bandwidth_nm': 1}])))
        bad = valid_preset(bands=[{'name': 'Blue', 'center_nm': 0, 'bandwidth_nm': 'wide'}])
        text = self.problems(bad)
        self.assertIn('Blue): center_nm must be a number above 0', text)
        self.assertIn('Blue): bandwidth_nm must be a number above 0', text)
        self.assertIn('process must be true or false', self.problems(valid_preset(
            bands=[{'name': 'B', 'center_nm': 1, 'bandwidth_nm': 1, 'process': 'no'}])))

    def test_duplicates_and_no_processed_band_are_reported(self):
        twice = valid_preset(bands=[{'name': 'B', 'center_nm': 1, 'bandwidth_nm': 1}] * 2)
        self.assertIn("band names used twice: ['B']", self.problems(twice))
        none = valid_preset(bands=[{'name': 'B', 'center_nm': 1, 'bandwidth_nm': 1, 'process': False}])
        self.assertIn('at least one band must be processed', self.problems(none))

    def test_a_bad_file_pattern_is_reported(self):
        self.assertIn('file_pattern', self.problems(valid_preset(file_pattern='(unclosed')))
        self.assertIn("group named 'capture'", self.problems(valid_preset(file_pattern=r'(.+)\.tif')))


class LoadPresetsTests(unittest.TestCase):

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.folder.cleanup()

    def write(self, folder, name, data):
        path = Path(folder) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data if isinstance(data, str) else json.dumps(data), encoding='utf-8')

    def test_presets_are_loaded_and_unusable_files_are_only_reported(self):
        self.write(self.folder.name, 'good.json', valid_preset())
        self.write(self.folder.name, 'broken.json', '{not json')
        self.write(self.folder.name, 'invalid.json', valid_preset(label=''))
        self.write(self.folder.name, 'notes.txt', 'ignored')
        presets, problems = cp.load_presets([self.folder.name])
        self.assertEqual(list(presets), ['good'])
        self.assertEqual(presets['good']['key'], 'good')
        self.assertTrue(presets['good']['source'].endswith('good.json'))
        self.assertEqual(len(problems), 2)
        self.assertTrue(any(problem.startswith('broken.json') for problem in problems))

    def test_a_later_folder_replaces_a_preset_with_the_same_file_name(self):
        self.write(Path(self.folder.name) / 'a', 'cam.json', valid_preset(label='From the program'))
        self.write(Path(self.folder.name) / 'b', 'cam.json', valid_preset(label='From the project'))
        self.write(Path(self.folder.name) / 'b', 'other.json', valid_preset(label='Other'))
        presets, _ = cp.load_presets([Path(self.folder.name) / 'a', Path(self.folder.name) / 'b',
                                      Path(self.folder.name) / 'missing'])
        self.assertEqual(presets['cam']['label'], 'From the project')
        self.assertEqual(sorted(presets), ['cam', 'other'])

    def test_the_folders_are_the_program_and_the_project(self):
        folders = cp.preset_folders(self.folder.name)
        self.assertEqual(folders[0], cp.BUILTIN_FOLDER)
        self.assertEqual(folders[1], Path(self.folder.name).resolve() / 'camera_presets')
        self.assertEqual(len(cp.preset_folders(ROOT)), 1)                  # the same folder is not searched twice


class BandRowsTests(unittest.TestCase):

    def test_rows_keep_the_typed_reflectance_of_bands_with_the_same_name(self):
        rows = cp.band_rows(valid_preset(), {'Blue': '0.477', 'Green': '0.478'})
        self.assertEqual(rows[0], {'name': 'Blue', 'center': '475', 'width': '32', 'reflectance': '0.477', 'process': True})
        self.assertEqual(rows[1], {'name': 'LWIR', 'center': '11000', 'width': '6000', 'reflectance': '', 'process': False})

    def test_without_typed_values_the_reflectance_is_empty(self):
        self.assertEqual(cp.band_rows(valid_preset())[0]['reflectance'], '')


class ShippedPresetsTests(unittest.TestCase):

    def setUp(self):
        self.presets, self.problems = cp.load_presets([cp.BUILTIN_FOLDER])

    def test_the_shipped_presets_are_valid(self):
        self.assertEqual(self.problems, [])
        self.assertEqual(sorted(self.presets), ['micasense_altum', 'micasense_rededge_mx', 'micasense_rededge_p'])

    def test_the_numbers_of_bands(self):
        counts = {key: len(preset['bands']) for key, preset in self.presets.items()}
        self.assertEqual(counts, {'micasense_rededge_p': 6, 'micasense_rededge_mx': 5, 'micasense_altum': 6})
        processed = {key: sum(band.get('process', True) for band in preset['bands']) for key, preset in self.presets.items()}
        self.assertEqual(processed, {'micasense_rededge_p': 6, 'micasense_rededge_mx': 5, 'micasense_altum': 5})

    def test_the_rededge_p_preset_agrees_with_the_example_images(self):
        try:
            detected = rct.read_band_definitions(rct.find_capture_paths(RAW, 'IMG_0010'), os.environ.get('exiftoolpath'))
        except ImportError:
            self.skipTest('pyexiftool is not installed')
        found = {band['name']: (band['center_nm'], band['bandwidth_nm']) for band in detected['bands']}
        preset = {band['name']: (float(band['center_nm']), float(band['bandwidth_nm']))
                  for band in self.presets['micasense_rededge_p']['bands']}
        self.assertEqual(preset, found)


if __name__ == '__main__':
    unittest.main()
