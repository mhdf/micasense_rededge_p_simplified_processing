"""
Tests for capture_picker.py: the reading of a picture without a display, the window with a hidden Tk root
(skipped if Tk cannot open one)

Run from the project folder with:  python -m unittest tests.test_capture_picker -v
"""

import time
import unittest
from pathlib import Path
from unittest import mock

import radiometric_correction_tools as rct
from radiometric_app import capture_picker as cp

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / 'images' / 'raw'


def paths_of(name):
    return rct.find_capture_paths(RAW, name)


class ThumbnailTests(unittest.TestCase):

    def test_a_raw_image_becomes_a_small_stretched_picture(self):
        picture = cp.thumbnail_image(paths_of('IMG_0005')[0])
        self.assertEqual(picture.mode, 'L')
        self.assertLessEqual(picture.width, cp.THUMBNAIL_SIZE[0])
        self.assertLessEqual(picture.height, cp.THUMBNAIL_SIZE[1])
        self.assertIn(max(picture.size), cp.THUMBNAIL_SIZE)                     # it fills the size in one direction
        low, high = picture.getextrema()
        self.assertGreater(high - low, 100)                                       # stretched: not a dark picture

    def test_the_size_can_be_chosen(self):
        picture = cp.thumbnail_image(paths_of('IMG_0010')[0], (60, 40))
        self.assertLessEqual(picture.width, 60)
        self.assertLessEqual(picture.height, 40)

    def test_a_flat_image_does_not_fail(self):
        import tempfile
        import numpy as np
        import tifffile
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'flat.tif'
            tifffile.imwrite(path, np.full((100, 120), 500, dtype=np.uint16))
            picture = cp.thumbnail_image(path)
        self.assertEqual(picture.getextrema(), (0, 0))

    def test_a_file_that_is_not_an_image_is_an_error(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'notes.tif'
            path.write_text('not an image')
            with self.assertRaises(Exception):
                cp.thumbnail_image(path)


class WindowTests(unittest.TestCase):

    def setUp(self):
        import tkinter as tk
        try:
            self.root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"no display for Tk: {error}")
        self.root.withdraw()

    def tearDown(self):
        try:
            self.root.destroy()
        except Exception:
            pass

    def open(self, **kwargs):
        import tkinter as tk
        window = tk.Toplevel(self.root)
        options = dict(names=['IMG_0005', 'IMG_0010'], paths_of=paths_of)
        options.update(kwargs)
        self.picker = cp.CapturePicker(window, **options)
        return self.picker

    def pump(self, condition, timeout=30):
        end = time.time() + timeout
        while not condition() and time.time() < end:
            self.root.update()
            time.sleep(0.03)
        self.assertTrue(condition(), 'timed out')

    def test_the_pictures_are_read_in_the_background_and_shown(self):
        picker = self.open()
        self.pump(lambda: len(picker.photos) == 2)
        for name in picker.names:
            self.assertIsNotNone(picker.tiles[name]['picture'].cget('image'))
        self.assertEqual(picker.tiles['IMG_0005']['picture'].cget('text'), '')
        self.pump(lambda: 'Click' in picker.status.cget('text'))

    def test_one_capture_is_chosen_by_clicking_its_picture(self):
        picker = self.open(selected=['IMG_0005'])
        self.assertEqual(picker.selection, ['IMG_0005'])
        picker.toggle('IMG_0010')
        self.assertEqual(picker.selection, ['IMG_0010'])
        picker.accept()
        self.assertEqual(picker.result, 'IMG_0010')
        self.assertFalse(picker.cancelled)

    def test_nothing_chosen_cannot_be_accepted_unless_there_is_a_choice_of_no_capture(self):
        picker = self.open()
        self.assertTrue(picker.ok_button.instate(['disabled']))
        picker.accept()
        self.assertIsNone(picker.result)
        self.assertTrue(picker.cancelled)                                          # the window is still open: not yet closed
        self.assertFalse(picker.closed)

    def test_the_choice_of_no_capture(self):
        picker = self.open(selected=['IMG_0005'], none_label='No second panel capture')
        self.assertIn('', picker.tiles)
        self.assertEqual(picker.tiles[''] ['picture'].cget('text'), 'No second panel capture')
        picker.toggle('')
        self.assertEqual(picker.selection, [])
        self.assertFalse(picker.ok_button.instate(['disabled']))
        picker.accept()
        self.assertEqual(picker.result, '')

    def test_several_captures_are_chosen_and_come_back_in_the_order_of_the_list(self):
        picker = self.open(names=['IMG_0005', 'IMG_0010'], multiple=True)
        picker.toggle('IMG_0010')
        picker.toggle('IMG_0005')
        self.assertEqual(picker.selection, ['IMG_0005', 'IMG_0010'])
        picker.toggle('IMG_0005')
        self.assertEqual(picker.selection, ['IMG_0010'])
        self.assertEqual(picker.count_label.cget('text'), '1 of 2 selected')
        picker.select_all()
        self.assertEqual(picker.count_label.cget('text'), '2 of 2 selected')
        picker.clear()
        self.assertEqual(picker.selection, [])
        picker.toggle('IMG_0010')
        picker.accept()
        self.assertEqual(picker.result, ['IMG_0010'])

    def test_cancel_returns_nothing(self):
        picker = self.open(selected=['IMG_0005'])
        picker.cancel()
        self.assertTrue(picker.cancelled)
        self.assertIsNone(picker.result)

    def test_the_band_shown_is_the_position_that_the_resolver_gives_and_a_new_band_reads_the_pictures_again(self):
        asked = []

        def resolve(band):
            asked.append(band)
            return {'Blue': 0, 'Red': 3}[band]

        files = []

        def fake_thumbnail(path, size=cp.THUMBNAIL_SIZE):
            from PIL import Image
            files.append(Path(path).name)
            return Image.new('L', (40, 30), 100)

        with mock.patch.object(cp, 'thumbnail_image', fake_thumbnail):
            picker = self.open(names=['IMG_0005'], bands=['Blue', 'Red'], resolve_band=resolve)
            self.pump(lambda: len(files) == 1)
            self.assertEqual(files[-1], 'IMG_0005_1.tif')
            picker.band_var.set('Red')
            picker.load()
            self.pump(lambda: len(files) == 2)
            self.assertEqual(files[-1], 'IMG_0005_4.tif')
            self.assertEqual(asked, ['Blue', 'Red'])

    def test_a_band_that_cannot_be_found_falls_back_to_the_first_file_and_says_so(self):
        def resolve(band):
            raise ValueError('no metadata')

        with mock.patch.object(cp, 'thumbnail_image', lambda path, size=cp.THUMBNAIL_SIZE: __import__('PIL.Image').Image.new('L', (4, 3))):
            picker = self.open(names=['IMG_0005'], bands=['Blue'], resolve_band=resolve)
            self.pump(lambda: 'could not be found' in picker.status.cget('text'))
            self.assertIn('first file of each capture', picker.status.cget('text'))

    def test_a_picture_that_cannot_be_read_says_why_in_its_tile(self):
        def broken(path, size=cp.THUMBNAIL_SIZE):
            raise OSError('cannot read')

        with mock.patch.object(cp, 'thumbnail_image', broken):
            picker = self.open(names=['IMG_0005'])
            self.pump(lambda: 'cannot read' in picker.tiles['IMG_0005']['picture'].cget('text'))

    def test_closing_stops_the_polling(self):
        picker = self.open()
        picker.cancel()
        self.assertTrue(picker.closed)


if __name__ == '__main__':
    unittest.main()
