"""
Tests for panel_selector.py: the logic is tested without a display; the window is tested with a hidden Tk root
(skipped if Tk cannot open one)

Run from this folder with:  python -m unittest test_panel_selector -v
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import radiometric_correction_tools as rct
import panel_selector as ps

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'images' / 'raw'
BANDS = ['Blue', 'Green', 'Panchro', 'Red', 'Red edge', 'NIR']


class PolygonModelTests(unittest.TestCase):

    def test_vertices_are_appended_and_inserted_in_edges(self):
        model = ps.PolygonModel([[0, 0], [100, 0], [100, 100]])
        self.assertEqual(model.add([300, 300], 5), 3)                  # far from everything: appended
        self.assertEqual(model.add([50, 2], 5), 1)                     # near the edge 0 -> 1: inserted after vertex 0
        self.assertEqual(model.vertices[1], [50.0, 2.0])
        self.assertEqual(len(model.vertices), 5)

    def test_the_closing_edge_is_an_edge(self):
        model = ps.PolygonModel([[0, 0], [100, 0], [100, 100]])
        self.assertEqual(model.nearest_edge([49, 51], 3), 2)           # edge 2 -> 0 (the diagonal)
        self.assertIsNone(model.nearest_edge([500, 500], 3))

    def test_nearest_vertex_within_a_radius(self):
        model = ps.PolygonModel([[0, 0], [100, 0], [100, 100]])
        self.assertEqual(model.nearest_vertex([98, 3], 5), 1)
        self.assertIsNone(model.nearest_vertex([50, 50], 5))

    def test_move_remove_clear_and_integers(self):
        model = ps.PolygonModel([[0, 0], [10, 0], [10, 10]])
        model.move(2, (11.6, 9.4))
        self.assertEqual(model.as_integers(), [[0, 0], [10, 0], [12, 9]])
        model.remove(0)
        self.assertEqual(len(model.vertices), 2)
        model.remove_last()
        model.clear()
        model.remove_last()                                             # nothing to remove: no error
        self.assertEqual(model.vertices, [])

    def test_copy_is_independent(self):
        model = ps.PolygonModel([[0, 0], [10, 0], [10, 10]])
        other = model.copy()
        other.move(0, (5, 5))
        self.assertEqual(model.vertices[0], [0.0, 0.0])


class ViewTransformTests(unittest.TestCase):

    def test_fit_centers_the_image(self):
        view = ps.ViewTransform()
        view.fit((200, 100), (400, 400))
        self.assertEqual(view.scale, 2.0)
        self.assertEqual(view.to_canvas((0, 0)), (0.0, 100.0))
        self.assertEqual(view.to_canvas((200, 100)), (400.0, 300.0))

    def test_canvas_and_image_coordinates_are_inverse(self):
        view = ps.ViewTransform(2.5, (13, -7))
        for point in ((0, 0), (10.5, 20.25), (-4, 99)):
            back = view.to_image(view.to_canvas(point))
            self.assertAlmostEqual(back[0], point[0])
            self.assertAlmostEqual(back[1], point[1])

    def test_zoom_keeps_the_point_under_the_cursor(self):
        view = ps.ViewTransform()
        view.fit((200, 100), (400, 400))
        before = view.to_image((150, 180))
        view.zoom_at((150, 180), 1.25)
        after = view.to_image((150, 180))
        self.assertAlmostEqual(before[0], after[0])
        self.assertAlmostEqual(before[1], after[1])
        self.assertAlmostEqual(view.scale, 2.5)

    def test_zoom_is_limited(self):
        view = ps.ViewTransform()
        view.fit((200, 100), (400, 400))
        for _ in range(100):
            view.zoom_at((10, 10), 2)
        self.assertEqual(view.scale, 2.0 * 60)
        for _ in range(100):
            view.zoom_at((10, 10), 0.5)
        self.assertEqual(view.scale, 1.0)

    def test_pan_moves_the_offset(self):
        view = ps.ViewTransform(1.0, (0, 0))
        view.pan(5, -3)
        self.assertEqual(view.to_canvas((0, 0)), (5.0, -3.0))


class PanelImagesMixin:
    """ Images of the panel capture, shared by the tests that need them """

    _images = None

    @classmethod
    def panel_images(cls):
        if PanelImagesMixin._images is None:
            files = rct.read_band_files(rct.find_capture_paths(RAW, 'IMG_0005'), BANDS, os.environ.get('exiftoolpath'))
            PanelImagesMixin._images = {band: rct.Image(*files[band]) for band in BANDS}
        return PanelImagesMixin._images


class SelectionSessionTests(PanelImagesMixin, unittest.TestCase):

    def session(self, saved=None):
        return ps.SelectionSession(self.panel_images(), saved)

    def test_the_automatic_region_is_the_starting_polygon(self):
        session = self.session()
        polygon = session.starting_polygon('Blue')
        self.assertEqual(len(polygon), 4)
        self.assertEqual(polygon, session.automatic_polygon('Blue'))
        self.assertIsNot(session.starting_polygon('Blue'), session.starting_polygon('Blue'))   # copies

    def test_a_saved_region_has_priority_over_the_automatic_one(self):
        saved = [[100, 100], [200, 100], [200, 200]]
        self.assertEqual(self.session({'Blue': saved}).starting_polygon('Blue'), saved)
        self.assertEqual(self.session({'Blue': saved}).starting_polygon('Green'), self.session().automatic_polygon('Green'))

    def test_a_target_capture_has_no_automatic_region_and_starts_empty(self):
        files = rct.read_band_files(rct.find_capture_paths(RAW, 'IMG_0010'), BANDS)
        session = ps.SelectionSession({'Blue': rct.Image(*files['Blue'])})
        self.assertIsNone(session.automatic_polygon('Blue'))
        self.assertEqual(session.starting_polygon('Blue'), [])

    def test_confirm_validates_and_rounds(self):
        session = self.session()
        with self.assertRaisesRegex(ValueError, "Band 'Blue'.*at least 3"):
            session.confirm('Blue', [[1, 1], [5, 5]])
        with self.assertRaisesRegex(ValueError, 'cross'):
            session.confirm('Blue', [[10, 10], [60, 50], [60, 10], [10, 50]])
        self.assertIsNone(session.decisions['Blue'])
        session.confirm('Blue', [[10.4, 10.6], [60, 10], [60, 50]])
        self.assertEqual(session.decisions['Blue'], 'manual')
        self.assertEqual(session.polygons['Blue'], [[10, 11], [60, 10], [60, 50]])

    def test_stats_of_the_automatic_region_and_of_an_invalid_one(self):
        session = self.session()
        stats = session.stats('Blue', session.automatic_polygon('Blue'))
        self.assertGreater(stats['n_pixels'], 1000)
        self.assertLess(stats['variation_percent'], 5)
        self.assertIsNone(session.stats('Blue', [[1, 1], [2, 2]]))

    def test_result_has_the_manual_polygons_and_none_for_the_automatic_bands(self):
        session = self.session()
        self.assertFalse(session.is_complete())
        for band in BANDS:
            if band == 'Green':
                session.use_automatic(band)
            else:
                session.confirm(band, session.automatic_polygon(band))
        self.assertTrue(session.is_complete())
        result = session.result()
        self.assertIsNone(result['Green'])
        self.assertEqual(result['Blue'], session.automatic_polygon('Blue'))
        self.assertEqual(list(result), BANDS)

    def test_navigation_is_clamped(self):
        session = self.session()
        session.go_to(-3)
        self.assertEqual(session.band, 'Blue')
        session.go_to(99)
        self.assertEqual(session.band, 'NIR')

    def test_a_region_chosen_here_gives_the_same_calibration_in_the_workflow(self):
        import radiometric_workflow as wf
        session = self.session()
        regions = {band: session.automatic_polygon(band) for band in BANDS}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'regions.json'
            wf.save_panel_regions(path, 'IMG_0005', regions)
            saved = wf.load_panel_regions(path)['IMG_0005']
        levels = wf.default_config()['PANEL.reflectance_by_band']
        image = rct.Image(self.panel_images()['Blue'].image_path, self.panel_images()['Blue'].metadata)
        automatic = image.detect_reference_panel(levels, 0.7)[1]
        manual = rct.Image(image.image_path, image.metadata).detect_reference_panel(levels, 0.7, manual_bounds=saved['Blue'])[1]
        self.assertEqual(manual, automatic)


class WindowTests(PanelImagesMixin, unittest.TestCase):

    def setUp(self):
        import tkinter as tk
        try:
            self.root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"no display for Tk: {error}")
        self.root.withdraw()
        self.session = ps.SelectionSession(self.panel_images())
        self.window = ps.PanelSelectorWindow(self.root, self.session)
        self.root.geometry('900x700')
        self.root.update()

    def tearDown(self):
        try:
            self.root.destroy()
        except Exception:
            pass

    def click(self, handler, image_point):
        x, y = self.window.view.to_canvas(image_point)
        handler(SimpleNamespace(x=x, y=y))

    def test_the_first_band_is_shown_with_the_automatic_region(self):
        self.assertIn('Band 1 of 6: Blue', self.window.title_label.cget('text'))
        self.assertEqual(self.window.model.as_integers(), self.session.automatic_polygon('Blue'))
        self.assertIsNotNone(self.window.photo)
        self.assertGreater(len(self.window.canvas.find_all()), 3)

    def test_clicks_add_drag_and_remove_vertices(self):
        window = self.window
        window.clear()
        for point in ([300, 300], [500, 300], [500, 450]):
            self.click(window._on_press, point)
            self.click(window._on_release, point)
        self.assertEqual(window.model.as_integers(), [[300, 300], [500, 300], [500, 450]])
        self.assertIn('pixels', window.stats_label.cget('text'))
        self.click(window._on_press, [500, 450])           # pick the last vertex and drag it
        self.click(window._on_drag, [520, 470])
        self.click(window._on_release, [520, 470])
        self.assertEqual(window.model.as_integers()[2], [520, 470])
        self.click(window._on_remove, [520, 470])
        self.assertEqual(len(window.model.vertices), 2)
        self.assertIn('at least 3', window.stats_label.cget('text'))

    def test_confirming_an_invalid_region_shows_the_reason_and_stays(self):
        self.window.clear()
        self.window.confirm()
        self.assertEqual(self.session.index, 0)
        self.assertIn('at least 3 vertices', self.window.message_label.cget('text'))

    def test_all_bands_can_be_confirmed_and_saved(self):
        window = self.window
        for _ in BANDS:
            window.confirm()
        self.assertTrue(self.session.is_complete())
        self.assertEqual(self.session.index, len(BANDS) - 1)
        window.finish()
        self.assertFalse(window.cancelled)
        self.assertEqual(self.session.result()['Red'], self.session.automatic_polygon('Red'))

    def test_finishing_before_all_bands_are_decided_is_refused(self):
        self.window.confirm()
        self.window.finish()
        self.assertFalse(self.window.finished)
        self.assertIn('Green', self.window.message_label.cget('text'))

    def test_skip_uses_the_automatic_detection_and_navigation_keeps_decisions(self):
        window = self.window
        window.skip()
        self.assertEqual(self.session.decisions['Blue'], 'automatic')
        self.assertEqual(self.session.band, 'Green')
        window.previous()
        self.assertEqual(self.session.band, 'Blue')
        self.assertIn('automatic', window.message_label.cget('text'))

    def test_reset_returns_to_the_automatic_region_and_undo_removes_a_vertex(self):
        window = self.window
        window.clear()
        self.assertEqual(window.model.vertices, [])
        window.reset()
        self.assertEqual(window.model.as_integers(), self.session.automatic_polygon('Blue'))
        window.undo()
        self.assertEqual(len(window.model.vertices), 3)

    def test_the_window_explains_every_button_and_command(self):
        legend = self.window.legend_label.cget('text')
        for text in ('Confirm', 'Use automatic detection', 'Reset to automatic', 'Clear', 'Previous', 'Next',
                     'Save and close', 'Cancel', 'a click adds a vertex', 'right click removes', 'wheel', 'middle button',
                     'Enter', 'Backspace', 'Esc'):
            self.assertIn(text, legend)
        self.assertEqual(set(ps.HINTS), set(self.window.buttons))

    def test_pointing_at_a_button_says_what_it_does(self):
        self.assertEqual(self.window.hint_label.cget('text'), ps.DEFAULT_HINT)
        for name, button in self.window.buttons.items():
            button.event_generate('<Enter>')
            self.root.update()
            self.assertEqual(self.window.hint_label.cget('text'), ps.HINTS[name], name)
            button.event_generate('<Leave>')
            self.root.update()
            self.assertEqual(self.window.hint_label.cget('text'), ps.DEFAULT_HINT)

    def test_the_explanation_stays_visible_below_the_image(self):
        self.root.attributes('-alpha', 0.0)
        self.root.deiconify()
        self.root.geometry('900x700')
        self.root.update()
        buttons_top = self.window.buttons['confirm'].winfo_rooty()
        self.assertLess(self.window.legend_label.winfo_rooty(), buttons_top)
        self.assertGreater(self.window.legend_label.winfo_rooty(), self.window.canvas.winfo_rooty())
        self.assertGreater(self.window.canvas.winfo_height(), 200)                   # the image keeps room

    def test_zoom_with_the_wheel_and_pan(self):
        window = self.window
        before = window.view.scale
        window._on_wheel(SimpleNamespace(x=200, y=200, delta=120))
        self.assertGreater(window.view.scale, before)
        window._on_pan_start(SimpleNamespace(x=10, y=10))
        offset = window.view.offset
        window._on_pan_move(SimpleNamespace(x=30, y=25))
        self.assertEqual(window.view.offset, (offset[0] + 20, offset[1] + 15))

    def test_cancel_returns_nothing(self):
        self.window.cancel()
        self.assertTrue(self.window.cancelled)


class CommandLineTests(unittest.TestCase):

    def test_a_configuration_file_that_does_not_exist_returns_2(self):
        self.assertEqual(ps.main(['--config', str(ROOT / 'missing_inputs.json'), '--root', str(ROOT)]), 2)

    def test_invalid_configuration_is_reported_and_returns_1(self):
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder) / 'inputs.json'
            config.write_text(json.dumps({'INPUTS.reference_panel_capture': 'IMG_9999'}), encoding='utf-8')
            self.assertEqual(ps.main(['--config', str(config), '--root', str(ROOT)]), 1)


if __name__ == '__main__':
    unittest.main()
