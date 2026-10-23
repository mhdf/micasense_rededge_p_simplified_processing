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


class TileTests(unittest.TestCase):

    def test_without_polygons_the_whole_image_is_shown(self):
        self.assertEqual(ps.tile_crop_box([], (1456, 1088), 340 / 250), (0, 0, 1456, 1088))

    def test_the_box_surrounds_the_polygon_with_the_aspect_of_the_tile_inside_the_image(self):
        polygon = [[700, 500], [800, 500], [800, 600], [700, 600]]
        left, top, right, bottom = ps.tile_crop_box([polygon], (1456, 1088), 340 / 250)
        self.assertTrue(left < 700 and right > 800 and top < 500 and bottom > 600)
        self.assertTrue(0 <= left and right <= 1456 and 0 <= top and bottom <= 1088)
        self.assertAlmostEqual((right - left) / (bottom - top), 340 / 250, delta=0.02)

    def test_a_polygon_near_the_corner_is_kept_inside_the_image(self):
        box = ps.tile_crop_box([[[5, 5], [60, 5], [60, 60]]], (400, 300), 340 / 250)
        self.assertEqual((box[0], box[1]), (0, 0))
        self.assertTrue(box[2] <= 400 and box[3] <= 300)

    def test_the_margin_has_a_minimum_for_small_polygons(self):
        left, top, right, bottom = ps.tile_crop_box([[[500, 500], [502, 500], [502, 502]]], (1456, 1088), 1.0)
        self.assertGreaterEqual(right - left, 120)

    def display(self):
        import numpy as np
        return np.full((300, 400), 90, dtype=np.uint8)

    def colored(self, picture, color):
        from PIL import ImageColor
        wanted = ImageColor.getrgb(color)
        return sum(1 for pixel in picture.getdata() if pixel == wanted)

    def test_a_tile_has_the_requested_size_and_shows_the_outlines(self):
        polygon = [[150, 100], [250, 100], [250, 200], [150, 200]]
        picture = ps.render_tile(self.display(), [(polygon, '#00ff00', False)], (340, 250))
        self.assertEqual((picture.size, picture.mode), ((340, 250), 'RGB'))
        self.assertGreater(self.colored(picture, '#00ff00'), 400)

    def test_a_dashed_outline_has_gaps(self):
        polygon = [[150, 100], [250, 100], [250, 200], [150, 200]]
        solid = self.colored(ps.render_tile(self.display(), [(polygon, '#ffcc00', False)], (340, 250)), '#ffcc00')
        dashed = self.colored(ps.render_tile(self.display(), [(polygon, '#ffcc00', True)], (340, 250)), '#ffcc00')
        self.assertTrue(0 < dashed < 0.8 * solid)

    def test_without_polygons_there_is_no_outline(self):
        picture = ps.render_tile(self.display(), [], (340, 250))
        self.assertEqual(self.colored(picture, '#00ff00') + self.colored(picture, '#ffcc00'), 0)

    def test_the_second_polygon_is_drawn_over_the_first(self):
        polygon = [[150, 100], [250, 100], [250, 200], [150, 200]]
        picture = ps.render_tile(self.display(), [(polygon, '#00ff00', False), (polygon, '#ffcc00', True)], (340, 250))
        self.assertGreater(self.colored(picture, '#00ff00'), 0)            # seen between the dashes
        self.assertGreater(self.colored(picture, '#ffcc00'), 0)


class PanelImagesMixin:
    """ Images of the panel capture, shared by the tests that need them """

    _images = None
    _target = None

    @classmethod
    def panel_images(cls):
        if PanelImagesMixin._images is None:
            files = rct.read_band_files(rct.find_capture_paths(RAW, 'IMG_0005'), BANDS, os.environ.get('exiftoolpath'))
            PanelImagesMixin._images = {band: rct.Image(*files[band]) for band in BANDS}
        return PanelImagesMixin._images

    @classmethod
    def target_images(cls):
        """ Images without a reference panel: there is no region to find from the QR code """
        if PanelImagesMixin._target is None:
            files = rct.read_band_files(rct.find_capture_paths(RAW, 'IMG_0010'), BANDS, os.environ.get('exiftoolpath'))
            PanelImagesMixin._target = {band: rct.Image(*files[band]) for band in BANDS}
        return PanelImagesMixin._target


class SelectionSessionTests(PanelImagesMixin, unittest.TestCase):

    def session(self, saved=None, origins=None):
        return ps.SelectionSession(self.panel_images(), saved, saved_origins=origins)

    def test_nothing_is_accepted_at_the_start_and_the_proposal_is_the_automatic_region(self):
        session = self.session()
        self.assertEqual(session.accepted_count(), 0)
        self.assertFalse(session.is_complete())
        polygon = session.proposed_polygon('Blue')
        self.assertEqual(len(polygon), 4)
        self.assertEqual(polygon, session.automatic_polygon('Blue'))
        self.assertIsNot(session.proposed_polygon('Blue'), session.proposed_polygon('Blue'))   # copies
        self.assertEqual({session.state(band) for band in BANDS}, {'pending'})

    def test_saved_regions_count_as_accepted_with_their_origin(self):
        automatic = self.session().automatic_polygon('Blue')
        edited = [[100, 100], [200, 100], [200, 200]]
        session = self.session({'Blue': automatic, 'Green': edited, 'Red': edited},
                               {'Red': 'accepted_automatic'})
        self.assertEqual(session.state('Blue'), 'accepted')
        self.assertEqual(session.origins['Blue'], 'accepted_automatic')       # worked out: it is the automatic region
        self.assertEqual(session.origins['Green'], 'edited')
        self.assertEqual(session.origins['Red'], 'accepted_automatic')        # as written in the file
        self.assertEqual(session.proposed_polygon('Green'), edited)
        self.assertEqual(session.state('NIR'), 'pending')
        self.assertEqual(session.accepted_count(), 3)

    def test_a_saved_region_that_is_not_valid_is_ignored_and_reported(self):
        session = self.session({'Blue': [[1, 1], [2, 2]], 'Green': [[10, 10], [9000, 10], [9000, 50]]})
        self.assertEqual(session.accepted_count(), 0)
        self.assertEqual(len(session.notes), 2)
        self.assertIn("band 'Blue'", session.notes[0])
        self.assertEqual(session.proposed_polygon('Blue'), session.automatic_polygon('Blue'))

    def test_a_band_in_the_file_that_is_not_in_the_images_is_left_out(self):
        session = self.session({'Infrared': [[10, 10], [20, 10], [20, 20]]})
        self.assertEqual(session.accepted_count(), 0)
        self.assertEqual(session.notes, [])

    def test_a_capture_without_a_panel_has_no_region_to_accept(self):
        session = ps.SelectionSession({'Blue': self.target_images()['Blue']})
        self.assertIsNone(session.automatic_polygon('Blue'))
        self.assertEqual(session.state('Blue'), 'no_region')
        with self.assertRaisesRegex(ValueError, 'at least 3'):
            session.accept('Blue')
        self.assertEqual(session.accept_all_automatic(), ['Blue'])
        session.accept('Blue', [[600, 300], [700, 300], [700, 400], [600, 400]])
        self.assertEqual((session.state('Blue'), session.origins['Blue']), ('accepted', 'edited'))

    def test_accepting_validates_rounds_and_works_out_the_origin(self):
        session = self.session()
        with self.assertRaisesRegex(ValueError, "Band 'Blue'.*at least 3"):
            session.accept('Blue', [[1, 1], [5, 5]])
        with self.assertRaisesRegex(ValueError, 'cross'):
            session.accept('Blue', [[10, 10], [60, 50], [60, 10], [10, 50]])
        self.assertEqual(session.state('Blue'), 'pending')
        session.accept('Blue', [[10.4, 10.6], [60, 10], [60, 50]])
        self.assertEqual(session.polygons['Blue'], [[10, 11], [60, 10], [60, 50]])
        self.assertEqual(session.origins['Blue'], 'edited')
        session.accept('Green')                                             # the region that is shown, as it is
        self.assertEqual(session.origins['Green'], 'accepted_automatic')
        self.assertEqual(session.polygons['Green'], session.automatic_polygon('Green'))

    def test_undoing_an_acceptance_goes_back_to_what_was_proposed(self):
        edited = [[100, 100], [200, 100], [200, 200]]
        session = self.session({'Blue': edited})
        session.accept('Blue', [[150, 150], [250, 150], [250, 250]])
        session.unaccept('Blue')
        self.assertEqual(session.state('Blue'), 'pending')
        self.assertEqual(session.proposed_polygon('Blue'), edited)           # the saved region, not accepted any more
        self.assertIsNone(session.origins['Blue'])

    def test_accept_all_takes_the_automatic_region_of_the_bands_that_are_not_accepted(self):
        edited = [[100, 100], [200, 100], [200, 200]]
        session = self.session({'Blue': edited})
        self.assertEqual(session.accept_all_automatic(), [])
        self.assertTrue(session.is_complete())
        self.assertEqual(session.polygons['Blue'], edited)                    # accepted ones are not touched
        self.assertEqual(session.polygons['Green'], session.automatic_polygon('Green'))
        self.assertEqual({session.origins[band] for band in BANDS if band != 'Blue'}, {'accepted_automatic'})

    def test_the_result_has_every_accepted_polygon_with_its_origin(self):
        session = self.session()
        session.accept('Blue')
        session.accept('Red', [[100, 100], [200, 100], [200, 200]])
        result = session.result()
        self.assertEqual(list(result), ['Blue', 'Red'])
        self.assertEqual(result.origins, {'Blue': 'accepted_automatic', 'Red': 'edited'})
        self.assertIsInstance(result, dict)

    def test_the_next_band_to_accept_goes_round(self):
        session = self.session()
        self.assertEqual(session.next_not_accepted(), 1)
        session.accept('Green')
        self.assertEqual(session.next_not_accepted(), 2)
        session.go_to(5)
        self.assertEqual(session.next_not_accepted(), 0)                      # round the end
        session.accept_all_automatic()
        self.assertIsNone(session.next_not_accepted())

    def test_stats_of_the_automatic_region_and_of_an_invalid_one(self):
        session = self.session()
        stats = session.stats('Blue', session.automatic_polygon('Blue'))
        self.assertGreater(stats['n_pixels'], 1000)
        self.assertLess(stats['variation_percent'], 5)
        self.assertIsNone(session.stats('Blue', [[1, 1], [2, 2]]))

    def test_navigation_is_clamped(self):
        session = self.session()
        session.go_to(-3)
        self.assertEqual(session.band, 'Blue')
        session.go_to(99)
        self.assertEqual(session.band, 'NIR')

    def test_a_region_accepted_here_gives_the_same_calibration_in_the_workflow(self):
        import radiometric_workflow as wf
        session = self.session()
        session.accept_all_automatic()
        result = session.result()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'regions.json'
            wf.save_panel_regions(path, 'IMG_0005', result, result.origins)
            saved = wf.load_panel_regions(path)['IMG_0005']
            origins = wf.load_panel_origins(path)['IMG_0005']
        self.assertEqual(origins['Blue'], 'accepted_automatic')
        levels = wf.default_config()['PANEL.reflectance_by_band']
        image = rct.Image(self.panel_images()['Blue'].image_path, self.panel_images()['Blue'].metadata)
        automatic = image.detect_reference_panel(levels, 0.7)[1]
        manual_image = rct.Image(image.image_path, image.metadata)
        manual = manual_image.detect_reference_panel(levels, 0.7, manual_bounds=saved['Blue'],
                                                     region_origin=origins['Blue'])[1]
        self.assertEqual(manual, automatic)
        self.assertEqual(manual_image.panel_audit['region_origin'], 'accepted_automatic')
        self.assertEqual(manual_image.panel_audit['region_source'], 'manual')


class WindowTests(PanelImagesMixin, unittest.TestCase):

    def setUp(self):
        import tkinter as tk
        try:
            self.root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"no display for Tk: {error}")
        self.root.withdraw()
        self.open_window()

    def open_window(self, saved=None, origins=None, images=None, start='overview'):
        self.session = ps.SelectionSession(images or self.panel_images(), saved, saved_origins=origins)
        self.window = ps.PanelSelectorWindow(self.root, self.session, start=start)
        self.root.geometry('1100x760')
        self.root.update()

    def tearDown(self):
        try:
            self.root.destroy()
        except Exception:
            pass

    def click(self, handler, image_point):
        x, y = self.window.view.to_canvas(image_point)
        handler(SimpleNamespace(x=x, y=y))

    def hover(self, widget):
        widget.event_generate('<Enter>')
        self.root.update()
        text = self.window.hint_label.cget('text')
        widget.event_generate('<Leave>')
        self.root.update()
        return text

    # --- overview ------------------------------------------------------------------------------------------

    def test_it_opens_on_an_overview_with_a_tile_for_each_band(self):
        window = self.window
        self.assertEqual(window.mode, 'overview')
        self.assertIn('overview', window.title_label.cget('text'))
        self.assertEqual(list(window.tiles), BANDS)
        for band, tile in window.tiles.items():
            self.assertEqual((tile['photo'].width(), tile['photo'].height()), ps.TILE_SIZE, band)
            self.assertEqual(tile['badge'].cget('text'), 'Not accepted (QR code)')
            self.assertIn('pixels', tile['stats'].cget('text'))
            self.assertEqual(tile['accept'].cget('text'), 'Accept')
        self.assertEqual(window.status_label.cget('text'), '0 of 6 bands accepted')
        self.assertTrue(window.overview_buttons['finish'].instate(['disabled']))
        self.assertFalse(window.overview_buttons['accept_all'].instate(['disabled']))

    def test_a_tile_can_be_accepted_and_the_acceptance_undone(self):
        window = self.window
        window.toggle_accept('Green')
        tile = window.tiles['Green']
        self.assertEqual(tile['accept'].cget('text'), 'Accepted (undo)')
        self.assertEqual(tile['badge'].cget('text'), 'Accepted (as found)')
        self.assertEqual(window.status_label.cget('text'), '1 of 6 bands accepted')
        self.assertIn('Still to accept: Blue, Panchro', window.overview_message.cget('text'))
        window.toggle_accept('Green')
        self.assertEqual(window.tiles['Green']['accept'].cget('text'), 'Accept')
        self.assertEqual(window.status_label.cget('text'), '0 of 6 bands accepted')

    def test_all_the_automatic_regions_can_be_accepted_without_opening_the_editor(self):
        window = self.window
        window.accept_all()
        self.assertEqual(window.mode, 'overview')
        self.assertTrue(self.session.is_complete())
        self.assertFalse(window.overview_buttons['finish'].instate(['disabled']))
        self.assertIn('Every region is accepted', window.overview_message.cget('text'))
        window.finish()
        self.assertFalse(window.cancelled)
        result = self.session.result()
        self.assertEqual(list(result), BANDS)
        self.assertEqual(set(result.origins.values()), {'accepted_automatic'})

    def test_finishing_before_every_band_is_accepted_is_refused(self):
        window = self.window
        window.toggle_accept('Blue')
        window.finish()
        self.assertFalse(window.finished)
        self.assertIn('Accept every band first', window.overview_message.cget('text'))
        self.assertIn('Green', window.overview_message.cget('text'))

    def test_the_regions_saved_before_are_shown_as_accepted(self):
        self.root.destroy()
        import tkinter as tk
        self.root = tk.Tk()
        self.root.withdraw()
        edited = [[700, 200], [850, 200], [850, 320], [700, 320]]
        self.open_window({'Blue': edited}, {'Blue': 'edited'})
        tile = self.window.tiles['Blue']
        self.assertEqual(tile['accept'].cget('text'), 'Accepted (undo)')
        self.assertEqual(tile['badge'].cget('text'), 'Accepted (edited)')
        self.assertEqual(self.window.status_label.cget('text'), '1 of 6 bands accepted')
        self.assertEqual(self.session.proposed_polygon('Blue'), edited)

    def test_a_saved_region_that_cannot_be_used_is_reported_in_the_overview(self):
        self.root.destroy()
        import tkinter as tk
        self.root = tk.Tk()
        self.root.withdraw()
        self.open_window({'Blue': [[1, 1], [2, 2]]})
        self.assertIn("The saved region of band 'Blue' is not valid", self.window.overview_message.cget('text'))

    def test_a_band_without_a_region_cannot_be_accepted_from_the_overview(self):
        self.root.destroy()
        import tkinter as tk
        self.root = tk.Tk()
        self.root.withdraw()
        self.open_window(images={'Blue': self.target_images()['Blue'], 'Green': self.target_images()['Green']})
        tile = self.window.tiles['Blue']
        self.assertEqual(tile['badge'].cget('text'), 'No region: use Edit')
        self.assertTrue(tile['accept'].instate(['disabled']))
        self.window.accept_all()
        self.assertIn('Blue, Green', self.window.overview_message.cget('text'))
        self.assertFalse(self.session.is_complete())

    def test_cancel_returns_nothing(self):
        self.window.cancel()
        self.assertTrue(self.window.cancelled)

    # --- editor --------------------------------------------------------------------------------------------

    def test_edit_opens_the_band_in_the_editor_with_the_proposed_region(self):
        window = self.window
        window.edit_band('Green')
        self.assertEqual(window.mode, 'editor')
        self.assertIn('Band 2 of 6: Green', window.title_label.cget('text'))
        self.assertEqual(window.model.as_integers(), self.session.automatic_polygon('Green'))
        self.assertIn('Not accepted yet', window.message_label.cget('text'))
        self.assertIn('pixels', window.stats_label.cget('text'))
        self.assertEqual(window.editor.winfo_manager(), 'pack')                 # the editor is shown, the overview is not
        self.assertEqual(window.overview.winfo_manager(), '')

    def test_the_editor_can_be_the_first_view(self):
        self.root.destroy()
        import tkinter as tk
        self.root = tk.Tk()
        self.root.withdraw()
        self.open_window(start='editor')
        self.assertEqual(self.window.mode, 'editor')
        self.assertIn('Band 1 of 6: Blue', self.window.title_label.cget('text'))

    def test_clicks_add_drag_and_remove_vertices(self):
        window = self.window
        window.edit_band('Blue')
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
        window = self.window
        window.edit_band('Blue')
        window.clear()
        window.confirm()
        self.assertEqual(window.mode, 'editor')
        self.assertEqual(self.session.state('Blue'), 'pending')
        self.assertIn('at least 3 vertices', window.message_label.cget('text'))

    def test_confirming_goes_to_the_next_band_that_is_not_accepted(self):
        window = self.window
        self.session.accept('Green')
        window.edit_band('Blue')
        window.confirm()                                                   # as it is: the automatic region
        self.assertEqual(self.session.origins['Blue'], 'accepted_automatic')
        self.assertEqual(window.mode, 'editor')
        self.assertIn('Band 3 of 6: Panchro', window.title_label.cget('text'))   # Green is already accepted

    def test_an_edited_region_is_accepted_as_edited(self):
        window = self.window
        window.edit_band('Blue')
        window.model.move(0, (window.model.vertices[0][0] + 12, window.model.vertices[0][1] + 7))
        window.confirm()
        self.assertEqual(self.session.origins['Blue'], 'edited')
        self.assertNotEqual(self.session.polygons['Blue'], self.session.automatic_polygon('Blue'))

    def test_confirming_the_last_band_that_is_not_accepted_returns_to_the_overview(self):
        window = self.window
        for band in BANDS[1:]:
            self.session.accept(band)
        window.edit_band('Blue')
        window.confirm()
        self.assertEqual(window.mode, 'overview')
        self.assertTrue(self.session.is_complete())
        self.assertEqual(window.tiles['Blue']['accept'].cget('text'), 'Accepted (undo)')
        self.assertFalse(window.overview_buttons['finish'].instate(['disabled']))

    def test_going_back_to_the_overview_discards_what_was_not_confirmed(self):
        window = self.window
        before = self.session.proposed_polygon('Blue')
        window.edit_band('Blue')
        window.clear()
        window.show_overview()
        self.assertEqual(window.mode, 'overview')
        self.assertEqual(self.session.state('Blue'), 'pending')
        self.assertEqual(self.session.proposed_polygon('Blue'), before)

    def test_the_editor_shows_the_accepted_region_to_compare(self):
        window = self.window
        self.session.accept('Blue', [[700, 200], [850, 200], [850, 320], [700, 320]])
        window.edit_band('Blue')
        self.assertEqual(window.model.as_integers(), [[700, 200], [850, 200], [850, 320], [700, 320]])
        self.assertIn('Accepted (drawn or edited)', window.message_label.cget('text'))
        window.reset()                                                      # back to the QR code region
        self.assertEqual(window.model.as_integers(), self.session.automatic_polygon('Blue'))
        window._redraw()                                                    # draws the accepted region too, to compare
        self.assertEqual(self.session.polygons['Blue'], [[700, 200], [850, 200], [850, 320], [700, 320]])

    def test_previous_and_next_move_between_bands_in_the_editor(self):
        window = self.window
        window.edit_band('Green')
        window.next()
        self.assertIn('Band 3 of 6', window.title_label.cget('text'))
        window.previous()
        window.previous()
        self.assertIn('Band 1 of 6', window.title_label.cget('text'))
        window.previous()                                                   # clamped
        self.assertIn('Band 1 of 6', window.title_label.cget('text'))

    def test_reset_returns_to_the_automatic_region_and_undo_removes_a_vertex(self):
        window = self.window
        window.edit_band('Blue')
        window.clear()
        self.assertEqual(window.model.vertices, [])
        window.reset()
        self.assertEqual(window.model.as_integers(), self.session.automatic_polygon('Blue'))
        window.undo()
        self.assertEqual(len(window.model.vertices), 3)

    def test_zoom_with_the_wheel_and_pan(self):
        window = self.window
        window.edit_band('Blue')
        before = window.view.scale
        window._on_wheel(SimpleNamespace(x=200, y=200, delta=120))
        self.assertGreater(window.view.scale, before)
        window._on_pan_start(SimpleNamespace(x=10, y=10))
        offset = window.view.offset
        window._on_pan_move(SimpleNamespace(x=30, y=25))
        self.assertEqual(window.view.offset, (offset[0] + 20, offset[1] + 15))

    def test_escape_goes_back_from_the_editor_and_cancels_from_the_overview(self):
        window = self.window
        window.edit_band('Blue')
        window._key_escape()
        self.assertEqual(window.mode, 'overview')
        self.assertFalse(window.finished)
        window._key_escape()
        self.assertTrue(window.cancelled and window.finished)

    def test_the_keys_of_the_editor_do_nothing_in_the_overview(self):
        window = self.window
        window._key_confirm()
        window._key_clear()
        window._key_next()
        self.assertEqual(window.mode, 'overview')
        self.assertEqual(self.session.accepted_count(), 0)

    # --- what each element does ------------------------------------------------------------------------------

    def test_there_is_no_fixed_explanation_only_the_hint_of_the_element_under_the_mouse(self):
        window = self.window
        self.assertFalse(hasattr(window, 'legend_label'))
        self.assertEqual(window.hint_label.cget('text'), ps.DEFAULT_HINT)
        self.assertNotIn('Mouse:', ps.DEFAULT_HINT)

    def test_every_button_and_image_has_a_hint(self):
        window = self.window
        names = set(window.buttons) | set(window.overview_buttons) | {'canvas', 'tile_accept', 'tile_edit', 'tile_image'}
        self.assertEqual(set(ps.HINTS), names)
        self.assertNotIn('skip', ps.HINTS)                                  # there is no separate automatic option

    def test_pointing_at_the_buttons_of_the_overview_says_what_they_do(self):
        window = self.window
        for name, button in window.overview_buttons.items():
            self.assertEqual(self.hover(button), ps.HINTS[name], name)
        tile = window.tiles['Blue']
        self.assertEqual(self.hover(tile['accept']), ps.HINTS['tile_accept'])
        self.assertEqual(self.hover(tile['edit']), ps.HINTS['tile_edit'])
        self.assertEqual(self.hover(tile['image']), ps.HINTS['tile_image'])
        self.assertEqual(window.hint_label.cget('text'), ps.DEFAULT_HINT)

    def test_pointing_at_the_buttons_and_the_image_of_the_editor_says_what_they_do(self):
        window = self.window
        self.root.attributes('-alpha', 0.0)             # a canvas gets the events of the mouse only when it is shown
        self.root.deiconify()
        window.edit_band('Blue')
        self.root.update()
        for name, button in window.buttons.items():
            self.assertEqual(self.hover(button), ps.HINTS[name], name)
        self.assertEqual(self.hover(window.canvas), ps.HINTS['canvas'])

    def test_the_hints_carry_the_important_information(self):
        hints = ps.HINTS
        for name, phrases in {'confirm': ('Enter', 'not accepted yet', '3 or more vertices', 'next band'),
                              'overview': ('Esc', 'lost'),
                              'reset': ('QR code', 'not accepted until'),
                              'clear': ('Backspace',),
                              'canvas': ('Click adds a vertex', 'right click', 'wheel', 'middle button', 'Cyan',
                                         'Yellow dashed', 'Green', 'variation'),
                              'accept_all': ('QR code', 'Bands without one', 'used for the reference panel'),
                              'finish': ('every band is accepted', 'exactly these regions'),
                              'cancel': ('save nothing',),
                              'tile_accept': ('undo',), 'tile_edit': ('editor',),
                              'tile_image': ('yellow dashed', 'green', 'cyan', 'variation')}.items():
            for phrase in phrases:
                self.assertIn(phrase.lower(), hints[name].lower(), f"{name}: {phrase}")


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
