"""
Tests for radiometric_window.py: the values being edited are tested without a display; the window with a hidden Tk
root (skipped if Tk cannot open one)

Run from this folder with:  python -m unittest test_radiometric_window -v
"""

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import radiometric_window as rw
import radiometric_workflow as wf

ROOT = Path(__file__).resolve().parent


class ConfigModelTests(unittest.TestCase):

    def test_defaults_are_valid_and_are_not_saved(self):
        model = rw.ConfigModel()
        config, errors = model.config()
        self.assertEqual(config, wf.default_config())
        self.assertEqual(errors, {})
        self.assertEqual(model.overrides(), {})

    def test_every_field_has_a_raw_value_of_the_right_shape(self):
        model = rw.ConfigModel()
        for field in wf.FIELDS:
            raw = model.get(field['key'])
            expected = {'band_values': dict, 'bool': bool}.get(field['kind'], str)
            self.assertIsInstance(raw, expected, field['key'])

    def test_numbers_are_converted_and_bad_text_is_reported_by_key(self):
        model = rw.ConfigModel()
        model.set('PANEL.percent_inner_area', '0.5')
        model.set('PANEL.saturation_threshold', '65000.5')
        model.set('CHECKS.jump_threshold', 'abc')
        model.set_band('PANEL.reflectance_by_band', 'Red', 'x')
        config, errors = model.config()
        self.assertEqual(config['PANEL.percent_inner_area'], 0.5)
        self.assertEqual(set(errors), {'PANEL.saturation_threshold', 'CHECKS.jump_threshold',
                                       'PANEL.reflectance_by_band'})
        self.assertEqual(errors['PANEL.saturation_threshold'], 'must be an integer')
        self.assertEqual(errors['CHECKS.jump_threshold'], 'must be a number')
        self.assertIn('Red', errors['PANEL.reflectance_by_band'])
        # an unreadable value keeps the default and is never saved
        self.assertEqual(config['PANEL.saturation_threshold'], 65000)
        self.assertEqual(model.overrides(), {'PANEL.percent_inner_area': 0.5})

    def test_an_integer_typed_with_a_decimal_point_is_accepted(self):
        model = rw.ConfigModel()
        model.set('PANEL.saturation_threshold', '60000.0')
        self.assertEqual(model.config()[0]['PANEL.saturation_threshold'], 60000)

    def test_only_changed_values_are_saved_and_loaded_back(self):
        model = rw.ConfigModel()
        model.set('INPUTS.reference_panel_capture', ' IMG_0007 ')
        model.set('CORRECTION.use_dls', True)
        model.set_band('PANEL.reflectance_by_band', 'Blue', '0.5')
        with tempfile.TemporaryDirectory() as folder:
            path = model.save(Path(folder) / 'inputs.json')
            saved = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(set(saved), {'INPUTS.reference_panel_capture', 'CORRECTION.use_dls',
                                          'PANEL.reflectance_by_band'})
            self.assertEqual(saved['INPUTS.reference_panel_capture'], 'IMG_0007')
            loaded = rw.ConfigModel.load(path)
            self.assertEqual(loaded.config()[0], model.config()[0])
            self.assertFalse(Path(str(path) + '.tmp').exists())

    def test_unknown_keys_in_a_saved_file_are_an_error(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'inputs.json'
            path.write_text(json.dumps({'INPUTS.typo': 1}), encoding='utf-8')
            with self.assertRaises(wf.ConfigError):
                rw.ConfigModel.load(path)

    def test_a_key_that_does_not_exist_cannot_be_set(self):
        with self.assertRaises(KeyError):
            rw.ConfigModel().set('INPUTS.typo', 'x')

    def test_every_field_belongs_to_a_page(self):
        prefixes = {prefix for prefix, _ in rw.SECTIONS}
        self.assertTrue(all(field['key'].split('.')[0] in prefixes for field in wf.FIELDS))


class WindowTests(unittest.TestCase):

    def setUp(self):
        import tkinter as tk
        try:
            self.root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"no display for Tk: {error}")
        self.root.withdraw()
        rw.apply_dark_theme(self.root)
        self.folder = tempfile.TemporaryDirectory()
        self.config_path = Path(self.folder.name) / 'inputs.json'
        self.frame = rw.RadiometricFrame(self.root, self.config_path, ROOT)
        self.frame.widget.pack(fill='both', expand=True)
        self.root.geometry('1000x700')
        self.root.update()

    def tearDown(self):
        try:
            self.root.destroy()
        except Exception:
            pass
        self.folder.cleanup()

    def pump(self, condition, timeout=240):
        end = time.time() + timeout
        while not condition() and time.time() < end:
            self.root.update()
            time.sleep(0.05)
        self.assertTrue(condition(), 'timed out')

    def results(self, name):
        return str(Path(self.folder.name) / name)

    def test_pages_and_fields_are_built_from_the_schema(self):
        self.assertEqual(self.frame.navigation.size(), len(rw.SECTIONS))
        self.assertEqual(set(self.frame.variables), set(wf.FIELD_KEYS))
        self.assertEqual(set(self.frame.pages), {prefix for prefix, _ in rw.SECTIONS})

    def test_typing_changes_the_model_and_marks_invalid_values(self):
        self.frame.set_value('PANEL.percent_inner_area', '0.4')
        self.assertEqual(self.frame.model.config()[0]['PANEL.percent_inner_area'], 0.4)
        self.assertEqual(str(self.frame.entries['PANEL.percent_inner_area'].cget('style')), 'TEntry')
        self.frame.set_value('PANEL.percent_inner_area', '0,4x')
        self.assertEqual(str(self.frame.entries['PANEL.percent_inner_area'].cget('style')), 'Invalid.TEntry')
        self.frame.set_value('PANEL.reflectance_by_band', {'Red': 'oops'})
        self.assertEqual(str(self.frame.entries['PANEL.reflectance_by_band']['Red'].cget('style')), 'Invalid.TEntry')
        self.assertEqual(str(self.frame.entries['PANEL.reflectance_by_band']['Blue'].cget('style')), 'TEntry')

    def test_the_check_box_is_a_boolean(self):
        self.frame.set_value('CORRECTION.use_dls', True)
        self.assertIs(self.frame.model.config()[0]['CORRECTION.use_dls'], True)

    def test_save_writes_only_the_changes(self):
        self.frame.set_value('INPUTS.reference_panel_capture', 'IMG_0007')
        self.assertTrue(self.frame.save())
        self.assertEqual(json.loads(self.config_path.read_text(encoding='utf-8')),
                         {'INPUTS.reference_panel_capture': 'IMG_0007'})
        self.assertIn('Saved to', self.frame.status.cget('text'))

    def test_a_saved_configuration_is_loaded_and_a_bad_one_is_reported(self):
        self.config_path.write_text(json.dumps({'CORRECTION.use_dls': True}), encoding='utf-8')
        frame = rw.RadiometricFrame(self.root, self.config_path, ROOT)
        self.assertTrue(frame.variables['CORRECTION.use_dls'].get())
        self.config_path.write_text('{not json', encoding='utf-8')
        frame = rw.RadiometricFrame(self.root, self.config_path, ROOT)
        self.assertEqual(frame.model.overrides(), {})
        self.assertTrue(any('was not used' in line for line in frame.log_lines()))

    def test_check_inputs_accepts_the_example_and_names_the_problems(self):
        config, problems = self.frame.check_inputs()
        self.assertEqual(problems, [])
        self.assertIn('The inputs are valid.', self.frame.log_lines())
        self.frame.set_value('INPUTS.reference_panel_capture', 'IMG_9999')
        self.frame.set_value('CHECKS.jump_threshold', 'abc')
        _, problems = self.frame.check_inputs()
        self.assertTrue(any('CHECKS.jump_threshold' in problem and 'must be a number' in problem for problem in problems))
        self.frame.set_value('CHECKS.jump_threshold', '0.1')
        _, problems = self.frame.check_inputs()
        self.assertTrue(any("Capture 'IMG_9999' not found" in problem for problem in problems))

    def test_running_the_workflow_reports_progress_and_the_result(self):
        self.frame.set_value('INPUTS.results_folder', self.results('run'))
        self.frame.set_value('CORRECTION.use_dls', True)
        self.frame.start_run()
        self.assertTrue(self.frame.running)
        self.assertEqual(str(self.frame.buttons['run'].cget('state')), 'disabled')
        self.pump(lambda: not self.frame.running)
        result = self.frame.last_result
        self.assertEqual(result['status'], 'completed', result['errors'])
        lines = '\n'.join(self.frame.log_lines())
        for stage in wf.STAGES[:4]:
            self.assertIn(f'[{stage}]', lines)
        self.assertIn('FLAG: IMG_0010', lines)
        self.assertIn('Status: completed. 6 image(s) written', self.frame.summary.cget('text'))
        self.assertEqual(str(self.frame.buttons['run'].cget('state')), 'normal')
        self.assertEqual(len(list(Path(self.results('run')).glob('IMG_0010_*.tif'))), 6)
        self.assertTrue(self.config_path.is_file())       # the run saves the configuration first

    def test_a_run_with_invalid_inputs_does_not_start(self):
        self.frame.set_value('PANEL.saturation_threshold', 'high')
        self.frame.start_run()
        self.assertFalse(self.frame.running)
        self.assertIsNone(self.frame.last_result)
        self.assertTrue(any('must be an integer' in line for line in self.frame.log_lines()))

    def test_drawing_regions_saves_them_and_sets_the_regions_file(self):
        self.frame.set_value('INPUTS.results_folder', self.results('regions'))
        polygon = [[700, 200], [850, 200], [850, 320], [700, 320]]
        selection = {band: (polygon if band == 'Blue' else None) for band in wf.default_config()['PANEL.reflectance_by_band']}
        with mock.patch('panel_selector.select_panel_regions', return_value=selection) as selector:
            self.frame.draw_regions()
            self.pump(lambda: selector.called and not self.frame.running)
        path = Path(self.results('regions')) / 'panel_regions.json'
        self.assertEqual(wf.load_panel_regions(path), {'IMG_0005': {'Blue': polygon}})
        self.assertEqual(self.frame.model.config()[0]['PANEL.manual_regions_file'], str(path))
        self.assertIn('1 region(s) drawn by hand and 5 band(s)', '\n'.join(self.frame.log_lines()))
        saved = json.loads(self.config_path.read_text(encoding='utf-8'))
        self.assertEqual(saved['PANEL.manual_regions_file'], str(path))

    def test_cancelling_the_dialog_saves_nothing(self):
        self.frame.set_value('INPUTS.results_folder', self.results('cancel'))
        with mock.patch('panel_selector.select_panel_regions', return_value=None) as selector:
            self.frame.draw_regions()
            self.pump(lambda: selector.called and not self.frame.running)
        self.assertFalse((Path(self.results('cancel')) / 'panel_regions.json').exists())
        self.assertIn('Cancelled: no region was saved.', self.frame.log_lines())

    def test_the_dialog_receives_the_saved_regions_and_the_panel_parameters(self):
        self.frame.set_value('INPUTS.results_folder', self.results('again'))
        self.frame.set_value('PANEL.percent_inner_area', '0.6')
        path = Path(self.results('again')) / 'panel_regions.json'
        path.parent.mkdir(parents=True)
        wf.save_panel_regions(path, 'IMG_0005', {'Red': [[10, 10], [50, 10], [50, 50]]})
        with mock.patch('panel_selector.select_panel_regions', return_value=None) as selector:
            self.frame.draw_regions()
            self.pump(lambda: selector.called and not self.frame.running)
        images, saved, percent, saturation = selector.call_args[0]
        self.assertEqual(set(images), set(wf.default_config()['PANEL.reflectance_by_band']))
        self.assertEqual(saved, {'Red': [[10, 10], [50, 10], [50, 50]]})
        self.assertEqual((percent, saturation), (0.6, 65000))

    def test_the_frame_can_be_placed_inside_another_window(self):
        import tkinter as tk
        from tkinter import ttk
        host = tk.Toplevel(self.root)
        notebook = ttk.Notebook(host)
        notebook.pack(fill='both', expand=True)
        frame = rw.RadiometricFrame(notebook, Path(self.folder.name) / 'embedded.json', ROOT)
        notebook.add(frame.widget, text='Radiometric')
        host.update()
        self.assertEqual(frame.widget.winfo_parent(), str(notebook))
        host.destroy()


if __name__ == '__main__':
    unittest.main()
