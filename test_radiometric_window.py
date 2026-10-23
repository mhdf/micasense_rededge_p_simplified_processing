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

import panel_spectrum as ps
import radiometric_window as rw
import radiometric_workflow as wf

ROOT = Path(__file__).resolve().parent


def rct_pattern():
    import radiometric_correction_tools as rct
    return rct.CAPTURE_FILE_PATTERN


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

    def test_choice_fields_hold_one_of_their_values(self):
        model = rw.ConfigModel()
        self.assertEqual(model.config()[0]['PANEL.reflectance_source'], 'values')
        model.set('PANEL.reflectance_source', 'spectral_file')
        self.assertEqual(model.overrides(), {'PANEL.reflectance_source': 'spectral_file'})

    def test_every_field_belongs_to_a_page(self):
        pages = {page for page, _ in rw.SECTIONS}
        self.assertTrue(all(rw.page_of(field['key']) in pages for field in wf.FIELDS))
        self.assertEqual(rw.page_of('CHECKS.jump_threshold'), 'CORRECTION')           # the DLS page has the checks too
        self.assertEqual(rw.page_of('PANEL.manual_regions_file'), 'INPUTS')
        self.assertEqual(rw.page_of('PANEL.percent_inner_area'), 'PANEL')


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
        names = [row['name'] for row in self.frame.band_table.get_rows()]
        styles = {name: str(row['widgets']['reflectance'].cget('style'))
                  for name, row in zip(names, self.frame.band_table.rows)}
        self.assertEqual(styles['Red'], 'Invalid.TEntry')
        self.assertEqual(styles['Blue'], 'TEntry')

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

    def accepted(self, polygons, origins):
        import panel_selector
        return panel_selector.SelectionResult(polygons, origins)

    def test_the_accepted_regions_are_saved_with_their_origin_and_the_regions_file_is_set(self):
        self.frame.set_value('INPUTS.results_folder', self.results('regions'))
        polygon = [[700, 200], [850, 200], [850, 320], [700, 320]]
        bands = list(wf.default_config()['PANEL.reflectance_by_band'])
        selection = {'IMG_0005': self.accepted({band: polygon for band in bands},
                                               {band: ('accepted_automatic' if band != 'Blue' else 'edited')
                                                for band in bands})}
        with mock.patch('panel_selector.select_panel_regions', return_value=selection) as selector:
            self.frame.draw_regions()
            self.pump(lambda: selector.called and not self.frame.running)
        path = Path(self.results('regions')) / 'panel_regions.json'
        self.assertEqual(wf.load_panel_regions(path), {'IMG_0005': {band: polygon for band in bands}})
        origins = wf.load_panel_origins(path)['IMG_0005']
        self.assertEqual((origins['Blue'], origins['Green']), ('edited', 'accepted_automatic'))
        self.assertEqual(self.frame.model.config()[0]['PANEL.manual_regions_file'], str(path))
        text = '\n'.join(self.frame.log_lines())
        self.assertIn('IMG_0005: 6 region(s), 5 as found from the QR code, 1 drawn or edited', text)
        self.assertIn('Saved the accepted regions', self.frame.inputs_status.cget('text'))
        saved = json.loads(self.config_path.read_text(encoding='utf-8'))
        self.assertEqual(saved['PANEL.manual_regions_file'], str(path))

    def test_cancelling_the_dialog_saves_nothing(self):
        self.frame.set_value('INPUTS.results_folder', self.results('cancel'))
        with mock.patch('panel_selector.select_panel_regions', return_value=None) as selector:
            self.frame.draw_regions()
            self.pump(lambda: selector.called and not self.frame.running)
        self.assertFalse((Path(self.results('cancel')) / 'panel_regions.json').exists())
        self.assertIn('Cancelled: no region was saved.', self.frame.log_lines())
        self.assertEqual(self.frame.inputs_status.cget('text'), 'Cancelled: no region was saved.')

    def test_the_dialog_receives_the_saved_regions_their_origins_and_the_panel_parameters(self):
        self.frame.set_value('INPUTS.results_folder', self.results('again'))
        self.frame.set_value('PANEL.percent_inner_area', '0.6')
        path = Path(self.results('again')) / 'panel_regions.json'
        path.parent.mkdir(parents=True)
        wf.save_panel_regions(path, 'IMG_0005', {'Red': [[10, 10], [50, 10], [50, 50]]}, {'Red': 'edited'})
        with mock.patch('panel_selector.select_panel_regions', return_value=None) as selector:
            self.frame.draw_regions()
            self.pump(lambda: selector.called and not self.frame.running)
        names, load, required, saved, origins, percent, saturation = selector.call_args[0]
        self.assertEqual(names, ['IMG_0005'])                                     # only the panel captures are listed
        self.assertEqual(required, ['IMG_0005'])
        self.assertEqual(saved, {'IMG_0005': {'Red': [[10, 10], [50, 10], [50, 50]]}})
        self.assertEqual(origins, {'IMG_0005': {'Red': 'edited'}})
        self.assertEqual((percent, saturation), (0.6, 65000))
        preloaded = selector.call_args.kwargs['preloaded']
        self.assertEqual(list(preloaded), ['IMG_0005'])
        self.assertEqual(set(preloaded['IMG_0005']), set(wf.default_config()['PANEL.reflectance_by_band']))
        other = load('IMG_0010')                                                  # the others are read when they are opened
        self.assertEqual(set(other), set(preloaded['IMG_0005']))

    def test_both_panel_captures_are_required_and_their_regions_are_saved(self):
        self.frame.set_value('INPUTS.results_folder', self.results('two'))
        self.frame.set_value('INPUTS.reference_panel_capture_after', 'IMG_0005')
        polygon = [[700, 200], [850, 200], [850, 320], [700, 320]]
        bands = list(wf.default_config()['PANEL.reflectance_by_band'])
        selection = {name: self.accepted({band: polygon for band in bands}, {band: 'edited' for band in bands})
                     for name in ('IMG_0005', 'IMG_0010')}
        with mock.patch('panel_selector.select_panel_regions', return_value=selection) as selector:
            self.frame.draw_regions()
            self.pump(lambda: selector.called and not self.frame.running)
        self.assertEqual(selector.call_args[0][2], ['IMG_0005'])                   # the same capture twice is one
        path = Path(self.results('two')) / 'panel_regions.json'
        self.assertEqual(sorted(wf.load_panel_regions(path)), ['IMG_0005', 'IMG_0010'])
        self.assertEqual(wf.load_panel_origins(path)['IMG_0010']['Blue'], 'edited')
        self.assertIn('IMG_0010: 6 region(s)', self.frame.inputs_status.cget('text'))

    # --- checking the inputs before the tool is opened ----------------------------------------------------------

    def test_the_inputs_can_be_checked_next_to_the_tool_without_leaving_the_page(self):
        self.frame.show_page('INPUTS')
        config, problems = self.frame.check_inputs_here()
        self.assertEqual(problems, [])
        self.assertIn('The inputs are valid', self.frame.inputs_status.cget('text'))
        self.assertEqual(str(self.frame.inputs_status.cget('style')), 'Ok.TLabel')
        self.frame.set_value('INPUTS.reference_panel_capture', 'IMG_9999')
        _, problems = self.frame.check_inputs_here()
        self.assertTrue(any("Capture 'IMG_9999' not found" in problem for problem in problems))
        self.assertIn("Capture 'IMG_9999' not found", self.frame.inputs_status.cget('text'))
        self.assertEqual(str(self.frame.inputs_status.cget('style')), 'Error.TLabel')
        self.assertEqual([line for line in self.frame.log_lines() if line], [])      # the Run page log is not used

    def test_the_check_button_is_next_to_the_tool_on_the_inputs_page(self):
        self.assertTrue(self.inside(self.frame.buttons['check_here'], 'INPUTS'))
        self.assertEqual(self.frame.buttons['check_here'].cget('text'), 'Check inputs')
        self.assertEqual(self.frame.buttons['check_here'].master, self.frame.buttons['regions'].master)

    def test_the_tool_is_not_opened_when_the_inputs_have_problems(self):
        self.frame.set_value('INPUTS.reference_panel_capture', 'IMG_9999')
        with mock.patch('panel_selector.select_panel_regions') as selector:
            self.frame.draw_regions()
        self.assertFalse(self.frame.running)
        self.assertFalse(selector.called)
        self.assertIn("Capture 'IMG_9999' not found", self.frame.inputs_status.cget('text'))
        self.assertEqual(self.frame.navigation.curselection(), (0,))      # still on the Inputs page

    def test_problems_in_the_band_table_are_shown_next_to_the_tool_too(self):
        self.frame.band_table.add_row({'name': 'Blue', 'center': '1', 'width': '1', 'reflectance': '0.5'})
        _, problems = self.frame.check_inputs_here()
        self.assertTrue(any('used by more than one band' in problem for problem in problems))
        self.assertIn('used by more than one band', self.frame.inputs_status.cget('text'))

    def test_both_buttons_are_disabled_while_the_workflow_is_busy(self):
        self.frame._set_busy(True)
        self.assertTrue(self.frame.buttons['check_here'].instate(['disabled']))
        self.assertTrue(self.frame.buttons['regions'].instate(['disabled']))
        self.frame._set_busy(False)
        self.assertFalse(self.frame.buttons['check_here'].instate(['disabled']))

    # --- spectral response of the panel and profiles ---------------------------------------------------------

    def write_spectrum(self, name='panel.csv', level=0.5, start=400, stop=900):
        path = Path(self.folder.name) / name
        path.write_text(''.join(f"{w},{level}\n" for w in range(start, stop + 1)), encoding='utf-8')
        return path

    def enabled(self, widgets):
        return [not widget.instate(['disabled']) for widget in widgets]

    def test_only_the_controls_of_the_chosen_source_can_be_edited(self):
        values = self.frame.band_table.reflectance_entries()
        spectral = self.frame.source_widgets['spectral_file']
        self.assertEqual(len(values), 6)
        self.assertEqual(len(spectral), 3)                      # entry, Browse and Import
        self.assertTrue(all(self.enabled(values)))
        self.assertFalse(any(self.enabled(spectral)))
        self.frame.set_value('PANEL.reflectance_source', 'spectral_file')
        self.assertFalse(any(self.enabled(values)))
        self.assertTrue(all(self.enabled(spectral)))

    def test_choices_are_drop_downs_and_dialog_keys_have_no_widgets_on_the_page(self):
        self.assertEqual(self.frame.entries['PANEL.reflectance_source'].winfo_class(), 'TCombobox')
        for key in ('PANEL.spectral_delimiter', 'PANEL.spectral_skip_rows', 'PANEL.band_center_wavelengths'):
            self.assertIn(key, self.frame.variables)
            self.assertNotIn(key, self.frame.entries)
        self.frame.set_value('PANEL.spectral_delimiter', 'semicolon')
        self.frame.set_value('PANEL.band_bandwidths', {'Blue': '40'})
        config = self.frame.model.config()[0]
        self.assertEqual(config['PANEL.spectral_delimiter'], 'semicolon')
        self.assertEqual(config['PANEL.band_bandwidths']['Blue'], 40.0)

    def test_the_summary_shows_the_values_read_from_the_file_or_the_problem(self):
        path = self.write_spectrum(level=0.5, stop=848)
        self.frame.set_value('PANEL.spectral_file', str(path))
        self.pump(lambda: 'From the file' in self.frame.spectral_summary.cget('text'), 5)
        text = self.frame.spectral_summary.cget('text')
        self.assertIn('Blue 0.5000', text)
        self.assertIn('NIR 61 %', text)                         # the window of NIR is partly covered
        self.frame.set_value('PANEL.spectral_wavelength_column', '9')
        self.pump(lambda: 'data row' in self.frame.spectral_summary.cget('text'), 5)
        self.frame.set_value('PANEL.spectral_wavelength_column', '1')
        self.frame.set_value('PANEL.spectral_min_coverage', '0.9')
        self.pump(lambda: "Band 'NIR'" in self.frame.spectral_summary.cget('text'), 5)
        self.frame.set_value('PANEL.spectral_file', '')
        self.pump(lambda: self.frame.spectral_summary.cget('text') == '', 5)

    def result_of_the_dialog(self, action, **kwargs):
        import panel_spectrum_dialog as psd
        path = kwargs.pop('path', None) or self.write_spectrum(level=0.48)
        return psd.ImportSession(path, **kwargs).result(action)

    def test_using_the_file_fills_the_fields_and_switches_the_source(self):
        frame = rw.RadiometricFrame(self.root, Path(self.folder.name) / 'inside.json', self.folder.name)
        path = self.write_spectrum(level=0.48)
        result = self.result_of_the_dialog('use_file', path=path, fmt={'skip_rows': 0},
                                           windows=dict(ps.REDEDGE_P_BANDS, Blue=(480.0, 40.0)))
        frame.apply_import(result)
        config = frame.model.config()[0]
        self.assertEqual(config['PANEL.reflectance_source'], 'spectral_file')
        self.assertEqual(frame.band_table.get_rows()[0]['center'], '480')              # the table shows the windows
        self.assertEqual(config['PANEL.spectral_file'], 'panel.csv')       # inside the project folder: relative
        self.assertEqual(config['PANEL.spectral_skip_rows'], '0')
        self.assertEqual((config['PANEL.spectral_skip_footer'], config['PANEL.spectral_comment_prefix']), (0, ''))
        self.assertEqual((config['PANEL.spectral_wavelength_min'], config['PANEL.spectral_reflectance_scale']), ('', 1.0))
        self.assertEqual(config['PANEL.band_center_wavelengths']['Blue'], 480.0)
        self.assertEqual(config['PANEL.band_bandwidths']['Blue'], 40.0)
        self.assertEqual(wf.validate_config(dict(config, **{'INPUTS.images_folder': str(ROOT / 'images' / 'raw')}),
                                            self.folder.name), [])
        levels, info = wf.panel_reflectance_levels(config, self.folder.name)
        self.assertAlmostEqual(levels['Blue'], 0.48)
        self.assertFalse(any(self.enabled(frame.band_table.reflectance_entries())))

    def test_the_reading_options_of_the_dialog_reach_the_fields(self):
        result = self.result_of_the_dialog('use_file', fmt={'comment_prefix': '#', 'skip_footer': 0, 'exclude_rows': '3',
                                                            'wavelength_min': 420.0, 'reflectance_scale': 1.0,
                                                            'delimiter': 'custom', 'delimiter_char': ','})
        self.frame.apply_import(result)
        config = self.frame.model.config()[0]
        self.assertEqual(config['PANEL.spectral_comment_prefix'], '#')
        self.assertEqual(config['PANEL.spectral_exclude_rows'], '3')
        self.assertEqual(config['PANEL.spectral_wavelength_min'], '420.0')
        self.assertEqual((config['PANEL.spectral_delimiter'], config['PANEL.spectral_delimiter_char']), ('custom', ','))
        self.assertEqual(wf.spectral_format(config)['wavelength_min'], 420.0)

    def test_a_file_outside_the_project_folder_keeps_its_full_path(self):
        other = tempfile.TemporaryDirectory()
        try:
            path = Path(other.name) / 'panel.csv'
            path.write_text(''.join(f"{w},0.5\n" for w in range(400, 901)), encoding='utf-8')
            self.frame.apply_import(self.result_of_the_dialog('use_file', path=path))
            self.assertEqual(self.frame.model.config()[0]['PANEL.spectral_file'], str(path))
        finally:
            other.cleanup()

    def test_copying_the_values_fills_the_typed_list_and_returns_to_the_typed_source(self):
        self.frame.set_value('PANEL.reflectance_source', 'spectral_file')
        self.frame.apply_import(self.result_of_the_dialog('copy_values'))
        config = self.frame.model.config()[0]
        self.assertEqual(config['PANEL.reflectance_source'], 'values')
        self.assertTrue(all(value == 0.48 for value in config['PANEL.reflectance_by_band'].values()))
        self.assertEqual(config['PANEL.spectral_file'], '')                # the file is not made the source
        self.assertTrue(any('copied' in line for line in self.frame.log_lines()))

    def test_the_import_button_opens_the_dialog_with_the_current_values_and_applies_the_answer(self):
        self.frame.set_value('PANEL.spectral_delimiter', 'comma')
        self.frame.set_value('PANEL.band_bandwidths', {'Red': '20'})
        answer = self.result_of_the_dialog('use_file')
        with mock.patch('panel_spectrum_dialog.import_spectrum', return_value=answer) as dialog:
            self.frame.import_spectrum()
        _, path, fmt, windows, coverage = dialog.call_args[0]
        self.assertEqual(path, '')
        self.assertEqual(fmt['delimiter'], 'comma')
        self.assertEqual(windows['Red'], (668.0, 20.0))
        self.assertEqual(coverage, 0.5)
        self.assertEqual(self.frame.model.config()[0]['PANEL.reflectance_source'], 'spectral_file')
        with mock.patch('panel_spectrum_dialog.import_spectrum', return_value=None):
            self.frame.import_spectrum()                                   # cancelled: nothing changes
        self.assertEqual(self.frame.model.config()[0]['PANEL.reflectance_source'], 'spectral_file')

    def test_a_profile_saves_every_value_and_restores_them(self):
        self.frame.set_value('INPUTS.reference_panel_capture', 'IMG_0007')
        self.frame.set_value('CORRECTION.use_dls', True)
        self.frame.set_value('PANEL.reflectance_by_band', {'Blue': '0.5'})
        path = Path(self.folder.name) / 'profiles' / 'mine.json'
        self.assertTrue(self.frame.save_profile(path))
        saved = json.loads(path.read_text(encoding='utf-8'))
        self.assertEqual(list(saved), wf.FIELD_KEYS)                         # all the keys, not only the changes
        self.assertEqual(saved['PANEL.reflectance_by_band']['Blue'], 0.5)
        self.frame.set_value('INPUTS.reference_panel_capture', 'IMG_0001')
        self.frame.set_value('CORRECTION.use_dls', False)
        self.assertTrue(self.frame.load_profile(path))
        config = self.frame.model.config()[0]
        self.assertEqual(config['INPUTS.reference_panel_capture'], 'IMG_0007')
        self.assertIs(config['CORRECTION.use_dls'], True)
        self.assertEqual(config['PANEL.reflectance_by_band']['Blue'], 0.5)
        self.assertIs(self.frame.variables['CORRECTION.use_dls'].get(), True)     # the check box follows

    def test_a_profile_with_only_some_keys_changes_the_others_to_the_defaults(self):
        path = Path(self.folder.name) / 'small.json'
        path.write_text(json.dumps({'PANEL.percent_inner_area': 0.4}), encoding='utf-8')
        self.frame.set_value('CORRECTION.use_dls', True)
        self.assertTrue(self.frame.load_profile(path))
        config = self.frame.model.config()[0]
        self.assertEqual(config['PANEL.percent_inner_area'], 0.4)
        self.assertIs(config['CORRECTION.use_dls'], False)

    def test_bad_profiles_are_refused_and_nothing_changes(self):
        path = Path(self.folder.name) / 'bad.json'
        path.write_text(json.dumps({'INPUTS.typo': 1}), encoding='utf-8')
        self.frame.set_value('INPUTS.reference_panel_capture', 'IMG_0007')
        self.assertFalse(self.frame.load_profile(path))
        self.assertFalse(self.frame.load_profile(Path(self.folder.name) / 'missing.json'))
        self.assertEqual(self.frame.model.config()[0]['INPUTS.reference_panel_capture'], 'IMG_0007')
        self.assertTrue(any('INPUTS.typo' in line for line in self.frame.log_lines()))

    def test_a_profile_is_not_saved_while_a_value_cannot_be_read(self):
        self.frame.set_value('PANEL.saturation_threshold', 'high')
        path = Path(self.folder.name) / 'profiles' / 'refused.json'
        self.assertFalse(self.frame.save_profile(path))
        self.assertFalse(path.exists())

    # --- bands of the camera ----------------------------------------------------------------------------------

    def names(self):
        return [row['name'] for row in self.frame.band_table.get_rows()]

    def row(self, name):
        return next(row for row in self.frame.band_table.rows if row['vars']['name'].get() == name)

    def test_the_table_shows_the_bands_of_the_configuration(self):
        rows = self.frame.band_table.get_rows()
        self.assertEqual([r['name'] for r in rows], ['Blue', 'Green', 'Panchro', 'Red', 'Red edge', 'NIR'])
        self.assertEqual(rows[0], {'name': 'Blue', 'center': '475.0', 'width': '32.0', 'reflectance': '0.477', 'process': True})
        self.assertEqual(self.frame.model.get('INPUTS.ignored_bands'), '')
        for key in rw.BAND_KEYS:
            self.assertIs(self.frame.variables[key], self.frame.band_table)

    def test_editing_the_table_writes_the_configuration(self):
        self.row('Blue')['vars']['center'].set('480')
        self.row('Blue')['vars']['reflectance'].set('0.5')
        config = self.frame.model.config()[0]
        self.assertEqual(config['PANEL.band_center_wavelengths']['Blue'], 480.0)
        self.assertEqual(config['PANEL.reflectance_by_band']['Blue'], 0.5)
        self.assertEqual(config['PANEL.band_bandwidths']['Blue'], 32.0)

    def test_a_band_that_is_not_used_moves_to_the_ignored_bands(self):
        self.row('Panchro')['vars']['process'].set(False)
        self.row('NIR')['vars']['process'].set(False)
        config = self.frame.model.config()[0]
        self.assertEqual(config['INPUTS.ignored_bands'], 'Panchro, NIR')
        for key in ('PANEL.reflectance_by_band', 'PANEL.band_center_wavelengths', 'PANEL.band_bandwidths'):
            self.assertEqual(list(config[key]), ['Blue', 'Green', 'Red', 'Red edge'])
        self.assertEqual(wf.validate_config(config, ROOT), [])
        self.row('NIR')['vars']['process'].set(True)                                 # and back
        self.assertEqual(self.frame.model.get('INPUTS.ignored_bands'), 'Panchro')
        self.assertIn('NIR', self.frame.model.get('PANEL.band_center_wavelengths'))

    def test_bands_can_be_added_renamed_and_removed(self):
        table = self.frame.band_table
        new = table.add_row({'name': 'Coastal', 'center': '444', 'width': '28', 'reflectance': '0.47'})
        self.assertEqual(self.frame.model.config()[0]['PANEL.band_center_wavelengths']['Coastal'], 444.0)
        new['vars']['name'].set('Coastal blue')
        config = self.frame.model.config()[0]
        self.assertIn('Coastal blue', config['PANEL.reflectance_by_band'])
        self.assertNotIn('Coastal', config['PANEL.reflectance_by_band'])
        table.remove(new)
        table.remove(self.row('Panchro'))
        config = self.frame.model.config()[0]
        self.assertEqual(list(config['PANEL.reflectance_by_band']), ['Blue', 'Green', 'Red', 'Red edge', 'NIR'])
        self.assertEqual(wf.validate_config(config, ROOT), [])

    def test_problems_in_the_table_stop_the_run_and_are_listed(self):
        self.frame.band_table.add_row({'name': 'Blue', 'center': '1', 'width': '1', 'reflectance': '0.5'})
        config, problems = self.frame.check_inputs()
        self.assertTrue(any("'Blue' is used by more than one band" in problem for problem in problems))
        self.frame.start_run()
        self.assertFalse(self.frame.running)
        self.assertIn('is used by more than one band', self.frame.band_info.cget('text'))

    def test_a_band_without_a_name_is_reported_and_not_written(self):
        self.frame.band_table.add_row({'center': '500', 'width': '10', 'reflectance': '0.5'})
        self.assertEqual(len(self.frame.model.get('PANEL.band_center_wavelengths')), 6)
        _, problems = self.frame.problems()
        self.assertTrue(any('needs a name' in problem for problem in problems))

    def test_setting_the_band_keys_updates_the_table(self):
        self.frame.set_value('PANEL.reflectance_by_band', {'Blue': '0.45'})                 # added to the bands
        self.assertEqual(self.frame.band_table.get_rows()[0]['reflectance'], '0.45')
        self.assertEqual(len(self.names()), 6)
        self.frame.set_value('INPUTS.ignored_bands', 'LWIR, Thermal')
        rows = self.frame.band_table.get_rows()
        self.assertEqual([(r['name'], r['process']) for r in rows[-2:]], [('LWIR', False), ('Thermal', False)])
        self.frame.set_value('PANEL.reflectance_by_band', {'Red': '0.4'}, replace=True)    # replaces the reflectances
        self.assertEqual(self.frame.band_table.get_rows()[0]['reflectance'], '')
        self.assertEqual(self.frame.model.get('PANEL.reflectance_by_band'), {'Red': '0.4'})

    def test_the_typed_reflectance_is_not_needed_when_the_spectral_file_is_the_source(self):
        self.row('Blue')['vars']['reflectance'].set('oops')
        self.assertIn('PANEL.reflectance_by_band', self.frame.model.config()[1])
        self.frame.set_value('PANEL.reflectance_source', 'spectral_file')
        config, errors = self.frame.model.config()
        self.assertNotIn('PANEL.reflectance_by_band', errors)
        self.assertEqual(config['PANEL.reflectance_by_band']['Blue'], 0.5)                 # placeholder, not used
        self.assertEqual(config['PANEL.reflectance_by_band']['Green'], 0.478)              # readable values are kept
        self.assertTrue(all(entry.instate(['disabled']) for entry in self.frame.band_table.reflectance_entries()))
        self.assertEqual(str(self.row('Blue')['widgets']['reflectance'].cget('style')), 'TEntry')

    def test_a_profile_replaces_the_bands(self):
        path = Path(self.folder.name) / 'profiles' / 'three.json'
        path.parent.mkdir()
        path.write_text(json.dumps({
            'PANEL.reflectance_by_band': {'Green': 0.5, 'Red': 0.5, 'NIR': 0.5},
            'PANEL.band_center_wavelengths': {'Green': 550.0, 'Red': 650.0, 'NIR': 800.0},
            'PANEL.band_bandwidths': {'Green': 40.0, 'Red': 40.0, 'NIR': 40.0},
            'INPUTS.ignored_bands': 'LWIR'}), encoding='utf-8')
        self.assertTrue(self.frame.load_profile(path))
        self.assertEqual(self.names(), ['Green', 'Red', 'NIR', 'LWIR'])
        self.assertEqual([r['process'] for r in self.frame.band_table.get_rows()], [True, True, True, False])
        config = self.frame.model.config()[0]
        self.assertEqual(list(config['PANEL.band_center_wavelengths']), ['Green', 'Red', 'NIR'])
        saved = Path(self.folder.name) / 'profiles' / 'back.json'
        self.assertTrue(self.frame.save_profile(saved))
        self.assertEqual(json.loads(saved.read_text(encoding='utf-8'))['INPUTS.ignored_bands'], 'LWIR')

    def test_a_preset_fills_the_table_and_keeps_the_typed_reflectance_of_the_same_bands(self):
        self.assertIn('micasense_rededge_mx', self.frame.presets)
        self.frame.apply_preset('micasense_rededge_mx')
        self.assertEqual(self.names(), ['Blue', 'Green', 'Red', 'NIR', 'Red edge'])
        self.assertEqual(self.frame.band_table.get_rows()[0]['reflectance'], '0.477')      # kept: same name
        self.assertEqual(self.frame.band_table.get_rows()[2]['width'], '14')
        self.assertNotIn('Panchro', self.frame.model.get('PANEL.reflectance_by_band'))      # the panchromatic band is gone
        self.assertIn('Preset: MicaSense RedEdge-M', self.frame.band_info.cget('text'))
        self.assertEqual(self.frame.model.config()[0]['INPUTS.file_pattern'], rct_pattern())

    def test_a_preset_can_leave_a_band_out(self):
        self.frame.apply_preset('micasense_altum')
        self.assertEqual(self.names(), ['Blue', 'Green', 'Red', 'NIR', 'Red edge', 'LWIR'])
        self.assertEqual(self.frame.model.config()[0]['INPUTS.ignored_bands'], 'LWIR')
        self.assertEqual(self.frame.band_table.get_rows()[-1]['process'], False)

    def test_the_selected_preset_is_applied_by_its_label(self):
        label = next(label for label, key in self.frame.preset_labels.items() if key == 'micasense_altum')
        self.frame.preset_var.set(label)
        self.frame.apply_selected_preset()
        self.assertEqual(len(self.names()), 6)
        self.frame.preset_var.set('')
        self.frame.apply_selected_preset()                                                  # nothing selected: nothing
        self.assertEqual(len(self.names()), 6)

    def test_the_bands_are_read_from_the_images_of_the_panel_capture(self):
        self.row('Blue')['vars']['reflectance'].set('0.4')
        self.frame.detect_bands()
        self.assertTrue(self.frame.detecting)
        self.pump(lambda: not self.frame.detecting, 60)
        rows = self.frame.band_table.get_rows()
        self.assertEqual([(r['name'], r['center'], r['width']) for r in rows],
                         [('Blue', '475', '32'), ('Green', '560', '27'), ('Red', '668', '14'),
                          ('NIR', '842', '57'), ('Red edge', '717', '12'), ('Panchro', '634.5', '463')])
        self.assertEqual(rows[0]['reflectance'], '0.4')                                     # typed values are kept
        self.assertEqual(rows[2]['reflectance'], '0.478')
        self.assertTrue(all(r['process'] for r in rows))
        text = self.frame.band_info.cget('text')
        self.assertIn('Read 6 bands from IMG_0005: MicaSense RedEdge-P', text)
        self.assertEqual(self.frame.band_buttons['detect'].instate(['disabled']), False)

    def test_detection_reports_a_capture_that_does_not_exist(self):
        self.frame.set_value('INPUTS.reference_panel_capture', 'IMG_9999')
        self.frame.detect_bands()
        self.pump(lambda: not self.frame.detecting, 60)
        self.assertIn('FileNotFoundError', self.frame.band_info.cget('text'))
        self.assertEqual(len(self.names()), 6)                                              # the table is not changed

    def test_detection_leaves_a_thermal_band_out(self):
        detected = {'camera': 'MicaSense Altum', 'serial': 'AL06-1', 'bands': [
            {'name': 'Blue', 'center_nm': 475.0, 'bandwidth_nm': 32.0, 'thermal': False},
            {'name': 'LWIR', 'center_nm': 11000.0, 'bandwidth_nm': 6000.0, 'thermal': True}]}
        self.frame.apply_detected_bands(detected, 'IMG_0001')
        self.assertEqual(self.names(), ['Blue', 'LWIR'])
        self.assertEqual(self.frame.model.config()[0]['INPUTS.ignored_bands'], 'LWIR')
        text = self.frame.band_info.cget('text')
        self.assertIn('serial AL06-1', text)
        self.assertIn('Not used because they look thermal: LWIR', text)

    def test_fewer_bands_can_be_run_from_the_window(self):
        for name in ('Panchro', 'Red edge', 'NIR'):
            self.row(name)['vars']['process'].set(False)
        self.frame.set_value('INPUTS.results_folder', self.results('three'))
        self.frame.start_run()
        self.pump(lambda: not self.frame.running)
        result = self.frame.last_result
        self.assertEqual(result['status'], 'completed', result['errors'])
        self.assertEqual(sorted(Path(path).name for path in result['outputs']),
                         ['IMG_0010_1.tif', 'IMG_0010_2.tif', 'IMG_0010_3.tif'])

    def test_a_long_page_scrolls_and_a_short_one_does_not(self):
        self.root.attributes('-alpha', 0.0)
        self.root.deiconify()
        self.root.geometry('1000x420')
        self.frame.show_page('PANEL')
        self.root.update()
        self.assertTrue(self.frame.scroll_pages['PANEL'].scrolls())
        self.root.geometry('1000x1900')
        self.root.update()
        self.assertFalse(self.frame.scroll_pages['PANEL'].scrolls())
        self.assertNotIn('RUN', self.frame.scroll_pages)                                   # the log scrolls by itself

    def inside(self, widget, page):
        return str(widget).startswith(str(self.frame.scroll_pages[page].body))

    def test_the_dls_correction_and_its_checks_are_one_page(self):
        titles = [self.frame.navigation.get(index).strip() for index in range(self.frame.navigation.size())]
        self.assertEqual(titles, ['Inputs', 'Reference panel', 'DLS', 'Run and results'])
        self.assertEqual(self.frame.scroll_pages['CORRECTION'].body.winfo_children()[0].cget('text'),
                         'DLS (Downwelling Light Sensor)')
        keys = ['CORRECTION.use_dls'] + [field['key'] for field in wf.FIELDS if field['key'].startswith('CHECKS.')]
        self.assertEqual(len(keys), 7)
        for key in keys:
            self.assertIn(key, self.frame.variables)
        for key in keys[1:]:
            self.assertTrue(self.inside(self.frame.entries[key], 'CORRECTION'), key)
        self.assertNotIn('CHECKS', self.frame.pages)

    def test_the_dls_page_has_a_heading_for_each_part(self):
        headings = [child.cget('text') for child in self.frame.scroll_pages['CORRECTION'].body.winfo_children()
                    if child.winfo_class() == 'TLabel']
        self.assertIn('Correction with the DLS', headings)
        self.assertIn('Checks of the DLS data (limits)', headings)
        self.assertLess(headings.index('Correction with the DLS'), headings.index('Checks of the DLS data (limits)'))

    def test_the_panel_region_tool_is_on_the_inputs_page_next_to_the_panel_captures(self):
        self.assertTrue(self.inside(self.frame.entries['PANEL.manual_regions_file'], 'INPUTS'))
        self.assertTrue(self.inside(self.frame.buttons['regions'], 'INPUTS'))
        self.assertFalse(self.inside(self.frame.buttons['regions'], 'PANEL'))
        self.assertNotIn('PANEL.manual_regions_file', [key for key in self.frame.entries
                                                       if self.inside(self.frame.entries[key], 'PANEL')])
        order = wf.FIELD_KEYS
        self.assertLess(order.index('INPUTS.reference_panel_capture_after'), order.index('PANEL.manual_regions_file'))
        self.assertLess(order.index('PANEL.manual_regions_file'), order.index('INPUTS.image_captures'))
        run_page_buttons = [child.cget('text') for child in self.frame.pages['RUN'].winfo_children()[1].winfo_children()]
        self.assertEqual(run_page_buttons, ['Check inputs', 'Run', 'Open results folder'])

    def test_the_inputs_page_has_headings_for_its_groups(self):
        headings = [child.cget('text') for child in self.frame.scroll_pages['INPUTS'].body.winfo_children()
                    if child.winfo_class() == 'TLabel']
        self.assertEqual(headings[1:], ['Folders and tools', 'Reference panel', 'Reference panel regions',
                                        'Images to correct'])

    def test_the_explanations_follow_the_width_of_the_page(self):
        self.root.attributes('-alpha', 0.0)
        self.root.deiconify()
        self.root.geometry('1000x760')
        self.root.update()
        self.assertGreater(len(self.frame.wrapped_labels), 20)
        narrow = {int(label.cget('wraplength')) for label in self.frame.wrapped_labels}
        self.assertEqual(len(narrow), 1)
        self.root.geometry('1600x760')
        self.root.update()
        wide = {int(label.cget('wraplength')) for label in self.frame.wrapped_labels}
        self.assertEqual(len(wide), 1)
        self.assertGreater(wide.pop(), narrow.pop() + 400)
        self.assertGreater(max(int(label.cget('wraplength')) for label in self.frame.wrapped_labels), 1200)

    def test_the_help_of_a_field_uses_the_width_of_the_page_on_every_page(self):
        self.root.attributes('-alpha', 0.0)
        self.root.deiconify()
        self.root.geometry('1500x760')
        self.root.update()
        width = self.frame.content.winfo_width() - 100
        for key in ('INPUTS.images_folder', 'CHECKS.jump_threshold', 'PANEL.panels_used'):
            page = rw.page_of(key)
            helps = [label for label in self.frame.wrapped_labels
                     if label.cget('text') == next(f for f in wf.FIELDS if f['key'] == key)['help']]
            self.assertEqual(len(helps), 1, key)
            self.assertEqual(int(helps[0].cget('wraplength')), width, key)

    def dls_labels(self):
        return [child.cget('text') for child in self.frame.scroll_pages['CORRECTION'].body.winfo_children()
                if child.winfo_class() == 'TLabel']

    def test_the_dls_page_explains_the_ratio_and_where_the_zenith_comes_from(self):
        texts = ' '.join(self.dls_labels())
        for phrase in ('E_panel / E_image', 'XMP:HorizontalIrradiance', 'Example: 0.885 means that the light was 13 % stronger',
                       'the reflectance is multiplied by 0.885', 'XMP:EstimatedDirectLightVector',
                       'the solar zenith, 90 degrees minus XMP:SolarElevation', 'tilted DLS or inconsistent data',
                       'Only the metadata of the image and of the panel capture is used'):
            self.assertIn(phrase, texts)
        self.assertIn('Checks of the DLS data (limits)', self.dls_labels())

    def test_each_part_of_the_dls_page_has_its_explanation_under_its_heading(self):
        labels = self.dls_labels()
        for group, start in (('Correction with the DLS', 'The panel photo sets the reflectance scale'),
                             ('Checks of the DLS data (limits)', 'Flags only')):
            self.assertEqual(labels[labels.index(group) + 1][:len(start)], start)

    def test_the_limits_say_what_they_mean(self):
        helps = {field['key']: field['help'] for field in wf.FIELDS if field['key'].startswith(('CHECKS.', 'CORRECTION.'))}
        self.assertIn('DLS ratio E_panel / E_image', helps['CORRECTION.use_dls'])
        self.assertIn('median ratio', helps['CHECKS.band_ratio_tolerance'])
        self.assertIn('twice as much', helps['CHECKS.ratio_range_min'])
        self.assertIn('half as much', helps['CHECKS.ratio_range_max'])
        self.assertIn('90 degrees minus XMP:SolarElevation', helps['CHECKS.max_sun_angle_discrepancy_deg'])
        self.assertIn('spike', helps['CHECKS.jump_threshold'])
        self.assertIn('step', helps['CHECKS.jump_threshold'])
        self.assertIn('second panel captures', helps['CHECKS.panel_drift_tolerance'])

    def test_the_use_of_two_panels_is_a_choice_in_the_panel_page(self):
        self.assertTrue(self.inside(self.frame.entries['PANEL.panels_used'], 'PANEL'))
        self.assertEqual(self.frame.entries['PANEL.panels_used'].winfo_class(), 'TCombobox')
        self.assertEqual(list(self.frame.entries['PANEL.panels_used'].cget('values')), ['first', 'mean', 'interpolate_time', 'nearest_time'])
        self.frame.set_value('PANEL.panels_used', 'mean')
        _, problems = self.frame.problems()
        self.assertTrue(any('no second panel capture' in problem for problem in problems))
        self.frame.set_value('INPUTS.reference_panel_capture_after', 'IMG_0005')
        self.assertEqual(self.frame.problems()[1], [])

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


class CaptureChoiceTests(unittest.TestCase):
    setUp = WindowTests.setUp
    tearDown = WindowTests.tearDown

    def test_panel_captures_are_chosen_among_the_captures_of_the_folder(self):
        box = self.frame.entries['INPUTS.reference_panel_capture']
        self.assertEqual(box.winfo_class(), 'TCombobox')
        self.assertEqual(self.frame.capture_names(), ['IMG_0005', 'IMG_0010'])
        self.frame._refresh_capture_choices('INPUTS.reference_panel_capture_after')
        self.assertEqual(list(self.frame.entries['INPUTS.reference_panel_capture_after'].cget('values')),
                         ['', 'IMG_0005', 'IMG_0010'])
        self.frame.set_value('INPUTS.images_folder', 'does/not/exist')
        self.assertEqual(self.frame.capture_names(), [])

    def test_captures_to_correct_are_chosen_from_a_list(self):
        with mock.patch.object(rw, 'pick_names', return_value=['IMG_0010']) as picker:
            self.frame.set_value('INPUTS.reference_panel_capture', 'IMG_0005')
            self.frame.choose_captures()
        self.assertEqual(picker.call_args[0][1], ['IMG_0010'])             # the panel capture is not offered
        self.assertEqual(self.frame.model.get('INPUTS.image_captures'), 'IMG_0010')      # shown in the field, even if it is all
        self.frame.set_value('INPUTS.reference_panel_capture', 'IMG_0001')
        with mock.patch.object(rw, 'pick_names', return_value=['IMG_0010']):
            self.frame.choose_captures()
        self.assertEqual(self.frame.model.get('INPUTS.image_captures'), 'IMG_0010')
        with mock.patch.object(rw, 'pick_names', return_value=None):
            self.frame.choose_captures()
        self.assertEqual(self.frame.model.get('INPUTS.image_captures'), 'IMG_0010')
        self.frame._update_captures_summary()
        self.assertEqual(self.frame.captures_summary.cget('text'), '1 capture(s) will be corrected: IMG_0010.')
        self.frame.set_value('INPUTS.image_captures', 'IMG_0010, IMG_7')
        self.frame._update_captures_summary()
        self.assertIn('Not found in the images folder: IMG_7', self.frame.captures_summary.cget('text'))
        self.frame.set_value('INPUTS.image_captures', '')
        self.frame.set_value('INPUTS.reference_panel_capture', 'IMG_0005')
        self.frame._update_captures_summary()
        self.assertEqual(self.frame.captures_summary.cget('text'),
                         'All the other captures of the folder will be corrected (1): IMG_0010.')

    def test_the_pattern_preview_and_the_suggestion(self):
        self.frame._update_pattern_preview()
        text = self.frame.pattern_preview.cget('text')
        self.assertIn('<capture>_<index>.tif', text)
        self.assertIn('IMG_0005_1.tif  ->  capture IMG_0005, file 1', text)
        self.assertIn('2 captures (2 with 6 files)', text)
        self.frame.set_value('INPUTS.file_pattern', r'(?P<capture>X.+)\.tif$')
        self.frame._update_pattern_preview()
        self.assertEqual(self.frame.pattern_preview.cget('style'), 'Error.TLabel')
        self.frame.suggest_pattern()
        self.assertEqual(self.frame.model.get('INPUTS.file_pattern'), r'(?P<capture>.+)_(?P<index>\d+)\.tif$')
        self.frame.set_value('INPUTS.file_pattern', '(unclosed')
        self.frame._update_pattern_preview()
        self.assertIn('not a valid regular expression', self.frame.pattern_preview.cget('text'))


    def test_the_source_of_the_panel_reflectance_is_a_dropdown_of_descriptions(self):
        box = self.frame.entries['PANEL.reflectance_source']
        self.assertEqual(box.winfo_class(), 'TCombobox')
        self.assertEqual(list(box.cget('values')), ['Values typed in the table of the bands', 'Spectral response file (.csv)'])
        self.assertEqual(box.get(), 'Values typed in the table of the bands')
        box.set('Spectral response file (.csv)')                                    # the stored name is what the program uses
        self.assertEqual(self.frame.model.get('PANEL.reflectance_source'), 'spectral_file')
        self.frame.set_value('PANEL.reflectance_source', 'values')
        self.assertEqual(box.get(), 'Values typed in the table of the bands')
        self.frame.variables['PANEL.reflectance_source'].set('spectral_file')
        self.assertEqual(box.get(), 'Spectral response file (.csv)')


if __name__ == '__main__':
    unittest.main()
