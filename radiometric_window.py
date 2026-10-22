"""
Window for the radiometric correction workflow

The pages are built from radiometric_workflow.FIELDS (so a new key appears here without changes) and the work is done
by radiometric_workflow.run() in a separate thread, so the window stays responsive. RadiometricFrame builds its
widgets inside a ttk.Frame (its .widget): it can be placed in any window, for example as a page of the photogrammetric
workflow window.

    python radiometric_window.py [--config radiometric_inputs.json] [--root folder]

Only the values that differ from the defaults are saved in the configuration file.

ConfigModel (the values being edited and their validation) has no user interface and is tested without a display.
"""

# Standard Library Imports
import argparse
import json
import os
import queue
import sys
import threading
import traceback
from pathlib import Path

# Local Imports
import radiometric_correction_tools as rct
import radiometric_workflow as wf
from ui_theme import COLORS, FONT, apply_dark_theme

DEFAULT_CONFIG_NAME = 'radiometric_inputs.json'

# Pages of the window: key prefix, title. The last page ('RUN') is not made of fields
SECTIONS = [('INPUTS', 'Inputs'), ('PANEL', 'Reference panel'), ('CORRECTION', 'DLS correction'),
            ('CHECKS', 'DLS checks'), ('RUN', 'Run and results')]


# ======================================================================================
# VALUES BEING EDITED (NO USER INTERFACE)
# ======================================================================================

class ConfigModel:
    """
    The text typed in each field and its validation. Values are kept as typed ('raw') and converted when asked,
    so that a half-typed number never loses what the user wrote
    """

    def __init__(self, saved=None):
        self.fields = {field['key']: field for field in wf.FIELDS}
        merged = wf.merge_config(saved or {})        # unknown keys raise wf.ConfigError
        self.raw = {key: self.to_raw(self.fields[key], merged[key]) for key in self.fields}

    @staticmethod
    def to_raw(field, value):
        if field['kind'] == 'band_values':
            return {band: str(number) for band, number in value.items()}
        if field['kind'] == 'bool':
            return bool(value)
        return str(value)

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding='utf-8')))

    def get(self, key):
        return self.raw[key]

    def set(self, key, raw):
        if key not in self.fields:
            raise KeyError(key)
        self.raw[key] = raw

    def set_band(self, key, band, text):
        self.raw[key][band] = text

    @staticmethod
    def parse(field, raw):
        """ Returns (value, None) or (None, message) """
        kind = field['kind']
        try:
            if kind == 'bool':
                return bool(raw), None
            if kind in ('folder', 'file', 'text', 'choice'):
                return str(raw).strip(), None
            if kind == 'float':
                return float(raw), None
            if kind == 'int':
                number = float(raw)
                if number != int(number):
                    raise ValueError
                return int(number), None
            if kind == 'band_values':
                values, problems = {}, []
                for band, text in raw.items():
                    try:
                        values[band] = float(text)
                    except ValueError:
                        problems.append(band)
                if problems:
                    return None, f"not a number for {', '.join(problems)}"
                return values, None
        except (TypeError, ValueError):
            return None, 'must be an integer' if kind == 'int' else 'must be a number'
        return None, f"unknown kind {kind}"

    def config(self):
        """ Returns (configuration with the values that could be read, {key: message} of those that could not) """
        config, errors = wf.default_config(), {}
        for key, field in self.fields.items():
            value, message = self.parse(field, self.raw[key])
            if message:
                errors[key] = message
            else:
                config[key] = value
        return config, errors

    def overrides(self):
        """ Only the valid values that differ from the defaults """
        config, errors = self.config()
        defaults = wf.default_config()
        return {key: value for key, value in config.items() if key not in errors and value != defaults[key]}

    def save(self, path):
        path = Path(path)
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(json.dumps(self.overrides(), indent=2), encoding='utf-8')
        os.replace(temporary, path)
        return path


# ======================================================================================
# WINDOW
# ======================================================================================

class RadiometricFrame:
    """
    The pages of the workflow. It is created with a parent widget and packed by the caller:
        frame = RadiometricFrame(parent, config_path, root); frame.widget.pack(fill='both', expand=True)
    """

    def __init__(self, parent, config_path=None, root='.'):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.ttk = tk, ttk
        self.parent = parent
        self.root_folder = Path(root).resolve()
        self.config_path = Path(config_path) if config_path else self.root_folder / DEFAULT_CONFIG_NAME
        self.load_message = None
        try:
            self.model = ConfigModel.load(self.config_path) if self.config_path.is_file() else ConfigModel()
        except (ValueError, OSError) as error:
            self.model = ConfigModel()
            self.load_message = f"{self.config_path} was not used: {error}"
        self.variables = {}
        self.entries = {}
        self.pages = {}
        self.groups = {}                  # last heading written in each page
        self.source_widgets = {'values': [], 'spectral_file': []}
        self.summary_after = None
        self.events = queue.Queue()
        self.running = False
        self.last_result = None
        self.widget = ttk.Frame(parent)
        self._after_id = None
        self.widget.bind('<Destroy>', lambda event: self.close() if event.widget is self.widget else None)
        self._build()
        self._show_page(SECTIONS[0][0])
        self._apply_source_state()
        self._update_spectral_summary()
        if self.load_message:
            self.log(self.load_message, 'error')
        self._after_id = self.parent.after(100, self._poll)

    def close(self):
        """ Stop looking for events of the worker threads (done when the frame is destroyed) """
        if self.summary_after is not None:
            try:
                self.parent.after_cancel(self.summary_after)
            except self.tk.TclError:
                pass
            self.summary_after = None
        if self._after_id is not None:
            try:
                self.parent.after_cancel(self._after_id)
            except self.tk.TclError:
                pass
            self._after_id = None

    # --- construction ---------------------------------------------------------------------------------------

    def _build(self):
        tk, ttk = self.tk, self.ttk
        bottom = ttk.Frame(self.widget, padding=(12, 6, 12, 10))
        bottom.pack(side='bottom', fill='x')
        self.status = ttk.Label(bottom, style='Muted.TLabel', text=f"Configuration file: {self.config_path}")
        self.status.pack(side='left')
        ttk.Button(bottom, text='Save configuration', command=self.save).pack(side='right')
        ttk.Button(bottom, text='Save profile...', command=self.save_profile_as).pack(side='right', padx=(0, 6))
        ttk.Button(bottom, text='Load profile...', command=self.load_profile_from).pack(side='right', padx=(0, 6))

        body = ttk.Frame(self.widget)
        body.pack(fill='both', expand=True)
        self.navigation = tk.Listbox(body, exportselection=False, activestyle='none', borderwidth=0,
                                     highlightthickness=0, width=22, font=FONT, background=COLORS['panel'],
                                     foreground=COLORS['foreground'], selectbackground=COLORS['accent'],
                                     selectforeground=COLORS['foreground'])
        self.navigation.pack(side='left', fill='y', padx=(12, 0), pady=(12, 0))
        for _, title in SECTIONS:
            self.navigation.insert('end', '  ' + title)
        self.navigation.bind('<<ListboxSelect>>', self._on_navigate)
        self.content = ttk.Frame(body, padding=(16, 10, 12, 0))
        self.content.pack(side='left', fill='both', expand=True)

        for prefix, title in SECTIONS:
            page = ttk.Frame(self.content)
            self.pages[prefix] = page
            ttk.Label(page, text=title, style='Title.TLabel').pack(anchor='w', pady=(0, 10))
            if prefix == 'RUN':
                self._build_run_page(page)
            else:
                for field in wf.FIELDS:
                    if field['key'].split('.')[0] == prefix:
                        self._build_field(page, field)

    def _build_hidden(self, field):
        """ A key that is edited in a dialog: its value lives in variables, without widgets on the page """
        tk = self.tk
        key = field['key']
        if field['kind'] == 'band_values':
            self.variables[key] = {}
            for band, text in self.model.get(key).items():
                variable = tk.StringVar(value=text)
                variable.trace_add('write', lambda *_, key=key, band=band, variable=variable:
                                   self._band_changed(key, band, variable.get()))
                self.variables[key][band] = variable
        else:
            variable = tk.StringVar(value=self.model.get(key))
            variable.trace_add('write', lambda *_, key=key, variable=variable: self._changed(key, variable.get()))
            self.variables[key] = variable

    def _build_field(self, page, field):
        tk, ttk = self.tk, self.ttk
        key, kind = field['key'], field['kind']
        group = field.get('group')
        if group and group != self.groups.get(page):
            self.groups[page] = group
            ttk.Label(page, text=group, style='Heading.TLabel').pack(anchor='w', pady=(10, 2))
            ttk.Separator(page).pack(fill='x', pady=(0, 8))
        if field.get('dialog'):
            self._build_hidden(field)
            return
        row = ttk.Frame(page)
        row.pack(fill='x', pady=(0, 9))
        if kind == 'bool':
            variable = tk.BooleanVar(value=self.model.get(key))
            ttk.Checkbutton(row, text=field['label'], variable=variable).pack(anchor='w')
            variable.trace_add('write', lambda *_, key=key, variable=variable: self._changed(key, variable.get()))
            self.variables[key] = variable
        else:
            ttk.Label(row, text=field['label'], style='Heading.TLabel').pack(anchor='w')
            if kind == 'band_values':
                grid = ttk.Frame(row)
                grid.pack(anchor='w', pady=(3, 0))
                self.variables[key], self.entries[key] = {}, {}
                for column, (band, text) in enumerate(self.model.get(key).items()):
                    ttk.Label(grid, text=band, style='Muted.TLabel').grid(row=0, column=column, padx=(0, 8), sticky='w')
                    variable = tk.StringVar(value=text)
                    entry = ttk.Entry(grid, textvariable=variable, width=9)
                    entry.grid(row=1, column=column, padx=(0, 8))
                    variable.trace_add('write', lambda *_, key=key, band=band, variable=variable:
                                       self._band_changed(key, band, variable.get()))
                    self.variables[key][band], self.entries[key][band] = variable, entry
                    if key == 'PANEL.reflectance_by_band':
                        self.source_widgets['values'].append(entry)
            else:
                line = ttk.Frame(row)
                line.pack(fill='x', pady=(3, 0))
                variable = tk.StringVar(value=self.model.get(key))
                if kind == 'choice':
                    entry = ttk.Combobox(line, textvariable=variable, values=field['choices'], state='readonly',
                                         width=18)
                else:
                    entry = ttk.Entry(line, textvariable=variable, width=12 if kind in ('float', 'int') else 56)
                entry.pack(side='left', fill='x', expand=kind in ('folder', 'file', 'text'))
                if kind in ('folder', 'file'):
                    browse = ttk.Button(line, text='Browse...', command=lambda key=key, kind=kind: self._browse(key, kind))
                    browse.pack(side='left', padx=(6, 0))
                    if key == 'PANEL.spectral_file':
                        importer = ttk.Button(line, text='Import / preview...', command=self.import_spectrum)
                        importer.pack(side='left', padx=(6, 0))
                        self.source_widgets['spectral_file'].extend([entry, browse, importer])
                variable.trace_add('write', lambda *_, key=key, variable=variable: self._changed(key, variable.get()))
                self.variables[key], self.entries[key] = variable, entry
        ttk.Label(row, text=field['help'], style='Muted.TLabel', wraplength=760, justify='left').pack(anchor='w')
        if key == 'PANEL.spectral_file':
            self.spectral_summary = ttk.Label(row, text='', style='Muted.TLabel', wraplength=760, justify='left')
            self.spectral_summary.pack(anchor='w', pady=(3, 0))

    def _build_run_page(self, page):
        tk, ttk = self.tk, self.ttk
        buttons = ttk.Frame(page)
        buttons.pack(fill='x')
        self.buttons = {}
        for name, text, command in (('check', 'Check inputs', self.check_inputs),
                                    ('regions', 'Draw panel regions...', self.draw_regions),
                                    ('run', 'Run', self.start_run),
                                    ('open', 'Open results folder', self.open_results)):
            self.buttons[name] = ttk.Button(buttons, text=text, command=command)
            self.buttons[name].pack(side='left', padx=(0, 6))
        self.progress = ttk.Progressbar(page, mode='determinate', maximum=1)
        self.progress.pack(fill='x', pady=(10, 4))
        self.summary = ttk.Label(page, text='', style='Muted.TLabel', wraplength=900, justify='left')
        self.summary.pack(anchor='w', pady=(0, 6))
        frame = ttk.Frame(page)
        frame.pack(fill='both', expand=True, pady=(0, 6))
        self.log_text = tk.Text(frame, height=14, wrap='word', state='disabled', borderwidth=0, highlightthickness=0,
                                background=COLORS['panel'], foreground=COLORS['foreground'], font=('Consolas', 10),
                                insertbackground=COLORS['foreground'])
        scroll = ttk.Scrollbar(frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right', fill='y')
        self.log_text.pack(side='left', fill='both', expand=True)
        for tag, color in (('error', COLORS['error']), ('ok', COLORS['ok']), ('muted', COLORS['muted']),
                           ('flag', COLORS['automatic'])):
            self.log_text.tag_configure(tag, foreground=color)

    # --- pages and values -----------------------------------------------------------------------------------

    def _on_navigate(self, event):
        selection = self.navigation.curselection()
        if selection:
            self._show_page(SECTIONS[selection[0]][0])

    def _show_page(self, prefix):
        for page in self.pages.values():
            page.pack_forget()
        self.pages[prefix].pack(fill='both', expand=True)
        self.navigation.selection_clear(0, 'end')
        self.navigation.selection_set([name for name, _ in SECTIONS].index(prefix))

    def show_page(self, prefix):
        self._show_page(prefix)

    def _mark(self, key):
        if key not in self.entries:
            return
        field = self.model.fields[key]
        _, message = self.model.parse(field, self.model.get(key))
        if field['kind'] == 'band_values':
            for band, entry in self.entries[key].items():
                _, bad = self.model.parse({'kind': 'float'}, self.model.get(key)[band])
                entry.configure(style='Invalid.TEntry' if bad else 'TEntry')
        elif key in self.entries:
            self.entries[key].configure(style='Invalid.TEntry' if message else 'TEntry')

    def _changed(self, key, raw):
        self.model.set(key, raw)
        self._mark(key)
        self._after_change(key)

    def _band_changed(self, key, band, text):
        self.model.set_band(key, band, text)
        self._mark(key)
        self._after_change(key)

    def _after_change(self, key):
        if key == 'PANEL.reflectance_source':
            self._apply_source_state()
        if key.startswith('PANEL.spectral_') or key in ('PANEL.band_center_wavelengths', 'PANEL.band_bandwidths',
                                                       'PANEL.reflectance_source'):
            if self.summary_after is not None:
                self.parent.after_cancel(self.summary_after)
            self.summary_after = self.parent.after(400, self._update_spectral_summary)

    def _apply_source_state(self):
        """ Only the controls of the chosen source of the panel reflectance can be edited """
        source = self.model.get('PANEL.reflectance_source')
        for name, widgets in self.source_widgets.items():
            for widget in widgets:
                widget.state(['!disabled'] if name == source else ['disabled'])

    def _update_spectral_summary(self):
        self.summary_after = None
        if not hasattr(self, 'spectral_summary'):
            return
        config, errors = self.model.config()
        if not config['PANEL.spectral_file'].strip():
            self.spectral_summary.configure(text='', style='Muted.TLabel')
            return
        if errors:
            self.spectral_summary.configure(text='Fix the values marked in red to read the file.', style='Error.TLabel')
            return
        try:
            config['PANEL.reflectance_source'] = 'spectral_file'
            levels, info = wf.panel_reflectance_levels(config, self.root_folder)
        except (ValueError, OSError) as error:
            self.spectral_summary.configure(text=str(error), style='Error.TLabel')
            return
        partial = [f"{band} {100 * values['coverage']:.0f} %" for band, values in info['bands'].items()
                   if values['coverage'] < 1]
        text = 'From the file: ' + ', '.join(f"{band} {level:.4f}" for band, level in levels.items())
        if partial:
            text += '. Part of the window covered: ' + ', '.join(partial)
        self.spectral_summary.configure(text=text, style='Muted.TLabel')

    def import_spectrum(self):
        """ Open the dialog to choose and preview the spectral response file, and apply what the user accepts """
        from panel_spectrum_dialog import import_spectrum
        config, errors = self.model.config()
        try:
            fmt = wf.spectral_format(config)
        except wf.ConfigError:
            fmt = None
        windows = {band: (center, config['PANEL.band_bandwidths'].get(band, 1.0))
                   for band, center in config['PANEL.band_center_wavelengths'].items()}
        path = config['PANEL.spectral_file']
        result = import_spectrum(self.parent.winfo_toplevel(), str(wf.resolve_path(self.root_folder, path)) if path else '',
                                 fmt, windows, config['PANEL.spectral_min_coverage'])
        if result is not None:
            self.apply_import(result)

    def stored_path(self, path):
        """ A path inside the project folder is kept relative to it """
        path = Path(path)
        try:
            return path.resolve().relative_to(self.root_folder).as_posix()
        except ValueError:
            return str(path)

    def apply_import(self, result):
        """ Put what the import dialog returned in the fields (see panel_spectrum_dialog) """
        if result['action'] == 'copy_values':
            for band, level in result['levels'].items():
                if band in self.variables['PANEL.reflectance_by_band']:
                    self.variables['PANEL.reflectance_by_band'][band].set(f"{level:.4f}")
            self.set_value('PANEL.reflectance_source', 'values')
            self.log('The values of the spectral file were copied to the typed list of reflectances.')
            return
        fmt = result['format']
        self.set_value('PANEL.spectral_file', self.stored_path(result['file']))
        for key, name in (('PANEL.spectral_delimiter', 'delimiter'), ('PANEL.spectral_delimiter_char', 'delimiter_char'),
                          ('PANEL.spectral_skip_rows', 'skip_rows'), ('PANEL.spectral_skip_footer', 'skip_footer'),
                          ('PANEL.spectral_comment_prefix', 'comment_prefix'),
                          ('PANEL.spectral_exclude_rows', 'exclude_rows'),
                          ('PANEL.spectral_wavelength_min', 'wavelength_min'),
                          ('PANEL.spectral_wavelength_max', 'wavelength_max'),
                          ('PANEL.spectral_reflectance_scale', 'reflectance_scale'),
                          ('PANEL.spectral_wavelength_column', 'wavelength_column'),
                          ('PANEL.spectral_reflectance_column', 'reflectance_column'),
                          ('PANEL.spectral_wavelength_unit', 'wavelength_unit'),
                          ('PANEL.spectral_reflectance_unit', 'reflectance_unit'),
                          ('PANEL.spectral_decimal', 'decimal')):
            self.set_value(key, '' if fmt[name] is None else str(fmt[name]))
        self.set_value('PANEL.band_center_wavelengths', {band: f"{value:g}" for band, value in result['centers'].items()})
        self.set_value('PANEL.band_bandwidths', {band: f"{value:g}" for band, value in result['bandwidths'].items()})
        self.set_value('PANEL.spectral_min_coverage', f"{result['min_coverage']:g}")
        self.set_value('PANEL.reflectance_source', 'spectral_file')
        self.log(f"The panel reflectance will be computed from {result['file']}.")

    def _browse(self, key, kind):
        from tkinter import filedialog
        current = self.model.get(key)
        start = wf.resolve_path(self.root_folder, current) if current else self.root_folder
        initial = start if start.is_dir() else start.parent
        chosen = (filedialog.askdirectory(initialdir=initial) if kind == 'folder'
                  else filedialog.askopenfilename(initialdir=initial))
        if chosen:
            self.variables[key].set(chosen)

    def set_value(self, key, value):
        """ Change a field as if it was typed (used by the program and the tests) """
        variable = self.variables[key]
        if isinstance(variable, dict):
            for band, text in value.items():
                variable[band].set(str(text))
        else:
            variable.set(value if isinstance(value, bool) else str(value))

    def save(self):
        try:
            self.model.save(self.config_path)
        except OSError as error:
            self.status.configure(text=f"Could not save: {error}", style='Error.TLabel')
            return False
        _, errors = self.model.config()
        note = f" ({len(errors)} invalid value(s) were not saved)" if errors else ''
        self.status.configure(text=f"Saved to {self.config_path}{note}", style='Muted.TLabel')
        return True

    # --- profiles ---------------------------------------------------------------------------------------------------

    @property
    def profiles_folder(self):
        return self.root_folder / 'profiles'

    def save_profile(self, path):
        """ Save every value (not only the changes) in a json file, so that the profile does not depend on the defaults """
        config, errors = self.model.config()
        if errors:
            self.log('The profile was not saved: ' + '; '.join(f"{key} {message}" for key, message in errors.items()), 'error')
            return False
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(json.dumps(config, indent=2), encoding='utf-8')
        os.replace(temporary, path)
        self.status.configure(text=f"Profile saved to {path}", style='Muted.TLabel')
        return True

    def load_profile(self, path):
        """ Replace the values of all the fields with those of a profile (a json file with any of the keys) """
        try:
            values = wf.merge_config(json.loads(Path(path).read_text(encoding='utf-8')))
        except (ValueError, OSError) as error:
            self.log(f"The profile {path} was not loaded: {error}", 'error')
            self.status.configure(text=f"The profile was not loaded: {error}", style='Error.TLabel')
            return False
        for key, field in self.model.fields.items():
            self.set_value(key, ConfigModel.to_raw(field, values[key]))
        self.status.configure(text=f"Profile loaded from {path}", style='Muted.TLabel')
        return True

    def save_profile_as(self):
        from tkinter import filedialog
        self.profiles_folder.mkdir(exist_ok=True)
        path = filedialog.asksaveasfilename(initialdir=self.profiles_folder, defaultextension='.json',
                                            initialfile='radiometric_profile.json',
                                            filetypes=(('JSON files', '*.json'),))
        if path:
            self.save_profile(path)

    def load_profile_from(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(initialdir=self.profiles_folder if self.profiles_folder.is_dir()
                                          else self.root_folder, filetypes=(('JSON files', '*.json'),))
        if path:
            self.load_profile(path)

    # --- log ------------------------------------------------------------------------------------------------

    def log(self, text, tag='muted'):
        self.log_text.configure(state='normal')
        self.log_text.insert('end', text + '\n', tag)
        self.log_text.configure(state='disabled')
        self.log_text.see('end')

    def clear_log(self):
        self.log_text.configure(state='normal')
        self.log_text.delete('1.0', 'end')
        self.log_text.configure(state='disabled')

    def log_lines(self):
        return self.log_text.get('1.0', 'end').splitlines()

    # --- checks and actions ---------------------------------------------------------------------------------

    def problems(self):
        """ All the problems of the current inputs: unreadable values first, then the checks of the workflow """
        config, errors = self.model.config()
        if errors:
            return config, [f"{self.model.fields[key]['label']} ({key}): {message}" for key, message in errors.items()]
        return config, wf.validate_config(config, self.root_folder)

    def check_inputs(self):
        self.show_page('RUN')
        self.clear_log()
        config, problems = self.problems()
        for problem in problems:
            self.log(problem, 'error')
        if not problems:
            self.log('The inputs are valid.', 'ok')
        self.summary.configure(text='')
        return config, problems

    def _set_busy(self, busy):
        self.running = busy
        for name in ('check', 'regions', 'run'):
            self.buttons[name].configure(state='disabled' if busy else 'normal')

    def start_run(self):
        if self.running:
            return
        config, problems = self.check_inputs()
        if problems:
            return
        self.save()
        self._set_busy(True)
        self.summary.configure(text='Running...', style='Muted.TLabel')
        self.progress.configure(value=0)
        threading.Thread(target=self._run_worker, args=(config,), daemon=True).start()

    def _run_worker(self, config):
        try:
            result = wf.run(config, self.root_folder, lambda event: self.events.put(('progress', event)))
            self.events.put(('done', result))
        except Exception:
            self.events.put(('crash', traceback.format_exc()))

    def draw_regions(self):
        if self.running:
            return
        config, problems = self.check_inputs()
        if problems:
            return
        self._set_busy(True)
        self.log('Reading the reference panel images...')
        threading.Thread(target=self._regions_worker, args=(config,), daemon=True).start()

    def regions_path(self, config):
        return wf.resolve_path(self.root_folder, config['PANEL.manual_regions_file'] or
                               Path(config['INPUTS.results_folder']) / 'panel_regions.json')

    def _regions_worker(self, config):
        try:
            capture = config['INPUTS.reference_panel_capture']
            images = wf.read_capture_images(config, self.root_folder, capture)
            path = self.regions_path(config)
            saved = wf.load_panel_regions(path).get(capture, {}) if path.is_file() else {}
            self.events.put(('images', (config, capture, images, saved, path)))
        except Exception as error:
            self.events.put(('regions_failed', f"{type(error).__name__}: {error}"))

    def _open_selector(self, payload):
        from panel_selector import select_panel_regions
        config, capture, images, saved, path = payload
        try:
            selected = select_panel_regions(images, saved, config['PANEL.percent_inner_area'],
                                            config['PANEL.saturation_threshold'], parent=self.parent.winfo_toplevel())
            if selected is None:
                self.log('Cancelled: no region was saved.')
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                wf.save_panel_regions(path, capture, selected)
                manual = sum(polygon is not None for polygon in selected.values())
                self.log(f"Saved {manual} region(s) drawn by hand and {len(selected) - manual} band(s) left to the "
                         f"automatic detection in {path}", 'ok')
                if not config['PANEL.manual_regions_file']:
                    self.set_value('PANEL.manual_regions_file', str(path))
                    self.log('PANEL.manual_regions_file was set to this file.')
                    self.save()
        finally:
            self._set_busy(False)

    def open_results(self):
        config, errors = self.model.config()
        folder = wf.resolve_path(self.root_folder, config['INPUTS.results_folder'])
        if folder.is_dir() and hasattr(os, 'startfile'):
            os.startfile(folder)
        elif not folder.is_dir():
            self.log(f"The results folder does not exist yet: {folder}", 'error')

    # --- events from the worker threads ---------------------------------------------------------------------

    def _poll(self):
        try:
            while True:
                kind, payload = self.events.get_nowait()
                self._handle(kind, payload)
        except queue.Empty:
            pass
        if self._after_id is not None:      # not closed
            try:
                self._after_id = self.parent.after(100, self._poll)
            except self.tk.TclError:    # the window was closed
                self._after_id = None

    def _handle(self, kind, payload):
        if kind == 'progress':
            total = payload['total'] or 1
            self.progress.configure(value=payload['current'] / total)
            self.log(f"[{payload['stage']}] {payload['message']}")
        elif kind == 'done':
            self._show_result(payload)
            self._set_busy(False)
        elif kind == 'crash':
            self.log(payload, 'error')
            self.summary.configure(text='The workflow stopped unexpectedly.', style='Error.TLabel')
            self._set_busy(False)
        elif kind == 'images':
            self._open_selector(payload)
        elif kind == 'regions_failed':
            self.log(payload, 'error')
            self._set_busy(False)

    def _show_result(self, result):
        self.last_result = result
        for error in result['errors']:
            self.log(f"ERROR: {error}", 'error')
        for flag in result['flags']:
            self.log(f"FLAG: {flag}", 'flag')
        done = len(result['outputs'])
        text = f"Status: {result['status']}. {done} image(s) written"
        if 'result_file' in result:
            text += f"; details in {result['result_file']}"
        self.summary.configure(text=text, style='Ok.TLabel' if result['status'] == 'completed' else 'Error.TLabel')
        self.progress.configure(value=1 if done else 0)


def main(argv=None):
    import tkinter as tk
    parser = argparse.ArgumentParser(description="Window for the radiometric correction workflow")
    parser.add_argument('--config', help=f"configuration json file (default: {DEFAULT_CONFIG_NAME} in --root)")
    parser.add_argument('--root', default='.', help="folder from which the relative paths are resolved")
    arguments = parser.parse_args(argv)
    window = tk.Tk()
    window.title("Radiometric correction - Micasense RedEdge-P")
    window.geometry("1100x760")
    window.minsize(980, 620)
    apply_dark_theme(window)
    frame = RadiometricFrame(window, arguments.config, arguments.root)
    frame.widget.pack(fill='both', expand=True)
    window.mainloop()
    return 0


if __name__ == '__main__':
    sys.exit(main())
