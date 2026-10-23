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
import camera_presets
import radiometric_correction_tools as rct
import radiometric_workflow as wf
from ui_theme import COLORS, FONT, apply_dark_theme, hide_title_icon
from ui_widgets import BandTable, ScrollablePage, model_from_rows, pick_names, rows_from_model

DEFAULT_CONFIG_NAME = 'radiometric_inputs.json'

# The keys that the band table edits: the reflectance, center wavelength and bandwidth of each band that is used, and the
# bands that are in the files but are left out
BAND_KEYS = ('PANEL.reflectance_by_band', 'PANEL.band_center_wavelengths', 'PANEL.band_bandwidths',
             'INPUTS.ignored_bands')

# Pages of the window: key prefix, title. The last page ('RUN') is not made of fields
SECTIONS = [('INPUTS', 'Inputs'), ('PANEL', 'Reference panel'), ('CORRECTION', 'DLS'),
            ('RUN', 'Run and results')]

# The keys whose prefix is one of these are shown in the page (the DLS page has the correction and its checks), and
# the keys that are shown in another page than their prefix says
PAGE_TITLES = {'CORRECTION': 'DLS (Downwelling Light Sensor)'}          # where it is not the title in the list of pages
PAGE_PREFIXES = {'CORRECTION': ('CORRECTION', 'CHECKS')}
FIELD_PAGE = {'PANEL.manual_regions_file': 'INPUTS'}

# The keys that hold the name of a capture, chosen from the captures of the images folder (or typed): whether an empty
# name is one of the choices
CAPTURE_KEYS = {'INPUTS.reference_panel_capture': False, 'INPUTS.reference_panel_capture_after': True}


def page_of(key):
    """ The page where a key is shown """
    if key in FIELD_PAGE:
        return FIELD_PAGE[key]
    prefix = key.split('.')[0]
    for page, _ in SECTIONS:
        if prefix in PAGE_PREFIXES.get(page, (page,)):
            return page
    return prefix


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
        # The typed reflectances are not used when the spectral file is the source: what cannot be read there is not an
        # error, and a placeholder (0.5) takes its place in the configuration
        if config['PANEL.reflectance_source'] == 'spectral_file' and 'PANEL.reflectance_by_band' in errors:
            del errors['PANEL.reflectance_by_band']
            config['PANEL.reflectance_by_band'] = {band: self._usable(text)
                                                    for band, text in self.raw['PANEL.reflectance_by_band'].items()}
        return config, errors

    @staticmethod
    def _usable(text):
        try:
            value = float(text)
        except ValueError:
            return 0.5
        return value if 0 < value <= 1 else 0.5

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
        self.band_table = None
        self.buttons = {}
        self.syncing = False
        self.detecting = False
        self.presets, self.preset_problems = camera_presets.load_presets(camera_presets.preset_folders(self.root_folder))
        self.summary_after = None
        self.events = queue.Queue()
        self.running = False
        self.last_result = None
        self.widget = ttk.Frame(parent)
        self._after_id = None
        self.widget.bind('<Destroy>', lambda event: self.close() if event.widget is self.widget else None)
        self._build()
        self.wrapped_labels = []
        self._collect_wrapped(self.content)
        self.content.bind('<Configure>', self._rewrap)
        self._table_from_model()
        self._show_page(SECTIONS[0][0])
        self._apply_source_state()
        self._update_spectral_summary()
        self.preview_after = None
        self._update_pattern_preview()
        if self.load_message:
            self.log(self.load_message, 'error')
        self._after_id = self.parent.after(100, self._poll)

    def close(self):
        """ Stop looking for events of the worker threads (done when the frame is destroyed) """
        if getattr(self, 'preview_after', None) is not None:
            try:
                self.parent.after_cancel(self.preview_after)
            except self.tk.TclError:
                pass
            self.preview_after = None
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
        ttk.Button(bottom, text='Save configuration', command=self.save).pack(side='right')
        ttk.Button(bottom, text='Save profile...', command=self.save_profile_as).pack(side='right', padx=(0, 6))
        ttk.Button(bottom, text='Load profile...', command=self.load_profile_from).pack(side='right', padx=(0, 6))
        self.status = ttk.Label(bottom, style='Muted.TLabel', text=f"Configuration file: {self.config_path}")
        self.status.pack(side='left')                       # packed last: a long path gives way to the buttons

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

        self.scroll_pages = {}
        for prefix, title in SECTIONS:
            if prefix == 'RUN':
                page = ttk.Frame(self.content)
                self.pages[prefix] = page
            else:
                scrolling = ScrollablePage(self.content)           # the pages of fields scroll when they are long
                self.scroll_pages[prefix] = scrolling
                self.pages[prefix] = scrolling.widget
                page = scrolling.body
            ttk.Label(page, text=PAGE_TITLES.get(prefix, title), style='Title.TLabel').pack(anchor='w', pady=(0, 10))
            if prefix == 'RUN':
                self._build_run_page(page)
            else:
                if prefix == 'PANEL':
                    self._build_band_section(page)
                for field in wf.FIELDS:
                    if page_of(field['key']) == prefix:
                        self._build_field(page, field)
        for key in BAND_KEYS:                                     # these keys live in the band table
            self.variables[key] = self.band_table

    def _build_band_section(self, page):
        tk, ttk = self.tk, self.ttk
        ttk.Label(page, text='Bands of the camera', style='Heading.TLabel').pack(anchor='w', pady=(6, 2))
        ttk.Separator(page).pack(fill='x', pady=(0, 8))
        ttk.Label(page, style='Muted.TLabel', wraplength=760, justify='left',
                  text="One row per band. The names must be those written by the camera (XMP:BandName). A band that is "
                       "not used is in the files but is left out of the processing, for example a thermal band. "
                       "The panel reflectance is the value provided with the panel for each band (see the reflectance of "
                       "the panel, below); it is typed here only when its source is 'values'.").pack(anchor='w', pady=(0, 6))
        self.band_table = BandTable(page, self._table_changed)
        self.band_table.widget.pack(anchor='w')
        tools = ttk.Frame(page)
        tools.pack(fill='x', pady=(8, 0))
        self.band_buttons = {}
        self.band_buttons['add'] = ttk.Button(tools, text='Add band', command=lambda: self.band_table.add_row())
        self.band_buttons['detect'] = ttk.Button(tools, text='Detect from capture', command=self.detect_bands)
        self.band_buttons['add'].pack(side='left')
        self.band_buttons['detect'].pack(side='left', padx=(6, 0))
        preset_line = ttk.Frame(page)                         # its own line: the descriptions of the presets are long
        preset_line.pack(fill='x', pady=(8, 0))
        ttk.Label(preset_line, text='Camera preset', style='Muted.TLabel').pack(side='left', padx=(0, 6))
        self.preset_labels = {preset['label']: key for key, preset in sorted(self.presets.items(),
                                                                              key=lambda item: item[1]['label'])}
        self.preset_var = tk.StringVar()
        self.band_buttons['preset'] = ttk.Button(preset_line, text='Apply preset', command=self.apply_selected_preset)
        self.band_buttons['preset'].pack(side='right', padx=(6, 0))
        self.preset_box = ttk.Combobox(preset_line, textvariable=self.preset_var, values=list(self.preset_labels),
                                       state='readonly')
        self.preset_box.pack(side='left', fill='x', expand=True)
        self.band_info = ttk.Label(page, text='', style='Muted.TLabel', wraplength=900, justify='left')
        self.band_info.pack(anchor='w', pady=(6, 0))
        if self.preset_problems:
            self.band_info.configure(text='Camera presets not used: ' + '; '.join(self.preset_problems), style='Error.TLabel')

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
        if key in BAND_KEYS:
            return                                           # edited in the band table
        group = field.get('group')
        if group and group != self.groups.get(page):
            self.groups[page] = group
            ttk.Label(page, text=group, style='Heading.TLabel').pack(anchor='w', pady=(10, 2))
            ttk.Separator(page).pack(fill='x', pady=(0, 8))
            if wf.GROUP_NOTES.get(group):
                ttk.Label(page, text=wf.GROUP_NOTES[group], style='Muted.TLabel', wraplength=760,
                          justify='left').pack(anchor='w', pady=(0, 10))
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
                if kind == 'choice' and field.get('labels'):      # a list of descriptions that stands for the stored names
                    labels = field['labels']
                    shown = tk.StringVar(value=labels[variable.get()] if variable.get() in labels else variable.get())
                    entry = ttk.Combobox(line, textvariable=shown, values=[labels[name] for name in field['choices']],
                                         state='readonly', width=38)
                    names = {label: name for name, label in labels.items()}
                    shown.trace_add('write', lambda *_, shown=shown, variable=variable, names=names:
                                    variable.set(names.get(shown.get(), shown.get())))
                    variable.trace_add('write', lambda *_, shown=shown, variable=variable, labels=labels:
                                       shown.set(labels.get(variable.get(), variable.get())))
                elif kind == 'choice':
                    entry = ttk.Combobox(line, textvariable=variable, values=field['choices'], state='readonly',
                                         width=18)
                elif key in CAPTURE_KEYS:                     # typed, or chosen among the captures of the folder
                    entry = ttk.Combobox(line, textvariable=variable, values=self.capture_names(CAPTURE_KEYS[key]),
                                         postcommand=lambda key=key: self._refresh_capture_choices(key))
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
                if key == 'INPUTS.image_captures':
                    ttk.Button(line, text='Select...', command=self.choose_captures).pack(side='left', padx=(6, 0))
                if key == 'INPUTS.file_pattern':
                    ttk.Button(line, text='Suggest from files', command=self.suggest_pattern).pack(side='left', padx=(6, 0))
                variable.trace_add('write', lambda *_, key=key, variable=variable: self._changed(key, variable.get()))
                self.variables[key], self.entries[key] = variable, entry
        ttk.Label(row, text=field['help'], style='Muted.TLabel', wraplength=760, justify='left').pack(anchor='w')
        if key == 'INPUTS.image_captures':
            self.captures_summary = ttk.Label(row, text='', style='Muted.TLabel', wraplength=760, justify='left')
            self.captures_summary.pack(anchor='w', pady=(3, 0))
        if key == 'INPUTS.file_pattern':
            self.pattern_preview = ttk.Label(row, text='', style='Muted.TLabel', wraplength=760, justify='left')
            self.pattern_preview.pack(anchor='w', pady=(3, 0))
        if key == 'PANEL.spectral_file':
            self.spectral_summary = ttk.Label(row, text='', style='Muted.TLabel', wraplength=760, justify='left')
            self.spectral_summary.pack(anchor='w', pady=(3, 0))
        if key == 'PANEL.manual_regions_file':
            self._build_regions_tool(page)

    def _build_regions_tool(self, page):
        """ The buttons to check the inputs and to open the tool that checks, accepts and draws the panel regions """
        ttk = self.ttk
        holder = ttk.Frame(page)
        holder.pack(fill='x', pady=(0, 9))
        line = ttk.Frame(holder)
        line.pack(anchor='w')
        self.buttons['check_here'] = ttk.Button(line, text='Check inputs', command=self.check_inputs_here)
        self.buttons['check_here'].pack(side='left')
        self.buttons['regions'] = ttk.Button(line, text='Check / draw panel regions...', command=self.draw_regions)
        self.buttons['regions'].pack(side='left', padx=(6, 0))
        self.inputs_status = ttk.Label(holder, text='', style='Muted.TLabel', wraplength=760, justify='left')
        self.inputs_status.pack(anchor='w', pady=(6, 0))
        ttk.Label(holder, style='Muted.TLabel', wraplength=760, justify='left',
                  text="Check inputs tells whether the folders, the panel capture and the bands are right, before anything "
                       "is opened. The second button opens an overview of the reference panel capture with the region "
                       "found from the QR code in each band (and the regions saved before): accept them there, or edit or "
                       "draw a band when the region is wrong or the QR code cannot be read. Every band has to be accepted. "
                       "The regions are saved in the file above (if it is empty, in panel_regions.json in the results "
                       "folder, which is then filled in above).").pack(anchor='w', pady=(4, 0))

    def _build_run_page(self, page):
        tk, ttk = self.tk, self.ttk
        buttons = ttk.Frame(page)
        buttons.pack(fill='x')
        for name, text, command in (('check', 'Check inputs', self.check_inputs),
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

    def _collect_wrapped(self, widget):
        """ The labels that wrap their text (explanations, notes, messages): they follow the width of the page """
        for child in widget.winfo_children():
            if child.winfo_class() == 'TLabel' and int(float(child.cget('wraplength') or 0)) > 0:
                self.wrapped_labels.append(child)
            self._collect_wrapped(child)

    def _rewrap(self, event=None):
        width = max(self.content.winfo_width() - 100, 300)           # the margins, the scroll bar and the indentation
        for label in self.wrapped_labels:
            label.configure(wraplength=width)

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

    def _table_changed(self):
        """ The user edited the band table: write its rows in the model """
        if self.syncing:
            return
        reflectance, centers, widths, ignored = model_from_rows(self.band_table.get_rows())
        self.model.raw['PANEL.reflectance_by_band'] = reflectance
        self.model.raw['PANEL.band_center_wavelengths'] = centers
        self.model.raw['PANEL.band_bandwidths'] = widths
        self.model.raw['INPUTS.ignored_bands'] = ignored
        problems = [message for _, _, message in self.band_table.problems()]
        if problems:
            self.band_info.configure(text='; '.join(problems), style='Error.TLabel')
        elif self.band_info.cget('style') == 'Error.TLabel':
            self.band_info.configure(text='', style='Muted.TLabel')
        self._after_change('PANEL.band_center_wavelengths')

    def _table_from_model(self):
        """ Show in the band table what the model has (after loading a profile, for example) """
        self.syncing = True
        try:
            self.band_table.set_rows(rows_from_model(
                self.model.get('PANEL.reflectance_by_band'), self.model.get('PANEL.band_center_wavelengths'),
                self.model.get('PANEL.band_bandwidths'), self.model.get('INPUTS.ignored_bands')))
        finally:
            self.syncing = False

    # --- the captures of the images folder ---------------------------------------------------------------------

    def capture_names(self, with_empty=False):
        """ Names of the captures that the file name pattern finds in the images folder ([] if it cannot be read) """
        try:
            names = rct.list_capture_names(wf.resolve_path(self.root_folder, self.model.get('INPUTS.images_folder')),
                                           self.model.get('INPUTS.file_pattern'))
        except (OSError, ValueError):
            names = []
        return ([''] if with_empty else []) + names

    def _refresh_capture_choices(self, key):
        self.entries[key].configure(values=self.capture_names(CAPTURE_KEYS[key]))

    def choose_captures(self):
        """ Choose the captures to correct in a list of those of the folder (the panel captures are left out) """
        names = [name for name in self.capture_names()
                 if name not in (self.model.get('INPUTS.reference_panel_capture').strip(),
                                 self.model.get('INPUTS.reference_panel_capture_after').strip())]
        if not names:
            self.log('There are no other captures in the images folder (check the folder and the file name pattern).', 'error')
            return
        chosen = pick_names(self.parent.winfo_toplevel(), names, wf.capture_names(self.model.get('INPUTS.image_captures')))
        if chosen is not None:
            self.set_value('INPUTS.image_captures', ', '.join(chosen))

    def suggest_pattern(self):
        folder = wf.resolve_path(self.root_folder, self.model.get('INPUTS.images_folder'))
        try:
            pattern = rct.suggest_file_pattern(path.name for path in folder.iterdir() if path.is_file())
        except OSError as error:
            self.log(f"The images folder cannot be read: {error}", 'error')
            return
        if pattern is None:
            self.log('No pattern could be suggested from the file names (expected names such as IMG_0005_1.tif).', 'error')
            return
        self.set_value('INPUTS.file_pattern', pattern)
        self.log(f"File name pattern suggested from the files of the images folder: {pattern}")

    def _update_captures_summary(self):
        """ Say which captures will be corrected (the typed ones, or all but the panel captures when the field is empty) """
        if not hasattr(self, 'captures_summary'):
            return
        typed = wf.capture_names(self.model.get('INPUTS.image_captures'))
        available = self.capture_names()
        panels = {self.model.get('INPUTS.reference_panel_capture').strip(),
                  self.model.get('INPUTS.reference_panel_capture_after').strip()}
        if typed:
            missing = [name for name in typed if name not in available]
            text = f"{len(typed)} capture(s) will be corrected: {', '.join(typed)}."
            if missing:
                text += f" Not found in the images folder: {', '.join(missing)}."
        else:
            others = [name for name in available if name not in panels]
            text = (f"All the other captures of the folder will be corrected ({len(others)}): {', '.join(others)}."
                    if others else 'There are no captures to correct in the images folder.')
        self.captures_summary.configure(text=text, style='Muted.TLabel')

    def _update_pattern_preview(self):
        """ Show how the file name pattern reads the files of the images folder """
        self.preview_after = None
        self._update_captures_summary()
        if not hasattr(self, 'pattern_preview'):
            return
        pattern = self.model.get('INPUTS.file_pattern')
        folder = wf.resolve_path(self.root_folder, self.model.get('INPUTS.images_folder'))
        try:
            template = rct.pattern_template(pattern)
            info = rct.preview_file_pattern(folder, pattern)
        except ValueError as error:
            self.pattern_preview.configure(text=str(error), style='Error.TLabel')
            return
        except OSError:
            self.pattern_preview.configure(text='The images folder cannot be read.', style='Error.TLabel')
            return
        lines = [f"The pattern reads names written as {template}: <capture> is the name of the capture, <index> the number "
                 f"of the file. In the images folder it matches {info['matched']} of {info['files']} files, in "
                 f"{info['captures']} captures"
                 + (' (' + ', '.join(f"{count} with {files} files" for files, count in info['files_per_capture'].items()) + ')'
                    if info['captures'] else '') + '.']
        for name, capture, index in info['examples']:
            lines.append(f"    {name}  ->  capture {capture}" + ('' if index is None else f", file {index}"))
        if info['unmatched']:
            lines.append('Not matched, for example: ' + ', '.join(info['unmatched']))
        self.pattern_preview.configure(text='\n'.join(lines),
                                       style='Muted.TLabel' if info['matched'] else 'Error.TLabel')

    def _after_change(self, key):
        if key in ('INPUTS.images_folder', 'INPUTS.file_pattern', 'INPUTS.image_captures', 'INPUTS.reference_panel_capture',
                   'INPUTS.reference_panel_capture_after'):
            if getattr(self, 'preview_after', None) is not None:
                self.parent.after_cancel(self.preview_after)
            self.preview_after = self.parent.after(400, self._update_pattern_preview)
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
        if self.band_table is not None:
            self.band_table.set_reflectance_state(source == 'values')

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
            self.set_value('PANEL.reflectance_source', 'values')
            current = self.model.get('PANEL.reflectance_by_band')
            self.set_value('PANEL.reflectance_by_band', {band: f"{level:.4f}" for band, level in result['levels'].items()
                                                         if band in current})
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

    def set_value(self, key, value, replace=False):
        """
        Change a field as if it was typed (used by the program and the tests). The keys of the band table take
        {band: text} dictionaries (the bands of the dictionary are added or changed; with replace=True the dictionary
        becomes the whole list of bands) and, for INPUTS.ignored_bands, text
        """
        if key in BAND_KEYS:
            if key == 'INPUTS.ignored_bands':
                self.model.raw[key] = str(value)
            else:
                merged = {} if replace else dict(self.model.raw[key])
                merged.update({str(band): str(text) for band, text in value.items()})
                self.model.raw[key] = merged
            self._table_from_model()
            self._after_change(key)
            return
        variable = self.variables[key]
        variable.set(value if isinstance(value, bool) else str(value))

    # --- bands: presets and detection ----------------------------------------------------------------------------

    def typed_reflectance(self):
        """ {band: text} of the reflectance now in the table (kept when the bands are replaced) """
        return {row['name'].strip(): row['reflectance'] for row in self.band_table.get_rows()
                if row['name'].strip() and row['reflectance'].strip()}

    def apply_preset(self, key):
        """ Fill the band table (and the file pattern) with a camera preset; the typed reflectances of bands with the
        same name are kept """
        preset = self.presets[key]
        self.band_table.set_rows(camera_presets.band_rows(preset, self.typed_reflectance()))
        if preset.get('file_pattern'):
            self.set_value('INPUTS.file_pattern', preset['file_pattern'])
        text = f"Preset: {preset['label']}."
        if preset.get('notes'):
            text += ' ' + preset['notes']
        self.band_info.configure(text=text, style='Muted.TLabel')
        self.log(f"Camera preset applied: {preset['label']}")

    def apply_selected_preset(self):
        key = self.preset_labels.get(self.preset_var.get())
        if key:
            self.apply_preset(key)

    def detect_bands(self):
        """ Read the bands from the metadata of the files of the panel capture (in a thread) """
        if self.detecting or self.running:
            return
        config, errors = self.model.config()
        relevant = [key for key in errors if key.startswith('INPUTS.')]
        if relevant:
            self.band_info.configure(text='Fix the inputs first: ' + ', '.join(relevant), style='Error.TLabel')
            return
        self.detecting = True
        self.band_buttons['detect'].state(['disabled'])
        self.band_info.configure(text='Reading the bands from the images...', style='Muted.TLabel')
        threading.Thread(target=self._detect_worker, args=(config,), daemon=True).start()

    def _detect_worker(self, config):
        try:
            self.events.put(('bands', (wf.detect_bands(config, self.root_folder),
                                       config['INPUTS.reference_panel_capture'].strip())))
        except Exception as error:
            self.events.put(('bands_failed', f"{type(error).__name__}: {error}"))

    def apply_detected_bands(self, detected, capture):
        typed = self.typed_reflectance()
        rows = [{'name': band['name'], 'center': f"{band['center_nm']:g}", 'width': f"{band['bandwidth_nm']:g}",
                 'reflectance': '' if band['thermal'] else typed.get(band['name'], ''), 'process': not band['thermal']}
                for band in detected['bands']]
        self.band_table.set_rows(rows)
        left_out = [band['name'] for band in detected['bands'] if band['thermal']]
        text = f"Read {len(rows)} bands from {capture}: {detected['camera'] or 'camera not named'}"
        if detected['serial']:
            text += f" (serial {detected['serial']})"
        text += '.'
        if left_out:
            text += f" Not used because they look thermal: {', '.join(left_out)}."
        missing = [row['name'] for row in rows if row['process'] and not row['reflectance']]
        if missing and self.model.get('PANEL.reflectance_source') == 'values':
            text += f" Type the panel reflectance of: {', '.join(missing)}."
        self.band_info.configure(text=text, style='Muted.TLabel')
        self.log(text)

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
            self.set_value(key, ConfigModel.to_raw(field, values[key]), replace=True)
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
        table = [message for _, _, message in self.band_table.problems()]
        if table:
            return config, ['Bands of the camera: ' + message for message in table]
        if errors:
            return config, [f"{self.model.fields[key]['label']} ({key}): {message}" for key, message in errors.items()]
        return config, wf.validate_config(config, self.root_folder)

    def check_inputs_here(self):
        """ Check the inputs and show the result next to the tool, without leaving the page. Returns (config, problems) """
        config, problems = self.problems()
        if problems:
            self.inputs_status.configure(text='\n'.join(problems), style='Error.TLabel')
        else:
            self.inputs_status.configure(text='The inputs are valid: the folders, the panel capture, the bands and the '
                                              'other values can be used.', style='Ok.TLabel')
        return config, problems

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
        for name in ('check', 'check_here', 'regions', 'run'):
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
        config, problems = self.check_inputs_here()                 # the inputs are checked first, in this page
        if problems:
            return
        self._set_busy(True)
        self.inputs_status.configure(text='Reading the reference panel images...', style='Muted.TLabel')
        self.log('Reading the reference panel images...')
        threading.Thread(target=self._regions_worker, args=(config,), daemon=True).start()

    def regions_path(self, config):
        return wf.resolve_path(self.root_folder, config['PANEL.manual_regions_file'] or
                               Path(config['INPUTS.results_folder']) / 'panel_regions.json')

    def _regions_worker(self, config):
        try:
            required = list(dict.fromkeys(name for name in (config['INPUTS.reference_panel_capture'].strip(),
                                                            config['INPUTS.reference_panel_capture_after'].strip())
                                          if name))
            preloaded = {name: wf.read_capture_images(config, self.root_folder, name) for name in required}
            names = list(required)                                   # the regions are only needed for the panel captures
            path = self.regions_path(config)
            saved = wf.load_panel_regions(path) if path.is_file() else {}
            origins = wf.load_panel_origins(path) if path.is_file() else {}
            self.events.put(('images', (config, names, required, preloaded, saved, origins, path)))
        except Exception as error:
            self.events.put(('regions_failed', f"{type(error).__name__}: {error}"))

    def _open_selector(self, payload):
        from panel_selector import select_panel_regions
        config, names, required, preloaded, saved, origins, path = payload
        try:
            selected = select_panel_regions(
                names, lambda name: wf.read_capture_images(config, self.root_folder, name), required, saved, origins,
                config['PANEL.percent_inner_area'], config['PANEL.saturation_threshold'],
                parent=self.parent.winfo_toplevel(), preloaded=preloaded)
            if selected is None:
                self.log('Cancelled: no region was saved.')
                self.inputs_status.configure(text='Cancelled: no region was saved.', style='Muted.TLabel')
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                parts = []
                for capture, result in selected.items():
                    found = getattr(result, 'origins', {})
                    wf.save_panel_regions(path, capture, result, found)
                    automatic = sum(origin == 'accepted_automatic' for origin in found.values())
                    parts.append(f"{capture}: {len(result)} region(s), {automatic} as found from the QR code, "
                                 f"{len(result) - automatic} drawn or edited")
                text = f"Saved the accepted regions in {path} ({'; '.join(parts)})."
                self.log(text, 'ok')
                self.inputs_status.configure(text=text, style='Ok.TLabel')
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
        elif kind == 'bands':
            self.detecting = False
            self.band_buttons['detect'].state(['!disabled'])
            self.apply_detected_bands(*payload)
        elif kind == 'bands_failed':
            self.detecting = False
            self.band_buttons['detect'].state(['!disabled'])
            self.band_info.configure(text=payload, style='Error.TLabel')
            self.log(payload, 'error')
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
    hide_title_icon(window)
    window.mainloop()
    return 0


if __name__ == '__main__':
    sys.exit(main())
