"""
Dialog to import the spectral response of the reference panel (.csv): choose the file and how to read it, see what
was read from each row and the reflectance computed for each band, then use it

    ImportSession   the file, the format and what was read from it (no user interface, tested without a display)
    import_spectrum(parent, ...)   opens the dialog above a window and returns what the user chose, or None

The result is a dictionary: {'action': 'use_file' or 'copy_values', 'file', 'format', 'centers', 'bandwidths',
'min_coverage', 'levels' (reflectance by band)}. 'use_file' makes the file the source of the reflectance;
'copy_values' only copies the computed values to the list that is typed by hand.
"""

# Standard Library Imports
from pathlib import Path

# Local Imports
import panel_spectrum as ps
from ui_theme import COLORS

PREVIEW_ROWS = 300


# ======================================================================================
# LOGIC WITHOUT USER INTERFACE
# ======================================================================================

class ImportSession:
    """
    Inputs: the file, the format, the spectral window of each band {band: (center, bandwidth)} and the minimum coverage
    After refresh(): rows (as read, for the preview), spectrum or error, band_table or band_error, warnings
    """

    def __init__(self, path='', fmt=None, windows=None, min_coverage=0.5):
        self.path = str(path or '')
        self.fmt = dict(ps.DEFAULT_FORMAT, **(fmt or {}))
        self.windows = dict(windows or ps.REDEDGE_P_BANDS)
        self.min_coverage = min_coverage
        self.notes = []
        self.rows = []
        self.spectrum = None
        self.error = None
        self.band_table = None
        self.band_error = None
        self.refresh()

    def set_path(self, path):
        self.path = str(path or '')
        self.notes = []
        self.refresh()

    def set_format(self, **changes):
        self.fmt.update(changes)
        self.refresh()

    def set_windows(self, windows, min_coverage=None):
        self.windows = dict(windows)
        if min_coverage is not None:
            self.min_coverage = min_coverage
        self.refresh()

    def detect(self):
        """ Guess the format from the file; returns the notes that say what was guessed """
        try:
            fmt, notes = ps.detect_format(self.path)
        except (OSError, ps.SpectrumError) as error:
            self.error, self.notes = str(error), []
            return []
        self.fmt, self.notes = dict(ps.DEFAULT_FORMAT, **fmt), notes
        self.refresh()
        return notes

    def refresh(self):
        self.rows, self.spectrum, self.error = [], None, None
        self.band_table, self.band_error = None, None
        path = Path(self.path) if self.path else None
        if path is None:
            self.error = "Choose a file."
            return
        if not path.is_file():
            self.error = f"File not found: {path}"
            return
        try:
            self.rows = ps.parse_rows(ps.read_text_lines(path), self.fmt)
            self.spectrum = ps.load_spectrum(path, self.fmt)
        except (OSError, ps.SpectrumError) as error:
            self.error = str(error)
            return
        try:
            self.band_table = ps.band_reflectances(self.spectrum, self.windows, self.min_coverage)
        except ps.SpectrumError as error:
            self.band_error = str(error)

    # --- what is shown --------------------------------------------------------------------------------------

    def preview_rows(self, limit=PREVIEW_ROWS):
        """ [(line, text, wavelength in nm or '', reflectance or '', status)] of the first lines of the file """
        shown = []
        for row in self.rows[:limit]:
            status = row['status']
            if status == 'ok' and not 0 < row['reflectance'] <= 1:
                status = 'not used (outside 0-1)'
            shown.append((row['line'], row['raw'],
                          '' if row['wavelength'] is None else f"{row['wavelength']:g}",
                          '' if row['reflectance'] is None else f"{row['reflectance']:.6g}", status))
        return shown

    def curve(self):
        """ (wavelengths, reflectances) of the samples that are used, or empty lists """
        if self.spectrum is None:
            return [], []
        valid = (self.spectrum['reflectance'] > 0) & (self.spectrum['reflectance'] <= 1)
        return (list(self.spectrum['wavelength'][valid]), list(self.spectrum['reflectance'][valid]))

    def band_rows(self):
        """ [(band, window text, samples, covered %, reflectance)] """
        if self.band_table is None:
            return []
        return [(band, f"{values['window_nm'][0]:g} - {values['window_nm'][1]:g}", values['samples'],
                 f"{100 * values['coverage']:.0f} %", f"{values['reflectance']:.4f}")
                for band, values in self.band_table.items()]

    def messages(self):
        """ [(text, 'error' | 'warning' | 'info')] about the file, the format and the bands """
        found = []
        if self.error:
            found.append((self.error, 'error'))
        if self.band_error:
            found.append((self.band_error, 'error'))
        if self.spectrum is not None:
            found.extend((warning, 'warning') for warning in self.spectrum['warnings'])
            header = sum(1 for row in self.rows if row['status'] == 'header' and row['raw'].strip())
            found.append((f"{len(self.spectrum['wavelength'])} samples, "
                          f"{self.spectrum['wavelength'][0]:g} - {self.spectrum['wavelength'][-1]:g} nm; "
                          f"{header} header row(s) skipped; delimiter {self.spectrum['delimiter']!r}", 'info'))
        if self.band_table:
            partial = [band for band, values in self.band_table.items() if values['coverage'] < 1]
            if partial:
                found.append(("Window only partly covered by the file (averaged over the part that exists): "
                              + ', '.join(partial), 'warning'))
        return found

    def result(self, action):
        """ What the dialog returns; ValueError if the file cannot give the reflectance of every band """
        if self.band_table is None:
            raise ValueError(self.error or self.band_error or "The reflectance of the bands could not be computed.")
        return {'action': action, 'file': self.path, 'format': dict(self.fmt),
                'centers': {band: window[0] for band, window in self.windows.items()},
                'bandwidths': {band: window[1] for band, window in self.windows.items()},
                'min_coverage': self.min_coverage,
                'levels': {band: values['reflectance'] for band, values in self.band_table.items()}}


# ======================================================================================
# DIALOG
# ======================================================================================

class ImportDialog:
    """ The window; use import_spectrum() to run it """

    def __init__(self, window, session):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.ttk, self.window, self.session = tk, ttk, window, session
        self.result = None
        self.pending = None
        self.updating = False
        window.title("Import the spectral response of the panel")
        window.geometry("1180x880")
        window.minsize(1000, 720)
        self._build()
        self._load_into_fields()
        self.refresh()

    # --- construction ---------------------------------------------------------------------------------------

    def _build(self):
        tk, ttk = self.tk, self.ttk
        window = self.window
        outer = ttk.Frame(window, padding=12)
        outer.pack(fill='both', expand=True)

        line = ttk.Frame(outer)
        line.pack(fill='x')
        ttk.Label(line, text='File', style='Heading.TLabel').pack(side='left')
        self.path_var = tk.StringVar()
        ttk.Entry(line, textvariable=self.path_var).pack(side='left', fill='x', expand=True, padx=8)
        ttk.Button(line, text='Browse...', command=self.browse).pack(side='left')
        ttk.Button(line, text='Detect format', command=self.detect).pack(side='left', padx=(6, 0))
        self.path_var.trace_add('write', lambda *_: self._on_path())

        options = ttk.Frame(outer)
        options.pack(fill='x', pady=(10, 0))
        self.format_vars = {}
        specs = (('delimiter', 'Delimiter', list(ps.DELIMITERS), 10), ('skip_rows', 'Header rows', None, 6),
                 ('wavelength_column', 'Wavelength column', None, 4), ('reflectance_column', 'Reflectance column', None, 4),
                 ('wavelength_unit', 'Wavelength unit', list(ps.WAVELENGTH_UNITS), 6),
                 ('reflectance_unit', 'Reflectance unit', list(ps.REFLECTANCE_UNITS), 9),
                 ('decimal', 'Decimal mark', list(ps.DECIMAL_MARKS), 3))
        for column, (key, label, choices, width) in enumerate(specs):
            ttk.Label(options, text=label, style='Muted.TLabel').grid(row=0, column=column, sticky='w', padx=(0, 10))
            variable = tk.StringVar()
            if choices:
                widget = ttk.Combobox(options, textvariable=variable, values=choices, state='readonly', width=width)
            else:
                widget = ttk.Entry(options, textvariable=variable, width=width)
            widget.grid(row=1, column=column, sticky='w', padx=(0, 10))
            variable.trace_add('write', lambda *_: self._on_format())
            self.format_vars[key] = variable

        windows = ttk.Frame(outer)
        windows.pack(fill='x', pady=(10, 0))
        ttk.Label(windows, text='Spectral window of each band (nm)', style='Heading.TLabel').grid(
            row=0, column=0, columnspan=8, sticky='w')
        self.center_vars, self.width_vars = {}, {}
        for column, band in enumerate(self.session.windows):
            ttk.Label(windows, text=band, style='Muted.TLabel').grid(row=1, column=column, sticky='w', padx=(0, 10))
            center, width = tk.StringVar(), tk.StringVar()
            frame = ttk.Frame(windows)
            frame.grid(row=2, column=column, sticky='w', padx=(0, 10))
            ttk.Entry(frame, textvariable=center, width=5).pack(side='left')
            ttk.Label(frame, text='±').pack(side='left', padx=2)
            ttk.Entry(frame, textvariable=width, width=4).pack(side='left')
            ttk.Label(frame, text='/2', style='Muted.TLabel').pack(side='left', padx=(2, 0))
            for variable in (center, width):
                variable.trace_add('write', lambda *_: self._on_windows())
            self.center_vars[band], self.width_vars[band] = center, width
        last = len(self.session.windows)
        ttk.Label(windows, text='Min. covered', style='Muted.TLabel').grid(row=1, column=last, sticky='w')
        self.coverage_var = tk.StringVar()
        ttk.Entry(windows, textvariable=self.coverage_var, width=5).grid(row=2, column=last, sticky='w')
        self.coverage_var.trace_add('write', lambda *_: self._on_windows())

        buttons = ttk.Frame(outer)
        buttons.pack(side='bottom', fill='x', pady=(10, 0))
        self.use_button = ttk.Button(buttons, text='Use the file as the panel reflectance',
                                     command=lambda: self.accept('use_file'))
        self.use_button.pack(side='left')
        self.copy_button = ttk.Button(buttons, text='Copy the values to the typed list',
                                      command=lambda: self.accept('copy_values'))
        self.copy_button.pack(side='left', padx=(6, 0))
        ttk.Button(buttons, text='Cancel', command=self.cancel).pack(side='right')

        self.message_box = tk.Text(outer, height=3, wrap='word', state='disabled', borderwidth=0, highlightthickness=0,
                                   background=COLORS['panel'], foreground=COLORS['foreground'], font=('Segoe UI', 10))
        self.message_box.pack(side='bottom', fill='x', pady=(10, 0))
        for tag, color in (('error', COLORS['error']), ('warning', COLORS['automatic']), ('info', COLORS['muted'])):
            self.message_box.tag_configure(tag, foreground=color)

        body = ttk.Frame(outer)
        body.pack(fill='both', expand=True, pady=(12, 0))
        left = ttk.Frame(body)
        left.pack(side='left', fill='both', expand=True)
        ttk.Label(left, text='What was read from each line', style='Heading.TLabel').pack(anchor='w')
        table = ttk.Frame(left)
        table.pack(fill='both', expand=True, pady=(4, 0))
        columns = ('line', 'text', 'wavelength', 'reflectance', 'status')
        self.rows_tree = ttk.Treeview(table, columns=columns, show='headings', selectmode='browse')
        for column, (title, width) in zip(columns, (('Line', 50), ('Text', 200), ('Wavelength nm', 135),
                                                    ('Reflectance', 100), ('Status', 180))):
            self.rows_tree.heading(column, text=title)
            self.rows_tree.column(column, width=width, anchor='w', stretch=column == 'text')
        scroll = ttk.Scrollbar(table, command=self.rows_tree.yview)
        self.rows_tree.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right', fill='y')
        self.rows_tree.pack(side='left', fill='both', expand=True)
        self.rows_tree.tag_configure('header', foreground=COLORS['muted'])
        self.rows_tree.tag_configure('error', foreground=COLORS['error'])
        self.rows_tree.tag_configure('blank', foreground=COLORS['disabled'])
        self.rows_tree.tag_configure('unused', foreground=COLORS['automatic'])

        right = ttk.Frame(body)
        right.pack(side='left', fill='both', expand=True, padx=(12, 0))
        ttk.Label(right, text='Reflectance by wavelength (nm) and band windows', style='Heading.TLabel').pack(anchor='w')
        self.plot = tk.Canvas(right, height=200, background=COLORS['panel'], highlightthickness=0)
        self.plot.pack(fill='x', pady=(4, 8))
        self.plot.bind('<Configure>', lambda event: self._draw_plot())
        ttk.Label(right, text='Reflectance of each band', style='Heading.TLabel').pack(anchor='w')
        self.bands_tree = ttk.Treeview(right, columns=('band', 'window', 'samples', 'covered', 'reflectance'),
                                       show='headings', height=7, selectmode='none')
        for column, (title, width) in zip(self.bands_tree['columns'], (('Band', 90), ('Window (nm)', 120),
                                                                     ('Samples', 70), ('Covered', 70),
                                                                     ('Reflectance', 100))):
            self.bands_tree.heading(column, text=title)
            self.bands_tree.column(column, width=width, anchor='w')
        self.bands_tree.pack(fill='x', pady=(4, 0))

        window.bind('<Escape>', lambda event: self.cancel())
        window.bind('<Destroy>', lambda event: self._cancel_pending() if event.widget is self.window else None)
        window.protocol('WM_DELETE_WINDOW', self.cancel)

    # --- values <-> fields ----------------------------------------------------------------------------------

    def _load_into_fields(self):
        self.updating = True
        self.path_var.set(self.session.path)
        for key, variable in self.format_vars.items():
            variable.set(str(self.session.fmt[key]))
        for band, (center, width) in self.session.windows.items():
            self.center_vars[band].set(f"{center:g}")
            self.width_vars[band].set(f"{width:g}")
        self.coverage_var.set(f"{self.session.min_coverage:g}")
        self.updating = False

    def _cancel_pending(self):
        if self.pending is not None:
            try:
                self.window.after_cancel(self.pending)
            except self.tk.TclError:
                pass
            self.pending = None

    def _schedule(self, action):
        if self.updating:
            return
        self._cancel_pending()
        self.pending = self.window.after(250, action)

    def _on_path(self):
        self._schedule(self.apply_all)

    def _on_format(self):
        self._schedule(self.apply_all)

    def _on_windows(self):
        self._schedule(self.apply_all)

    def _read_format(self):
        changes = {}
        for key, variable in self.format_vars.items():
            text = variable.get().strip()
            if key == 'skip_rows':
                changes[key] = 'auto' if text.lower() == 'auto' else int(text)
            elif key.endswith('_column'):
                changes[key] = int(text)
            else:
                changes[key] = text
        return changes

    def _read_windows(self):
        windows = {band: (float(self.center_vars[band].get()), float(self.width_vars[band].get()))
                   for band in self.session.windows}
        coverage = float(self.coverage_var.get())
        if not 0 < coverage <= 1 or any(width <= 0 for _, width in windows.values()):
            raise ValueError
        return windows, coverage

    def apply_all(self):
        """
        Read the file, the format and the windows from the fields, in one step. What is typed that cannot be read is
        reported and the last values that could be read are kept
        """
        self._cancel_pending()
        session = self.session
        path = self.path_var.get().strip()
        problems = False
        try:
            changes = self._read_format()
        except ValueError:
            session.error = "The header rows and the columns must be whole numbers (the header rows can be 'auto')."
            session.rows, session.spectrum, session.band_table = [], None, None
            problems = True
        try:
            windows, coverage = self._read_windows()
        except ValueError:
            session.band_table = None
            session.band_error = "The windows need numbers (bandwidth above 0) and the coverage must be in (0, 1]."
            problems = True
        if not problems:
            session.path = path
            session.fmt.update(changes)
            session.windows, session.min_coverage = windows, coverage
            session.refresh()
        self.refresh()

    # --- actions --------------------------------------------------------------------------------------------

    def browse(self):
        from tkinter import filedialog
        start = Path(self.path_var.get()).parent if self.path_var.get() else Path('.')
        chosen = filedialog.askopenfilename(parent=self.window, initialdir=start if start.is_dir() else '.',
                                            filetypes=(('CSV files', '*.csv'), ('All files', '*.*')))
        if chosen:
            self.path_var.set(chosen)

    def detect(self):
        notes = self.session.detect()
        self._load_into_fields()
        self.refresh()
        if notes:
            self.show_messages(self.messages_with([('Detected: ' + '; '.join(notes), 'info')]))

    def messages_with(self, extra):
        return extra + self.session.messages()

    def accept(self, action):
        self.apply_all()
        try:
            self.result = self.session.result(action)
        except ValueError as error:
            self.show_messages([(str(error), 'error')])
            return
        self.window.destroy()

    def cancel(self):
        self.result = None
        self.window.destroy()

    # --- drawing --------------------------------------------------------------------------------------------

    def refresh(self):
        """ Show the state of the session: rows as read, band table, messages, buttons and plot """
        session = self.session
        self.rows_tree.delete(*self.rows_tree.get_children())
        for line, text, wavelength, reflectance, status in session.preview_rows():
            tag = ('error' if status.startswith('error') else 'unused' if status.startswith('not used')
                   else status if status in ('header', 'blank') else '')
            self.rows_tree.insert('', 'end', values=(line, text, wavelength, reflectance, status), tags=(tag,))
        self.bands_tree.delete(*self.bands_tree.get_children())
        for row in session.band_rows():
            self.bands_tree.insert('', 'end', values=row)
        self.show_messages(session.messages())
        ready = session.band_table is not None
        for button in (self.use_button, self.copy_button):
            button.state(['!disabled'] if ready else ['disabled'])
        self._draw_plot()

    def show_messages(self, messages):
        self.message_box.configure(state='normal')
        self.message_box.delete('1.0', 'end')
        for text, tag in messages:
            self.message_box.insert('end', text + '\n', tag)
        self.message_box.configure(state='disabled')

    def message_lines(self):
        return self.message_box.get('1.0', 'end').splitlines()

    def _draw_plot(self):
        canvas = self.plot
        canvas.delete('all')
        width, height = max(canvas.winfo_width(), 50), max(canvas.winfo_height(), 50)
        rows = len(self.session.windows)
        left, right, top, bottom = 46, 12, 8 + 9 * rows, 24
        wavelengths, reflectances = self.session.curve()
        if not wavelengths:
            canvas.create_text(width / 2, height / 2, text='No curve to show', fill=COLORS['muted'])
            return
        low = min(min(wavelengths), min(c - w / 2 for c, w in self.session.windows.values()))
        high = max(max(wavelengths), max(c + w / 2 for c, w in self.session.windows.values()))
        r_low, r_high = min(reflectances), max(reflectances)
        pad = max((r_high - r_low) * 0.15, 0.01)
        r_low, r_high = max(0.0, r_low - pad), min(1.0, r_high + pad) if r_high + pad < 1 else 1.0

        def x_of(value):
            return left + (value - low) / (high - low) * (width - left - right)

        def y_of(value):
            return top + (1 - (value - r_low) / (r_high - r_low)) * (height - top - bottom)

        canvas.create_rectangle(left, top, width - right, height - bottom, outline=COLORS['border'])
        # spectral window of each band, one thin bar per band above the plot
        for row, (band, (center, bandwidth)) in enumerate(self.session.windows.items()):
            x1, x2 = x_of(center - bandwidth / 2), x_of(center + bandwidth / 2)
            y = 6 + 9 * row + 4
            if x2 - x1 > 160:                  # a wide window: the name is written inside, before the bar
                canvas.create_text(x1 + 2, y, text=band, anchor='w', fill=COLORS['muted'], font=('Segoe UI', 8))
                canvas.create_line(x1 + 10 + 5 * len(band), y, x2, y, fill=COLORS['automatic'], width=4)
            else:
                canvas.create_line(x1, y, max(x2, x1 + 2), y, fill=COLORS['automatic'], width=4)
                canvas.create_text(x2 + 5, y, text=band, anchor='w', fill=COLORS['muted'], font=('Segoe UI', 8))                     if x2 < width - 70 else                     canvas.create_text(x1 - 5, y, text=band, anchor='e', fill=COLORS['muted'], font=('Segoe UI', 8))
        points = [coordinate for w, r in zip(wavelengths, reflectances) for coordinate in (x_of(w), y_of(r))]
        if len(points) >= 4:
            canvas.create_line(points, fill=COLORS['manual'], width=2)
        else:
            canvas.create_oval(points[0] - 3, points[1] - 3, points[0] + 3, points[1] + 3, fill=COLORS['manual'])
        for value in (low, (low + high) / 2, high):
            canvas.create_text(x_of(value), height - bottom + 12, text=f"{value:.0f}", fill=COLORS['muted'],
                               font=('Segoe UI', 8))
        for value in (r_low, (r_low + r_high) / 2, r_high):
            canvas.create_text(left - 4, y_of(value), text=f"{value:.3f}", fill=COLORS['muted'], anchor='e',
                               font=('Segoe UI', 8))


def import_spectrum(parent, path='', fmt=None, windows=None, min_coverage=0.5):
    """
    Open the dialog above a window and wait for the user
    Outputs: the dictionary described at the top of this file, or None if the user cancelled
    """
    import tkinter as tk
    from ui_theme import apply_dark_theme
    window = tk.Toplevel(parent)
    window.transient(parent)
    apply_dark_theme(window)
    dialog = ImportDialog(window, ImportSession(path, fmt, windows, min_coverage))
    window.grab_set()
    parent.wait_window(window)
    return dialog.result
