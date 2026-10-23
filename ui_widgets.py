"""
Widgets shared by the windows of this project

    ScrollablePage   a frame whose content scrolls (mouse wheel and scroll bar) when it is taller than the window
    BandTable        the bands of the camera: use, name, spectral window and panel reflectance, one row per band

The conversion between the rows of the table and the configuration (rows_from_model, model_from_rows) and the
validation of the rows (validate_rows) are plain functions, tested without a display.
"""

# Standard Library Imports
import math

# Local Imports
from ui_theme import COLORS, FONT

# ======================================================================================
# THE ROWS OF THE BAND TABLE (NO USER INTERFACE)
# ======================================================================================
# A row is {'name', 'center', 'width', 'reflectance' (texts), 'process' (bool)}. The bands that are used are written in the
# three dictionaries of the configuration (reflectance, center wavelength and bandwidth by band); the bands that are in
# the files but are not used are the names in the list of ignored bands.


def split_names(text):
    return [name.strip() for name in str(text).split(',') if name.strip()]


def rows_from_model(reflectance, centers, widths, ignored_text=''):
    """ Rows from the raw values of the configuration: {band: text} dictionaries and the text of the ignored bands """
    names = []
    for source in (centers, widths, reflectance):
        names.extend(name for name in source if name not in names)
    rows = [{'name': name, 'center': str(centers.get(name, '')), 'width': str(widths.get(name, '')),
             'reflectance': str(reflectance.get(name, '')), 'process': True} for name in names]
    rows.extend({'name': name, 'center': '', 'width': '', 'reflectance': '', 'process': False}
                for name in split_names(ignored_text) if name not in names)
    return rows


def model_from_rows(rows):
    """ The raw values of the configuration from rows: (reflectance, centers, widths, ignored text) """
    reflectance, centers, widths, ignored = {}, {}, {}, []
    for row in rows:
        name = row['name'].strip()
        if not name:
            continue
        if row['process']:
            reflectance[name], centers[name], widths[name] = row['reflectance'], row['center'], row['width']
        elif name not in ignored:
            ignored.append(name)
    return reflectance, centers, widths, ', '.join(ignored)


def _number(text):
    try:
        value = float(str(text).strip())
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def validate_rows(rows, check_reflectance=True):
    """
    Problems of the rows: [(row number from 0 or None, field, message)]. The window of a processed band must be
    positive; the reflectance is checked only when it is used (check_reflectance)
    """
    problems = []
    names = [row['name'].strip() for row in rows]
    for number, row in enumerate(rows):
        name = names[number]
        if not name:
            problems.append((number, 'name', 'the band needs a name'))
        elif names.count(name) > 1:
            problems.append((number, 'name', f"the name '{name}' is used by more than one band"))
        if not row['process']:
            continue
        for field, label in (('center', 'center wavelength'), ('width', 'bandwidth')):
            value = _number(row[field])
            if value is None or value <= 0:
                problems.append((number, field, f"{name or 'band'}: the {label} must be a number above 0"))
        if check_reflectance:
            value = _number(row['reflectance'])
            if value is None or not 0 < value <= 1:
                problems.append((number, 'reflectance', f"{name or 'band'}: the panel reflectance must be a number in (0, 1]"))
    if rows and not any(row['process'] and row['name'].strip() for row in rows):
        problems.append((None, 'process', 'at least one band must be used'))
    return problems


# ======================================================================================
# WIDGETS
# ======================================================================================

class ScrollablePage:
    """
    A page that scrolls. Put the widgets in .body; place .widget (the outer frame) where the page goes.
    The scroll bar is shown only when the content is taller than the page
    """

    def __init__(self, parent):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.ttk = tk, ttk
        self.widget = ttk.Frame(parent)
        self.canvas = tk.Canvas(self.widget, highlightthickness=0, borderwidth=0, background=COLORS['background'])
        self.scrollbar = ttk.Scrollbar(self.widget, orient='vertical', command=self.canvas.yview)
        self.body = ttk.Frame(self.canvas, padding=(0, 0, 18, 0))          # room before the scroll bar
        self.window_id = self.canvas.create_window((0, 0), window=self.body, anchor='nw')
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side='left', fill='both', expand=True)
        self.scrollbar_shown = False
        self.body.bind('<Configure>', lambda event: self._update())
        self.canvas.bind('<Configure>', self._on_canvas)
        self.widget.bind('<Enter>', lambda event: self.canvas.bind_all('<MouseWheel>', self._on_wheel))
        self.widget.bind('<Leave>', lambda event: self.canvas.unbind_all('<MouseWheel>'))

    def _on_canvas(self, event):
        self.canvas.itemconfigure(self.window_id, width=event.width)
        self._update()

    def _update(self):
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))
        needed = self.body.winfo_reqheight() > self.canvas.winfo_height() > 1
        if needed and not self.scrollbar_shown:
            self.scrollbar.pack(side='right', fill='y')
            self.scrollbar_shown = True
        elif not needed and self.scrollbar_shown:
            self.scrollbar.pack_forget()
            self.scrollbar_shown = False
            self.canvas.yview_moveto(0)

    def _on_wheel(self, event):
        if self.scrollbar_shown:
            self.canvas.yview_scroll(-1 if event.delta > 0 else 1, 'units')

    def scrolls(self):
        """ True while the content is taller than the page """
        return self.scrollbar_shown


class BandTable:
    """
    The bands of the camera, one row each: Use (a band that is not used is in the files but is left out), name (as
    written by the camera), center wavelength, bandwidth and panel reflectance. on_change is called after every edit
    """

    COLUMNS = (('process', 'Use', 4), ('name', 'Band name', 18), ('center', 'Center (nm)', 10), ('width', 'Bandwidth (nm)', 12),
               ('reflectance', 'Panel reflectance', 14))

    def __init__(self, parent, on_change=None):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.ttk = tk, ttk
        self.on_change = on_change
        self.reflectance_enabled = True
        self.quiet = False
        self.rows = []                      # {'vars': {...}, 'widgets': {...}}
        self.widget = ttk.Frame(parent)
        self.grid_frame = ttk.Frame(self.widget)
        self.grid_frame.pack(anchor='w')
        for column, (key, title, width) in enumerate(self.COLUMNS):
            ttk.Label(self.grid_frame, text=title, style='Muted.TLabel').grid(row=0, column=column, sticky='w', padx=(0, 8))

    # --- rows ---------------------------------------------------------------------------------------------------

    def set_rows(self, rows):
        """ Replace all the rows (the change is notified once) """
        self.quiet = True
        try:
            for row in self.rows:
                for widget in row['widgets'].values():
                    widget.destroy()
            self.rows = []
            for row in rows:
                self.add_row(row)
        finally:
            self.quiet = False
        self._refresh()
        self._notify()

    def get_rows(self):
        return [{'name': row['vars']['name'].get(), 'center': row['vars']['center'].get(),
                 'width': row['vars']['width'].get(), 'reflectance': row['vars']['reflectance'].get(),
                 'process': bool(row['vars']['process'].get())} for row in self.rows]

    def add_row(self, values=None):
        tk, ttk = self.tk, self.ttk
        values = dict({'name': '', 'center': '', 'width': '', 'reflectance': '', 'process': True}, **(values or {}))
        number = len(self.rows) + 1
        variables = {'process': tk.BooleanVar(value=values['process'])}
        widgets = {}
        for column, (key, title, width) in enumerate(self.COLUMNS):
            if key == 'process':
                widgets[key] = ttk.Checkbutton(self.grid_frame, variable=variables[key])
                widgets[key].grid(row=number, column=column, sticky='w', pady=1)
            else:
                variables[key] = tk.StringVar(value=values[key])
                widgets[key] = ttk.Entry(self.grid_frame, textvariable=variables[key], width=width)
                widgets[key].grid(row=number, column=column, sticky='w', padx=(0, 8), pady=1)
        row = {'vars': variables, 'widgets': widgets}
        widgets['remove'] = ttk.Button(self.grid_frame, text='Remove', width=8, command=lambda row=row: self.remove(row))
        widgets['remove'].grid(row=number, column=len(self.COLUMNS), sticky='w', pady=1)
        for variable in variables.values():
            variable.trace_add('write', lambda *_: self._edited())
        self.rows.append(row)
        if not self.quiet:
            self._refresh()
            self._notify()
        return row

    def remove(self, row):
        if row not in self.rows:
            return
        self.rows.remove(row)
        for widget in row['widgets'].values():
            widget.destroy()
        for number, other in enumerate(self.rows, start=1):          # close the gap
            for widget in other['widgets'].values():
                widget.grid_configure(row=number)
        self._refresh()
        self._notify()

    def remove_at(self, index):
        self.remove(self.rows[index])

    # --- state ------------------------------------------------------------------------------------------------

    def set_reflectance_state(self, enabled):
        """ The panel reflectance is typed only when it is the source of the reflectance """
        self.reflectance_enabled = enabled
        self._refresh()

    def reflectance_entries(self):
        return [row['widgets']['reflectance'] for row in self.rows]

    def problems(self):
        return validate_rows(self.get_rows(), self.reflectance_enabled)

    def _edited(self):
        if self.quiet:
            return
        self._refresh()
        self._notify()

    def _notify(self):
        if self.on_change is not None:
            self.on_change()

    def _refresh(self):
        """ Mark the invalid entries and disable the ones that a band that is not used does not need """
        marked = {(number, field) for number, field, _ in self.problems() if number is not None}
        for number, row in enumerate(self.rows):
            used = bool(row['vars']['process'].get())
            for field in ('name', 'center', 'width', 'reflectance'):
                widget = row['widgets'][field]
                widget.configure(style='Invalid.TEntry' if (number, field) in marked else 'TEntry')
                disabled = (field != 'name' and not used) or (field == 'reflectance' and not self.reflectance_enabled)
                widget.state(['disabled'] if disabled else ['!disabled'])


def pick_names(parent, names, selected=(), title='Choose captures'):
    """
    A modal list in which several of the names can be selected (click, ctrl-click, shift-click)
    Inputs: the names to show and those that start selected
    Outputs: the list of the names that were selected, in the order of the list, or None if the dialog was cancelled
    """
    import tkinter as tk
    from tkinter import ttk
    window = tk.Toplevel(parent)
    window.title(title)
    window.configure(background=COLORS['background'])
    window.transient(parent)
    outcome = {'value': None}
    ttk.Label(window, text='Select the captures to correct. Selecting all of them is the same as leaving the field empty.',
              style='Muted.TLabel').pack(anchor='w', padx=12, pady=(12, 6))
    holder = ttk.Frame(window)
    holder.pack(fill='both', expand=True, padx=12)
    listbox = tk.Listbox(holder, selectmode='extended', exportselection=False, activestyle='none', height=16, width=34,
                         font=FONT, background=COLORS['input'], foreground=COLORS['foreground'], borderwidth=0,
                         highlightthickness=0, selectbackground=COLORS['accent'], selectforeground=COLORS['foreground'])
    scrollbar = ttk.Scrollbar(holder, orient='vertical', command=listbox.yview)
    listbox.configure(yscrollcommand=scrollbar.set)
    listbox.pack(side='left', fill='both', expand=True)
    scrollbar.pack(side='right', fill='y')
    for name in names:
        listbox.insert('end', name)
    for number, name in enumerate(names):
        if name in selected:
            listbox.selection_set(number)

    def accept():
        outcome['value'] = [names[number] for number in listbox.curselection()]
        window.destroy()

    buttons = ttk.Frame(window)
    buttons.pack(fill='x', padx=12, pady=12)
    ttk.Button(buttons, text='OK', command=accept).pack(side='right')
    ttk.Button(buttons, text='Cancel', command=window.destroy).pack(side='right', padx=(0, 6))
    ttk.Button(buttons, text='Select all', command=lambda: listbox.selection_set(0, 'end')).pack(side='left')
    ttk.Button(buttons, text='Clear', command=lambda: listbox.selection_clear(0, 'end')).pack(side='left', padx=(6, 0))
    window.grab_set()
    window.wait_window()
    return outcome['value']
