"""
Choose captures by looking at them

A window with one picture for each capture of the images folder (one band of it, chosen in a box), to choose the capture
that shows the reference panel or the captures that are corrected. The pictures are read one after the other in the
background, so the window opens at once; the band can be changed and the pictures are read again.

    pick_captures(parent, names, paths_of, selected=(), multiple=False, none_label=None, bands=(), resolve_band=None)

The reading of a picture (thumbnail_image) has no user interface and is tested without a display.
"""

# Standard Library Imports
import queue
import threading

# Local Imports
from .ui_theme import COLORS, FONT, FONT_BOLD, hide_title_icon
from .ui_widgets import ScrollablePage

THUMBNAIL_SIZE = (168, 126)
COLUMNS = 5


def thumbnail_image(path, size=THUMBNAIL_SIZE):
    """
    A small picture of an image file (raw .tif of one band): reduced first, then stretched between its 1 % and 99.5 %
    percentiles so that dark raw images can be recognized. Returns a PIL image (8 bits, grey) that fits in size
    """
    import numpy as np
    import tifffile
    from PIL import Image
    data = np.squeeze(tifffile.imread(str(path)))
    if data.ndim == 3:                                  # color or several channels: the first one
        data = data[..., 0] if data.shape[-1] <= 4 else data[0]
    if data.ndim != 2:
        raise ValueError(f"{path} is not an image")
    step = max(1, int(min(data.shape[0] / size[1], data.shape[1] / size[0])))
    small = data[::step, ::step].astype(np.float32)
    low, high = np.percentile(small, (1, 99.5))
    if high <= low:
        high = low + 1
    picture = Image.fromarray((np.clip((small - low) / (high - low), 0, 1) * 255).astype(np.uint8))
    picture.thumbnail(size)
    return picture


class CapturePicker:
    """
    The window. names: the captures to show, in order; paths_of(name): the image files of a capture in the order of their
    numbers; selected: the names that start selected; multiple: several captures can be chosen (otherwise one);
    none_label: for a choice that may be empty, the text of the tile that stands for no capture; bands and resolve_band:
    the names of the bands that can be shown and a function that gives the position (from 0) of a band among the files
    of a capture (it may be slow: it is called in the background)
    """

    def __init__(self, root, names, paths_of, selected=(), multiple=False, none_label=None, bands=(), resolve_band=None,
                 title='Select captures', message=''):
        import tkinter as tk
        from tkinter import ttk
        from PIL import ImageTk
        self.tk, self.ttk, self.ImageTk = tk, ttk, ImageTk
        self.root = root
        self.names = list(names)
        self.paths_of = paths_of
        self.multiple = multiple
        self.none_label = none_label
        self.bands = list(bands)
        self.resolve_band = resolve_band
        self.selection = [name for name in self.names if name in set(selected)]
        if not multiple:
            self.selection = self.selection[:1]
        self.result = None
        self.cancelled = True
        self.closed = False
        self.generation = 0
        self.results = queue.Queue()
        self.tiles = {}
        self.photos = {}
        self.root.title(title)
        self.root.geometry('1000x700')
        self.root.minsize(620, 420)
        self.root.configure(background=COLORS['background'])
        self._build(message)
        hide_title_icon(self.root)
        self.root.protocol('WM_DELETE_WINDOW', self.cancel)
        self.root.bind('<Escape>', lambda event: self.cancel())
        self._refresh()
        self.load()
        self._poll()

    # --- construction ---------------------------------------------------------------------------------------

    def _build(self, message):
        tk, ttk = self.tk, self.ttk
        top = ttk.Frame(self.root, padding=(12, 10, 12, 4))
        top.pack(fill='x')
        ttk.Label(top, text=message or ('Click the captures to correct' if self.multiple else 'Click the capture to use'),
                  style='Heading.TLabel').pack(side='left')
        self.band_var = tk.StringVar(value=self.bands[0] if self.bands else '')
        if self.bands:
            self.band_box = ttk.Combobox(top, textvariable=self.band_var, values=self.bands, state='readonly', width=18)
            self.band_box.pack(side='right')
            ttk.Label(top, text='Band shown', style='Muted.TLabel').pack(side='right', padx=(0, 6))
            self.band_box.bind('<<ComboboxSelected>>', lambda event: self.load())
        self.status = ttk.Label(self.root, text='', style='Muted.TLabel', padding=(12, 0, 12, 4))
        self.status.pack(fill='x')

        bottom = ttk.Frame(self.root, padding=(12, 6, 12, 10))
        bottom.pack(side='bottom', fill='x')
        self.ok_button = ttk.Button(bottom, text='OK', command=self.accept)
        self.ok_button.pack(side='right')
        ttk.Button(bottom, text='Cancel', command=self.cancel).pack(side='right', padx=(0, 6))
        if self.multiple:
            ttk.Button(bottom, text='Select all', command=self.select_all).pack(side='left')
            ttk.Button(bottom, text='Clear', command=self.clear).pack(side='left', padx=(6, 0))
        self.count_label = ttk.Label(bottom, style='Muted.TLabel')
        self.count_label.pack(side='left', padx=(12, 0))

        self.page = ScrollablePage(self.root)
        self.page.widget.pack(fill='both', expand=True, padx=(12, 0))
        entries = ([None] if self.none_label else []) + self.names
        for number, name in enumerate(entries):
            row, column = divmod(number, COLUMNS)
            tile = tk.Frame(self.page.body, background=COLORS['panel'], highlightthickness=2,
                            highlightbackground=COLORS['border'], padx=4, pady=4)
            tile.grid(row=row, column=column, padx=(0, 10), pady=(0, 10), sticky='n')
            picture = tk.Label(tile, background=COLORS['panel'], foreground=COLORS['muted'], font=FONT,
                               width=THUMBNAIL_SIZE[0] // 7, height=THUMBNAIL_SIZE[1] // 16,
                               text=(self.none_label if name is None else 'reading...'), wraplength=THUMBNAIL_SIZE[0] - 10)
            picture.pack()
            caption = tk.Label(tile, text='No capture' if name is None else name, background=COLORS['panel'],
                               foreground=COLORS['foreground'], font=FONT_BOLD)
            caption.pack(pady=(3, 0))
            key = '' if name is None else name
            for widget in (tile, picture, caption):
                widget.bind('<Button-1>', lambda event, key=key: self.toggle(key))
                widget.bind('<Double-Button-1>', lambda event, key=key: self.double(key))
            self.tiles[key] = {'frame': tile, 'picture': picture, 'caption': caption}

    # --- the choice ----------------------------------------------------------------------------------------------

    def toggle(self, key):
        """ Click on a tile: select it (several: add it or take it away); the tile 'no capture' has the key '' """
        if self.multiple:
            if key in self.selection:
                self.selection.remove(key)
            else:
                self.selection = [name for name in self.names if name in set(self.selection) | {key}]
        else:
            self.selection = [key] if key else []
        self._refresh()

    def double(self, key):
        if not self.multiple:
            self.selection = [key] if key else []
            self._refresh()
            self.accept()

    def select_all(self):
        self.selection = list(self.names)
        self._refresh()

    def clear(self):
        self.selection = []
        self._refresh()

    def _refresh(self):
        chosen = set(self.selection)
        for key, tile in self.tiles.items():
            selected = (key in chosen) if key else (not self.multiple and not self.selection)
            color = COLORS['tile_selected'] if selected else COLORS['panel']
            for widget in (tile['frame'], tile['picture'], tile['caption']):
                widget.configure(background=color)
            tile['frame'].configure(highlightbackground=COLORS['manual'] if selected else COLORS['border'])
        if self.multiple:
            self.count_label.configure(text=f"{len(self.selection)} of {len(self.names)} selected")
        else:
            self.count_label.configure(text=self.selection[0] if self.selection else
                                       ('No capture' if self.none_label else 'Nothing selected'))
        usable = bool(self.selection) or bool(self.none_label) or self.multiple
        self.ok_button.state(['!disabled'] if usable else ['disabled'])

    def accept(self):
        if self.multiple:
            self.result = [name for name in self.names if name in set(self.selection)]
        elif self.selection:
            self.result = self.selection[0]
        elif self.none_label:
            self.result = ''
        else:
            return
        self.cancelled = False
        self._close()

    def cancel(self):
        self.cancelled = True
        self._close()

    def _close(self):
        self.closed = True
        self.generation += 1
        self.root.destroy()

    # --- the pictures ----------------------------------------------------------------------------------------------

    def load(self):
        """ Read the pictures of the band that is shown, one after the other, in the background """
        self.generation += 1
        generation, band = self.generation, self.band_var.get()
        for key, tile in self.tiles.items():
            if key:
                tile['picture'].configure(image='', text='reading...')
        self.status.configure(text=f"Reading the pictures{' of the band ' + band if band else ''}...")
        threading.Thread(target=self._read_all, args=(generation, band), daemon=True).start()

    def _read_all(self, generation, band):
        position, problem = 0, ''
        if band and self.resolve_band is not None:
            try:
                position = self.resolve_band(band)
            except Exception as error:
                problem = f"The band {band} could not be found ({type(error).__name__}: {error}): the first file of each capture is shown."
        for name in self.names:
            if self.generation != generation:
                return
            try:
                paths = self.paths_of(name)
                self.results.put((generation, name, thumbnail_image(paths[min(position, len(paths) - 1)]), ''))
            except Exception as error:
                self.results.put((generation, name, None, f"{type(error).__name__}: {error}"))
        self.results.put((generation, None, None, problem))

    def _poll(self):
        if self.closed:
            return
        try:
            while True:
                generation, name, picture, problem = self.results.get_nowait()
                if generation != self.generation:
                    continue
                if name is None:
                    self.status.configure(text=problem or 'Click a picture to choose it.' if not self.multiple
                                          else (problem or 'Click the pictures to choose them (Select all and Clear help).'))
                elif picture is None:
                    self.tiles[name]['picture'].configure(image='', text=problem[:60])
                else:
                    self.photos[name] = self.ImageTk.PhotoImage(picture)
                    self.tiles[name]['picture'].configure(image=self.photos[name], text='', width=0, height=0)
        except queue.Empty:
            pass
        try:
            self.root.after(40, self._poll)
        except self.tk.TclError:
            self.closed = True


def pick_captures(parent, names, paths_of, selected=(), multiple=False, none_label=None, bands=(), resolve_band=None,
                  title='Select captures', message=''):
    """
    Open the window above parent and wait for the user
    Outputs: for several captures, the list of the names chosen (in the order of names); for one capture, its name ('' if
             the tile for no capture was chosen); None if the user cancelled
    """
    import tkinter as tk
    window = tk.Toplevel(parent)
    window.transient(parent)
    picker = CapturePicker(window, names, paths_of, selected, multiple, none_label, bands, resolve_band, title, message)
    window.grab_set()
    parent.wait_window(window)
    return None if picker.cancelled else picker.result
