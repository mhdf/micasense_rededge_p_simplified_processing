"""
Choose a folder while looking at what it contains: the files of the folder (folders first, with their sizes) and, at the
right, a picture of the image file that is selected, so that the folder of the raw images, or the one of the results, can be
recognised before it is chosen.

    ask_folder(parent, title, initialdir)   the dialog; returns the path of the folder that was chosen, or '' if cancelled

What does not need a window is tested without a display:
    list_folder(destination)    the contents of a folder, sorted
    preview_picture(path)       a picture of an image file (stretched between its 1 % and 99.5 % percentiles)
    size_text(size)             a size in bytes as text
"""

# Standard Library Imports
import os
import queue
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk

# Local Imports
from .ui_theme import COLORS, FONT, hide_title_icon

IMAGE_SUFFIXES = ('.tif', '.tiff', '.png', '.jpg', '.jpeg')
PREVIEW_BOUNDS = (1600, 1200)             # the picture is reduced to this before it is stretched and drawn
NO_PREVIEW = "Select an image to preview it."


# ======================================================================================
# LOGIC WITHOUT USER INTERFACE
# ======================================================================================

def size_text(size):
    """ A size in bytes as text (None, for a folder, is an empty text) """
    if size is None:
        return ''
    for unit in ('B', 'KiB', 'MiB', 'GiB', 'TiB'):
        if size < 1024 or unit == 'TiB':
            return f"{size:.0f} {unit}" if unit == 'B' else f"{size:.1f} {unit}"
        size /= 1024


def list_folder(destination, base=None):
    """
    The contents of a folder: (the folder, [(name, is_folder, size in bytes or None), ...]) with the folders first and then the
    files, each group by name without regard to case. destination is a path (a relative one is taken from base; the path of a
    file stands for its folder). OSError if the folder cannot be read
    """
    target = Path(str(destination).strip().strip('"')).expanduser()
    if not target.is_absolute() and base is not None:
        target = Path(base) / target
    target = Path(os.path.abspath(target))
    if target.is_file():
        target = target.parent
    rows = []
    with os.scandir(target) as contents:
        for entry in contents:
            try:
                folder = entry.is_dir()
                size = None if folder else entry.stat().st_size
            except OSError:
                folder, size = False, None
            rows.append((entry.name, folder, size))
    rows.sort(key=lambda row: (not row[1], row[0].casefold()))
    return target, rows


def is_image(path):
    return Path(path).suffix.lower() in IMAGE_SUFFIXES


def preview_picture(path, bounds=PREVIEW_BOUNDS):
    """
    A picture of an image file for the eye only (the file is not changed): one band of a raw or reflectance .tif (16 bits,
    floating point, with or without invalid values) is reduced and stretched between its 1 % and 99.5 % percentiles, a
    colour picture (.png, .jpg) is shown as it is
    Outputs: (PIL image that fits in bounds, (width, height) of the file, text of the type of its values);
             ValueError if the file is not an image, and what the readers raise if it cannot be read
    """
    import numpy as np
    from PIL import Image
    path = Path(path)
    if path.suffix.lower() in ('.tif', '.tiff'):
        import tifffile
        data = np.squeeze(tifffile.imread(str(path)))
        kind = str(data.dtype)
    else:
        with Image.open(path) as source:
            kind = source.mode
            if source.mode not in ('L', 'I', 'I;16', 'F'):
                width, height = source.size
                picture = source.convert('RGB')
                picture.thumbnail(bounds)
                return picture, (width, height), kind
            data = np.asarray(source)
    if data.ndim == 3:                                  # several channels: the first one
        data = data[..., 0] if data.shape[-1] <= 4 else data[0]
    if data.ndim != 2:
        raise ValueError(f"{path.name} is not an image")
    height, width = data.shape
    step = max(1, int(max(width / bounds[0], height / bounds[1])))
    small = data[::step, ::step].astype(np.float32)
    valid = small[np.isfinite(small)]
    if valid.size == 0:
        raise ValueError("the image has no valid values")
    low, high = np.percentile(valid, (1, 99.5))
    if high <= low:
        high = low + 1
    stretched = np.clip((np.nan_to_num(small, nan=low) - low) / (high - low), 0, 1)
    picture = Image.fromarray((stretched * 255).astype(np.uint8)).convert('RGB')
    picture.thumbnail(bounds)
    return picture, (width, height), kind


# ======================================================================================
# WINDOW
# ======================================================================================

class ImagePreview(ttk.Frame):
    """ The picture of an image file and what it is; the file is read in another thread, so the list stays quick to use """

    def __init__(self, parent):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, background=COLORS['panel'], width=260, height=220, highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)
        self.status = tk.StringVar(value=NO_PREVIEW)
        self.label = ttk.Label(self, textvariable=self.status, wraplength=300, justify='left')
        self.label.pack(fill='x', pady=(7, 0))
        self.photo = self.picture = self.path = None
        self.requests, self.results = queue.Queue(), queue.Queue()
        self.serial = 0
        self.closed = False
        self.canvas.bind('<Configure>', self._resize)
        self.bind('<Destroy>', self._destroyed, add='+')
        threading.Thread(target=self._read, args=(self.requests, self.results), daemon=True).start()
        self.poll = self.after(50, self._poll)

    def show(self, path=None):
        """ Show the picture of a file (None: nothing); a file that is asked for while another is being read replaces it """
        self.serial += 1
        self.path = Path(path) if path else None
        self.picture = self.photo = None
        self.canvas.delete('all')
        self.status.set(f"Reading {self.path.name}..." if self.path else NO_PREVIEW)
        if self.path is not None:
            self.requests.put((self.serial, self.path))

    @staticmethod
    def _read(requests, results):
        while True:
            item = requests.get()
            while not requests.empty():                  # only the last one that was asked for is of interest
                item = requests.get()
            if item is None:
                return
            serial, path = item
            try:
                answer, error = preview_picture(path), None
            except Exception as exc:                     # a file that cannot be read is said, not raised
                answer, error = None, f"{type(exc).__name__}: {exc}"
            results.put((serial, answer, error))

    def _poll(self):
        if self.closed:
            return
        try:
            while True:
                serial, answer, error = self.results.get_nowait()
                if serial != self.serial:
                    continue
                if error:
                    self.status.set(f"{self.path.name}\nNo preview: {error}")
                else:
                    self.picture, (width, height), kind = answer
                    self.status.set(f"{self.path.name}\n{width} × {height} pixels; {kind}. Preview contrast only; "
                                    "source values are unchanged.")
                    self._draw()
        except queue.Empty:
            pass
        self.poll = self.after(50, self._poll)

    def _resize(self, event=None):
        self.label.configure(wraplength=max(100, self.winfo_width() - 4))
        self._draw()

    def _draw(self):
        if self.picture is None:
            return
        from PIL import ImageTk
        picture = self.picture.copy()
        picture.thumbnail((max(1, self.canvas.winfo_width() - 12), max(1, self.canvas.winfo_height() - 12)))
        self.photo = ImageTk.PhotoImage(picture, master=self)
        self.canvas.delete('all')
        self.canvas.create_image(self.canvas.winfo_width() / 2, self.canvas.winfo_height() / 2, image=self.photo,
                                 anchor='center')

    def _destroyed(self, event):
        if event.widget is self:
            self.closed = True
            self.requests.put(None)
            self.after_cancel(self.poll)


def ask_name(parent, title, prompt):
    """ A small window that asks for a name; the name, or '' if cancelled """
    window = tk.Toplevel(parent)
    window.withdraw()
    window.transient(parent)
    window.title(title)
    window.configure(background=COLORS['background'])
    body = ttk.Frame(window, padding=14)
    body.pack(fill='both', expand=True)
    ttk.Label(body, text=prompt).pack(anchor='w', pady=(0, 6))
    name = tk.StringVar(window)
    entry = ttk.Entry(body, textvariable=name, width=40)
    entry.pack(fill='x')
    answer = []
    buttons = ttk.Frame(body)
    buttons.pack(fill='x', pady=(12, 0))
    ttk.Button(buttons, text='Cancel', command=window.destroy).pack(side='right')
    ttk.Button(buttons, text='OK', command=lambda: (answer.append(name.get().strip()), window.destroy())).pack(
        side='right', padx=(0, 6))
    window.bind('<Return>', lambda event: (answer.append(name.get().strip()), window.destroy()))
    window.bind('<Escape>', lambda event: window.destroy())
    hide_title_icon(window)
    window.deiconify()
    window.wait_visibility()
    window.grab_set()
    entry.focus_set()
    parent.wait_window(window)
    return answer[0] if answer else ''


class FolderBrowser(tk.Toplevel):
    """ The window; use ask_folder() to run it """

    def __init__(self, parent, title, initialdir):
        super().__init__(parent)
        self.withdraw()
        self.title(title)
        self.transient(parent)
        self.parent = parent
        self.result = ''
        self.current = Path(initialdir)
        self.entries = {}
        self.configure(background=COLORS['background'])
        self.minsize(720, 420)
        width, height = min(940, self.winfo_screenwidth() - 60), min(620, self.winfo_screenheight() - 100)
        self.geometry(f"{width}x{height}")
        self.protocol('WM_DELETE_WINDOW', self.cancel)
        self.bind('<Escape>', lambda event: self.cancel())

        body = ttk.Frame(self, padding=14)
        body.pack(fill='both', expand=True)
        navigation = ttk.Frame(body)
        navigation.pack(fill='x', pady=(0, 10))
        self.up_button = ttk.Button(navigation, text='Up', command=self.up)
        self.up_button.pack(side='left')
        ttk.Button(navigation, text='Home', command=lambda: self.navigate(Path.home())).pack(side='left', padx=(6, 0))
        ttk.Button(navigation, text='Refresh', command=lambda: self.navigate(self.current)).pack(side='left', padx=6)
        ttk.Button(navigation, text='New Folder...', command=self.new_folder).pack(side='left')

        address = ttk.Frame(body)
        address.pack(fill='x', pady=(0, 10))
        self.location = tk.StringVar(self, str(self.current))
        self.address = ttk.Entry(address, textvariable=self.location)
        self.address.pack(side='left', fill='x', expand=True)
        self.address.bind('<Return>', lambda event: self.navigate(self.location.get()))
        ttk.Button(address, text='Go', command=lambda: self.navigate(self.location.get())).pack(side='left', padx=(6, 0))

        self.panes = tk.PanedWindow(body, orient='horizontal', sashwidth=6, sashrelief='flat', borderwidth=0,
                                    background=COLORS['border'], opaqueresize=True)
        self.panes.pack(fill='both', expand=True)
        listing = ttk.Frame(self.panes)
        self.panes.add(listing, minsize=300, stretch='always')
        listing.rowconfigure(0, weight=1)
        listing.columnconfigure(0, weight=1)
        self.tree = ttk.Treeview(listing, columns=('kind', 'size'), selectmode='browse')
        self.tree.heading('#0', text='Name', anchor='w')
        self.tree.heading('kind', text='Type', anchor='w')
        self.tree.heading('size', text='Size', anchor='e')
        self.tree.column('#0', width=260, minwidth=140, stretch=True)
        self.tree.column('kind', width=65, minwidth=55, stretch=False)
        self.tree.column('size', width=85, minwidth=65, anchor='e', stretch=False)
        self.tree.grid(row=0, column=0, sticky='nsew')
        vertical = ttk.Scrollbar(listing, orient='vertical', command=self.tree.yview)
        vertical.grid(row=0, column=1, sticky='ns')
        self.tree.configure(yscrollcommand=vertical.set)
        self.tree.bind('<Double-1>', self.open_selected)
        self.tree.bind('<Return>', self.open_selected)
        self.tree.bind('<BackSpace>', lambda event: self.up())
        self.tree.bind('<<TreeviewSelect>>', self._selection_changed)
        self.preview = ImagePreview(self.panes)
        self.panes.add(self.preview, minsize=240, stretch='always')

        self.summary = tk.StringVar(self)
        self.selection = tk.StringVar(self)
        summary_label = ttk.Label(body, textvariable=self.summary, style='Muted.TLabel')
        selection_label = ttk.Label(body, textvariable=self.selection, wraplength=width - 45)
        actions = ttk.Frame(body)
        self.accept_button = ttk.Button(actions, text='Select Folder', command=self.accept)
        self.accept_button.pack(side='right')
        ttk.Button(actions, text='Cancel', command=self.cancel).pack(side='right', padx=(0, 6))
        for widget, options in ((actions, {'fill': 'x'}), (selection_label, {'fill': 'x', 'pady': (6, 10)}),
                                (summary_label, {'anchor': 'w', 'pady': (7, 0)})):
            widget.pack(side='bottom', before=self.panes, **options)       # before the list: they are never cut off
        body.bind('<Configure>', lambda event: selection_label.configure(wraplength=max(100, event.width - 4)), add='+')
        hide_title_icon(self)
        self.navigate(initialdir)

    def navigate(self, destination):
        """ Show the contents of a folder; False (with the reason shown) if it cannot be read """
        from tkinter import messagebox
        try:
            target, rows = list_folder(destination, self.current)
        except (OSError, ValueError) as error:
            messagebox.showerror('Cannot open folder', str(error), parent=self)
            self.location.set(str(self.current))
            return False
        children = self.tree.get_children()
        if children:
            self.tree.delete(*children)
        self.entries.clear()
        self.current = target
        self.location.set(str(target))
        folders = 0
        for index, (name, folder, size) in enumerate(rows):
            self.entries[str(index)] = (target / name, folder)
            self.tree.insert('', 'end', iid=str(index), text=name,
                             values=('Folder' if folder else 'File', size_text(size)))
            folders += folder
        self.summary.set(f"{folders} folders, {len(rows) - folders} files. Double-click a folder to open it.")
        self.up_button.state(['disabled'] if target.parent == target else ['!disabled'])
        self._selection_changed()
        return True

    def selected_folder(self):
        """ The folder that is selected in the list, or else the one that is open """
        selected = self.tree.selection()
        if selected and selected[0] in self.entries:
            path, folder = self.entries[selected[0]]
            if folder:
                return path
        return self.current

    def _selection_changed(self, event=None):
        self.selection.set(f"Folder to use: {self.selected_folder()}")
        selected = self.tree.selection()
        path, folder = self.entries.get(selected[0], (None, True)) if selected else (None, True)
        path = path if not folder and is_image(path) else None
        if path != self.preview.path:
            self.preview.show(path)

    def open_selected(self, event=None):
        selected = self.tree.selection()
        if selected and selected[0] in self.entries:
            path, folder = self.entries[selected[0]]
            if folder:
                self.navigate(path)
        return 'break'

    def up(self):
        self.navigate(self.current.parent)

    def new_folder(self):
        """ Make a folder in the one that is open and open it """
        from tkinter import messagebox
        name = ask_name(self, 'New Folder', 'Folder name:')
        if not name:
            return
        if name in ('.', '..') or Path(name).anchor or any(character in name for character in ('/', '\\')):
            messagebox.showerror('Invalid folder name', 'Enter a single folder name.', parent=self)
            return
        try:
            target = self.current / name
            target.mkdir()
        except (OSError, ValueError) as error:
            messagebox.showerror('Cannot create folder', str(error), parent=self)
            return
        self.navigate(target)

    def accept(self):
        from tkinter import messagebox
        if self.location.get().strip().strip('"') != str(self.current):      # a path that was typed and not opened yet
            if not self.navigate(self.location.get()):
                return
        target = self.selected_folder()
        if not target.is_dir():
            messagebox.showerror('Folder unavailable', 'Select an existing folder.', parent=self)
            return
        self.result = str(target)
        self.destroy()

    def cancel(self):
        self.result = ''
        self.destroy()


def ask_folder(parent, title, initialdir):
    """ Open the window above another one and wait; the folder that was chosen as text, or '' if it was cancelled """
    dialog = FolderBrowser(parent, title, initialdir)
    dialog.deiconify()
    dialog.wait_visibility()
    dialog.grab_set()
    dialog.tree.focus_set()
    parent.wait_window(dialog)
    return dialog.result
