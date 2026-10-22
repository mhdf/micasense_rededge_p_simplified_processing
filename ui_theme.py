"""
Dark theme shared by the windows of this project (panel_selector.py, radiometric_window.py)

The colors and fonts follow the configuration window of the photogrammetric workflow (compact, neutral dark controls),
so that the windows can be merged into it later: when that happens this file can be replaced by its theme.
"""

COLORS = {'background': '#242424', 'panel': '#2b2b2b', 'input': '#595d61', 'foreground': '#dedede',
          'muted': '#aaaeb2', 'disabled': '#777b7e', 'border': '#444444', 'accent': '#485968',
          'manual': '#35c4e8', 'automatic': '#f2c84b', 'error': '#e8785a', 'ok': '#7cc67c'}

FONT = ('Segoe UI', 10)
FONT_BOLD = ('Segoe UI', 10, 'bold')
FONT_TITLE = ('Segoe UI', 12, 'bold')


def apply_dark_theme(root):
    """ Configure the ttk styles of a Tk root (or toplevel) and its background; returns the colors """
    from tkinter import ttk
    c = COLORS
    root.configure(background=c['background'])
    root.option_add('*TCombobox*Listbox.background', c['input'])
    root.option_add('*TCombobox*Listbox.foreground', c['foreground'])
    style = ttk.Style(root)
    style.theme_use('clam')
    style.configure('TFrame', background=c['background'])
    style.configure('Panel.TFrame', background=c['panel'])
    style.configure('TLabelframe', background=c['background'], bordercolor=c['border'], lightcolor=c['border'],
                    darkcolor=c['border'], borderwidth=1, relief='solid')
    style.configure('TLabelframe.Label', background=c['background'], foreground=c['foreground'], font=FONT)
    style.configure('TLabel', background=c['background'], foreground=c['foreground'], font=FONT)
    style.configure('Muted.TLabel', foreground=c['muted'])
    style.configure('Title.TLabel', font=FONT_TITLE)
    style.configure('Heading.TLabel', font=FONT_BOLD)
    style.configure('Error.TLabel', foreground=c['error'])
    style.configure('Ok.TLabel', foreground=c['ok'])
    style.configure('TButton', background=c['input'], foreground=c['foreground'], bordercolor=c['border'],
                    lightcolor=c['input'], darkcolor=c['input'], padding=(10, 4), font=FONT)
    style.map('TButton', background=[('active', c['accent']), ('disabled', c['panel'])],
              foreground=[('disabled', c['disabled'])])
    style.configure('TEntry', fieldbackground=c['input'], foreground=c['foreground'], bordercolor=c['border'],
                    lightcolor=c['input'], darkcolor=c['input'], insertcolor=c['foreground'], padding=3)
    style.map('TEntry', fieldbackground=[('disabled', c['panel'])], foreground=[('disabled', c['disabled'])])
    style.configure('Invalid.TEntry', fieldbackground='#6b3f36')
    style.configure('TCheckbutton', background=c['background'], foreground=c['foreground'], font=FONT)
    style.map('TCheckbutton', background=[('active', c['background'])])
    style.configure('TProgressbar', background=c['accent'], troughcolor=c['panel'], bordercolor=c['border'],
                    lightcolor=c['accent'], darkcolor=c['accent'])
    style.configure('TScrollbar', background=c['input'], troughcolor=c['panel'], bordercolor=c['border'],
                    arrowcolor=c['foreground'])
    style.configure('TCombobox', fieldbackground=c['input'], background=c['input'], foreground=c['foreground'],
                    arrowcolor=c['foreground'], bordercolor=c['border'], lightcolor=c['input'], darkcolor=c['input'],
                    selectbackground=c['input'], selectforeground=c['foreground'], padding=3)
    style.map('TCombobox', fieldbackground=[('readonly', c['input']), ('disabled', c['panel'])],
              foreground=[('disabled', c['disabled'])], selectbackground=[('readonly', c['input'])],
              selectforeground=[('readonly', c['foreground'])])
    style.configure('Treeview', background=c['panel'], fieldbackground=c['panel'], foreground=c['foreground'],
                    bordercolor=c['border'], lightcolor=c['panel'], darkcolor=c['panel'], rowheight=22, font=FONT)
    style.map('Treeview', background=[('selected', c['accent'])], foreground=[('selected', c['foreground'])])
    style.configure('Treeview.Heading', background=c['input'], foreground=c['foreground'], bordercolor=c['border'],
                    lightcolor=c['input'], darkcolor=c['input'], font=FONT)
    style.map('Treeview.Heading', background=[('active', c['accent'])])
    return c
