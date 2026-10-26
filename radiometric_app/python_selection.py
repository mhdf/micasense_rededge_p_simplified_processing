"""
Confirm the Python environment before the window opens: a small dialog with the Python that was used the last time (or
the one that is running), where it can be confirmed, pasted or browsed. The Python is tried before it is accepted
(version, and a Tk window can be created), and saved as the one to use (see environment.remember_python).

    choose_python_environment(parent, executable, root)   the dialog; returns the Path that was confirmed, or None
"""

# Standard Library Imports
from pathlib import Path

# Local Imports
from . import environment
from .ui_theme import hide_title_icon


class PythonSelectionDialog:
    """ The window; use choose_python_environment() to run it """

    def __init__(self, parent, executable, root='.', remember=None):
        import tkinter as tk
        from tkinter import ttk
        from .ui_theme import apply_dark_theme
        apply_dark_theme(parent)
        self.tk, self.ttk = tk, ttk
        self.root_folder = root
        self.remember = remember or environment.remember_python        # what accepts a Python (replaceable in tests)
        self.result = None
        self.window = tk.Toplevel(parent)
        self.window.withdraw()
        self.window.title("Select Python Environment")
        self.window.geometry("820x275")
        self.window.minsize(620, 275)
        self.window.resizable(True, False)
        self.window.protocol("WM_DELETE_WINDOW", self.window.destroy)
        self.window.bind("<Escape>", lambda _event: self.window.destroy())
        self.window.bind("<Return>", lambda _event: self.confirm())
        body = ttk.Frame(self.window, padding=16)
        body.pack(fill="both", expand=True)
        self.path = tk.StringVar(value=str(executable))
        ttk.Label(body, text="Confirm the Python environment for this launch",
                  font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 9))
        message = ("The selected Python environment must exist on this computer. "
                   "Confirm the path below, paste a different path, or browse.")
        message_label = ttk.Label(body, text=message, wraplength=700)
        message_label.pack(anchor="w", fill="x", pady=(0, 9))
        ttk.Label(body, text="Python executable: select the python.exe file inside your environment folder.").pack(
            anchor="w", pady=(0, 4))
        row = ttk.Frame(body)
        row.pack(fill="x")
        self.entry = ttk.Entry(row, textvariable=self.path)
        self.entry.pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Browse...", command=self.browse).pack(side="left", padx=(8, 0))
        self.status = tk.StringVar(value="Python 3.10+ with Tkinter is required. The packages and ExifTool are checked next.")
        status_label = ttk.Label(body, textvariable=self.status, wraplength=700)
        status_label.pack(anchor="w", fill="x", pady=(9, 10))

        def wrap_labels(event):
            for label in (message_label, status_label):
                label.configure(wraplength=max(100, event.width - 4))
        body.bind("<Configure>", wrap_labels, add="+")
        buttons = ttk.Frame(body)
        buttons.pack(side="bottom", fill="x")
        ttk.Button(buttons, text="Cancel", command=self.window.destroy).pack(side="right")
        self.confirm_button = ttk.Button(buttons, text="Confirm and open window", command=self.confirm)
        self.confirm_button.pack(side="right", padx=(0, 8))
        hide_title_icon(self.window)
        self.window.deiconify()
        self.entry.focus_set()

    def browse(self):
        from tkinter import filedialog
        current = Path(self.path.get().strip().strip('"'))
        selected = filedialog.askopenfilename(
            parent=self.window, title="Select Python executable (python.exe)",
            initialdir=current.parent if current.parent.is_dir() else None,
            initialfile=current.name if current.is_file() else None,
            filetypes=(("Python executable", "python.exe pythonw.exe python python3"), ("All files", "*.*")))
        if selected:
            self.path.set(selected)

    def confirm(self):
        from tkinter import messagebox
        self.status.set("Checking Python and Tkinter...")
        self.window.update_idletasks()
        try:
            report = self.remember(self.root_folder, self.path.get().strip().strip('"'))
        except (OSError, ValueError, RuntimeError) as error:
            self.status.set("Choose a Python 3.10+ environment with working Tkinter.")
            messagebox.showerror("Python environment cannot open the window", str(error), parent=self.window)
            return
        self.result = Path(report["executable"])
        self.window.destroy()


def choose_python_environment(parent, executable, root='.'):
    """ Run the dialog above a window and wait for the user; the Path of the Python that was confirmed, or None """
    dialog = PythonSelectionDialog(parent, executable, root)
    dialog.window.grab_set()
    parent.wait_window(dialog.window)
    return dialog.result
