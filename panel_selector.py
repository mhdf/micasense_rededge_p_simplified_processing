"""
Manual selection of the reference panel region, band by band

The bands of the panel capture are shown one after the other; the region of the panel is a polygon that starts from the
one found automatically (or from a region saved before), and that can be edited and confirmed. The vertices are pixels
of the undistorted image and are saved in the file used by the workflow (PANEL.manual_regions_file).

    python panel_selector.py --config radiometric_inputs.json [--root folder] [--capture IMG_0005]

Keys: Enter confirm (and go to the next band), S use the automatic detection for this band, R back to the automatic region, C clear,
Backspace remove the last vertex, Left/Right previous/next band, Escape cancel.
Mouse: click adds a vertex (on an edge it inserts one), drag moves a vertex, right click removes it, wheel zooms,
middle button drags the image.

The logic (PolygonModel, ViewTransform, SelectionSession) has no user interface and is tested without a display.
"""

# Standard Library Imports
import argparse
import math
import sys
from pathlib import Path

# Third-Party Imports
import numpy as np

# Local Imports
import radiometric_correction_tools as rct
from ui_theme import COLORS, apply_dark_theme

VERTEX_RADIUS_PIXELS = 7      # distance on the screen to pick a vertex
EDGE_DISTANCE_PIXELS = 6      # distance on the screen to insert a vertex on an edge


# ======================================================================================
# LOGIC WITHOUT USER INTERFACE
# ======================================================================================

class PolygonModel:
    """ Vertices [x, y] in image pixels """

    def __init__(self, vertices=None):
        self.vertices = [[float(x), float(y)] for x, y in (vertices or [])]

    def copy(self):
        return PolygonModel(self.vertices)

    def clear(self):
        self.vertices = []

    def move(self, index, point):
        self.vertices[index] = [float(point[0]), float(point[1])]

    def remove(self, index):
        del self.vertices[index]

    def remove_last(self):
        if self.vertices:
            self.vertices.pop()

    def nearest_vertex(self, point, radius):
        """ Index of the closest vertex within radius (image pixels), or None """
        best, best_distance = None, radius
        for index, (x, y) in enumerate(self.vertices):
            distance = math.hypot(x - point[0], y - point[1])
            if distance <= best_distance:
                best, best_distance = index, distance
        return best

    def nearest_edge(self, point, distance):
        """ Index i of the closest edge (vertex i to vertex i + 1, the last one closing the polygon) within distance """
        count = len(self.vertices)
        if count < 2:
            return None
        best, best_distance = None, distance
        for index in range(count if count > 2 else 1):
            (ax, ay), (bx, by) = self.vertices[index], self.vertices[(index + 1) % count]
            length_squared = (bx - ax) ** 2 + (by - ay) ** 2
            t = 0.0 if length_squared == 0 else max(0.0, min(1.0, ((point[0] - ax) * (bx - ax) +
                                                                  (point[1] - ay) * (by - ay)) / length_squared))
            d = math.hypot(point[0] - (ax + t * (bx - ax)), point[1] - (ay + t * (by - ay)))
            if d <= best_distance:
                best, best_distance = index, d
        return best

    def add(self, point, radius):
        """
        Add a vertex: inserted in the edge if the point is within radius (image pixels) of one, otherwise appended
        Returns the index of the new vertex
        """
        edge = self.nearest_edge(point, radius)
        if edge is None:
            self.vertices.append([float(point[0]), float(point[1])])
            return len(self.vertices) - 1
        self.vertices.insert(edge + 1, [float(point[0]), float(point[1])])
        return edge + 1

    def as_integers(self):
        return [[int(round(x)), int(round(y))] for x, y in self.vertices]


class ViewTransform:
    """ Zoom and pan: canvas = scale * image + offset """

    def __init__(self, scale=1.0, offset=(0.0, 0.0)):
        self.scale, self.offset = float(scale), (float(offset[0]), float(offset[1]))
        self.fit_scale = self.scale

    def to_canvas(self, point):
        return (point[0] * self.scale + self.offset[0], point[1] * self.scale + self.offset[1])

    def to_image(self, point):
        return ((point[0] - self.offset[0]) / self.scale, (point[1] - self.offset[1]) / self.scale)

    def fit(self, image_size, canvas_size):
        """ Show the whole image, centered """
        self.scale = self.fit_scale = min(canvas_size[0] / image_size[0], canvas_size[1] / image_size[1])
        self.offset = ((canvas_size[0] - image_size[0] * self.scale) / 2, (canvas_size[1] - image_size[1] * self.scale) / 2)

    def zoom_at(self, canvas_point, factor, maximum=60.0):
        """ Zoom keeping the image point under canvas_point where it is """
        anchor = self.to_image(canvas_point)
        self.scale = max(self.fit_scale * 0.5, min(self.fit_scale * maximum, self.scale * factor))
        self.offset = (canvas_point[0] - anchor[0] * self.scale, canvas_point[1] - anchor[1] * self.scale)

    def pan(self, dx, dy):
        self.offset = (self.offset[0] + dx, self.offset[1] + dy)


class SelectionSession:
    """
    The decisions of the user, band by band
    Inputs: {band: rct.Image} of the panel capture, optionally the regions saved before {band: polygon} and the
            parameters used to find the automatic region
    Each band ends as 'manual' (a confirmed polygon) or 'automatic' (the region is found from the QR code at run time)
    """

    def __init__(self, images, saved=None, percent_inner_area=0.7, sat_threshold=65000):
        self.images = dict(images)
        self.bands = list(self.images)
        self.saved = saved or {}
        self.percent_inner_area = percent_inner_area
        self.sat_threshold = sat_threshold
        self.index = 0
        self.decisions = {band: None for band in self.bands}
        self.polygons = {band: None for band in self.bands}
        self._automatic = {}

    @property
    def band(self):
        return self.bands[self.index]

    def image_size(self, band=None):
        image = self.images[band or self.band]
        return (image.image.shape[1], image.image.shape[0])

    def automatic_polygon(self, band):
        """ Region found from the QR code in the undistorted image, or None """
        if band not in self._automatic:
            located = self.images[band].locate_reference_panel(self.percent_inner_area, self.sat_threshold)
            polygon = None
            if located is not None and located[2] == 'undistorted':
                polygon = [[int(x), int(y)] for x, y in located[0]]
            self._automatic[band] = polygon
        return self._automatic[band]

    def starting_polygon(self, band):
        """ What is drawn when a band is shown for the first time: confirmed polygon, saved one or automatic one """
        if self.polygons[band] is not None:
            return [list(vertex) for vertex in self.polygons[band]]
        if self.saved.get(band):
            return [list(vertex) for vertex in self.saved[band]]
        return [list(vertex) for vertex in (self.automatic_polygon(band) or [])]

    def stats(self, band, polygon):
        """ Pixels, saturated pixels and variation (%) inside a polygon, or None while it is not a valid region """
        try:
            rct.validate_polygon(polygon, self.image_size(band), band)
        except ValueError:
            return None
        return self.images[band].panel_region_stats(polygon, self.sat_threshold)

    def confirm(self, band, polygon):
        """ Confirm a polygon for a band; raises ValueError with the reason if it is not a valid region """
        rct.validate_polygon(polygon, self.image_size(band), f"Band '{band}'")
        self.polygons[band] = [[int(round(x)), int(round(y))] for x, y in polygon]
        self.decisions[band] = 'manual'

    def use_automatic(self, band):
        self.polygons[band] = None
        self.decisions[band] = 'automatic'

    def go_to(self, index):
        self.index = max(0, min(len(self.bands) - 1, index))

    def is_complete(self):
        return all(decision is not None for decision in self.decisions.values())

    def result(self):
        """ {band: polygon or None (use the automatic detection)} """
        return {band: self.polygons[band] if self.decisions[band] == 'manual' else None for band in self.bands}


# ======================================================================================
# WINDOW
# ======================================================================================

class PanelSelectorWindow:
    """ Dialog to draw the regions; use select_panel_regions() to run it """

    def __init__(self, root, session, title="Reference panel regions"):
        import tkinter as tk
        from tkinter import ttk
        from PIL import Image as PILImage, ImageTk
        self.tk, self.ttk, self.PILImage, self.ImageTk = tk, ttk, PILImage, ImageTk
        self.root, self.session = root, session
        self.cancelled = True
        self.finished = False
        self.model = PolygonModel()
        self.view = ViewTransform()
        self.display = None
        self.photo = None
        self.drag_vertex = None
        self.pan_start = None
        self.fitted_band = None
        self.root.title(title)
        self.root.geometry("1180x800")
        self.root.minsize(1060, 520)
        apply_dark_theme(self.root)
        self._build()
        self._load_band()

    # --- construction ---------------------------------------------------------------------------------------

    def _build(self):
        tk, ttk = self.tk, self.ttk
        top = ttk.Frame(self.root, padding=(12, 8))
        top.pack(fill='x')
        self.title_label = ttk.Label(top, style='Title.TLabel')
        self.title_label.pack(side='left')
        self.status_label = ttk.Label(top, style='Muted.TLabel')
        self.status_label.pack(side='right')

        buttons = ttk.Frame(self.root, padding=(12, 4, 12, 12))
        buttons.pack(side='bottom', fill='x')
        info = ttk.Frame(self.root, padding=(12, 6))
        info.pack(side='bottom', fill='x')
        self.stats_label = ttk.Label(info)
        self.stats_label.pack(anchor='w')
        self.message_label = ttk.Label(info, style='Muted.TLabel')
        self.message_label.pack(anchor='w')

        self.canvas = tk.Canvas(self.root, background=COLORS['panel'], highlightthickness=0, cursor='crosshair')
        self.canvas.pack(fill='both', expand=True, padx=12)
        self.canvas.bind('<Configure>', self._on_configure)
        self.canvas.bind('<Button-1>', self._on_press)
        self.canvas.bind('<B1-Motion>', self._on_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_release)
        self.canvas.bind('<Button-3>', self._on_remove)
        self.canvas.bind('<Button-2>', self._on_pan_start)
        self.canvas.bind('<B2-Motion>', self._on_pan_move)
        self.canvas.bind('<MouseWheel>', self._on_wheel)
        self.buttons = {}
        # editing buttons on the left, decisions on the right (packed from the right edge inwards)
        for name, text, command in (('previous', 'Previous', self.previous), ('next', 'Next', self.next),
                                    ('reset', 'Reset to automatic', self.reset), ('clear', 'Clear', self.clear)):
            self.buttons[name] = ttk.Button(buttons, text=text, command=command)
            self.buttons[name].pack(side='left', padx=(0, 6))
        for name, text, command in (('cancel', 'Cancel', self.cancel), ('finish', 'Save and close', self.finish),
                                    ('skip', 'Use automatic detection', self.skip),
                                    ('confirm', 'Confirm', self.confirm)):
            self.buttons[name] = ttk.Button(buttons, text=text, command=command)
            self.buttons[name].pack(side='right', padx=(6, 0))

        for sequence, command in (('<Return>', self.confirm), ('<s>', self.skip), ('<r>', self.reset),
                                  ('<c>', self.clear), ('<BackSpace>', self.undo), ('<Left>', self.previous),
                                  ('<Right>', self.next), ('<Escape>', self.cancel)):
            self.root.bind(sequence, lambda event, command=command: command())
        self.root.protocol('WM_DELETE_WINDOW', self.cancel)

    # --- band ---------------------------------------------------------------------------------------------

    def _load_band(self):
        session = self.session
        band = session.band
        image = session.images[band]
        self.display = self.PILImage.fromarray(image.display_image())
        self.model = PolygonModel(session.starting_polygon(band))
        self.fitted_band = None
        self._fit_if_possible()
        decision = session.decisions[band]
        self.title_label.configure(text=f"Band {session.index + 1} of {len(session.bands)}: {band}")
        automatic = session.automatic_polygon(band)
        self.message(f"Decision: {decision or 'pending'}. " +
                     ("The yellow dashed line is the region found from the QR code." if automatic else
                      "No region was found from the QR code: draw the panel."), error=False)
        self.status_label.configure(text=self._status_text())
        self._redraw()

    def _status_text(self):
        done = sum(decision is not None for decision in self.session.decisions.values())
        return f"{done} of {len(self.session.bands)} bands decided"

    def _fit_if_possible(self):
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        if width > 20 and height > 20 and self.fitted_band != self.session.band:
            self.view.fit(self.session.image_size(), (width, height))
            self.fitted_band = self.session.band

    # --- drawing ------------------------------------------------------------------------------------------

    def _redraw(self):
        canvas, view = self.canvas, self.view
        width, height = max(self.canvas.winfo_width(), 1), max(self.canvas.winfo_height(), 1)
        canvas.delete('all')
        if self.display is not None:
            scale = view.scale
            data = (1 / scale, 0, -view.offset[0] / scale, 0, 1 / scale, -view.offset[1] / scale)
            shown = self.display.transform((width, height), self.PILImage.AFFINE, data, resample=self.PILImage.BILINEAR)
            self.photo = self.ImageTk.PhotoImage(shown)
            canvas.create_image(0, 0, image=self.photo, anchor='nw')
        vertices = [view.to_canvas(vertex) for vertex in self.model.vertices]
        if len(vertices) >= 3:
            canvas.create_polygon([c for vertex in vertices for c in vertex], outline=COLORS['manual'], fill='', width=2)
        elif len(vertices) == 2:
            canvas.create_line([c for vertex in vertices for c in vertex], fill=COLORS['manual'], width=2)
        # the automatic region is drawn last, so that it can be seen over an identical drawn one
        automatic = self.session.automatic_polygon(self.session.band)
        if automatic:
            points = [coordinate for vertex in automatic for coordinate in view.to_canvas(vertex)]
            canvas.create_polygon(points, outline=COLORS['automatic'], fill='', dash=(6, 4), width=2)
        for x, y in vertices:
            canvas.create_rectangle(x - 4, y - 4, x + 4, y + 4, outline=COLORS['manual'], fill=COLORS['background'])

    def _update_stats(self):
        stats = self.session.stats(self.session.band, self.model.vertices) if len(self.model.vertices) >= 3 else None
        if stats is None:
            self.stats_label.configure(text="Region: draw at least 3 vertices without crossing edges")
        else:
            self.stats_label.configure(
                text=f"Region: {stats['n_pixels']} pixels, {stats['n_saturated']} saturated, "
                     f"variation {stats['variation_percent']:.2f} % (lower is more uniform)")

    def message(self, text, error):
        self.message_label.configure(text=text, style='Error.TLabel' if error else 'Muted.TLabel')

    # --- mouse --------------------------------------------------------------------------------------------

    def _image_point(self, event):
        return self.view.to_image((event.x, event.y))

    def _on_configure(self, event):
        self._fit_if_possible()
        self._redraw()

    def _on_press(self, event):
        self.canvas.focus_set()
        point = self._image_point(event)
        radius = VERTEX_RADIUS_PIXELS / self.view.scale
        index = self.model.nearest_vertex(point, radius)
        if index is None:
            index = self.model.add(point, EDGE_DISTANCE_PIXELS / self.view.scale)
        self.drag_vertex = index
        self._redraw()

    def _on_drag(self, event):
        if self.drag_vertex is not None:
            self.model.move(self.drag_vertex, self._image_point(event))
            self._redraw()

    def _on_release(self, event):
        self.drag_vertex = None
        self._update_stats()
        self._redraw()

    def _on_remove(self, event):
        index = self.model.nearest_vertex(self._image_point(event), VERTEX_RADIUS_PIXELS / self.view.scale)
        if index is not None:
            self.model.remove(index)
            self._update_stats()
            self._redraw()

    def _on_pan_start(self, event):
        self.pan_start = (event.x, event.y)

    def _on_pan_move(self, event):
        if self.pan_start is not None:
            self.view.pan(event.x - self.pan_start[0], event.y - self.pan_start[1])
            self.pan_start = (event.x, event.y)
            self._redraw()

    def _on_wheel(self, event):
        self.view.zoom_at((event.x, event.y), 1.25 if event.delta > 0 else 1 / 1.25)
        self._redraw()

    # --- commands -----------------------------------------------------------------------------------------

    def confirm(self):
        try:
            self.session.confirm(self.session.band, self.model.vertices)
        except ValueError as error:
            self.message(str(error), error=True)
            return
        self.message("Region confirmed.", error=False)
        self._advance()

    def skip(self):
        self.session.use_automatic(self.session.band)
        self.message("The automatic detection will be used for this band.", error=False)
        self._advance()

    def _advance(self):
        self.status_label.configure(text=self._status_text())
        if self.session.index < len(self.session.bands) - 1:
            self.session.go_to(self.session.index + 1)
            self._load_band()
        else:
            self._update_stats()
            self._redraw()

    def previous(self):
        self.session.go_to(self.session.index - 1)
        self._load_band()
        self._update_stats()

    def next(self):
        self.session.go_to(self.session.index + 1)
        self._load_band()
        self._update_stats()

    def reset(self):
        automatic = self.session.automatic_polygon(self.session.band)
        self.model = PolygonModel(automatic or [])
        self._update_stats()
        self._redraw()

    def clear(self):
        self.model.clear()
        self._update_stats()
        self._redraw()

    def undo(self):
        self.model.remove_last()
        self._update_stats()
        self._redraw()

    def finish(self):
        if not self.session.is_complete():
            pending = [band for band, decision in self.session.decisions.items() if decision is None]
            self.message(f"Confirm or skip the bands first: {', '.join(pending)}", error=True)
            return
        self.cancelled = False
        self.finished = True
        self.root.destroy()

    def cancel(self):
        self.cancelled = True
        self.finished = True
        self.root.destroy()


def select_panel_regions(images, saved=None, percent_inner_area=0.7, sat_threshold=65000, parent=None):
    """
    Open the dialog and wait for the user
    Inputs: {band: rct.Image} of the panel capture, the regions saved before {band: polygon} (optional) and, when
            another window is open, that window (the dialog is then modal and opens above it)
    Outputs: {band: polygon or None (use the automatic detection)}, or None if the user cancelled
    """
    import tkinter as tk
    session = SelectionSession(images, saved, percent_inner_area, sat_threshold)
    if parent is None:
        window_root = tk.Tk()
        window = PanelSelectorWindow(window_root, session)
        window_root.mainloop()
    else:
        window_root = tk.Toplevel(parent)
        window_root.transient(parent)
        window = PanelSelectorWindow(window_root, session)
        window_root.grab_set()
        parent.wait_window(window_root)
    return None if window.cancelled else session.result()


def main(argv=None):
    import radiometric_workflow as wf
    parser = argparse.ArgumentParser(description="Draw the reference panel region of each band")
    parser.add_argument('--config', required=True, help="configuration json file of the workflow")
    parser.add_argument('--root', default='.', help="folder from which the relative paths are resolved")
    parser.add_argument('--capture', help="panel capture (default: INPUTS.reference_panel_capture)")
    arguments = parser.parse_args(argv)

    config = wf.load_config(arguments.config)
    problems = wf.validate_config(config, arguments.root)
    if problems:
        for problem in problems:
            print(f"ERROR: {problem}", file=sys.stderr)
        return 1
    root = Path(arguments.root)
    capture = arguments.capture or config['INPUTS.reference_panel_capture'].strip()
    regions_path = wf.resolve_path(root, config['PANEL.manual_regions_file'] or
                                   Path(config['INPUTS.results_folder']) / 'panel_regions.json')
    exiftool_path = wf.exiftool_executable(config, root)
    bands = list(config['PANEL.reflectance_by_band'])
    print(f"Reading {capture} ...", flush=True)
    files = rct.read_band_files(rct.find_capture_paths(wf.resolve_path(root, config['INPUTS.images_folder']), capture),
                                bands, exiftool_path)
    images = {band: rct.Image(*files[band]) for band in bands}
    saved = wf.load_panel_regions(regions_path).get(capture, {}) if regions_path.is_file() else {}
    selected = select_panel_regions(images, saved, config['PANEL.percent_inner_area'],
                                    config['PANEL.saturation_threshold'])
    if selected is None:
        print("Cancelled: nothing was saved.")
        return 1
    regions_path.parent.mkdir(parents=True, exist_ok=True)
    wf.save_panel_regions(regions_path, capture, selected)
    print(f"PANEL_REGIONS_JSON={regions_path.resolve()}")
    if config['PANEL.manual_regions_file'] == '':
        print("Set PANEL.manual_regions_file to this file in the configuration to use it.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
