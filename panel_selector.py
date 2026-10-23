"""
Check, accept and (if needed) draw the reference panel region of each band

The window opens on an overview with one tile per band: the image around the panel with the region found from the QR code
(yellow, dashed) and the region that is accepted or saved (green), and the pixels, saturated pixels and variation inside.
Each band must be accepted: "Accept" takes the region as it is shown, "Edit" opens the band in an editor where the polygon
can be moved, extended or drawn. The regions that were saved before are shown as accepted, so the overview is also a way
to look at them. The vertices are pixels of the undistorted image and are saved in the file used by the workflow
(PANEL.manual_regions_file), with the origin of each region (accepted as found, or edited).

    python panel_selector.py [--config radiometric_inputs.json] [--root folder] [--capture IMG_0005]

Without --config, radiometric_inputs.json in the root folder is used if it exists, otherwise the defaults.

What every element does is shown at the bottom of the window when the mouse is over it.

The logic (PolygonModel, ViewTransform, SelectionSession, render_tile) has no user interface and is tested without a display.
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
from ui_theme import COLORS, FONT, apply_dark_theme, hide_title_icon

DEFAULT_HINT = "Point at a button or at an image to see what it does."

# What each element does, shown while the mouse is over it
HINTS = {
    # --- editor
    'overview': "Back to overview (Esc): leave this band and return to the overview. The polygon that you are editing is "
                "not accepted and the changes made to it are lost; what was accepted before stays accepted.",
    'previous': "Previous band (Left arrow): show the previous band in this editor, without accepting or changing anything "
                "(the changes made to this polygon and not confirmed are lost).",
    'next': "Next band (Right arrow): show the next band in this editor, without accepting or changing anything "
            "(the changes made to this polygon and not confirmed are lost).",
    'reset': "Reset to automatic (R): put the polygon back on the region found from the QR code (yellow dashed line). It is "
             "not accepted until you press Confirm.",
    'clear': "Clear (C): remove all the vertices to draw the region from scratch with clicks. Backspace removes only the "
             "last vertex.",
    'confirm': "Confirm (Enter): accept the polygon as it is shown for this band, then go to the next band that is not "
               "accepted yet (or back to the overview if none is left). It needs 3 or more vertices, inside the image and "
               "without crossing edges. The file keeps exactly this polygon.",
    'canvas': "Image of this band. Click adds a vertex (on an edge it inserts one); drag a vertex to move it; right click "
              "removes it; the wheel zooms; the middle button drags the image. Cyan: the polygon that you are editing. "
              "Yellow dashed: the region found from the QR code. Green: the region accepted or saved before. The numbers "
              "above are the pixels, the saturated pixels and the variation (std / mean) inside the polygon: lower means "
              "a more uniform panel.",
    # --- both views
    'capture': "Capture: choose which capture of the images folder to look at. The panel captures of the configuration (the "
               "first one and the one after the flight, if there is one) come first and all their bands must be accepted "
               "before saving; any other capture is saved only if all its bands are accepted. What you accepted is kept "
               "for each capture while you switch.",
    # --- overview
    'accept_all': "Accept all automatic regions: for every band that is not accepted yet, accept the region found from the "
                  "QR code (yellow outline). Bands without one are left for you to draw with Edit. Look at the outlines "
                  "first: accepting means that this region is the one used for the reference panel.",
    'finish': "Save and close: write the accepted regions (and whether each was accepted as found or edited) to the "
              "regions file; the workflow then uses exactly these regions. Available when every band is accepted.",
    'cancel': "Cancel: close the window and save nothing.",
    'tile_accept': "Accept / Accepted: use the region outlined in this tile for the band (the green or cyan one; if there "
                   "is none, the yellow one found from the QR code). Press it again on an accepted band to undo the "
                   "acceptance.",
    'tile_edit': "Edit: open this band in the editor to move, add or delete vertices, or to draw the region when none was "
                 "found; confirm it there to accept it.",
    'tile_image': "Outlines: yellow dashed = region found from the QR code; green = region accepted or saved; cyan = region "
                  "proposed but not accepted yet. Below the image: the pixels, the saturated pixels and the variation "
                  "(std / mean) inside the region shown; lower variation means a more uniform panel.",
}

VERTEX_RADIUS_PIXELS = 7      # distance on the screen to pick a vertex
EDGE_DISTANCE_PIXELS = 6      # distance on the screen to insert a vertex on an edge
TILE_SIZE = (320, 225)        # image of a tile of the overview
TILE_COLUMNS = 3

ORIGIN_TEXT = {'accepted_automatic': 'as found from the QR code', 'edited': 'drawn or edited', None: ''}
BADGE_TEXT = {'accepted_automatic': 'Accepted (as found)', 'edited': 'Accepted (edited)', None: 'Accepted'}


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


def tile_crop_box(polygons, image_size, aspect, margin=0.8, minimum=60):
    """
    The part of the image to show in a tile: around the polygons with a margin, with the aspect ratio (width / height)
    of the tile where possible, and always inside the image. Returns (left, top, right, bottom) in pixels
    """
    width, height = image_size
    points = [point for polygon in polygons for point in polygon]
    if not points:
        return (0, 0, width, height)
    xs, ys = zip(*points)
    left, right, top, bottom = min(xs), max(xs), min(ys), max(ys)
    pad = max(minimum, margin * max(right - left, bottom - top))
    left, right, top, bottom = left - pad, right + pad, top - pad, bottom + pad
    box_width, box_height = right - left, bottom - top
    if box_width / box_height < aspect:
        extra = box_height * aspect - box_width
        left, right = left - extra / 2, right + extra / 2
    else:
        extra = box_width / aspect - box_height
        top, bottom = top - extra / 2, bottom + extra / 2
    return (int(max(0, round(left))), int(max(0, round(top))), int(min(width, round(right))), int(min(height, round(bottom))))


def _dashed_line(draw, start, end, color, width, dash=7, gap=5):
    length = math.hypot(end[0] - start[0], end[1] - start[1])
    if length == 0:
        return
    ux, uy = (end[0] - start[0]) / length, (end[1] - start[1]) / length
    position = 0.0
    while position < length:
        stop = min(position + dash, length)
        draw.line([(start[0] + ux * position, start[1] + uy * position), (start[0] + ux * stop, start[1] + uy * stop)],
                  fill=color, width=width)
        position += dash + gap


def render_tile(display, polygons, size=TILE_SIZE):
    """
    The picture of a tile: the image around the polygons, with their outlines
    Inputs: the image shown (2D uint8 array, undistorted), [(polygon [[x, y], ...], color, dashed)] and the size of the tile
    Outputs: PIL image of that size (RGB)
    """
    from PIL import Image, ImageDraw
    image = Image.fromarray(display)
    box = tile_crop_box([polygon for polygon, _, _ in polygons], image.size, size[0] / size[1])
    crop = image.crop(box).convert('RGB')
    scale = min(size[0] / crop.width, size[1] / crop.height)
    shown = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), Image.BILINEAR)
    tile = Image.new('RGB', size, COLORS['panel'])
    offset = ((size[0] - shown.width) // 2, (size[1] - shown.height) // 2)
    tile.paste(shown, offset)
    draw = ImageDraw.Draw(tile)
    for polygon, color, dashed in polygons:
        if len(polygon) < 2:
            continue
        points = [((x - box[0]) * scale + offset[0], (y - box[1]) * scale + offset[1]) for x, y in polygon]
        for start, end in zip(points, points[1:] + points[:1] if len(points) > 2 else points[1:]):
            if dashed:
                _dashed_line(draw, start, end, color, 2)
            else:
                draw.line([start, end], fill=color, width=2)
    return tile


class SelectionResult(dict):
    """ {band: polygon} of the accepted regions, with .origins = {band: 'accepted_automatic' or 'edited'} """

    def __init__(self, regions, origins):
        super().__init__(regions)
        self.origins = dict(origins)


class SelectionSession:
    """
    The decisions of the user, band by band
    Inputs: {band: rct.Image} of the panel capture, the regions saved before {band: polygon} (they count as accepted, with
            the origins {band: origin} written with them) and the parameters used to find the automatic region
    Each band has to be accepted: with the region found from the QR code as it is, or with a polygon that was edited.
    The origin of an accepted region is 'accepted_automatic' if it is the region found from the QR code, else 'edited'
    """

    def __init__(self, images, saved=None, percent_inner_area=0.7, sat_threshold=65000, saved_origins=None):
        self.images = dict(images)
        self.bands = list(self.images)
        self.saved = {band: polygon for band, polygon in (saved or {}).items() if band in self.images}
        self.percent_inner_area = percent_inner_area
        self.sat_threshold = sat_threshold
        self.index = 0
        self.decisions = {band: None for band in self.bands}          # None or 'accepted'
        self.polygons = {band: None for band in self.bands}           # the accepted polygons
        self.origins = {band: None for band in self.bands}
        self.notes = []                                               # what could not be used from the saved regions
        self._automatic = {}
        for band, polygon in list(self.saved.items()):
            try:
                self.accept(band, polygon, origin=(saved_origins or {}).get(band))
            except ValueError as error:
                self.notes.append(f"The saved region of band '{band}' is not valid and was ignored: {error}")
                del self.saved[band]

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

    def proposed_polygon(self, band):
        """ The region shown for a band: the accepted one, else the saved one, else the one found from the QR code """
        if self.polygons[band] is not None:
            return [list(vertex) for vertex in self.polygons[band]]
        if self.saved.get(band):
            return [list(vertex) for vertex in self.saved[band]]
        return [list(vertex) for vertex in (self.automatic_polygon(band) or [])]

    starting_polygon = proposed_polygon

    def state(self, band):
        """ 'accepted', 'pending' (there is a region to accept) or 'no_region' (it has to be drawn) """
        if self.decisions[band] == 'accepted':
            return 'accepted'
        return 'pending' if self.proposed_polygon(band) else 'no_region'

    def stats(self, band, polygon):
        """ Pixels, saturated pixels and variation (%) inside a polygon, or None while it is not a valid region """
        try:
            rct.validate_polygon(polygon, self.image_size(band), band)
        except ValueError:
            return None
        return self.images[band].panel_region_stats(polygon, self.sat_threshold)

    def accept(self, band, polygon=None, origin=None):
        """
        Accept a polygon for a band (default: the one that is shown); ValueError with the reason if it is not a valid region
        The origin is worked out when it is not given: 'accepted_automatic' if it is the region found from the QR code
        """
        if polygon is None:
            polygon = self.proposed_polygon(band)
        rct.validate_polygon(polygon, self.image_size(band), f"Band '{band}'")
        polygon = [[int(round(x)), int(round(y))] for x, y in polygon]
        if origin is None:
            origin = 'accepted_automatic' if polygon == self.automatic_polygon(band) else 'edited'
        self.polygons[band] = polygon
        self.decisions[band] = 'accepted'
        self.origins[band] = origin

    confirm = accept

    def unaccept(self, band):
        self.polygons[band] = None
        self.decisions[band] = None
        self.origins[band] = None

    def accept_all_automatic(self):
        """ Accept the region found from the QR code for every band that is not accepted; returns the bands without one """
        left = []
        for band in self.bands:
            if self.decisions[band] == 'accepted':
                continue
            automatic = self.automatic_polygon(band)
            if automatic:
                self.accept(band, automatic, origin='accepted_automatic')
            else:
                left.append(band)
        return left

    def go_to(self, index):
        self.index = max(0, min(len(self.bands) - 1, index))

    def accepted_count(self):
        return sum(decision == 'accepted' for decision in self.decisions.values())

    def is_complete(self):
        return all(decision == 'accepted' for decision in self.decisions.values())

    def next_not_accepted(self, after=None):
        """ Index of the next band (after the current one, going round) that is not accepted, or None """
        start = self.index if after is None else after
        for step in range(1, len(self.bands) + 1):
            index = (start + step) % len(self.bands)
            if self.decisions[self.bands[index]] != 'accepted':
                return index
        return None

    def result(self):
        """ The accepted regions {band: polygon}, with .origins """
        accepted = [band for band in self.bands if self.decisions[band] == 'accepted']
        return SelectionResult({band: self.polygons[band] for band in accepted},
                               {band: self.origins[band] for band in accepted if self.origins[band]})


class CaptureSet:
    """
    The captures that can be looked at, and the session of each one (opened when it is chosen)
    Inputs: the names of the captures, a function that reads the images of a capture ({band: rct.Image}), the captures that
            must be fully accepted before the regions are saved (the panel captures), the regions and origins saved before
            for every capture ({capture: {band: ...}}) and images that were read already ({capture: {band: rct.Image}})
    """

    def __init__(self, names, load, required=(), saved=None, saved_origins=None, percent_inner_area=0.7,
                 sat_threshold=65000, preloaded=None, current=None):
        self.required = list(dict.fromkeys(required))
        self.names = self.required + [name for name in dict.fromkeys(names) if name not in self.required]
        if not self.names:
            raise ValueError("There is no capture to look at")
        self.load = load
        self.saved, self.saved_origins = saved or {}, saved_origins or {}
        self.percent_inner_area, self.sat_threshold = percent_inner_area, sat_threshold
        self.sessions = {}
        for name, images in (preloaded or {}).items():
            self.sessions[name] = self._make(name, images)
        self.current = current if current in self.names else self.names[0]

    @classmethod
    def single(cls, session, name='capture'):
        """ A set with one capture whose session exists already """
        captures = cls([name], None, required=[name])
        captures.sessions[name] = session
        return captures

    def _make(self, name, images):
        return SelectionSession(images, self.saved.get(name), self.percent_inner_area, self.sat_threshold,
                                self.saved_origins.get(name))

    def session(self, name):
        """ The session of a capture; its images are read the first time (any error of the reading is raised) """
        if name not in self.sessions:
            if self.load is None:
                raise KeyError(name)
            self.sessions[name] = self._make(name, self.load(name))
        return self.sessions[name]

    def summary(self, name):
        if name not in self.sessions:
            return 'not opened'
        session = self.sessions[name]
        return f"{session.accepted_count()} of {len(session.bands)} accepted"

    def not_ready(self):
        """ The captures that must be accepted and are not: {capture: bands still to accept, or None if not opened} """
        missing = {}
        for name in self.required:
            session = self.sessions.get(name)
            if session is None:
                missing[name] = None
            elif not session.is_complete():
                missing[name] = [band for band in session.bands if session.decisions[band] != 'accepted']
        return missing

    def is_ready(self):
        return not self.not_ready()

    def result(self):
        """ {capture: SelectionResult} of the captures whose bands are all accepted """
        return {name: session.result() for name, session in self.sessions.items() if session.is_complete()}


def count_wrapped_lines(measure, text, width):
    """ Lines that a text takes when it is wrapped at words in a width; measure(text) gives the width of a text """
    total = 0
    for paragraph in text.split('\n'):
        line = ''
        total += 1
        for word in paragraph.split(' '):
            candidate = f"{line} {word}" if line else word
            if line and measure(candidate) > width:
                total += 1
                line = word
            else:
                line = candidate
    return total


# ======================================================================================
# WINDOW
# ======================================================================================

class PanelSelectorWindow:
    """ The overview and the editor; use select_panel_regions() to run it """

    def __init__(self, root, source, title="Reference panel regions", start='overview'):
        import tkinter as tk
        from tkinter import ttk
        from PIL import Image as PILImage, ImageTk
        self.tk, self.ttk, self.PILImage, self.ImageTk = tk, ttk, PILImage, ImageTk
        self.root = root
        self.captures = CaptureSet.single(source) if isinstance(source, SelectionSession) else source
        self.session = self.captures.session(self.captures.current)
        self.capture_labels = {}
        self.cancelled = True
        self.finished = False
        self.mode = None
        self.model = PolygonModel()
        self.view = ViewTransform()
        self.display = None
        self.photo = None
        self.drag_vertex = None
        self.pan_start = None
        self.fitted_band = None
        self.display_cache = {}
        self.tiles = {}
        self.buttons = {}                     # buttons of the editor
        self.overview_buttons = {}
        self.root.title(title)
        self.root.geometry("1180x860")
        self.root.minsize(1060, 560)
        apply_dark_theme(self.root)
        self._build()
        hide_title_icon(self.root)
        if start == 'editor':
            self.show_editor(self.session.index)
        else:
            self.show_overview()

    # --- construction ---------------------------------------------------------------------------------------

    def _build(self):
        tk, ttk = self.tk, self.ttk
        top = ttk.Frame(self.root, padding=(12, 8))
        top.pack(fill='x')
        self.title_label = ttk.Label(top, style='Title.TLabel')
        self.title_label.pack(side='left')
        self.status_label = ttk.Label(top, style='Muted.TLabel')
        self.status_label.pack(side='right')
        self.capture_var = tk.StringVar()
        self.capture_box = ttk.Combobox(top, textvariable=self.capture_var, state='readonly', width=44)
        if len(self.captures.names) > 1:
            self.capture_box.pack(side='right', padx=(6, 12))
            ttk.Label(top, text='Capture', style='Muted.TLabel').pack(side='right')
        self.capture_box.bind('<<ComboboxSelected>>', lambda event: self._on_capture_chosen())
        self._hover(self.capture_box, 'capture')

        # The explanation takes the whole width and has the height of the longest explanation, so that pointing at an
        # element never changes the size of what is above it (which made the image blink)
        footer = ttk.Frame(self.root, padding=(12, 0, 12, 6))
        footer.pack(side='bottom', fill='x')
        footer.pack_propagate(False)
        self.footer = footer
        self.hint_label = ttk.Label(footer, text=DEFAULT_HINT, foreground=COLORS['foreground'], justify='left')
        self.hint_label.pack(anchor='w', fill='x')
        footer.configure(height=self.hint_height(1156))
        footer.bind('<Configure>', self._on_footer_configure)

        self.container = ttk.Frame(self.root)
        self.container.pack(fill='both', expand=True)
        self._build_overview()
        self._build_editor()
        for sequence, command in (('<Return>', self._key_confirm), ('<r>', self._key_reset), ('<c>', self._key_clear),
                                  ('<BackSpace>', self._key_undo), ('<Left>', self._key_previous),
                                  ('<Right>', self._key_next), ('<Escape>', self._key_escape)):
            self.root.bind(sequence, lambda event, command=command: command())
        self.root.protocol('WM_DELETE_WINDOW', self.cancel)

    def hint_height(self, wrap):
        """ Height in pixels of the footer: the lines of the longest explanation when it is wrapped at wrap pixels """
        from tkinter import font as tkfont
        font = tkfont.Font(root=self.root, font=FONT)
        lines = max(count_wrapped_lines(font.measure, text, wrap) for text in [DEFAULT_HINT, *HINTS.values()])
        return lines * font.metrics('linespace') + 12

    def _on_footer_configure(self, event):
        wrap = max(event.width - 24, 200)
        self.hint_label.configure(wraplength=wrap)
        height = self.hint_height(wrap)
        if int(self.footer.cget('height')) != height:
            self.footer.configure(height=height)

    def _hover(self, widget, name):
        """ Show what an element does while the mouse is over it """
        widget.bind('<Enter>', lambda event: self.hint_label.configure(text=HINTS[name]), add='+')
        widget.bind('<Leave>', lambda event: self.hint_label.configure(text=DEFAULT_HINT), add='+')

    def _build_overview(self):
        ttk = self.ttk
        from ui_widgets import ScrollablePage
        self.overview = ttk.Frame(self.container)
        buttons = ttk.Frame(self.overview, padding=(12, 4, 12, 8))
        buttons.pack(side='bottom', fill='x')
        self.overview_message = ttk.Label(self.overview, style='Muted.TLabel', padding=(12, 4))
        self.overview_message.pack(side='bottom', anchor='w')
        for name, text, command, side in (('cancel', 'Cancel', self.cancel, 'right'),
                                          ('finish', 'Save and close', self.finish, 'right'),
                                          ('accept_all', 'Accept all automatic regions', self.accept_all, 'left')):
            self.overview_buttons[name] = ttk.Button(buttons, text=text, command=command)
            self.overview_buttons[name].pack(side=side, padx=(0, 6) if side == 'left' else (6, 0))
            self._hover(self.overview_buttons[name], name)
        self.tile_page = ScrollablePage(self.overview)
        self.tile_page.widget.pack(fill='both', expand=True, padx=(12, 0))
        self.tile_bands = list(self.session.bands)
        for number, band in enumerate(self.session.bands):
            self._build_tile(band, number)

    def _build_tile(self, band, number):
        ttk = self.ttk
        row, column = divmod(number, TILE_COLUMNS)
        frame = ttk.Frame(self.tile_page.body, padding=(0, 0, 12, 12))
        frame.grid(row=row, column=column, sticky='nw')
        head = ttk.Frame(frame)
        head.pack(fill='x')
        title = ttk.Label(head, text=band, style='Heading.TLabel')
        title.pack(side='left')
        badge = ttk.Label(head, style='Muted.TLabel')
        badge.pack(side='right')
        image = ttk.Label(frame)
        image.pack(pady=(3, 3))
        stats = ttk.Label(frame, style='Muted.TLabel', wraplength=TILE_SIZE[0])
        stats.pack(anchor='w')
        actions = ttk.Frame(frame)
        actions.pack(fill='x', pady=(4, 0))
        accept = ttk.Button(actions, text='Accept', command=lambda band=band: self.toggle_accept(band))
        accept.pack(side='left')
        edit = ttk.Button(actions, text='Edit...', command=lambda band=band: self.edit_band(band))
        edit.pack(side='left', padx=(6, 0))
        self._hover(image, 'tile_image')
        self._hover(accept, 'tile_accept')
        self._hover(edit, 'tile_edit')
        self.tiles[band] = {'frame': frame, 'title': title, 'badge': badge, 'image': image, 'stats': stats,
                            'accept': accept, 'edit': edit, 'photo': None}

    def _build_editor(self):
        tk, ttk = self.tk, self.ttk
        self.editor = ttk.Frame(self.container)
        buttons = ttk.Frame(self.editor, padding=(12, 4, 12, 8))
        buttons.pack(side='bottom', fill='x')
        info = ttk.Frame(self.editor, padding=(12, 6))
        info.pack(side='bottom', fill='x')
        self.stats_label = ttk.Label(info)
        self.stats_label.pack(anchor='w')
        self.message_label = ttk.Label(info, style='Muted.TLabel')
        self.message_label.pack(anchor='w')

        self.canvas = tk.Canvas(self.editor, background=COLORS['panel'], highlightthickness=0, cursor='crosshair')
        self.canvas.pack(fill='both', expand=True, padx=12)
        self.canvas.bind('<Configure>', self._on_configure)
        self.canvas.bind('<Button-1>', self._on_press)
        self.canvas.bind('<B1-Motion>', self._on_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_release)
        self.canvas.bind('<Button-3>', self._on_remove)
        self.canvas.bind('<Button-2>', self._on_pan_start)
        self.canvas.bind('<B2-Motion>', self._on_pan_move)
        self.canvas.bind('<MouseWheel>', self._on_wheel)
        self._hover(self.canvas, 'canvas')
        # editing buttons on the left, decisions on the right (packed from the right edge inwards)
        for name, text, command in (('previous', 'Previous', self.previous), ('next', 'Next', self.next),
                                    ('reset', 'Reset to automatic', self.reset), ('clear', 'Clear', self.clear)):
            self.buttons[name] = ttk.Button(buttons, text=text, command=command)
            self.buttons[name].pack(side='left', padx=(0, 6))
        for name, text, command in (('overview', 'Back to overview', self.show_overview),
                                    ('confirm', 'Confirm', self.confirm)):
            self.buttons[name] = ttk.Button(buttons, text=text, command=command)
            self.buttons[name].pack(side='right', padx=(6, 0))
        for name, button in self.buttons.items():
            self._hover(button, name)

    # --- views ---------------------------------------------------------------------------------------------

    def _status_text(self):
        return f"{self.session.accepted_count()} of {len(self.session.bands)} bands accepted"

    def display_image(self, band):
        key = (self.captures.current, band)
        if key not in self.display_cache:
            self.display_cache[key] = self.session.images[band].display_image()
        return self.display_cache[key]

    def _capture_label(self, name):
        kind = 'panel, ' if name in self.captures.required else ''
        return f"{name}  ({kind}{self.captures.summary(name)})"

    def _refresh_capture_box(self):
        self.capture_labels = {self._capture_label(name): name for name in self.captures.names}
        self.capture_box.configure(values=list(self.capture_labels))
        self.capture_var.set(self._capture_label(self.captures.current))

    def _on_capture_chosen(self):
        name = self.capture_labels.get(self.capture_var.get())
        if name is not None:
            self.switch_capture(name)

    def switch_capture(self, name):
        """ Look at another capture (its images are read the first time). Returns False if it could not be opened """
        if name == self.captures.current:
            return True
        self.overview_message.configure(text=f"Reading {name}...", style='Muted.TLabel')
        self.root.update_idletasks()
        try:
            session = self.captures.session(name)
            if session.bands != self.tile_bands:
                raise ValueError(f"{name} has the bands {session.bands}, not {self.tile_bands}")
        except Exception as error:
            self._refresh_capture_box()
            self.overview_message.configure(text=f"{name} could not be opened: {type(error).__name__}: {error}",
                                            style='Error.TLabel')
            return False
        self.captures.current = name
        self.session = session
        if self.mode == 'editor':
            self.show_overview()
        else:
            self.refresh_overview()
        return True

    def show_overview(self):
        self.mode = 'overview'
        self.editor.pack_forget()
        self.overview.pack(fill='both', expand=True)
        self.title_label.configure(text="Reference panel regions: overview")
        self.refresh_overview()

    def refresh_overview(self):
        for band in self.session.bands:
            self.refresh_tile(band)
        self._update_overview_state()

    def tile_polygons(self, band):
        """ What a tile draws: the region found from the QR code, and the one that is accepted or proposed """
        session = self.session
        shown = []
        proposed = session.proposed_polygon(band)
        if proposed:
            shown.append((proposed, COLORS['ok'] if session.decisions[band] == 'accepted' else COLORS['manual'], False))
        automatic = session.automatic_polygon(band)
        if automatic:
            shown.append((automatic, COLORS['automatic'], True))
        return shown

    def refresh_tile(self, band):
        session, tile = self.session, self.tiles[band]
        state = session.state(band)
        polygon = session.proposed_polygon(band)
        picture = render_tile(self.display_image(band), self.tile_polygons(band))
        tile['photo'] = self.ImageTk.PhotoImage(picture)
        tile['image'].configure(image=tile['photo'])
        if state == 'accepted':
            tile['badge'].configure(text=BADGE_TEXT.get(session.origins[band], 'Accepted'), style='Ok.TLabel')
            tile['accept'].configure(text='Accepted (undo)')
        elif state == 'pending':
            origin = 'saved' if session.saved.get(band) else 'QR code'
            tile['badge'].configure(text=f"Not accepted ({origin})", style='Muted.TLabel')
            tile['accept'].configure(text='Accept')
        else:
            tile['badge'].configure(text='No region: use Edit', style='Error.TLabel')
            tile['accept'].configure(text='Accept')
        tile['accept'].state(['disabled'] if state == 'no_region' else ['!disabled'])
        stats = session.stats(band, polygon) if len(polygon) >= 3 else None
        tile['stats'].configure(text='' if stats is None else
                                f"{stats['n_pixels']} pixels, {stats['n_saturated']} saturated, "
                                f"variation {stats['variation_percent']:.2f} %")

    def _update_overview_state(self):
        session = self.session
        self.status_label.configure(text=self._status_text())
        ready = self.captures.is_ready()
        self.overview_buttons['finish'].state(['!disabled'] if ready else ['disabled'])
        pending = [band for band in session.bands if session.decisions[band] != 'accepted']
        text = (f"{self.captures.current}: {session.accepted_count()} of {len(session.bands)} bands accepted. " +
                ("Every region is accepted." if not pending else f"Still to accept: {', '.join(pending)}."))
        missing = self.captures.not_ready()
        if len(self.captures.required) > 1 or (missing and list(missing) != [self.captures.current]):
            text += (" Panel captures: " + ', '.join(f"{name} ({self.captures.summary(name)})"
                                                     for name in self.captures.required) + '.')
        if ready:
            text += " Save and close writes the accepted captures."
        if session.notes:
            text += ' ' + ' '.join(session.notes)
        self.overview_message.configure(text=text, style='Muted.TLabel')
        self._refresh_capture_box()

    def show_editor(self, band):
        """ Open a band (its name or number) in the editor """
        index = band if isinstance(band, int) else self.session.bands.index(band)
        self.session.go_to(index)
        self.mode = 'editor'
        self.overview.pack_forget()
        self.editor.pack(fill='both', expand=True)
        self._load_band()

    # --- overview actions -----------------------------------------------------------------------------------

    def toggle_accept(self, band):
        session = self.session
        if session.decisions[band] == 'accepted':
            session.unaccept(band)
        else:
            try:
                session.accept(band)
            except ValueError as error:
                self.overview_message.configure(text=str(error), style='Error.TLabel')
                return
        self.refresh_tile(band)
        self._update_overview_state()

    def edit_band(self, band):
        self.show_editor(band)

    def accept_all(self):
        left = self.session.accept_all_automatic()
        self.refresh_overview()
        if left:
            self.overview_message.configure(
                text=f"No region was found from the QR code for: {', '.join(left)}. Draw them with Edit.", style='Error.TLabel')

    # --- editor: band ---------------------------------------------------------------------------------------

    def _load_band(self):
        session = self.session
        band = session.band
        self.display = self.PILImage.fromarray(self.display_image(band))
        self.model = PolygonModel(session.proposed_polygon(band))
        self.fitted_band = None
        self._fit_if_possible()
        self.title_label.configure(text=f"Band {session.index + 1} of {len(session.bands)}: {band}")
        state = session.state(band)
        automatic = session.automatic_polygon(band)
        if state == 'accepted':
            origin = ORIGIN_TEXT.get(session.origins[band], '')
            text = "Accepted" + (f" ({origin})" if origin else '') + ". Confirm again to replace it."
        elif automatic:
            text = "Not accepted yet. Confirm accepts the polygon as shown."
        else:
            text = "No region was found from the QR code: draw the panel (3 or more clicks) and Confirm."
        self.message(text, error=False)
        self.status_label.configure(text=self._status_text())
        self._update_stats()
        self._redraw()

    def _fit_if_possible(self):
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        if width > 20 and height > 20 and self.fitted_band != self.session.band:
            self.view.fit(self.session.image_size(), (width, height))
            self.fitted_band = self.session.band

    # --- editor: drawing --------------------------------------------------------------------------------------

    def _outline(self, polygon, color, **options):
        points = [coordinate for vertex in polygon for coordinate in self.view.to_canvas(vertex)]
        if len(polygon) >= 3:
            self.canvas.create_polygon(points, outline=color, fill='', **options)
        elif len(polygon) == 2:
            self.canvas.create_line(points, fill=color, **options)

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
        session, band = self.session, self.session.band
        reference = session.polygons[band] if session.decisions[band] == 'accepted' else session.saved.get(band)
        if reference:
            self._outline(reference, COLORS['ok'], width=2)               # accepted or saved before, to compare
        self._outline(self.model.vertices, COLORS['manual'], width=2)
        # the region found from the QR code is drawn last, so that it can be seen over an identical drawn one
        automatic = session.automatic_polygon(band)
        if automatic:
            self._outline(automatic, COLORS['automatic'], dash=(6, 4), width=2)
        for x, y in (view.to_canvas(vertex) for vertex in self.model.vertices):
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

    # --- editor: mouse -----------------------------------------------------------------------------------------

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

    # --- editor: commands ---------------------------------------------------------------------------------------

    def confirm(self):
        """ Accept the polygon shown for this band, then go to the next band that is not accepted (or to the overview) """
        try:
            self.session.accept(self.session.band, self.model.vertices)
        except ValueError as error:
            self.message(str(error), error=True)
            return
        self.status_label.configure(text=self._status_text())
        self._refresh_capture_box()
        following = self.session.next_not_accepted()
        if following is None:
            self.show_overview()
        else:
            self.show_editor(following)

    def previous(self):
        self.show_editor(self.session.index - 1)

    def next(self):
        self.show_editor(self.session.index + 1)

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

    # --- keys -----------------------------------------------------------------------------------------------------

    def _key_confirm(self):
        if self.mode == 'editor':
            self.confirm()

    def _key_reset(self):
        if self.mode == 'editor':
            self.reset()

    def _key_clear(self):
        if self.mode == 'editor':
            self.clear()

    def _key_undo(self):
        if self.mode == 'editor':
            self.undo()

    def _key_previous(self):
        if self.mode == 'editor':
            self.previous()

    def _key_next(self):
        if self.mode == 'editor':
            self.next()

    def _key_escape(self):
        if self.mode == 'editor':
            self.show_overview()
        else:
            self.cancel()

    # --- end ----------------------------------------------------------------------------------------------------

    def finish(self):
        missing = self.captures.not_ready()
        if missing:
            parts = [f"{name}: " + ('not opened' if bands is None else f"still to accept {', '.join(bands)}")
                     for name, bands in missing.items()]
            self.overview_message.configure(text="Accept every band of the panel captures first. " + '; '.join(parts),
                                            style='Error.TLabel')
            return
        self.cancelled = False
        self.finished = True
        self.root.destroy()

    def cancel(self):
        self.cancelled = True
        self.finished = True
        self.root.destroy()


def select_panel_regions(captures, load, required=None, saved=None, saved_origins=None, percent_inner_area=0.7,
                         sat_threshold=65000, parent=None, preloaded=None, current=None, start='overview'):
    """
    Open the window and wait for the user
    Inputs: the names of the captures that can be looked at, a function that reads the images of a capture, the panel
            captures that must be fully accepted (default: the first one), the regions and origins saved before for every
            capture (they are shown as accepted), the images that were read already and, when another window is open,
            that window (the window is then modal and opens above it)
    Outputs: {capture: SelectionResult} for every capture whose bands are all accepted (the panel captures at least), or
             None if the user cancelled
    """
    import tkinter as tk
    required = list(captures[:1]) if required is None else list(required)
    capture_set = CaptureSet(captures, load, required, saved, saved_origins, percent_inner_area, sat_threshold,
                             preloaded, current)
    if parent is None:
        window_root = tk.Tk()
        window = PanelSelectorWindow(window_root, capture_set, start=start)
        window_root.mainloop()
    else:
        window_root = tk.Toplevel(parent)
        window_root.transient(parent)
        window = PanelSelectorWindow(window_root, capture_set, start=start)
        window_root.grab_set()
        parent.wait_window(window_root)
    return None if window.cancelled else capture_set.result()


def main(argv=None):
    import radiometric_workflow as wf
    parser = argparse.ArgumentParser(description="Check, accept and draw the reference panel region of each band")
    parser.add_argument('--config', help="configuration json file of the workflow (default: radiometric_inputs.json "
                                         "in --root if it exists, otherwise the defaults)")
    parser.add_argument('--root', default='.', help="folder from which the relative paths are resolved")
    parser.add_argument('--capture', help="capture to show first; it is added to the panel captures if it is not one "
                                          "(default: INPUTS.reference_panel_capture)")
    arguments = parser.parse_args(argv)

    config_file = Path(arguments.config) if arguments.config else Path(arguments.root) / 'radiometric_inputs.json'
    if arguments.config and not config_file.is_file():
        print(f"ERROR: configuration file not found: {config_file}", file=sys.stderr)
        return 2
    config = wf.load_config(config_file) if config_file.is_file() else wf.default_config()
    problems = wf.validate_config(config, arguments.root)
    if problems:
        for problem in problems:
            print(f"ERROR: {problem}", file=sys.stderr)
        return 1
    root = Path(arguments.root)
    required = [name for name in (config['INPUTS.reference_panel_capture'].strip(),
                                  config['INPUTS.reference_panel_capture_after'].strip()) if name]
    if arguments.capture and arguments.capture not in required:
        required.insert(0, arguments.capture)
    first = arguments.capture or required[0]
    regions_path = wf.resolve_path(root, config['PANEL.manual_regions_file'] or
                                   Path(config['INPUTS.results_folder']) / 'panel_regions.json')
    images_folder = wf.resolve_path(root, config['INPUTS.images_folder'])
    names = rct.list_capture_names(images_folder, config['INPUTS.file_pattern'])
    preloaded = {}
    for name in required:
        print(f"Reading {name} ...", flush=True)
        preloaded[name] = wf.read_capture_images(config, root, name)
    saved = wf.load_panel_regions(regions_path) if regions_path.is_file() else {}
    origins = wf.load_panel_origins(regions_path) if regions_path.is_file() else {}
    selected = select_panel_regions(names, lambda name: wf.read_capture_images(config, root, name), required, saved,
                                    origins, config['PANEL.percent_inner_area'], config['PANEL.saturation_threshold'],
                                    preloaded=preloaded, current=first)
    if selected is None:
        print("Cancelled: nothing was saved.")
        return 1
    regions_path.parent.mkdir(parents=True, exist_ok=True)
    for name, result in selected.items():
        wf.save_panel_regions(regions_path, name, result, result.origins)
    print(f"PANEL_REGIONS_JSON={regions_path.resolve()}")
    if config['PANEL.manual_regions_file'] == '':
        print("Set PANEL.manual_regions_file to this file in the configuration to use it.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
