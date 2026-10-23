# %%
"""
RedEdge Image Class for Radiometric Correction

Simplified from the Micasense RedEdge Image Processing Library to avoid issues with missing 
Exif metadata in RedEdge-P images (frameware v1.3.1 and v1.4.5)

For the original/complete library, see:
https://github.com/micasense/imageprocessing

The Micasense imagerocessing Library is licensed under the MIT License
"""

# Standard Library Imports
import hashlib
import json
import os
import platform
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path

# Third-Party Imports
import cv2
import matplotlib.pyplot as plt
# import matplotlib.patches as patches
import numpy as np
from pyzbar.pyzbar import decode, ZBarSymbol
from shapely.geometry import Point as PointS, LinearRing as LinearR, LineString as LineS
from shapely.geometry.polygon import Polygon as PolyS
from skimage import measure


# Polygons drawn on the images (reference panel regions)
def _segments_intersect(a, b, c, d):
    """ True if the segments ab and cd cross (touching at a shared end point is not crossing) """
    def orientation(p, q, r):
        value = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
        return int(value > 0) - int(value < 0)
    return orientation(a, b, c) * orientation(a, b, d) < 0 and orientation(c, d, a) * orientation(c, d, b) < 0


def validate_polygon(points, image_size, name='polygon'):
    """
    Check a polygon [[x, y], ...] drawn on an image of size (width, height): at least 3 vertices, inside the image,
    with area and without crossing edges. Raises ValueError with the reason
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError(f"{name}: a region needs at least 3 vertices [x, y]")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name}: the region has non-finite coordinates")
    width, height = image_size
    if (points < 0).any() or (points[:, 0] > width - 1).any() or (points[:, 1] > height - 1).any():
        raise ValueError(f"{name}: the region is not inside the {width} x {height} image")
    count = len(points)
    for i in range(count):
        for j in range(i + 1, count):
            if j == i + 1 or (i == 0 and j == count - 1):
                continue
            if _segments_intersect(points[i], points[(i + 1) % count], points[j], points[(j + 1) % count]):
                raise ValueError(f"{name}: the edges of the region cross each other")
    if cv2.contourArea(points.astype(np.float32)) <= 0:
        raise ValueError(f"{name}: the region has no area")


# Define the camera matrix
def cv2_camera_matrix(principal_point, focal_length, focal_plane_resolution_px_per_mm): 
    center_x = principal_point[0] * focal_plane_resolution_px_per_mm[0]
    center_y = principal_point[1] * focal_plane_resolution_px_per_mm[1]

    # set up camera matrix for cv2
    cam_mat = np.zeros((3, 3))
    cam_mat[0, 0] = focal_length * focal_plane_resolution_px_per_mm[0]
    cam_mat[1, 1] = focal_length * focal_plane_resolution_px_per_mm[1]
    cam_mat[2, 2] = 1.0
    cam_mat[0, 2] = center_x
    cam_mat[1, 2] = center_y

    # set up distortion coefficients for cv2
    return cam_mat


# Metadata validation helpers - report the file and the field when metadata are missing or invalid
class MetadataError(ValueError):
    """ Raised when image metadata are missing or invalid, naming the file and the field """


def _metadata_value(metadata, key, image_path):
    if key not in metadata or metadata[key] is None:
        raise MetadataError(f"{image_path}: missing metadata field '{key}'")
    return metadata[key]


def _metadata_floats(metadata, key, image_path, count=None):
    """ Return a field as a list of finite floats, accepting a list or a string separated by commas/spaces """
    value = _metadata_value(metadata, key, image_path)
    if isinstance(value, str):
        value = value.replace(',', ' ').split()
    elif np.isscalar(value):
        value = [value]
    try:
        numbers = [float(val) for val in value]
    except (TypeError, ValueError):
        raise MetadataError(f"{image_path}: metadata field '{key}' is not numeric: {metadata[key]!r}")
    if len(numbers) == 0 or not np.all(np.isfinite(numbers)):
        raise MetadataError(f"{image_path}: metadata field '{key}' has no finite values: {metadata[key]!r}")
    if count is not None and len(numbers) != count:
        raise MetadataError(f"{image_path}: metadata field '{key}' has {len(numbers)} values, expected {count}")
    return numbers


def _metadata_positive(metadata, key, image_path):
    """ Return a field (unchanged) after checking that it is a single positive finite number """
    value = _metadata_value(metadata, key, image_path)
    number = _metadata_floats(metadata, key, image_path, count=1)[0]
    if number <= 0:
        raise MetadataError(f"{image_path}: metadata field '{key}' must be positive, got {value!r}")
    return value


# Map the files of one capture to their bands using the band name in the metadata (not the file order)
def map_band_files(image_paths, metadata_list, expected_bands=None, ignore_bands=()):
    """
    Inputs: image paths of one capture, their metadata (same order), optionally the expected band names and the names of
            the bands that are in the files but must be left out (for example a thermal band)
    Outputs: dictionary {band name: (image path, metadata)}, with exactly one file per band
    """
    ignored = set(ignore_bands or ())
    if len(image_paths) != len(metadata_list):
        raise ValueError("image_paths and metadata_list must have the same length")
    band_files = {}
    for image_path, metadata in zip(image_paths, metadata_list):
        band_name = _metadata_value(metadata, 'XMP:BandName', image_path)
        if band_name in ignored:
            continue
        if band_name in band_files:
            raise MetadataError(f"Band '{band_name}' appears twice: {band_files[band_name][0]} and {image_path}")
        band_files[band_name] = (image_path, metadata)
    capture_ids = { metadata.get('XMP:CaptureId') for _, metadata in band_files.values() }
    if len(capture_ids) > 1:
        raise MetadataError(f"Files belong to different captures (XMP:CaptureId): {sorted(map(str, capture_ids))}")
    if expected_bands is not None:
        missing = [band for band in expected_bands if band not in band_files]
        unexpected = [band for band in band_files if band not in expected_bands]
        if missing or unexpected:
            raise MetadataError(f"Band mismatch - missing: {missing}, unexpected: {unexpected}")
    return band_files


# DLS correction - ratio between the irradiance at the panel capture and at the image capture
def dls_irradiance_ratio(panel_image, image):
    """
    Inputs: panel and target rct.Image objects of the SAME band
    Outputs: E_panel / E_image, to be multiplied by the panel-calibrated reflectance. It is 1 for the panel capture
    The Panchro band is treated like the others, using the irradiance written in its own file.
    A ratio outside DLS_CHECKS['ratio_range'] gives a warning
    """
    if panel_image.band_name != image.band_name:
        raise MetadataError(f"DLS correction needs the same band: panel is '{panel_image.band_name}' "
                            f"({panel_image.image_path}), image is '{image.band_name}' ({image.image_path})")
    if (panel_image.irradiance_scale_to_si is None) != (image.irradiance_scale_to_si is None):
        raise MetadataError(f"'XMP:IrradianceScaleToSIUnits' exists in only one of {panel_image.image_path} "
                            f"and {image.image_path}")
    ratio = panel_image.dls_irradiance() / image.dls_irradiance()
    low, high = DLS_CHECKS['ratio_range']
    if not low <= ratio <= high:
        warnings.warn(f"{image.image_path}: DLS ratio {ratio:.3f} is outside the plausible range {low}-{high}",
                      RuntimeWarning)
    return ratio


def capture_dls_ratios(panel_images, images):
    """
    DLS ratios (E_panel / E_image) of all the bands of a capture, Panchro included
    Inputs: dictionaries {band name: rct.Image} for the panel capture and for the image capture
    Outputs: dictionary {band name: (ratio, source)}, with source 'metadata'. A missing or invalid irradiance in
             any band raises MetadataError: no ratio is estimated from other bands
    """
    return { band: (dls_irradiance_ratio(panel_images[band], image), 'metadata') for band, image in images.items() }


# Checks of the DLS data - thresholds (adjust them if needed). They are guesses, not validated limits
DLS_CHECKS = { 'ratio_range': (0.5, 2.0),            # plausible E_panel / E_image
                'band_ratio_tolerance': 0.03,         # largest relative departure of a band from the median ratio
                'max_sun_angle_discrepancy_deg': 10.0,  # see Image.dls_sun_angle_discrepancy_deg
                'jump_threshold': 0.10,               # relative change between consecutive captures
                'panel_drift_tolerance': 0.05 }       # relative change of the panel factor (and of E_DLS / E_panel_estimate)
                                                      # between two panel captures


def check_dls_capture(panel_images, images, ratios=None):
    """
    Check the DLS data of one capture against the panel (see DLS_CHECKS for the limits)
    Inputs: dictionaries {band name: rct.Image} for the panel capture (after detect_reference_panel) and the image
            capture, and optionally the output of capture_dls_ratios
    Outputs: dictionary with the values checked and 'flags', a list of messages (also issued as RuntimeWarning).
             Data that are not available are reported in 'unavailable' instead of raising errors
    """
    flags, unavailable = [], []
    if ratios is None:
        ratios = {}
        for band, image in images.items():
            try:
                ratios[band] = (dls_irradiance_ratio(panel_images[band], image), 'metadata')
            except MetadataError as error:
                unavailable.append(str(error))
    values = { band: ratio for band, (ratio, _) in ratios.items() }
    result = { 'band_ratios': values, 'ratio_sources': { band: source for band, (_, source) in ratios.items() } }

    # 1. Same illumination change in all bands
    if values:
        median = float(np.median(list(values.values())))
        departures = { band: value / median - 1 for band, value in values.items() }
        result['median_ratio'] = median
        result['band_ratio_departures'] = departures
        for band, departure in departures.items():
            if abs(departure) > DLS_CHECKS['band_ratio_tolerance']:
                flags.append(f"Band '{band}' DLS ratio departs {100 * departure:.1f} % from the median of the bands")

    # 2. Irradiance of the DLS compared with the irradiance estimated from the panel. It differs between bands
    #    (14 % in the example, a calibration difference between camera and DLS), so it is only reported here; compare it
    #    between two panel captures with compare_panel_captures()
    estimates = {}
    for band, panel in panel_images.items():
        audit = getattr(panel, 'panel_audit', None)
        try:
            if audit is not None:
                estimates[band] = panel.dls_irradiance() / audit['panel_irradiance_estimate']
        except MetadataError as error:
            unavailable.append(str(error))
    if len(estimates) > 1:
        cv = float(np.std(list(estimates.values())) / np.mean(list(estimates.values())))
        result['dls_over_panel_estimate'] = estimates
        result['dls_over_panel_estimate_variation_across_bands'] = cv

    # 3. DLS pointing: sun angle seen by the DLS against the solar zenith
    result['sun_angle_discrepancy_deg'] = {}
    for name, group in (('panel', panel_images), ('image', images)):
        angles = [ image.dls_sun_angle_discrepancy_deg() for image in group.values() ]
        angles = [ angle for angle in angles if angle is not None ]
        if not angles:
            unavailable.append(f"sun angle discrepancy of the {name} capture (solar elevation or light vector missing)")
            continue
        result['sun_angle_discrepancy_deg'][name] = float(np.median(angles))
        if np.median(angles) > DLS_CHECKS['max_sun_angle_discrepancy_deg']:
            flags.append(f"DLS of the {name} capture: sun angle differs {np.median(angles):.1f} deg from the solar zenith "
                         f"(DLS tilted at least that much, or the angles are inconsistent)")

    # 5. Plausible range (also warned by dls_irradiance_ratio)
    low, high = DLS_CHECKS['ratio_range']
    for band, value in values.items():
        if not low <= value <= high:
            flags.append(f"Band '{band}' DLS ratio {value:.3f} is outside the plausible range {low}-{high}")

    result['flags'], result['unavailable'] = flags, unavailable
    for flag in flags:
        warnings.warn(flag, RuntimeWarning)
    return result


# 4. Changes of the irradiance along a flight
def classify_irradiance_changes(irradiances, threshold=None):
    """
    Find the changes between consecutive captures of the irradiance of one band, in acquisition order
    Inputs: positive irradiances; threshold is the relative change to report (default DLS_CHECKS['jump_threshold'])
    Outputs: list of dictionaries {'index', 'kind', 'change'}: 'spike' is a value that jumps and comes back to the level
             of its neighbours (a single capture, e.g. a shaded or tilted DLS); 'step' is a change that persists
             (e.g. the sun going behind a cloud). 'index' is the first capture after the change; 'change' is the
             relative change from the previous capture
    """
    if threshold is None:
        threshold = DLS_CHECKS['jump_threshold']
    values = [float(value) for value in irradiances]
    if any(not np.isfinite(value) or value <= 0 for value in values):
        raise ValueError("irradiances must be positive and finite")
    events, skip = [], set()
    for i in range(1, len(values)):
        if i in skip:
            continue
        change = values[i] / values[i - 1] - 1
        if abs(change) <= threshold:
            continue
        if i + 1 < len(values):
            back = values[i + 1] / values[i] - 1
            if abs(back) > threshold and back * change < 0 and abs(values[i + 1] / values[i - 1] - 1) <= threshold:
                events.append({'index': i, 'kind': 'spike', 'change': change})
                skip.add(i + 1)
                continue
        events.append({'index': i, 'kind': 'step', 'change': change})
    return events


# 6. Panel captured before and after the flight
def compare_panel_captures(before, after, tolerance=None):
    """
    Compare two panel captures (e.g. before and after the flight)
    Inputs: dictionaries {band name: rct.Image} after detect_reference_panel for each capture
    Outputs: dictionary {'bands': {band: {'factor_ratio', 'irradiance_ratio' and 'dls_over_panel_estimate_ratio' (if
             the DLS is available)}}, 'flags'}. A factor ratio far from 1 shows a change of the light or a problem with
             the panel (shadow, dirt, handling); the DLS ratio of ratios far from 1 shows that the DLS did not follow
             the light seen by the camera
    """
    if tolerance is None:
        tolerance = DLS_CHECKS['panel_drift_tolerance']
    result, flags = {}, []
    for band in before:
        if band not in after:
            flags.append(f"Band '{band}' is missing in the second panel capture")
            continue
        entry = { 'factor_ratio': after[band].panel_audit['radiance_to_reflectance_factor'] /
                                  before[band].panel_audit['radiance_to_reflectance_factor'] }
        try:
            entry['irradiance_ratio'] = after[band].dls_irradiance() / before[band].dls_irradiance()
            # DLS and camera should change together: this ratio of ratios is 1 if the DLS followed the light
            estimate_ratio = after[band].panel_audit['panel_irradiance_estimate'] /                              before[band].panel_audit['panel_irradiance_estimate']
            entry['dls_over_panel_estimate_ratio'] = entry['irradiance_ratio'] / estimate_ratio
        except MetadataError:
            pass
        result[band] = entry
        if abs(entry['factor_ratio'] - 1) > tolerance:
            flags.append(f"Band '{band}': panel factor changed {100 * (entry['factor_ratio'] - 1):.1f} % between captures")
        if abs(entry.get('dls_over_panel_estimate_ratio', 1) - 1) > tolerance:
            flags.append(f"Band '{band}': DLS irradiance / panel irradiance estimate changed "
                         f"{100 * (entry['dls_over_panel_estimate_ratio'] - 1):.1f} % between captures")
    for flag in flags:
        warnings.warn(flag, RuntimeWarning)
    return { 'bands': result, 'flags': flags }


# Provenance record - what was used to produce the outputs, saved next to them
def _jsonable(value):
    """ Convert numpy values (and containers of them) to plain Python types for json """
    if isinstance(value, dict):
        return { str(key): _jsonable(val) for key, val in value.items() }
    if isinstance(value, (list, tuple)):
        return [ _jsonable(val) for val in value ]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


# Two panel captures (before and after the flight)
PANELS_USED = ('first', 'mean', 'interpolate_time', 'nearest_time')


def capture_time(metadata, name='image'):
    """ Time of a capture, in seconds, from EXIF:DateTimeOriginal (written to the second); MetadataError if missing """
    from datetime import datetime
    text = metadata.get('EXIF:DateTimeOriginal')
    try:
        return datetime.strptime(str(text).strip(), '%Y:%m:%d %H:%M:%S').timestamp()
    except ValueError:
        raise MetadataError(f"{name}: EXIF:DateTimeOriginal is missing or not a date: {text!r}")


def panel_weight(mode, image_time=None, first_time=None, second_time=None):
    """
    Weight of the second panel in the calibration of an image
    mode: 'first' (0: only the first panel), 'mean' (0.5), 'interpolate_time' (the position in time of the image between
    the two panel captures, from 0 at the first to 1 at the second; outside them the nearest panel is used) or
    'nearest_time' (0 or 1: the panel captured closest in time to the image; the first one if they are equally close or
    were captured at the same time)
    """
    if mode == 'first':
        return 0.0
    if mode == 'mean':
        return 0.5
    if mode == 'interpolate_time':
        if second_time == first_time:
            return 0.5
        return float(min(1.0, max(0.0, (image_time - first_time) / (second_time - first_time))))
    if mode == 'nearest_time':
        return 1.0 if abs(image_time - second_time) < abs(image_time - first_time) else 0.0
    raise ValueError(f"panels_used must be one of {list(PANELS_USED)}, got {mode!r}")


def effective_panel_factor(first_factor, second_factor=None, weight=0.0, first_irradiance=None, second_irradiance=None):
    """
    The radiance to reflectance factor of an image from two panels. Without the DLS the factors are averaged with the weight
    of the second one. With the DLS (irradiances of the panels given) what is averaged is factor x irradiance, which is a
    constant of the camera that does not depend on the light, and the result is expressed for the irradiance of the
    first panel, the one that the DLS ratio refers to: factor = ((1 - w) f1 E1 + w f2 E2) / E1
    """
    if weight == 0 or second_factor is None:
        return first_factor
    if first_irradiance is None or second_irradiance is None:
        return (1 - weight) * first_factor + weight * second_factor
    return ((1 - weight) * first_factor * first_irradiance + weight * second_factor * second_irradiance) / first_irradiance


def band_record(output_path, panel_image, image, reflectance, second_panel=None, second_weight=0.0, effective_factor=None):
    """
    Describe how one output band was produced
    Inputs: path of the saved file, panel and target rct.Image of the band (after detect_reference_panel() and
            digital_numbers_to_reflectance()) and the reflectance array that was saved; optionally the image of the second
            panel, its weight and the factor that was used when it was combined with the first
    Outputs: dictionary that can be saved as json (see write_provenance)
    """
    return _jsonable({ 'band': image.band_name,
                       'second_panel_file': str(second_panel.image_path) if second_panel is not None else None,
                       'second_panel': getattr(second_panel, 'panel_audit', None),
                       'second_panel_weight': second_weight,
                       'effective_factor': effective_factor,
                       'output_file': str(output_path),
                       'output_dtype': str(reflectance.dtype),
                       'output_shape': reflectance.shape,
                       'image_file': str(image.image_path),
                       'image_capture_id': image.metadata.get('XMP:CaptureId'),
                       'panel_file': str(panel_image.image_path),
                       'panel_capture_id': panel_image.metadata.get('XMP:CaptureId'),
                       'panel': getattr(panel_image, 'panel_audit', None),
                       'dls': { 'used': image.dls_ratio is not None,
                                'ratio_panel_over_image': image.dls_ratio,
                                'ratio_source': image.dls_ratio_source,
                                'attitude_as_written': { key: image.metadata.get('XMP:' + key)
                                                                    for key in ('Roll', 'Pitch', 'Yaw') },
                                'sun_angle_discrepancy_deg': image.dls_sun_angle_discrepancy_deg(),
                                'panel_horizontal_irradiance': panel_image.horizontal_irradiance,
                                'image_horizontal_irradiance': image.horizontal_irradiance,
                                'panel_irradiance_scale_to_si': panel_image.irradiance_scale_to_si,
                                'image_irradiance_scale_to_si': image.irradiance_scale_to_si } })


def write_provenance(path, band_records, parameters=None, checks=None):
    """
    Save a json file with the records of the bands (see band_record), the parameters of the run and the software used
    The module is identified by the hash of its text (line endings normalized), so changes in the code are noticed
    """
    import cv2
    code = Path(__file__).read_bytes().replace(b'\r\n', b'\n')
    dls_used = any(record['dls']['used'] for record in band_records)
    record = { 'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
               'method': 'panel_and_dls' if dls_used else 'panel_only',
               'software': { 'radiometric_correction_tools_sha256': hashlib.sha256(code).hexdigest(),
                             'python': platform.python_version(),
                             'numpy': np.__version__,
                             'opencv': cv2.__version__ },
               'parameters': _jsonable(parameters or {}),
               'dls_checks_limits': _jsonable(DLS_CHECKS),
               'checks': _jsonable(checks or {}),
               'bands': band_records }
    path = Path(path)
    path.write_text(json.dumps(record, indent=2), encoding='utf-8')
    return path


# Names of the files: a regular expression (case-insensitive) with a group 'capture' (the name of the capture, shared by
# the files of the bands) and, optionally, a group 'index' (the number of the file in the capture). The band of each
# file is not taken from the name but from the metadata
CAPTURE_FILE_PATTERN = r'(?P<capture>.+)_(?P<index>\d+)\.tif$'


def compile_file_pattern(pattern=None):
    """ The compiled pattern of the file names; ValueError with the reason if it cannot be used """
    try:
        compiled = re.compile(pattern or CAPTURE_FILE_PATTERN, re.IGNORECASE)
    except re.error as error:
        raise ValueError(f"the file name pattern is not a valid regular expression: {error}")
    if 'capture' not in compiled.groupindex:
        raise ValueError("the file name pattern needs a group named 'capture', e.g. (?P<capture>.+)_(?P<index>\\d+)\\.tif$")
    return compiled


def _capture_files(images_folder, pattern=None):
    """ [(capture name, index or None, path)] of the files of the folder whose names match the pattern """
    compiled = compile_file_pattern(pattern)
    found = []
    for path in Path(images_folder).iterdir():
        match = compiled.match(path.name) if path.is_file() else None
        if match:
            index = match.groupdict().get('index')
            found.append((match.group('capture'), int(index) if index and index.isdigit() else None, path))
    return found


def list_capture_names(images_folder, pattern=None):
    """ Return the sorted names of the captures in a folder, e.g. IMG_0005_1.tif ... IMG_0005_6.tif -> 'IMG_0005' """
    return sorted({name for name, _, _ in _capture_files(images_folder, pattern)})


def find_capture_paths(images_folder, capture_name, pattern=None):
    """ Return the image paths of a capture, in the order of their numbers: IMG_0005_1.tif, IMG_0005_2.tif, ... """
    files = [(index, path) for name, index, path in _capture_files(images_folder, pattern) if name == capture_name]
    if not files:
        raise FileNotFoundError(f"No images of the capture '{capture_name}' matching the file name pattern "
                                f"{pattern or CAPTURE_FILE_PATTERN} in {images_folder}")
    files.sort(key=lambda item: (item[0] is None, item[0] or 0, item[1].name))
    return [str(path) for _, path in files]


def pattern_template(pattern):
    """
    The file name pattern written as a name: the groups become <capture> and <index>, the anchors and the escapes of the
    regular expression are removed. r'(?P<capture>.+)_(?P<index>\\d+)\\.tif$' -> '<capture>_<index>.tif'
    """
    text = re.sub(r'\(\?P<(\w+)>(?:[^()]|\([^()]*\))*\)', lambda match: f"<{match.group(1)}>", pattern or CAPTURE_FILE_PATTERN)
    text = text.lstrip('^').rstrip('$')
    return re.sub(r'\\(.)', r'\1', text)


def preview_file_pattern(images_folder, pattern=None, examples=3):
    """
    How a file name pattern reads the files of a folder
    Outputs: {'files' (all the files), 'matched', 'captures', 'files_per_capture' ({number of files: number of captures}),
             'examples' ([(file name, capture, index or None)], the first of each of the first captures), 'unmatched'
             (names of a few files that do not match)}. ValueError if the pattern is not valid
    """
    compiled = compile_file_pattern(pattern)
    names = sorted(path.name for path in Path(images_folder).iterdir() if path.is_file())
    found = {}
    unmatched = []
    for name in names:
        match = compiled.match(name)
        if match:
            index = match.groupdict().get('index')
            found.setdefault(match.group('capture'), []).append((name, int(index) if index and index.isdigit() else None))
        else:
            unmatched.append(name)
    per_capture = {}
    for files in found.values():
        per_capture[len(files)] = per_capture.get(len(files), 0) + 1
    shown = [(files[0][0], capture, files[0][1]) for capture, files in sorted(found.items())[:examples]]
    return {'files': len(names), 'matched': sum(len(files) for files in found.values()), 'captures': len(found),
            'files_per_capture': dict(sorted(per_capture.items())), 'examples': shown, 'unmatched': unmatched[:3]}


def suggest_file_pattern(file_names):
    """
    A file name pattern for the names of a folder, or None if there is no obvious one. The names are expected to be
    <capture name><separator><number of the file>.<extension> (IMG_0005_1.tif); the most common extension and separator
    are used and the pattern is kept only if most of the files follow it
    """
    file_names = list(file_names)
    stems = {}
    for name in file_names:
        stem, dot, extension = str(name).rpartition('.')
        if not dot or not stem:
            continue
        match = re.match(r'^(.+?)([_\-. ])(\d+)$', stem)
        if match:
            stems.setdefault(extension.lower(), []).append(match.group(2))
    if not stems:
        return None
    extension = max(stems, key=lambda key: len(stems[key]))
    separators = stems[extension]
    separator = max(set(separators), key=separators.count)
    if len(separators) < 0.5 * len(file_names):
        return None
    return rf"(?P<capture>.+){re.escape(separator)}(?P<index>\d+)\.{re.escape(extension)}$"


def _read_metadata(image_paths, exiftool_path=None):
    import exiftool
    with exiftool.ExifToolHelper(executable=exiftool_path) as et:
        return [ et.get_metadata(image_path, params=['-b', '-F'])[0] for image_path in image_paths ]


def read_band_files(image_paths, expected_bands=None, exiftool_path=None, ignore_bands=()):
    """ Read the metadata with ExifTool and map the files of one capture to their bands (see map_band_files) """
    return map_band_files(image_paths, _read_metadata(image_paths, exiftool_path), expected_bands, ignore_bands)


def read_band_definitions(image_paths, exiftool_path=None):
    """
    Read the bands of a capture from the metadata of its files: name (XMP:BandName), central wavelength and bandwidth
    (XMP:CentralWavelength, XMP:WavelengthFWHM, in nm) and the position of the camera in the rig (XMP:RigCameraIndex)
    Outputs: {'camera', 'serial', 'bands': [{'name', 'center_nm', 'bandwidth_nm', 'index', 'file', 'thermal'}]} with the
             bands in the order of the rig. 'thermal' is a guess (center above 2500 nm or a name like LWIR) to help to
             leave out bands that are not reflective. MetadataError if a field is missing
    """
    metadata_list = _read_metadata(image_paths, exiftool_path)
    bands = []
    for image_path, metadata in zip(image_paths, metadata_list):
        name = _metadata_value(metadata, 'XMP:BandName', image_path)
        center = _metadata_floats(metadata, 'XMP:CentralWavelength', image_path, count=1)[0]
        width = _metadata_floats(metadata, 'XMP:WavelengthFWHM', image_path, count=1)[0]
        rig = metadata.get('XMP:RigCameraIndex')
        bands.append({'name': name, 'center_nm': center, 'bandwidth_nm': width,
                      'index': int(rig) if isinstance(rig, (int, float)) else None, 'file': str(image_path),
                      'thermal': center > 2500 or name.strip().lower() in ('lwir', 'thermal')})
    names = [band['name'] for band in bands]
    duplicated = sorted({name for name in names if names.count(name) > 1})
    if duplicated:
        raise MetadataError(f"Band name(s) used by more than one file: {duplicated}")
    bands.sort(key=lambda band: (band['index'] is None, band['index'] if band['index'] is not None else 0, band['name']))
    first = metadata_list[0] if metadata_list else {}
    return {'camera': ' '.join(str(first.get(key, '')).strip() for key in ('EXIF:Make', 'EXIF:Model')).strip(),
            'serial': str(first.get('EXIF:SerialNumber', '')), 'bands': bands}


#
class Image(object):

    """
    Create an image object from the path for a micasense rededge image and corresponding metadata
    """

    def __init__(self, image_path, image_metadata):

        # Import image and store general information
        self.image_path = image_path
        self.metadata = image_metadata
        self.image = plt.imread( image_path )
        self.size = _metadata_positive(self.metadata, 'EXIF:ImageWidth', image_path), \
                    _metadata_positive(self.metadata, 'EXIF:ImageHeight', image_path)
        self.bits_per_pixel = _metadata_positive(self.metadata, 'EXIF:BitsPerSample', image_path)
        self.band_name = _metadata_value(self.metadata, 'XMP:BandName', image_path)

        # Parameters for camera matrix calculation
        self.principal_point = _metadata_floats(self.metadata, 'XMP:PrincipalPoint', image_path, count=2)
        self.focal_plane_resolution_px_per_mm = [ _metadata_positive(self.metadata, 'EXIF:FocalPlaneXResolution', image_path) ]*2
        self.focal_length = _metadata_positive(self.metadata, 'EXIF:FocalLength', image_path)
        self.distortion_parameters = np.array(
            _metadata_floats(self.metadata, 'XMP:PerspectiveDistortion', image_path, count=5) )[[0, 1, 3, 4, 2]]
        self.camera_matrix = cv2_camera_matrix(self.principal_point, self.focal_length , self.focal_plane_resolution_px_per_mm)

        # Pixel-wise polynomial 2D vignetting coefficients, according to functions provided by Micasense (vignette_map)
        self.k = _metadata_floats(self.metadata, 'XMP:VignettingPolynomial2D', image_path) # pol2DCoeff
        self.e = _metadata_floats(self.metadata, 'XMP:VignettingPolynomial2DName', image_path) # pol2DExponents
        if len(self.e) != 2 * len(self.k):
            raise MetadataError(f"{image_path}: 'XMP:VignettingPolynomial2DName' needs {2 * len(self.k)} exponents, "
                                f"found {len(self.e)}")

        # Parameters for radiometric calibration - radiometric sensitivity
        self.a1, self.a2, self.a3 = _metadata_floats(self.metadata, 'XMP:RadiometricCalibration', image_path, count=3)
        if self.a1 <= 0:
            raise MetadataError(f"{image_path}: first 'XMP:RadiometricCalibration' coefficient must be positive")

        # Dark current pixel values
        self.black_levels = np.array( _metadata_floats(self.metadata, 'EXIF:BlackLevel', image_path) )

        # Exposure time & gain (gain = ISO/100)
        self.exposure_time = float( _metadata_positive(self.metadata, 'EXIF:ExposureTime', image_path) )
        self.gain          = float( _metadata_positive(self.metadata, 'EXIF:ISOSpeed', image_path) ) / 100.0

        # Downwelling light sensor (DLS) - optional, only needed when the DLS correction is requested
        self.horizontal_irradiance = self.metadata.get('XMP:HorizontalIrradiance')
        self.irradiance_scale_to_si = self.metadata.get('XMP:IrradianceScaleToSIUnits')
        self.dls_ratio = None   # set by digital_numbers_to_reflectance() when the DLS correction is used
        self.dls_ratio_source = None


    # Undistort the image
    def undistorted(self, image=None):
        """ return the undistorted version of the image (default: the input image, or any array of the same size) """
        if image is None:
            image = self.image
        # get the optimal new camera matrix
        new_cam_mat, _ = cv2.getOptimalNewCameraMatrix( self.camera_matrix,
                                                        self.distortion_parameters,
                                                        self.size,
                                                        1 )
        map1, map2 = cv2.initUndistortRectifyMap( self.camera_matrix,
                                                  self.distortion_parameters,
                                                  np.eye(3),
                                                  new_cam_mat,
                                                  self.size,
                                                  cv2.CV_32F)  # cv2.CV_32F for 32 bit floats
        # Compute the undistorted image (16-bit for the raw image)
        image_undistorted = cv2.remap(image, map1, map2, cv2.INTER_LINEAR)
        return image_undistorted


    # Calculate region statistics
    def region_stats(self, region, sat_threshold=None, image=None):
        """
        Provide regional statistics for an image over a region
        Inputs: region is a skimage shape, image is any image ndarray (default: the input image)
        Outputs: mean, std, count, and saturated count tuple for the region
        """
        if image is None:
            image = self.image
        rev_panel_pts = np.fliplr( region )  # image coords are reversed
        w, h = image.shape[0:2]
        mask = measure.grid_points_in_poly((w, h), rev_panel_pts)
        num_pixels = mask.sum()
        panel_pixels = image[mask]
        stdev = panel_pixels.std()
        mean_value = panel_pixels.mean()
        saturated_count = 0
        if sat_threshold is not None:
            saturated_count = (panel_pixels > sat_threshold).sum()
        return mean_value, stdev, num_pixels, saturated_count


    # Convert DNs to radiance
    def digital_numbers_to_radiance(self):
        
        # Pixel-wise polynomial 2D vignetting coefficients, according to functions provided by Micasense (vignette_map)
        k = self.k
        e = self.e

        # Dark current pixel values
        dark_level = self.black_levels.mean()
        
        # Coordinate grid across image, seem swapped because of transposed vignette
        x, y = np.meshgrid( np.arange(self.size[1]), np.arange(self.size[0]) )

        # Vignette
        # meshgrid returns transposed arrays
        x = x.T
        y = y.T
        ##
        xv = x.T / self.size[1]
        yv = y.T / self.size[0]
        p2 = np.zeros_like(xv, dtype=float)
        for valT, c in enumerate(k):
            ex  = e[2 * valT]
            ey  = e[2 * valT + 1]
            p2 += c * xv ** ex * yv ** ey
        V = (1. / p2).T  #
        # plt.imshow( V )

        # apply image correction methods to raw image
        # step 1 - row gradient correction, vignette & radiometric calibration:
        
        # row gradient correction
        R = 1.0 / (1.0 + self.a2 * y / self.exposure_time - self.a3 * y)
        
        # subtract the dark level and adjust for vignette and row gradient
        # LT = VT  * RT * ( (imgT/normFactT) - (dark_level/normFactT) )
        L = V * R * ( self.image - dark_level )
        
        # Floor any negative radiance's to zero (can happen due to noise around black_level)
        L[L < 0] = 0
        # L = np.round(L).astype(np.uint16)
        # plt.imshow( L )
        
        # Apply the radiometric calibration - i.e. scale by the gain-exposure product and
        # multiply with the radiometric calibration coefficient
        # need to normalize by 2^16 for 16 bit images
        # because coefficients are scaled to work with input values of max 1.0
        dn_maximum     = float(2**self.bits_per_pixel)
        image_radiance = ( L.astype(float) / (self.gain * self.exposure_time) * self.a1 ) / dn_maximum
        # plt.imshow( image_radiance )

        #
        return image_radiance


    # Horizontal irradiance measured by the DLS for this band
    def dls_irradiance(self):
        """
        Return the DLS horizontal irradiance of this band (XMP:HorizontalIrradiance), multiplied by
        XMP:IrradianceScaleToSIUnits when it exists. The scale is only needed to express the value in SI units:
        it cancels when two images of the same camera are compared, so it is not guessed when it is absent
        """
        irradiance = _metadata_positive(self.metadata, 'XMP:HorizontalIrradiance', self.image_path)
        scale = 1.0
        if self.irradiance_scale_to_si is not None:
            scale = float( _metadata_positive(self.metadata, 'XMP:IrradianceScaleToSIUnits', self.image_path) )
        return float(irradiance) * scale


    # Angle between the sun direction seen by the DLS and the solar zenith
    def dls_sun_angle_discrepancy_deg(self):
        """
        The DLS estimates the direction of the direct light in its own frame (XMP:EstimatedDirectLightVector) and the
        solar elevation is written in radians (XMP:SolarElevation). With a level DLS the angle between the light vector
        and the DLS axis equals the solar zenith, so their difference is a lower bound of the DLS tilt towards or away
        from the sun. Returns degrees, or None if the fields are missing. The conventions were checked only against the
        example panel capture (level DLS: 0.3 deg); keep the limit in DLS_CHECKS generous
        """
        try:
            vector = np.array( _metadata_floats(self.metadata, 'XMP:EstimatedDirectLightVector', self.image_path, count=3) )
            elevation = _metadata_floats(self.metadata, 'XMP:SolarElevation', self.image_path, count=1)[0]
        except MetadataError:
            return None
        norm = np.linalg.norm(vector)
        if norm == 0:
            return None
        incidence = np.degrees(np.arccos(np.clip(-vector[2] / norm, -1, 1)))
        return float(abs(incidence - (90 - np.degrees(elevation))))


    # Convert radiance to reflectance, optionally correcting the illumination change with the DLS
    def digital_numbers_to_reflectance(self, radiance_to_reflectance_factor, panel_image=None,
                                       dls_ratio=None, dls_ratio_source=None):
        """
        Inputs: radiance to reflectance factor from detect_reference_panel() of the same band and, optionally,
                the panel image (rct.Image) of that band. When given, the DLS is used (see dls_irradiance_ratio)
                and the result is panel-anchored: reflectance = radiance * factor * (E_panel / E_image)
                Alternatively, give a ready DLS ratio (e.g. from capture_dls_ratios) and where it came from
        Outputs: reflectance image (float)
        """
        reflectance = self.digital_numbers_to_radiance() * radiance_to_reflectance_factor
        self.dls_ratio = None
        self.dls_ratio_source = None
        if dls_ratio is not None:
            self.dls_ratio = float(dls_ratio)
            self.dls_ratio_source = dls_ratio_source or 'given'
        elif panel_image is not None:
            self.dls_ratio = dls_irradiance_ratio(panel_image, self)
            self.dls_ratio_source = 'metadata'
        if self.dls_ratio is not None:
            reflectance = reflectance * self.dls_ratio
        return reflectance


    # Undistorted raw image, kept because the panel region is measured on it
    def undistorted_dn(self):
        if getattr(self, '_undistorted_dn', None) is None:
            self._undistorted_dn = self.undistorted()
        return self._undistorted_dn


    # Image to show when a panel region is drawn by hand: undistorted, stretched to 8 bits
    def display_image(self, low_percentile=1, high_percentile=99.5):
        image = self.undistorted_dn().astype(float)
        valid = image[image > 0]
        low, high = np.percentile(valid if valid.size else image, [low_percentile, high_percentile])
        return np.clip((image - low) / max(high - low, 1) * 255, 0, 255).astype(np.uint8)


    # Locate the reference panel from its QR code
    def locate_reference_panel(self, percent_inner_area, sat_threshold=65000):
        """
        Outputs: (panel bounds, QR code data, geometry where the QR code was found) or None if there is no QR code
                 The bounds are in the undistorted image when the geometry is 'undistorted', otherwise in the original
        """
        # Undistorted image
        image_undistorted = self.undistorted_dn()
        # plt.imshow( image_undistorted ) 

        # Convert from uint16 to uint8
        gray_undistorted = (image_undistorted // 256).astype(np.uint8)
        # plt.imshow( gray )

        # Detect and decode barcodes
        barcode = decode(gray_undistorted, symbols=[ZBarSymbol.QRCODE])
        if( len(barcode) > 0 ):
            barcode = barcode[0]
            barcode_source = 'undistorted'
        else: # if barcode is not detected in the undistroted image, use the QR code from the original image
            gray = (self.image // 256).astype(np.uint8)
            barcode = decode(gray, symbols=[ZBarSymbol.QRCODE])
            if( len(barcode) == 0 ):
                return None
            barcode = barcode[0]
            barcode_source = 'original'
        barcode_data = barcode.data.decode("utf-8")

        # Get bounds of the barcode
        barcode_bounds = []
        for point in barcode.polygon:
            barcode_bounds.append([point.x, point.y])
        barcode_bounds = np.asarray(barcode_bounds, np.int32)

        # Measures describing panel size and position
        if int(barcode_data[2:4]) < 3:
            # use the actual panel measures here - we use units of [mm]
            # the panel is 154.4 x 152.4 mm , vs. the 84 x 84 mm for the QR code
            # it is left 143.20 mm from the QR code 
            # use the inner 50% square of the panel
            s = 76.2
            p = 42
            T = np.array([-143.2, 0])
        elif (int(barcode_data[2:4]) >= 3) and (int(barcode_data[2:4]) < 6):
            s = 50
            p = 45
            T = np.array([-145.8, 0])
        elif int(barcode_data[2:4]) >= 6:
            # use the actual panel measures here - we use units of [mm]
            # the panel is 100 x 100 mm , vs. the 91 x 91 mm for the QR code
            # it is down 125.94 mm from the QR code 
            # use the inner 50% square of the panel
            p = 41
            s = 50
            T = np.array([0, -130.84])

        #
        reference_panel_pts = np.asarray([[-s, s], [s, s], [s, -s], [-s, -s]], dtype=np.float32) * percent_inner_area + T
        reference_qr_pts    = np.asarray([[-p, p], [p, p], [p, -p], [-p, -p]], dtype=np.float32)
        bounds = []
        costs  = []

        # The panel region must be applied to the same geometry where the QR code was found: the undistorted
        # image, or the original image when the QR code was only found there
        if barcode_source == 'undistorted':
            panel_dn = image_undistorted
        else:
            panel_dn = self.image
        for rotation in range(0, 4):
            qr_points = np.roll(reference_qr_pts, rotation, axis=0)

            src = np.asarray([tuple(row) for row in qr_points[:]], np.float32)
            dst = np.asarray([tuple(row) for row in barcode_bounds[:]], np.float32)  

            # we determine the homography from the 4 corner points
            warp_matrix = cv2.getPerspectiveTransform(src, dst)

            pts = np.asarray([reference_panel_pts], 'float32')
            panel_bounds = cv2.convexHull(cv2.perspectiveTransform(pts, warp_matrix), clockwise=False)
            panel_bounds = np.squeeze(panel_bounds)  # remove nested lists

            # Keep the rotation with the most homogeneous region (lowest coefficient of variation)
            mean, std, _, _ = self.region_stats( region=panel_bounds,
                                                 sat_threshold=sat_threshold,
                                                 image=panel_dn )
            bounds.append( panel_bounds.astype(np.int32) )
            costs.append(std / mean)

        idx = costs.index(min(costs))
        panel_bounds = bounds[idx]

        # # plot image to check
        # fig, ax = plt.subplots()
        # ax.imshow( image_undistorted )
        # polygon = patches.Polygon(panel_bounds, closed=True, edgecolor='r', facecolor='none')
        # ax.add_patch(polygon)

        return panel_bounds, barcode_data, barcode_source


    # Statistics of a region drawn by hand, to give feedback while it is drawn
    def panel_region_stats(self, panel_bounds, sat_threshold=65000):
        """
        Inputs: polygon [[x, y], ...] in the undistorted image
        Outputs: dictionary with the number of pixels, the saturated pixels and the variation (std / mean, in %) of the
                 raw values inside; None if the polygon has no pixels
        """
        panel_bounds = np.asarray(panel_bounds, dtype=np.int32)
        mean, std, count, saturated = self.region_stats(panel_bounds, sat_threshold, image=self.undistorted_dn())
        if count == 0 or mean == 0:
            return None
        return {'n_pixels': int(count), 'n_saturated': int(saturated), 'variation_percent': float(100 * std / mean)}


    # Calibrate the panel from a region: radiance to reflectance factor
    def calibrate_panel_region(self, panel_bounds, geometry, reflectance_level_reference_panel, sat_threshold=65000,
                               audit_extra=None, validate=True):
        """
        Inputs: polygon [[x, y], ...], the geometry of the image where it is drawn ('undistorted' or 'original'), the
                reflectance of the panel by band and the saturation threshold
                validate: check the polygon (see validate_polygon); the regions found from the QR code are not checked
                because a panel close to the border of the image may be partly outside it
        Outputs: the radiance to reflectance factor (also in self.panel_audit, with audit_extra)
        """
        panel_bounds = np.asarray(panel_bounds, dtype=np.int32)
        if geometry == 'undistorted':
            panel_dn = self.undistorted_dn()
        elif geometry == 'original':
            panel_dn = self.image
        else:
            raise ValueError(f"unknown geometry {geometry!r}")
        if validate:
            validate_polygon(panel_bounds, panel_dn.shape[1::-1], self.image_path)

        ##
        # Calculate radiance for reference panel image
        image_panel_radiance = self.digital_numbers_to_radiance()
        if geometry == 'undistorted':
            image_panel_radiance = self.undistorted(image_panel_radiance)
        # plt.imshow( image_panel_radiance  )

        # Create an empty mask for panel region
        panel_mask = np.zeros_like(panel_dn, dtype=np.uint8)

        # Fill the polygon in the mask
        cv2.fillPoly(panel_mask, [panel_bounds], 255)
        # plt.imshow( panel_mask  )

        # Extract pixel values inside the polygon, leaving out saturated pixels
        inside_panel = panel_mask == 255
        saturated = inside_panel & (panel_dn > sat_threshold)
        panel_radiance = image_panel_radiance[ inside_panel & ~saturated ]
        if panel_radiance.size == 0:
            raise ValueError(f"{self.image_path}: every pixel in the reference panel region is saturated")
        masked_values_average = panel_radiance.mean()
        # print('Mean Radiance in panel region: {:1.4f} W/m^2/nm/sr'.format( masked_values_average ))

        # Specific calibrated reflectance level for the band - provided by Micasense
        if self.band_name not in reflectance_level_reference_panel:
            raise KeyError(f"{self.image_path}: band '{self.band_name}' not in the reference panel reflectance levels "
                           f"{list(reflectance_level_reference_panel)}")
        panel_reflectance_level = reflectance_level_reference_panel[ self.band_name ]

        # Calculate radiance to reflectance conversion factor
        radiance_to_reflectance_factor = panel_reflectance_level / masked_values_average

        # Keep what was used, so the calibration can be audited (self.panel_audit)
        self.panel_audit = { 'image_path': str(self.image_path),
                             'band_name': self.band_name,
                             'geometry': geometry,
                             'panel_bounds': panel_bounds,
                             'sat_threshold': sat_threshold,
                             'n_pixels': int(inside_panel.sum()),
                             'n_saturated': int(saturated.sum()),
                             'radiance_mean': float(masked_values_average),
                             'radiance_std': float(panel_radiance.std()),
                             'reflectance_level': float(panel_reflectance_level),
                             # irradiance that the panel receives, from its radiance: E = pi * L / reflectance
                             'panel_irradiance_estimate': float(np.pi * masked_values_average / panel_reflectance_level),
                             'radiance_to_reflectance_factor': float(radiance_to_reflectance_factor) }
        self.panel_audit.update(audit_extra or {})
        return radiance_to_reflectance_factor


    # Detect the reference panel (or use a region drawn by hand) and return radiance to reflectance conversion factor
    def detect_reference_panel(self, reflectance_level_reference_panel, percent_inner_area, sat_threshold=65000,
                               manual_bounds=None, region_origin=None):
        """
        Inputs: reflectance of the panel by band, the inner fraction of the panel to measure, the saturation threshold
                and, optionally, a polygon [[x, y], ...] accepted or drawn by hand in the undistorted image (then no QR
                code is needed and percent_inner_area is not used) with where it came from (region_origin: 'edited' or
                'accepted_automatic', recorded in the audit)
        Outputs: panel bounds, radiance to reflectance factor, QR code data (None if manual) and where the region was
                 found ('undistorted', 'original' or 'manual')
        """
        if manual_bounds is not None:
            panel_bounds = np.asarray(manual_bounds, dtype=np.int32)
            barcode_data, barcode_source, geometry = None, 'manual', 'undistorted'
            region_source = 'manual'
        else:
            located = self.locate_reference_panel(percent_inner_area, sat_threshold)
            if located is None:
                raise ValueError(f"{self.image_path}: no QR code found for the reference panel "
                                 f"(band {self.band_name}), in either the undistorted or the original image")
            panel_bounds, barcode_data, barcode_source = located
            geometry, region_source = barcode_source, 'automatic'
        factor = self.calibrate_panel_region(panel_bounds, geometry, reflectance_level_reference_panel, sat_threshold,
                                             audit_extra={'region_source': region_source,
                                                          'barcode_data': barcode_data,
                                                          'barcode_source': barcode_source,
                                                          'percent_inner_area': percent_inner_area,
                                                          'region_origin': region_origin},
                                             validate=manual_bounds is not None)
        return panel_bounds, factor, barcode_data, barcode_source

