"""
Radiometric correction workflow for Micasense RedEdge-P captures - configuration, validation and runner

This module has no user interface. Everything the workflow needs is in a flat configuration dictionary
(FIELDS describes each key, so that a window can be built from it) and the result is a dictionary that is also saved
as json. It can be used from Python or from the command line:

    python radiometric_workflow.py --write-default-config radiometric_inputs.json
    python radiometric_workflow.py --config radiometric_inputs.json
    python radiometric_workflow.py --request-json request.json

The configuration keys and the stage identifiers in STAGES are stable: labels and descriptions may change, the keys
and identifiers may not, because saved inputs and results refer to them.
"""

# Standard Library Imports
import argparse
import json
import os
import sys
import traceback
import warnings
from pathlib import Path

# Third-Party Imports
import tifffile

# Local Imports
import panel_spectrum
import radiometric_correction_tools as rct

REQUEST_VERSION = 1
REGIONS_VERSION = 1
RESULT_FILE_NAME = 'radiometric_result.json'
STAGES = ('LOAD_CAPTURES', 'PANEL_CALIBRATION', 'REFLECTANCE', 'DLS_CHECKS', 'PROVENANCE')

BAND_NAMES = ('Blue', 'Green', 'Panchro', 'Red', 'Red edge', 'NIR')

# Description of every configuration key: default value, kind of value (for validation and for the editor), label
# and description. Kinds: folder, file, text, bool, float, int, choice (one of 'choices'), band_values (dictionary
# {band name: number}, with an optional 'range' (low exclusive, high inclusive)). Optional 'group' is a heading for the
# window and 'dialog' marks the keys that the window edits in a dialog instead of on the page
# What the groups of fields of the window are about (the window shows them under the heading of the group)
GROUP_NOTES = {
    'Correction with the DLS':
        "The DLS (downwelling light sensor) measures, for each band, the irradiance on a horizontal surface: the light that "
        "arrives from the sky. The DLS ratio of a band is E_panel / E_image: the irradiance that the DLS measured when the "
        "reference panel was photographed divided by the irradiance that it measured when the image was taken. The panel "
        "calibrates the camera for the light of that moment, so an image taken in more light than the panel (ratio below 1) "
        "would look too bright; multiplying by the ratio corrects it. Example: a ratio of 0.885 means that the DLS measured "
        "13 % more light at the image than at the panel, so the reflectance is lowered by 11.5 %. The ratio is 1 at the panel "
        "capture. The irradiance is read from XMP:HorizontalIrradiance of each file; the scale to SI units, when it is "
        "written, cancels in the ratio.",
    'Checks of the DLS data (limits)':
        "These checks look for DLS data that should not be trusted. They only raise flags: they never change the result. "
        "All the bands must have nearly the same ratio, because a change of the light changes all of them together. The "
        "plausible range catches ratios that are not credible (a covered DLS, a DLS pointing at the ground). The sun angle "
        "check uses two things written in each file: the direction of the direct sunlight in the frame of the DLS "
        "(XMP:EstimatedDirectLightVector) and the solar elevation (XMP:SolarElevation, in radians; it is taken as written, "
        "not recomputed from position and time). The solar zenith is 90 degrees minus the elevation. If the DLS were level, "
        "the angle between the sunlight and the axis of the DLS (the arccosine of minus the z component of the unit "
        "vector) would be equal to the zenith, so the difference between the two is at least the tilt of the DLS, or a sign "
        "that the two are inconsistent. The jump and panel limits apply to several captures along a flight and to two panel "
        "captures.",
}

FIELDS = [
    {'key': 'INPUTS.images_folder', 'group': 'Folders and tools', 'kind': 'folder', 'default': 'images/raw',
     'label': 'Images folder', 'help': 'Folder with the raw .tif images (one file per band, e.g. IMG_0010_1.tif).'},
    {'key': 'INPUTS.results_folder', 'group': 'Folders and tools', 'kind': 'folder', 'default': 'results/reflectance',
     'label': 'Results folder', 'help': 'Folder for the reflectance images, the provenance records and the result file.'},
    {'key': 'INPUTS.exiftool_path', 'group': 'Folders and tools', 'kind': 'file', 'default': '',
     'label': 'ExifTool executable', 'help': 'Empty: use the environment variable exiftoolpath or the system PATH.'},
    {'key': 'INPUTS.reference_panel_capture', 'group': 'Reference panel', 'kind': 'text', 'default': 'IMG_0005',
     'label': 'Reference panel capture', 'help': 'Name of the capture that shows the reference panel, e.g. IMG_0005.'},
    {'key': 'INPUTS.reference_panel_capture_after', 'group': 'Reference panel', 'kind': 'text', 'default': '',
     'label': 'Second panel capture (optional)',
     'help': 'Panel captured at the end of the flight. It is only compared with the first one; the first is used '
             'for the correction.'},
    {'key': 'PANEL.manual_regions_file', 'kind': 'file', 'default': '', 'group': 'Reference panel regions',
     'label': 'Panel regions drawn by hand (optional)',
     'help': "json file written by the Draw panel regions tool. A band with a region there uses it instead of the "
             "region found from the QR code; the other bands are detected automatically."},
    {'key': 'INPUTS.image_captures', 'group': 'Captures to correct', 'kind': 'text', 'default': '',
     'label': 'Captures to correct', 'help': 'Names separated by commas. Empty: all the other captures in the folder.'},
    {'key': 'INPUTS.file_pattern', 'group': 'Captures to correct', 'kind': 'text', 'default': rct.CAPTURE_FILE_PATTERN,
     'label': 'File name pattern',
     'help': "Regular expression for the names of the image files, with a group named capture (the name of the "
             "capture, shared by the files of its bands) and, optionally, a group named index (the number of the "
             "file). The band of each file is read from its metadata, not from its name."},
    {'key': 'INPUTS.ignored_bands', 'kind': 'text', 'default': '',
     'label': 'Bands to ignore',
     'help': "Names (as written by the camera, XMP:BandName) separated by commas, of bands that are in the files but "
             "must not be processed, for example a thermal band."},
    {'key': 'PANEL.reflectance_source', 'kind': 'choice', 'choices': ['values', 'spectral_file'],
     'default': 'values', 'group': 'Reflectance of the panel',
     'label': 'Source of the panel reflectance',
     'help': "values: the reflectance of each band typed below (printed on the panel). spectral_file: computed from "
             "the spectral response file (.csv) of the panel, below."},
    {'key': 'PANEL.reflectance_by_band', 'kind': 'band_values', 'group': 'Reflectance of the panel', 'range': (0, 1),
     'default': {'Blue': 0.477, 'Green': 0.478, 'Panchro': 0.479, 'Red': 0.478, 'Red edge': 0.478, 'NIR': 0.477},
     'label': 'Panel reflectance by band', 'help': 'Values provided by Micasense with the reference panel. The band '
                                                    'names must be those written by the camera (XMP:BandName).'},
    {'key': 'PANEL.spectral_file', 'kind': 'file', 'default': '', 'group': 'Spectral response file (.csv)',
     'label': 'Spectral response file',
     'help': 'Reflectance of the panel by wavelength (.csv). Use Import / preview to choose how to read it.'},
    {'key': 'PANEL.spectral_delimiter', 'kind': 'choice', 'choices': list(panel_spectrum.DELIMITERS), 'dialog': True,
     'default': 'auto', 'group': 'Spectral response file (.csv)', 'label': 'Delimiter',
     'help': 'Character that separates the columns; auto detects it.'},
    {'key': 'PANEL.spectral_delimiter_char', 'kind': 'text', 'default': '', 'dialog': True,
     'group': 'Spectral response file (.csv)', 'label': 'Custom delimiter',
     'help': "The separator when the delimiter is custom (one or more characters)."},
    {'key': 'PANEL.spectral_skip_rows', 'kind': 'text', 'default': 'auto', 'dialog': True,
     'group': 'Spectral response file (.csv)', 'label': 'Header rows to skip',
     'help': "auto: the rows at the start that are not numbers; or the number of rows."},
    {'key': 'PANEL.spectral_skip_footer', 'kind': 'int', 'default': 0, 'dialog': True,
     'group': 'Spectral response file (.csv)', 'label': 'Footer rows to skip',
     'help': "Number of rows at the end of the file that are not data."},
    {'key': 'PANEL.spectral_comment_prefix', 'kind': 'text', 'default': '', 'dialog': True,
     'group': 'Spectral response file (.csv)', 'label': 'Comment prefix',
     'help': "Lines that start with this text are ignored, for example #."},
    {'key': 'PANEL.spectral_exclude_rows', 'kind': 'text', 'default': '', 'dialog': True,
     'group': 'Spectral response file (.csv)', 'label': 'Lines to exclude',
     'help': "Line numbers of the file that are not used, for example 5, 10-12."},
    {'key': 'PANEL.spectral_wavelength_min', 'kind': 'text', 'default': '', 'dialog': True,
     'group': 'Spectral response file (.csv)', 'label': 'Smallest wavelength used (nm)',
     'help': "Empty: no limit."},
    {'key': 'PANEL.spectral_wavelength_max', 'kind': 'text', 'default': '', 'dialog': True,
     'group': 'Spectral response file (.csv)', 'label': 'Largest wavelength used (nm)',
     'help': "Empty: no limit."},
    {'key': 'PANEL.spectral_reflectance_scale', 'kind': 'float', 'default': 1.0, 'dialog': True,
     'group': 'Spectral response file (.csv)', 'label': 'Reflectance multiplier',
     'help': "Applied after the reflectance unit, for example 0.001 for per mille."},
    {'key': 'PANEL.spectral_wavelength_column', 'kind': 'int', 'default': 1, 'dialog': True,
     'group': 'Spectral response file (.csv)', 'label': 'Wavelength column', 'help': 'Column number, from 1.'},
    {'key': 'PANEL.spectral_reflectance_column', 'kind': 'int', 'default': 2, 'dialog': True,
     'group': 'Spectral response file (.csv)', 'label': 'Reflectance column', 'help': 'Column number, from 1.'},
    {'key': 'PANEL.spectral_wavelength_unit', 'kind': 'choice', 'choices': list(panel_spectrum.WAVELENGTH_UNITS),
     'default': 'nm', 'dialog': True, 'group': 'Spectral response file (.csv)', 'label': 'Wavelength unit',
     'help': 'nm or um (micrometers).'},
    {'key': 'PANEL.spectral_reflectance_unit', 'kind': 'choice', 'choices': list(panel_spectrum.REFLECTANCE_UNITS),
     'default': 'fraction', 'dialog': True, 'group': 'Spectral response file (.csv)', 'label': 'Reflectance unit',
     'help': 'fraction (0 to 1) or percent.'},
    {'key': 'PANEL.spectral_decimal', 'kind': 'choice', 'choices': list(panel_spectrum.DECIMAL_MARKS), 'default': '.',
     'dialog': True, 'group': 'Spectral response file (.csv)', 'label': 'Decimal mark', 'help': 'Point or comma.'},
    {'key': 'PANEL.band_center_wavelengths', 'kind': 'band_values', 'range': (0, 5000), 'dialog': True,
     'default': {band: window[0] for band, window in panel_spectrum.REDEDGE_P_BANDS.items()},
     'group': 'Spectral response file (.csv)', 'label': 'Center wavelength of each band (nm)',
     'help': 'Spectral window of each band: the reflectance of a band is the mean of the curve over '
             'center +/- bandwidth / 2.'},
    {'key': 'PANEL.band_bandwidths', 'kind': 'band_values', 'range': (0, 5000), 'dialog': True,
     'default': {band: window[1] for band, window in panel_spectrum.REDEDGE_P_BANDS.items()},
     'group': 'Spectral response file (.csv)', 'label': 'Bandwidth of each band (nm)',
     'help': 'Width of the spectral window of each band.'},
    {'key': 'PANEL.spectral_min_coverage', 'kind': 'float', 'default': 0.5, 'dialog': True,
     'group': 'Spectral response file (.csv)', 'label': 'Smallest covered part of a band window',
     'help': 'Fraction (0 to 1] of the window of a band that must have valid samples in the file.'},
    {'key': 'PANEL.percent_inner_area', 'kind': 'float', 'default': 0.7, 'group': 'Panel measurement',
     'label': 'Inner area of the panel', 'help': 'Fraction (0 to 1] of the panel used to measure its radiance.'},
    {'key': 'PANEL.panels_used', 'kind': 'choice', 'choices': list(rct.PANELS_USED), 'default': 'first',
     'group': 'Panel measurement', 'label': 'Panels used when there are two',
     'help': "With a second panel capture: first = only the first panel calibrates (the second is only compared with it); "
             "mean = the average of the two; interpolate_time = weighted by the time of each image between the two panel "
             "captures; nearest_time = the panel captured closest in time to each image (times from "
             "EXIF:DateTimeOriginal). With the DLS the combination is of factor x irradiance, which does not depend on "
             "the light, so no panel is the reference of the DLS: only the panels that are used matter."},
    {'key': 'PANEL.saturation_threshold', 'kind': 'int', 'default': 65000, 'group': 'Panel measurement',
     'label': 'Saturation threshold', 'help': 'Pixels above this value are left out of the panel radiance.'},
    {'key': 'CORRECTION.use_dls', 'group': 'Correction with the DLS', 'kind': 'bool', 'default': False,
     'label': 'Correct illumination changes with the DLS',
     'help': 'Multiply the reflectance of each band (Panchro included) by the DLS ratio E_panel / E_image. Off: panel '
             'only.'},
    {'key': 'CHECKS.band_ratio_tolerance', 'group': 'Checks of the DLS data (limits)', 'kind': 'float', 'default': rct.DLS_CHECKS['band_ratio_tolerance'],
     'label': 'Band ratio tolerance', 'help': 'Largest relative departure of the DLS ratio of a band from the median ratio of the bands (0.03 = 3 %).'},
    {'key': 'CHECKS.ratio_range_min', 'group': 'Checks of the DLS data (limits)', 'kind': 'float', 'default': rct.DLS_CHECKS['ratio_range'][0],
     'label': 'Smallest plausible DLS ratio', 'help': 'A DLS ratio below this value is flagged: the image would have more than 1 / value times the light of '
             'the panel (0.5 = twice as much).'},
    {'key': 'CHECKS.ratio_range_max', 'group': 'Checks of the DLS data (limits)', 'kind': 'float', 'default': rct.DLS_CHECKS['ratio_range'][1],
     'label': 'Largest plausible DLS ratio', 'help': 'A DLS ratio above this value is flagged: the image would have less than 1 / value times the light of '
             'the panel (2 = half as much).'},
    {'key': 'CHECKS.max_sun_angle_discrepancy_deg', 'group': 'Checks of the DLS data (limits)', 'kind': 'float',
     'default': rct.DLS_CHECKS['max_sun_angle_discrepancy_deg'],
     'label': 'Largest sun angle discrepancy (deg)',
     'help': 'Degrees. Difference between the angle of the sunlight seen by the DLS and the solar zenith (90 degrees '
             'minus XMP:SolarElevation): the DLS was tilted at least that much, or the angles are inconsistent.'},
    {'key': 'CHECKS.jump_threshold', 'group': 'Checks of the DLS data (limits)', 'kind': 'float', 'default': rct.DLS_CHECKS['jump_threshold'],
     'label': 'Irradiance jump threshold', 'help': 'Relative change (0.1 = 10 %) of the irradiance of a band between two consecutive captures that is '
             'reported: a spike if it comes back (one capture, for example a shaded or tilted DLS), a step if it stays '
             '(for example a cloud).'},
    {'key': 'CHECKS.panel_drift_tolerance', 'group': 'Checks of the DLS data (limits)', 'kind': 'float', 'default': rct.DLS_CHECKS['panel_drift_tolerance'],
     'label': 'Panel drift tolerance', 'help': 'Largest relative change (0.05 = 5 %) between the first and the second panel captures of the panel factor '
             'of a band, and of the DLS irradiance relative to the irradiance estimated from the panel.'},
]

FIELD_KEYS = [field['key'] for field in FIELDS]


class ConfigError(ValueError):
    """ Raised when the configuration is not valid """


def default_config():
    """ A copy of the default value of every key """
    return json.loads(json.dumps({field['key']: field['default'] for field in FIELDS}))


def load_config(path):
    """ Default configuration with the values of a json file on top; unknown keys are an error """
    saved = json.loads(Path(path).read_text(encoding='utf-8'))
    return merge_config(saved)


def merge_config(values):
    unknown = sorted(set(values) - set(FIELD_KEYS))
    if unknown:
        raise ConfigError(f"Unknown configuration keys: {unknown}")
    config = default_config()
    config.update(values)
    return config


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == value and abs(value) != float('inf')


def resolve_path(root, value):
    path = Path(value)
    return path if path.is_absolute() else Path(root) / path


def exiftool_executable(config, root='.'):
    """ ExifTool to use: the configured file, else the environment variable exiftoolpath, else None (system PATH) """
    if config['INPUTS.exiftool_path']:
        return str(resolve_path(root, config['INPUTS.exiftool_path']))
    return os.environ.get('exiftoolpath')


def read_capture_images(config, root, capture, exiftool_path=None):
    """ {band: rct.Image} of one capture, with the file pattern, the bands and the ignored bands of the configuration """
    images_folder = resolve_path(root, config['INPUTS.images_folder'])
    if exiftool_path is None:
        exiftool_path = exiftool_executable(config, root)
    return _read_capture(images_folder, capture, panel_bands(config), exiftool_path,
                         config['INPUTS.file_pattern'], capture_names(config['INPUTS.ignored_bands']))


def detect_bands(config, root, capture=None):
    """
    Read the bands from the metadata of the files of a capture (default: the panel capture)
    Outputs: the dictionary of rct.read_band_definitions
    """
    images_folder = resolve_path(root, config['INPUTS.images_folder'])
    capture = capture or config['INPUTS.reference_panel_capture'].strip()
    paths = rct.find_capture_paths(images_folder, capture, config['INPUTS.file_pattern'])
    return rct.read_band_definitions(paths, exiftool_executable(config, root))


def capture_names(text):
    """ Names separated by commas -> list """
    return [name.strip() for name in str(text).split(',') if name.strip()]


def spectral_format(config):
    """ Format of the spectral response file from the configuration; ConfigError if it is not valid """
    skip = config['PANEL.spectral_skip_rows'].strip().lower()
    try:
        skip = 'auto' if skip == 'auto' else int(skip)
    except ValueError:
        raise ConfigError("PANEL.spectral_skip_rows must be 'auto' or a whole number")
    limits = []
    for key in ('PANEL.spectral_wavelength_min', 'PANEL.spectral_wavelength_max'):
        text = config[key].strip()
        try:
            limits.append(float(text) if text else None)
        except ValueError:
            raise ConfigError(f"{key} must be a number or empty")
    fmt = {'delimiter': config['PANEL.spectral_delimiter'], 'delimiter_char': config['PANEL.spectral_delimiter_char'],
           'skip_rows': skip, 'skip_footer': int(config['PANEL.spectral_skip_footer']),
           'comment_prefix': config['PANEL.spectral_comment_prefix'], 'exclude_rows': config['PANEL.spectral_exclude_rows'],
           'wavelength_min': limits[0], 'wavelength_max': limits[1],
           'reflectance_scale': float(config['PANEL.spectral_reflectance_scale']),
           'wavelength_column': int(config['PANEL.spectral_wavelength_column']),
           'reflectance_column': int(config['PANEL.spectral_reflectance_column']),
           'wavelength_unit': config['PANEL.spectral_wavelength_unit'],
           'reflectance_unit': config['PANEL.spectral_reflectance_unit'], 'decimal': config['PANEL.spectral_decimal']}
    problems = panel_spectrum.validate_format(fmt)
    if problems:
        raise ConfigError('Spectral response format: ' + '; '.join(problems))
    return fmt


def panel_bands(config):
    """ Names of the bands that are processed: those of the typed reflectances, or those that have a spectral window """
    if config['PANEL.reflectance_source'] == 'spectral_file':
        return list(config['PANEL.band_center_wavelengths'])
    return list(config['PANEL.reflectance_by_band'])


def panel_reflectance_levels(config, root='.'):
    """
    Reflectance of the panel by band
    Outputs: ({band: reflectance}, information). The values are the typed ones, or the mean of the spectral response
             file over the window of each band (the information then has the file, its hash, the format and, for each
             band, the window, the samples and the covered part of the window)
    """
    if config['PANEL.reflectance_source'] == 'values':
        return dict(config['PANEL.reflectance_by_band']), {'source': 'values'}
    fmt = spectral_format(config)
    path = resolve_path(root, config['PANEL.spectral_file'])
    spectrum = panel_spectrum.load_spectrum(path, fmt)
    windows = {band: (center, config['PANEL.band_bandwidths'][band])
               for band, center in config['PANEL.band_center_wavelengths'].items()}
    stats = panel_spectrum.band_reflectances(spectrum, windows, config['PANEL.spectral_min_coverage'])
    info = {'source': 'spectral_file', 'file': spectrum['file'], 'sha256': spectrum['sha256'], 'format': fmt,
            'min_coverage': config['PANEL.spectral_min_coverage'], 'warnings': spectrum['warnings'], 'bands': stats}
    return {band: values['reflectance'] for band, values in stats.items()}, info


def load_panel_regions(path):
    """
    Read the regions of the reference panel drawn by hand
    Outputs: {capture name: {band name: [[x, y], ...]}}, the vertices being pixels of the undistorted image
    """
    saved = json.loads(Path(path).read_text(encoding='utf-8'))
    if saved.get('version') != REGIONS_VERSION or saved.get('geometry') != 'undistorted':
        raise ConfigError(f"{path}: not a panel regions file (version {REGIONS_VERSION}, undistorted geometry)")
    regions = saved.get('captures')
    if not isinstance(regions, dict) or not all(isinstance(bands, dict) for bands in regions.values()):
        raise ConfigError(f"{path}: 'captures' must be {{capture: {{band: [[x, y], ...]}}}}")
    return regions


def load_panel_origins(path):
    """
    Where each region of the file came from: {capture name: {band name: 'accepted_automatic' or 'edited'}}.
    'accepted_automatic' is a region found from the QR code that the user accepted as it was; files that do not say it
    give an empty dictionary
    """
    saved = json.loads(Path(path).read_text(encoding='utf-8'))
    origins = saved.get('origins', {})
    return origins if isinstance(origins, dict) else {}


def save_panel_regions(path, capture, regions, origins=None):
    """
    Save the regions of one capture ({band: [[x, y], ...]}, None or missing bands are not saved) and, optionally, where
    each came from ({band: 'accepted_automatic' or 'edited'}), keeping the other captures of the file. The file is
    replaced atomically
    """
    path = Path(path)
    captures = load_panel_regions(path) if path.is_file() else {}
    all_origins = load_panel_origins(path) if path.is_file() else {}
    kept = {band: [[int(round(x)), int(round(y))] for x, y in polygon]
            for band, polygon in regions.items() if polygon is not None}
    if kept:
        captures[capture] = kept
        known = {band: origin for band, origin in (origins or {}).items() if band in kept and origin}
        if known:
            all_origins[capture] = known
        else:
            all_origins.pop(capture, None)
    else:
        captures.pop(capture, None)
        all_origins.pop(capture, None)
    content = {'version': REGIONS_VERSION, 'geometry': 'undistorted', 'captures': captures}
    if all_origins:
        content['origins'] = all_origins
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(content, indent=2), encoding='utf-8')
    os.replace(temporary, path)
    return path


def validate_config(config, root='.'):
    """ Return the list of problems of a configuration (empty if it is valid). Paths are resolved from root """
    problems = []
    for key in sorted(set(config) - set(FIELD_KEYS)):
        problems.append(f"Unknown configuration key {key}")
    if problems:
        return problems
    for field in FIELDS:
        key, kind, value = field['key'], field['kind'], config.get(field['key'], field['default'])
        if kind in ('folder', 'file', 'text') and not isinstance(value, str):
            problems.append(f"{key} must be text")
        elif kind == 'bool' and not isinstance(value, bool):
            problems.append(f"{key} must be true or false")
        elif kind == 'float' and not _is_number(value):
            problems.append(f"{key} must be a number")
        elif kind == 'int' and (not _is_number(value) or int(value) != value):
            problems.append(f"{key} must be an integer")
        elif kind == 'choice' and value not in field['choices']:
            problems.append(f"{key} must be one of {field['choices']}, got {value!r}")
        elif kind == 'band_values':
            low, high = field.get('range', (0, 1))
            if not isinstance(value, dict) or not value:
                problems.append(f"{key} must be a dictionary {{band name: number}}")
            else:
                for band, number in value.items():
                    if not _is_number(number) or not low < number <= high:
                        problems.append(f"{key}['{band}'] must be a number in ({low}, {high}], got {number!r}")
    if problems:
        return problems

    if not 0 < config['PANEL.percent_inner_area'] <= 1:
        problems.append("PANEL.percent_inner_area must be in (0, 1]")
    if config['PANEL.panels_used'] != 'first' and not config['INPUTS.reference_panel_capture_after'].strip():
        problems.append(f"PANEL.panels_used is '{config['PANEL.panels_used']}' but there is no second panel capture "
                        "(INPUTS.reference_panel_capture_after)")
    if not 0 < config['PANEL.spectral_min_coverage'] <= 1:
        problems.append("PANEL.spectral_min_coverage must be in (0, 1]")
    band_sets = {frozenset(config[key]) for key in ('PANEL.reflectance_by_band', 'PANEL.band_center_wavelengths',
                                                      'PANEL.band_bandwidths')}
    ignored = capture_names(config['INPUTS.ignored_bands'])
    try:
        rct.compile_file_pattern(config['INPUTS.file_pattern'])
        pattern_problem = None
    except ValueError as error:
        pattern_problem = f"INPUTS.file_pattern: {error}"
        problems.append(pattern_problem)
    if len(band_sets) > 1:
        problems.append("PANEL.reflectance_by_band, PANEL.band_center_wavelengths and PANEL.band_bandwidths must "
                        "have the same bands")
    elif set(ignored) & set(config['PANEL.reflectance_by_band']):
        problems.append(f"INPUTS.ignored_bands has bands that are also processed: "
                        f"{sorted(set(ignored) & set(config['PANEL.reflectance_by_band']))}")
    elif config['PANEL.reflectance_source'] == 'spectral_file':
        spectral_file = config['PANEL.spectral_file'].strip()
        if not spectral_file:
            problems.append("PANEL.spectral_file is empty (PANEL.reflectance_source is spectral_file)")
        elif not resolve_path(root, spectral_file).is_file():
            problems.append(f"Spectral response file not found: {resolve_path(root, spectral_file)}")
        else:
            try:
                panel_reflectance_levels(config, root)
            except (ValueError, OSError) as error:
                problems.append(str(error))
    if config['PANEL.saturation_threshold'] <= 0:
        problems.append("PANEL.saturation_threshold must be positive")
    for key in ('CHECKS.band_ratio_tolerance', 'CHECKS.max_sun_angle_discrepancy_deg', 'CHECKS.jump_threshold',
                'CHECKS.panel_drift_tolerance', 'CHECKS.ratio_range_min', 'CHECKS.ratio_range_max'):
        if config[key] <= 0:
            problems.append(f"{key} must be positive")
    if config['CHECKS.ratio_range_min'] >= config['CHECKS.ratio_range_max']:
        problems.append("CHECKS.ratio_range_min must be smaller than CHECKS.ratio_range_max")

    images = resolve_path(root, config['INPUTS.images_folder'])
    if not images.is_dir():
        problems.append(f"Images folder not found: {images}")
    if not config['INPUTS.results_folder'].strip():
        problems.append("INPUTS.results_folder is empty")
    if config['INPUTS.exiftool_path'] and not resolve_path(root, config['INPUTS.exiftool_path']).is_file():
        problems.append(f"ExifTool not found: {resolve_path(root, config['INPUTS.exiftool_path'])}")
    if config['PANEL.manual_regions_file']:
        regions_file = resolve_path(root, config['PANEL.manual_regions_file'])
        if not regions_file.is_file():
            problems.append(f"Panel regions file not found: {regions_file}")
        else:
            try:
                regions = load_panel_regions(regions_file)
                for capture, bands in regions.items():
                    for band in bands:
                        if band not in panel_bands(config):
                            problems.append(f"{regions_file}: band '{band}' of {capture} is not one of the panel bands "
                                            f"{panel_bands(config)}")
            except (ValueError, OSError) as error:
                problems.append(str(error))
    panel = config['INPUTS.reference_panel_capture'].strip()
    if not panel:
        problems.append("INPUTS.reference_panel_capture is empty")
    if images.is_dir() and pattern_problem is None:
        found = rct.list_capture_names(images, config['INPUTS.file_pattern'])
        named = [panel, config['INPUTS.reference_panel_capture_after'].strip()] + capture_names(config['INPUTS.image_captures'])
        for name in named:
            if name and name not in found:
                problems.append(f"Capture '{name}' not found in {images}")
        if not capture_names(config['INPUTS.image_captures']) and not [n for n in found if n not in named[:2]]:
            problems.append(f"No captures to correct in {images}")
    return problems


def _progress(callback, stage, message, current=0, total=0):
    if callback is not None:
        callback({'stage': stage, 'message': message, 'current': current, 'total': total})


def irradiance_series_changes(series_by_band, threshold):
    """
    Spikes and steps of the horizontal irradiance along the captures
    Inputs: {band: [(capture name, irradiance), ...]} in acquisition order
    Outputs: {band: [event with the capture name, ...]}; bands with fewer than 3 captures or with invalid values
             are skipped
    """
    changes = {}
    for band, series in series_by_band.items():
        if len(series) < 3:
            continue
        try:
            events = rct.classify_irradiance_changes([value for _, value in series], threshold)
        except ValueError:
            continue
        changes[band] = [dict(event, capture=series[event['index']][0]) for event in events]
    return changes


def run(config, root='.', progress=None):
    """
    Run the workflow
    Inputs: configuration (see FIELDS; missing keys take the default), the folder from which the relative paths are
            resolved and, optionally, a function that receives {'stage', 'message', 'current', 'total'}
    Outputs: result dictionary, also saved as radiometric_result.json in the results folder. 'status' is 'completed',
             'completed_with_errors' (some captures failed) or 'failed' (nothing was produced)
    """
    result = {'status': 'failed', 'stages': list(STAGES), 'errors': [], 'flags': [], 'captures': {},
              'panel': {}, 'outputs': []}
    results_folder = None
    limits = dict(rct.DLS_CHECKS)
    try:
        config = merge_config(config)
        result['config'] = config
        problems = validate_config(config, root)
        if problems:
            result['errors'] = problems
            return result
        root = Path(root)
        results_folder = resolve_path(root, config['INPUTS.results_folder'])
        results_folder.mkdir(parents=True, exist_ok=True)
        rct.DLS_CHECKS.update({'band_ratio_tolerance': config['CHECKS.band_ratio_tolerance'],
                               'ratio_range': (config['CHECKS.ratio_range_min'], config['CHECKS.ratio_range_max']),
                               'max_sun_angle_discrepancy_deg': config['CHECKS.max_sun_angle_discrepancy_deg'],
                               'jump_threshold': config['CHECKS.jump_threshold'],
                               'panel_drift_tolerance': config['CHECKS.panel_drift_tolerance']})
        _run_stages(config, root, results_folder, result, progress)
    except Exception as error:   # the result must always say what happened
        result['errors'].append(f"{type(error).__name__}: {error}")
        result['traceback'] = traceback.format_exc()
        result['status'] = 'failed'
    finally:
        rct.DLS_CHECKS.clear()
        rct.DLS_CHECKS.update(limits)
    if results_folder is not None:
        result['result_file'] = str(results_folder / RESULT_FILE_NAME)
        _write_result(results_folder / RESULT_FILE_NAME, result)
    return result


def _second_panel_weight(mode, second_present, image, first_panel, second_panel, names):
    """ Weight of the second panel for one image (see rct.panel_weight); names are those of the image and the two panels """
    if not second_present or mode == 'first':
        return 0.0
    if mode not in ('interpolate_time', 'nearest_time'):
        return rct.panel_weight(mode)
    return rct.panel_weight(mode, rct.capture_time(image.metadata, names[0]), rct.capture_time(first_panel.metadata, names[1]),
                            rct.capture_time(second_panel.metadata, names[2]))


def _read_capture(images_folder, name, bands, exiftool_path, pattern=None, ignored=()):
    files = rct.read_band_files(rct.find_capture_paths(images_folder, name, pattern), bands, exiftool_path, ignored)
    return {band: rct.Image(*files[band]) for band in bands}


def _run_stages(config, root, results_folder, result, progress):
    images_folder = resolve_path(root, config['INPUTS.images_folder'])
    exiftool_path = exiftool_executable(config, root)
    ignored = capture_names(config['INPUTS.ignored_bands'])
    levels, reflectance_info = panel_reflectance_levels(config, root)
    result['panel']['reflectance'] = rct._jsonable({'levels': levels, **reflectance_info})
    bands = list(levels)
    use_dls = config['CORRECTION.use_dls']

    # LOAD_CAPTURES
    panel_name = config['INPUTS.reference_panel_capture'].strip()
    panel_after_name = config['INPUTS.reference_panel_capture_after'].strip()
    names = capture_names(config['INPUTS.image_captures'])
    if not names:
        names = [name for name in rct.list_capture_names(images_folder, config['INPUTS.file_pattern'])
                 if name not in (panel_name, panel_after_name)]
    _progress(progress, 'LOAD_CAPTURES', f"{len(names)} capture(s) to correct; panel capture {panel_name}", 0, len(names))

    # PANEL_CALIBRATION
    regions, origins = {}, {}
    if config['PANEL.manual_regions_file']:
        regions_file = resolve_path(root, config['PANEL.manual_regions_file'])
        regions, origins = load_panel_regions(regions_file), load_panel_origins(regions_file)
    panels = {}
    for label, name in (('before', panel_name), ('after', panel_after_name)):
        if not name:
            continue
        _progress(progress, 'PANEL_CALIBRATION', f"Detecting the reference panel in {name}")
        objects = _read_capture(images_folder, name, bands, exiftool_path, config['INPUTS.file_pattern'], ignored)
        for band, image in objects.items():
            try:
                image.detect_reference_panel(levels, config['PANEL.percent_inner_area'],
                                             config['PANEL.saturation_threshold'],
                                             manual_bounds=regions.get(name, {}).get(band),
                                             region_origin=origins.get(name, {}).get(band))
            except Exception as error:
                raise RuntimeError(f"Panel of band '{band}' in {name}: {error}") from error
        panels[label] = objects
        result['panel'][label] = {'capture': name, 'bands': {band: rct._jsonable(image.panel_audit)
                                                              for band, image in objects.items()}}
    if 'after' in panels:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            comparison = rct.compare_panel_captures(panels['before'], panels['after'])
        result['panel']['comparison'] = rct._jsonable(comparison)
        result['flags'].extend(comparison['flags'])

    # REFLECTANCE and DLS_CHECKS of each capture
    series = {band: [] for band in bands}
    parameters = {'config': config, 'panel_reflectance_levels': levels, 'panel_reflectance_source': reflectance_info}
    for number, name in enumerate(names, start=1):
        _progress(progress, 'REFLECTANCE', f"Correcting {name}", number - 1, len(names))
        entry = {'outputs': [], 'flags': [], 'errors': []}
        result['captures'][name] = entry
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                objects = _read_capture(images_folder, name, bands, exiftool_path, config['INPUTS.file_pattern'], ignored)
                ratios = rct.capture_dls_ratios(panels['before'], objects) if use_dls else {}
                second = panels.get('after')
                weight = _second_panel_weight(config['PANEL.panels_used'], second is not None, objects[bands[0]],
                                              panels['before'][bands[0]], second[bands[0]] if second else None,
                                              (name, panel_name, panel_after_name))
                entry['panels'] = {'used': config['PANEL.panels_used'] if second is not None else 'first',
                                   'weight_of_the_second_panel': weight}
                records = []
                for band in bands:
                    image, panel = objects[band], panels['before'][band]
                    ratio, source = ratios.get(band, (None, None))
                    factor = panel.panel_audit['radiance_to_reflectance_factor']
                    if weight > 0:
                        factor = rct.effective_panel_factor(
                            factor, second[band].panel_audit['radiance_to_reflectance_factor'], weight,
                            panel.dls_irradiance() if use_dls else None, second[band].dls_irradiance() if use_dls else None)
                    reflectance = image.digital_numbers_to_reflectance(factor, dls_ratio=ratio, dls_ratio_source=source)
                    output_path = results_folder / Path(image.image_path).name
                    tifffile.imwrite(output_path, reflectance)
                    records.append(rct.band_record(output_path, panel, image, reflectance,
                                                   second[band] if weight > 0 else None, weight,
                                                   factor if weight > 0 else None))
                    entry['outputs'].append(str(output_path))
                checks = rct.check_dls_capture(panels['before'], objects, ratios or None)
                entry['checks'] = rct._jsonable(checks)
                entry['flags'] = list(checks['flags'])
                provenance = rct.write_provenance(results_folder / (name + '_provenance.json'), records,
                                                  parameters=parameters, checks=checks)
                entry['provenance'] = str(provenance)
            for band in bands:
                try:
                    series[band].append((name, objects[band].dls_irradiance()))
                except rct.MetadataError:
                    pass
            result['outputs'].extend(entry['outputs'])
        except Exception as error:
            entry['errors'].append(f"{type(error).__name__}: {error}")
            result['errors'].append(f"{name}: {error}")
        result['flags'].extend(f"{name}: {flag}" for flag in entry['flags'])

    _progress(progress, 'DLS_CHECKS', "Checking the irradiance along the captures", len(names), len(names))
    result['irradiance_changes'] = irradiance_series_changes(series, config['CHECKS.jump_threshold'])
    for band, events in result['irradiance_changes'].items():
        for event in events:
            result['flags'].append(f"{event['capture']}: {band} irradiance {event['kind']} "
                                   f"({100 * event['change']:+.1f} % from the previous capture)")

    _progress(progress, 'PROVENANCE', "Saving the result file", len(names), len(names))
    produced = any(entry['outputs'] for entry in result['captures'].values())
    failed = any(entry['errors'] for entry in result['captures'].values())
    result['status'] = 'failed' if not produced else ('completed_with_errors' if failed else 'completed')


def _write_result(path, result):
    """ Write the result atomically, so that a reader never finds a partial file """
    path = Path(path)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(rct._jsonable(result), indent=2), encoding='utf-8')
    os.replace(temporary, path)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Radiometric correction of Micasense RedEdge-P captures")
    parser.add_argument('--config', help="json file with the values to change from the defaults")
    parser.add_argument('--request-json', help="json request: {'version': 1, 'root': folder, 'config': {...}}")
    parser.add_argument('--root', help="folder from which the relative paths are resolved (default: current folder)")
    parser.add_argument('--write-default-config', metavar='PATH', help="write the default configuration and exit")
    parser.add_argument('--detect-bands', nargs='?', const='', metavar='CAPTURE',
                        help="print the bands read from the metadata of a capture (default: the panel capture) and exit")
    arguments = parser.parse_args(argv)

    if arguments.write_default_config:
        Path(arguments.write_default_config).write_text(json.dumps(default_config(), indent=2), encoding='utf-8')
        print(f"Default configuration written to {Path(arguments.write_default_config).resolve()}")
        return 0
    root, values = arguments.root or '.', {}
    try:
        if arguments.request_json:
            request = json.loads(Path(arguments.request_json).read_text(encoding='utf-8'))
            if request.get('version') != REQUEST_VERSION:
                raise ConfigError(f"Unsupported request version {request.get('version')!r}")
            root, values = request.get('root', root), request.get('config', {})
        elif arguments.config:
            values = json.loads(Path(arguments.config).read_text(encoding='utf-8'))
    except (OSError, ValueError) as error:
        print(f"Invalid input: {error}", file=sys.stderr)
        return 2

    if arguments.detect_bands is not None:
        try:
            detected = detect_bands(merge_config(values), root, arguments.detect_bands or None)
        except Exception as error:
            print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
            return 1
        print(json.dumps(rct._jsonable(detected), indent=2))
        return 0

    def report(event):
        print(f"[{event['stage']}] {event['message']}", flush=True)

    result = run(values, root, report)
    for error in result['errors']:
        print(f"ERROR: {error}", file=sys.stderr)
    for flag in result['flags']:
        print(f"FLAG: {flag}")
    if 'result_file' in result:
        print(f"RADIOMETRIC_RESULT_JSON={Path(result['result_file']).resolve()}")
    print(f"Status: {result['status']}")
    return 0 if result['status'] == 'completed' else 1


if __name__ == '__main__':
    sys.exit(main())
