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
import radiometric_correction_tools as rct

REQUEST_VERSION = 1
REGIONS_VERSION = 1
RESULT_FILE_NAME = 'radiometric_result.json'
STAGES = ('LOAD_CAPTURES', 'PANEL_CALIBRATION', 'REFLECTANCE', 'DLS_CHECKS', 'PROVENANCE')

BAND_NAMES = ('Blue', 'Green', 'Panchro', 'Red', 'Red edge', 'NIR')

# Description of every configuration key: default value, kind of value (for validation and for the editor), label
# and description. Kinds: folder, file, text, bool, float, int, band_values (dictionary {band name: number})
FIELDS = [
    {'key': 'INPUTS.images_folder', 'kind': 'folder', 'default': 'images/raw',
     'label': 'Images folder', 'help': 'Folder with the raw .tif images (one file per band, e.g. IMG_0010_1.tif).'},
    {'key': 'INPUTS.results_folder', 'kind': 'folder', 'default': 'results/reflectance',
     'label': 'Results folder', 'help': 'Folder for the reflectance images, the provenance records and the result file.'},
    {'key': 'INPUTS.exiftool_path', 'kind': 'file', 'default': '',
     'label': 'ExifTool executable', 'help': 'Empty: use the environment variable exiftoolpath or the system PATH.'},
    {'key': 'INPUTS.reference_panel_capture', 'kind': 'text', 'default': 'IMG_0005',
     'label': 'Reference panel capture', 'help': 'Name of the capture that shows the reference panel, e.g. IMG_0005.'},
    {'key': 'INPUTS.reference_panel_capture_after', 'kind': 'text', 'default': '',
     'label': 'Second panel capture (optional)',
     'help': 'Panel captured at the end of the flight. It is only compared with the first one; the first is used '
             'for the correction.'},
    {'key': 'INPUTS.image_captures', 'kind': 'text', 'default': '',
     'label': 'Captures to correct', 'help': 'Names separated by commas. Empty: all the other captures in the folder.'},
    {'key': 'PANEL.reflectance_by_band', 'kind': 'band_values',
     'default': {'Blue': 0.477, 'Green': 0.478, 'Panchro': 0.479, 'Red': 0.478, 'Red edge': 0.478, 'NIR': 0.477},
     'label': 'Panel reflectance by band', 'help': 'Values provided by Micasense with the reference panel. The band '
                                                    'names must be those written by the camera (XMP:BandName).'},
    {'key': 'PANEL.percent_inner_area', 'kind': 'float', 'default': 0.7,
     'label': 'Inner area of the panel', 'help': 'Fraction (0 to 1] of the panel used to measure its radiance.'},
    {'key': 'PANEL.saturation_threshold', 'kind': 'int', 'default': 65000,
     'label': 'Saturation threshold', 'help': 'Pixels above this value are left out of the panel radiance.'},
    {'key': 'PANEL.manual_regions_file', 'kind': 'file', 'default': '',
     'label': 'Panel regions drawn by hand (optional)',
     'help': 'json file written by panel_selector.py. A band with a region there uses it instead of the QR code; '
             'the other bands are detected automatically.'},
    {'key': 'CORRECTION.use_dls', 'kind': 'bool', 'default': False,
     'label': 'Correct illumination changes with the DLS',
     'help': 'Multiply the reflectance by E_panel / E_image of each band (Panchro included). Off: panel only.'},
    {'key': 'CHECKS.band_ratio_tolerance', 'kind': 'float', 'default': rct.DLS_CHECKS['band_ratio_tolerance'],
     'label': 'Band ratio tolerance', 'help': 'Largest relative departure of a band DLS ratio from the median of the bands.'},
    {'key': 'CHECKS.ratio_range_min', 'kind': 'float', 'default': rct.DLS_CHECKS['ratio_range'][0],
     'label': 'Smallest plausible DLS ratio', 'help': 'Ratios below this value are flagged.'},
    {'key': 'CHECKS.ratio_range_max', 'kind': 'float', 'default': rct.DLS_CHECKS['ratio_range'][1],
     'label': 'Largest plausible DLS ratio', 'help': 'Ratios above this value are flagged.'},
    {'key': 'CHECKS.max_sun_angle_discrepancy_deg', 'kind': 'float',
     'default': rct.DLS_CHECKS['max_sun_angle_discrepancy_deg'],
     'label': 'Largest sun angle discrepancy (deg)',
     'help': 'Difference between the sun direction seen by the DLS and the solar zenith.'},
    {'key': 'CHECKS.jump_threshold', 'kind': 'float', 'default': rct.DLS_CHECKS['jump_threshold'],
     'label': 'Irradiance jump threshold', 'help': 'Relative change between consecutive captures reported as spike or step.'},
    {'key': 'CHECKS.panel_drift_tolerance', 'kind': 'float', 'default': rct.DLS_CHECKS['panel_drift_tolerance'],
     'label': 'Panel drift tolerance', 'help': 'Largest relative change between the two panel captures.'},
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


def capture_names(text):
    """ Names separated by commas -> list """
    return [name.strip() for name in str(text).split(',') if name.strip()]


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


def save_panel_regions(path, capture, regions):
    """
    Save the regions of one capture ({band: [[x, y], ...]}, None or missing bands are not saved), keeping the other
    captures of the file. The file is replaced atomically
    """
    path = Path(path)
    captures = load_panel_regions(path) if path.is_file() else {}
    kept = {band: [[int(round(x)), int(round(y))] for x, y in polygon]
            for band, polygon in regions.items() if polygon is not None}
    if kept:
        captures[capture] = kept
    else:
        captures.pop(capture, None)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps({'version': REGIONS_VERSION, 'geometry': 'undistorted', 'captures': captures},
                                    indent=2), encoding='utf-8')
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
        elif kind == 'band_values':
            if not isinstance(value, dict) or not value:
                problems.append(f"{key} must be a dictionary {{band name: number}}")
            else:
                for band, number in value.items():
                    if not _is_number(number) or not 0 < number <= 1:
                        problems.append(f"{key}['{band}'] must be a reflectance in (0, 1], got {number!r}")
    if problems:
        return problems

    if not 0 < config['PANEL.percent_inner_area'] <= 1:
        problems.append("PANEL.percent_inner_area must be in (0, 1]")
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
                        if band not in config['PANEL.reflectance_by_band']:
                            problems.append(f"{regions_file}: band '{band}' of {capture} is not in PANEL.reflectance_by_band")
            except (ValueError, OSError) as error:
                problems.append(str(error))
    panel = config['INPUTS.reference_panel_capture'].strip()
    if not panel:
        problems.append("INPUTS.reference_panel_capture is empty")
    if images.is_dir():
        found = rct.list_capture_names(images)
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


def _read_capture(images_folder, name, bands, exiftool_path):
    files = rct.read_band_files(rct.find_capture_paths(images_folder, name), bands, exiftool_path)
    return {band: rct.Image(*files[band]) for band in bands}


def _run_stages(config, root, results_folder, result, progress):
    images_folder = resolve_path(root, config['INPUTS.images_folder'])
    exiftool_path = exiftool_executable(config, root)
    levels = config['PANEL.reflectance_by_band']
    bands = list(levels)
    use_dls = config['CORRECTION.use_dls']

    # LOAD_CAPTURES
    panel_name = config['INPUTS.reference_panel_capture'].strip()
    panel_after_name = config['INPUTS.reference_panel_capture_after'].strip()
    names = capture_names(config['INPUTS.image_captures'])
    if not names:
        names = [name for name in rct.list_capture_names(images_folder) if name not in (panel_name, panel_after_name)]
    _progress(progress, 'LOAD_CAPTURES', f"{len(names)} capture(s) to correct; panel capture {panel_name}", 0, len(names))

    # PANEL_CALIBRATION
    regions = {}
    if config['PANEL.manual_regions_file']:
        regions = load_panel_regions(resolve_path(root, config['PANEL.manual_regions_file']))
    panels = {}
    for label, name in (('before', panel_name), ('after', panel_after_name)):
        if not name:
            continue
        _progress(progress, 'PANEL_CALIBRATION', f"Detecting the reference panel in {name}")
        objects = _read_capture(images_folder, name, bands, exiftool_path)
        for band, image in objects.items():
            try:
                image.detect_reference_panel(levels, config['PANEL.percent_inner_area'],
                                             config['PANEL.saturation_threshold'],
                                             manual_bounds=regions.get(name, {}).get(band))
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
    parameters = {'config': config}
    for number, name in enumerate(names, start=1):
        _progress(progress, 'REFLECTANCE', f"Correcting {name}", number - 1, len(names))
        entry = {'outputs': [], 'flags': [], 'errors': []}
        result['captures'][name] = entry
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                objects = _read_capture(images_folder, name, bands, exiftool_path)
                ratios = rct.capture_dls_ratios(panels['before'], objects) if use_dls else {}
                records = []
                for band in bands:
                    image, panel = objects[band], panels['before'][band]
                    ratio, source = ratios.get(band, (None, None))
                    reflectance = image.digital_numbers_to_reflectance(
                        panel.panel_audit['radiance_to_reflectance_factor'], dls_ratio=ratio, dls_ratio_source=source)
                    output_path = results_folder / Path(image.image_path).name
                    tifffile.imwrite(output_path, reflectance)
                    records.append(rct.band_record(output_path, panel, image, reflectance))
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
