"""
Camera presets: the bands of a camera as a JSON file, so that the band table can be filled for a known camera

A preset is a starting point, not a requirement: the band table can also be read from the images ("Detect from
capture") and edited by hand. Presets are read from the folder camera_presets next to this file and from the one in the
project folder; add a file there to add a camera. The format of a file:

    {
      "label": "Name shown in the window",
      "notes": "Anything worth knowing (optional)",
      "file_pattern": "template of the file names, e.g. {capture}_{index}.tif, or a regular expression (optional)",
      "bands": [ {"name": "Blue", "center_nm": 475, "bandwidth_nm": 32},
                 {"name": "LWIR", "center_nm": 11000, "bandwidth_nm": 6000, "process": false} ]
    }

"name" is the band name written by the camera in XMP:BandName. "process": false keeps a band that is in the files out
of the processing (for example a thermal band). The panel reflectance of each band is not part of a preset, because it
belongs to the panel, not to the camera.

This module has no user interface and is tested without a display.
"""

# Standard Library Imports
import json
from pathlib import Path

# Local Imports
import radiometric_correction_tools as rct

BUILTIN_FOLDER = Path(__file__).resolve().parent / 'camera_presets'


def preset_folders(root='.'):
    """ The folders that are searched: the one that comes with the program, then the one of the project """
    folders = [BUILTIN_FOLDER, Path(root).resolve() / 'camera_presets']
    unique = []
    for folder in folders:
        if folder not in unique:
            unique.append(folder)
    return unique


def validate_preset(data, name='preset'):
    """ Problems of a preset read from a file (empty list if it is valid) """
    problems = []
    if not isinstance(data, dict):
        return [f"{name}: the file must contain a JSON object"]
    if not isinstance(data.get('label'), str) or not data['label'].strip():
        problems.append(f"{name}: 'label' is missing")
    pattern = data.get('file_pattern')
    if pattern is not None:
        try:
            rct.compile_file_pattern(pattern)
        except ValueError as error:
            problems.append(f"{name}: file_pattern: {error}")
    bands = data.get('bands')
    if not isinstance(bands, list) or not bands:
        problems.append(f"{name}: 'bands' must be a list with at least one band")
        return problems
    names = []
    for number, band in enumerate(bands, start=1):
        where = f"{name}: band {number}"
        if not isinstance(band, dict) or not isinstance(band.get('name'), str) or not band['name'].strip():
            problems.append(f"{where} needs a name")
            continue
        names.append(band['name'])
        for key in ('center_nm', 'bandwidth_nm'):
            value = band.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
                problems.append(f"{where} ({band['name']}): {key} must be a number above 0")
        if 'process' in band and not isinstance(band['process'], bool):
            problems.append(f"{where} ({band['name']}): process must be true or false")
    duplicated = sorted({n for n in names if names.count(n) > 1})
    if duplicated:
        problems.append(f"{name}: band names used twice: {duplicated}")
    if names and not any(band.get('process', True) for band in bands if isinstance(band, dict)):
        problems.append(f"{name}: at least one band must be processed")
    return problems


def load_presets(folders):
    """
    Read every .json file of the folders (later folders replace earlier presets with the same file name)
    Outputs: ({key: preset}, [problems]) where the key is the file name without extension and the preset has the
             contents of the file plus 'key' and 'source' (the file); files that cannot be used are only reported
    """
    presets, problems = {}, []
    for folder in folders:
        folder = Path(folder)
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob('*.json')):
            try:
                data = json.loads(path.read_text(encoding='utf-8'))
            except (OSError, ValueError) as error:
                problems.append(f"{path.name}: {error}")
                continue
            found = validate_preset(data, path.name)
            if found:
                problems.extend(found)
                continue
            presets[path.stem] = dict(data, key=path.stem, source=str(path))
    return presets, problems


def band_rows(preset, typed_reflectance=None):
    """
    The rows of the band table for a preset
    Inputs: the preset and, optionally, the reflectance already typed {band name: text}, kept for the bands that have
            the same name (the reflectance belongs to the panel, not to the camera)
    Outputs: [{'name', 'center', 'width', 'reflectance', 'process'}] with texts, in the order of the preset
    """
    typed = typed_reflectance or {}
    return [{'name': band['name'], 'center': f"{band['center_nm']:g}", 'width': f"{band['bandwidth_nm']:g}",
             'reflectance': typed.get(band['name'], '') if band.get('process', True) else '',
             'process': bool(band.get('process', True))} for band in preset['bands']]
