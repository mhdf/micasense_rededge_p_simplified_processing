"""
Radiometric correction workflow for Micasense RedEdge-P captures - configuration, validation and runner

This module has no user interface. Everything the workflow needs is in a flat configuration dictionary
(FIELDS describes each key, so that a window can be built from it) and the result is a dictionary that is also saved
as json. It can be used from Python or from the command line:

    python -m radiometric_app.radiometric_workflow --write-default-config config/radiometric_inputs.json
    python -m radiometric_app.radiometric_workflow --config config/radiometric_inputs.json
    python -m radiometric_app.radiometric_workflow --request-json request.json

The configuration keys and the stage identifiers in STAGES are stable: labels and descriptions may change, the keys
and identifiers may not, because saved inputs and results refer to them.
"""

# Standard Library Imports
import argparse
import json
import os
import statistics
import sys
import traceback
import warnings
from pathlib import Path

# Third-Party Imports
import tifffile

# Local Imports
from . import environment, panel_spectrum
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
    "Reflectance of the panel":
        "The reference panel reflects a known fraction of the light that reaches it in each band (for example "
        "0.478 in the green band: it reflects 47.8 % of it).\n"
        "That known value is what turns camera readings into reflectance: the reflectance of a pixel is its "
        "radiance divided by the radiance of the panel, multiplied by the reflectance of the panel in that "
        "band.\n"
        "The manufacturer provides the values with each panel, printed on it and as a spectral curve file. "
        "They differ a little between panels and change with age, so use the ones of your own panel.\n"
        "# Two ways to give it\n"
        "- Typed: one value per band, in the 'Panel reflectance' column of the table of the bands.\n"
        "- Computed: from the spectral curve of the panel (a .csv file), as the mean of the curve over the "
        "spectral window of each band.",
    "Correction with the DLS":
        "The panel photo sets the reflectance scale for the light at the moment it was taken. The DLS "
        "(downwelling light sensor) measures the irradiance, the light that arrives from the sky, so it shows "
        "how the light changed between the panel photo and each image.\n"
        "# The correction\n"
        "- For each band: DLS ratio = E_panel / E_image, with E read from XMP:HorizontalIrradiance of the "
        "panel capture and of the image.\n"
        "- Reflectance = reflectance from the panel x DLS ratio.\n"
        "- Example: a ratio of 0.885 means that the light was 13 % stronger at the image than at the panel, "
        "so the reflectance is multiplied by 0.885.\n"
        "Only the metadata of the image and of the panel capture is used: nothing is averaged along the "
        "flight and no value is estimated from other bands. A missing irradiance in any band, Panchro "
        "included, is an error.",
    "Checks of the DLS data (limits)":
        "These checks look for DLS data that should not be trusted. They only raise flags: the correction is "
        "applied anyway and the images are written. The flags are shown in the log, and saved in the result "
        "file and in the provenance record of each capture, so that you can look at those captures.\n"
        "# What is checked\n"
        "- Band ratio: the DLS ratio of each band is compared with the median of the ratios of all the bands "
        "of the same capture. The light changes all the bands together, so they should agree.\n"
        "- Plausible range: a ratio far from 1 means that the light at the image was very different from the "
        "light at the panel.\n"
        "- Sun angle: the sun direction measured by the DLS against the solar zenith computed from "
        "XMP:SolarElevation; they differ when the DLS is tilted. It is checked in the panel capture and in "
        "the image capture.\n"
        "- Jump: change of the irradiance of a band between consecutive captures: a spike if it comes back, a "
        "step if it stays.\n"
        "- Panel drift: the two panel captures against each other (only with a second panel capture).",
    "Acquisition geometry (limits)":
        "These checks look at how the camera was pointed in each capture. They only raise flags: they never "
        "change the result.\n"
        "# What is measured\n"
        "- Camera tilt: the angle between the optical axis of the camera and the vertical, from XMP:Roll and "
        "XMP:Pitch. A camera that looks straight down has 0.\n"
        "- Heading and the sun: the angle between the heading of the camera, which is the direction of the "
        "top of its image (XMP:Yaw), and the direction of the sun (XMP:SolarAzimuth).\n"
        "# How to read the angle between the heading and the sun\n"
        "- 0 degrees: the camera faces the sun.\n"
        "- 90 degrees: the sun is at the side of the camera.\n"
        "- 180 degrees: the camera has its back to the sun.\n"
        "# What is expected\n"
        "- The reference panel: facing the sun or with the back to it, close to 0 or to 180.\n"
        "- The images to correct: with the sun at the side, close to 90, which lessens the directional "
        "effects of the surfaces.\n"
        "It is the heading of the camera that counts, not the direction of the flight line. If the aircraft "
        "keeps its heading during the flight, the angle is the same in every line. If its nose follows the "
        "lines of a crossed pattern, the angle of one set of lines differs by 90 degrees from the other: the "
        "sun cannot be at the side in both. Set the limit for that, or turn the pattern 45 degrees from the "
        "sun.",
}

FIELDS = [
    {'key': 'INPUTS.images_folder', 'group': 'Folders and tools', 'kind': 'folder', 'default': 'images/raw',
     'label': 'Images folder', 'help': ("Folder with the raw .tif images of the camera.\n"
                  "The camera writes one file per band, for example IMG_0010_1.tif to IMG_0010_6.tif; the files of one "
                  "shot are a capture.\n"
                  "A relative path is read from the project folder. Browse... chooses a folder.")},
    {'key': 'INPUTS.results_folder', 'group': 'Folders and tools', 'kind': 'folder', 'default': 'results/reflectance',
     'label': 'Results folder', 'help': ("Folder where everything that the correction produces is written.\n"
                  "# What goes there\n"
                  "- the reflectance image of each band of each corrected capture, with the name of the raw file;\n"
                  "- a provenance record (.json) of each capture: what was used to produce it, including the panel "
                  "factors, the DLS ratios and the checks;\n"
                  "- radiometric_result.json, with the status, the outputs and the flags of the run;\n"
                  "- panel_regions.json, when the panel regions are checked and no other file was chosen.")},
    {'key': 'INPUTS.exiftool_path', 'group': 'Folders and tools', 'kind': 'file', 'default': '',
     'label': 'ExifTool executable',
     'help': ("ExifTool is a separate program (exiftool.org) that reads the metadata of the images: the bands, the "
                  "QR code of the panel and the DLS irradiance. It must be installed before the correction can run.\n"
                  "This field is only needed if the program is not found by itself. Empty: the environment variable "
                  "exiftoolpath is used, else the system PATH, so it is enough that 'exiftool -ver' works in a "
                  "terminal.\n"
                  "# On Windows\n"
                  "- Download the 'Windows Executable' from exiftool.org.\n"
                  "- Rename exiftool(-k).exe to exiftool.exe and keep it next to its exiftool_files folder.\n"
                  "- Choose it here, or add its folder to the PATH.\n"
                  "The Python package pyexiftool must be installed as well (pip install pyexiftool).\n"
                  "The window looks for both when it opens and each time this field changes, and says below what it found: "
                  "the program, how it was found, its version and whether the package is installed.")},
    {'key': 'INPUTS.reference_panel_capture', 'group': 'Reference panel', 'kind': 'text', 'default': 'IMG_0005',
     'label': 'Reference panel capture', 'help': ("The capture that shows the reference panel: choose it from the captures of the images folder, or "
                  "type its name (for example IMG_0005).\n"
                  "The panel is photographed with its QR code in view. The program finds the QR code and, from it, the "
                  "region of the panel in each band; the regions can be checked, accepted or drawn by hand with Check / "
                  "draw panel regions, below.\n"
                  "This capture also gives the irradiance of the panel (E_panel) for the DLS correction.")},
    {'key': 'INPUTS.reference_panel_capture_after', 'group': 'Reference panel', 'kind': 'text', 'default': '',
     'label': 'Second panel capture (optional)',
     'help': ("A second capture of the panel, taken at the end of the flight (optional).\n"
                  "It is always compared with the first panel capture: a change of the panel factor of a band, or of "
                  "the DLS irradiance relative to the panel, larger than the panel drift tolerance raises a flag.\n"
                  "Whether it also calibrates the images is chosen in 'Panel selection approach' (Reference panel"
                  "page).")},
    {'key': 'INPUTS.image_captures', 'group': 'Images to correct', 'kind': 'text', 'default': '',
     'label': 'Select captures',
     'help': ("The captures to correct: type their names separated by commas, or choose them from the images folder "
                  "with Select...\n"
                  "Empty: all the captures of the folder except the panel captures.")},
    {'key': 'INPUTS.file_pattern', 'group': 'Images to correct', 'kind': 'text', 'default': rct.CAPTURE_FILE_PATTERN,
     'label': 'File name pattern',
     'help': ("How the program tells which files belong to the same capture. The camera writes one file per band "
                  "(IMG_0005_1.tif ... IMG_0005_6.tif).\n"
                  "# Writing the pattern\n"
                  "- {capture}: the name of the capture, the part shared by the files of its bands.\n"
                  "- {index}: the number of the file in the capture (optional).\n"
                  "- *: any other text.\n"
                  "For example {capture}_{index}.tif. The match is not case sensitive. The band of a file is read from "
                  "its metadata, never from its name.\n"
                  "Change it only if your files are named differently; Suggest from files writes it from the names in "
                  "the folder. A regular expression with a group named capture also works.")},
    {'key': 'INPUTS.ignored_bands', 'kind': 'text', 'default': '',
     'label': 'Bands to ignore',
     'help': ("Names (as written by the camera, XMP:BandName) separated by commas, of bands that are in the files "
                  "but must not be processed, for example a thermal band.")},
    {'key': 'PANEL.reflectance_source', 'kind': 'choice', 'choices': ['values', 'spectral_file'],
     'labels': {'values': 'Values typed in the table of the bands', 'spectral_file': 'Spectral response file (.csv)'},
     'default': 'values', 'group': 'Reflectance of the panel',
     'label': 'Source of the panel reflectance',
     'help': ("Where the reflectance of the panel in each band comes from.\n"
                  "- Values typed in the table of the bands: the 'Panel reflectance' column of the table above. The "
                  "spectral file is not used.\n"
                  "- Spectral response file: computed from the .csv file of the panel, below. The reflectance column of "
                  "the table is not used.\n"
                  "The spectral file reproduces the printed values within about 0.3 %. Use it when you have the file of "
                  "your panel.")},
    {'key': 'PANEL.reflectance_by_band', 'kind': 'band_values', 'group': 'Reflectance of the panel', 'range': (0, 1),
     'default': {'Blue': 0.477, 'Green': 0.478, 'Panchro': 0.479, 'Red': 0.478, 'Red edge': 0.478, 'NIR': 0.477},
     'label': 'Panel reflectance by band', 'help': ("Values provided by the manufacturer with the reference panel. The band names must be those written "
                  "by the camera (XMP:BandName).")},
    {'key': 'PANEL.spectral_file', 'kind': 'file', 'default': '', 'group': 'Reflectance of the panel',
     'label': 'Spectral response file',
     'help': ("The spectral response of the panel: its reflectance at each wavelength, in a .csv file provided by "
                  "the manufacturer.\n"
                  "Browse... chooses the file and opens a window with a preview and a plot, where its format is detected "
                  "and can be changed (delimiter, header rows, units, ...); nothing changes until the file is accepted "
                  "there. Preview and format... opens the same window for the file that is already chosen.\n"
                  "The reflectance of a band is the mean of the curve over its spectral window: the center of the band "
                  "plus or minus half of its bandwidth, from the table of the bands.")},
    {'key': 'PANEL.spectral_delimiter', 'kind': 'choice', 'choices': list(panel_spectrum.DELIMITERS), 'dialog': True,
     'default': 'auto', 'group': 'Reflectance of the panel', 'label': 'Delimiter',
     'help': 'Character that separates the columns; auto detects it.'},
    {'key': 'PANEL.spectral_delimiter_char', 'kind': 'text', 'default': '', 'dialog': True,
     'group': 'Reflectance of the panel', 'label': 'Custom delimiter',
     'help': "The separator when the delimiter is custom (one or more characters)."},
    {'key': 'PANEL.spectral_skip_rows', 'kind': 'text', 'default': 'auto', 'dialog': True,
     'group': 'Reflectance of the panel', 'label': 'Header rows to skip',
     'help': "auto: the rows at the start that are not numbers; or the number of rows."},
    {'key': 'PANEL.spectral_skip_footer', 'kind': 'int', 'default': 0, 'dialog': True,
     'group': 'Reflectance of the panel', 'label': 'Footer rows to skip',
     'help': "Number of rows at the end of the file that are not data."},
    {'key': 'PANEL.spectral_comment_prefix', 'kind': 'text', 'default': '', 'dialog': True,
     'group': 'Reflectance of the panel', 'label': 'Comment prefix',
     'help': "Lines that start with this text are ignored, for example #."},
    {'key': 'PANEL.spectral_exclude_rows', 'kind': 'text', 'default': '', 'dialog': True,
     'group': 'Reflectance of the panel', 'label': 'Lines to exclude',
     'help': "Line numbers of the file that are not used, for example 5, 10-12."},
    {'key': 'PANEL.spectral_wavelength_min', 'kind': 'text', 'default': '', 'dialog': True,
     'group': 'Reflectance of the panel', 'label': 'Smallest wavelength used (nm)',
     'help': "Empty: no limit."},
    {'key': 'PANEL.spectral_wavelength_max', 'kind': 'text', 'default': '', 'dialog': True,
     'group': 'Reflectance of the panel', 'label': 'Largest wavelength used (nm)',
     'help': "Empty: no limit."},
    {'key': 'PANEL.spectral_reflectance_scale', 'kind': 'float', 'default': 1.0, 'dialog': True,
     'group': 'Reflectance of the panel', 'label': 'Reflectance multiplier',
     'help': "Applied after the reflectance unit, for example 0.001 for per mille."},
    {'key': 'PANEL.spectral_wavelength_column', 'kind': 'int', 'default': 1, 'dialog': True,
     'group': 'Reflectance of the panel', 'label': 'Wavelength column', 'help': 'Column number, from 1.'},
    {'key': 'PANEL.spectral_reflectance_column', 'kind': 'int', 'default': 2, 'dialog': True,
     'group': 'Reflectance of the panel', 'label': 'Reflectance column', 'help': 'Column number, from 1.'},
    {'key': 'PANEL.spectral_wavelength_unit', 'kind': 'choice', 'choices': list(panel_spectrum.WAVELENGTH_UNITS),
     'default': 'nm', 'dialog': True, 'group': 'Reflectance of the panel', 'label': 'Wavelength unit',
     'help': 'nm or um (micrometers).'},
    {'key': 'PANEL.spectral_reflectance_unit', 'kind': 'choice', 'choices': list(panel_spectrum.REFLECTANCE_UNITS),
     'default': 'fraction', 'dialog': True, 'group': 'Reflectance of the panel', 'label': 'Reflectance unit',
     'help': 'fraction (0 to 1) or percent.'},
    {'key': 'PANEL.spectral_decimal', 'kind': 'choice', 'choices': list(panel_spectrum.DECIMAL_MARKS), 'default': '.',
     'dialog': True, 'group': 'Reflectance of the panel', 'label': 'Decimal mark', 'help': 'Point or comma.'},
    {'key': 'PANEL.band_center_wavelengths', 'kind': 'band_values', 'range': (0, 5000), 'dialog': True,
     'default': {band: window[0] for band, window in panel_spectrum.REDEDGE_P_BANDS.items()},
     'group': 'Reflectance of the panel', 'label': 'Center wavelength of each band (nm)',
     'help': 'Spectral window of each band: the reflectance of a band is the mean of the curve over '
             'center +/- bandwidth / 2.'},
    {'key': 'PANEL.band_bandwidths', 'kind': 'band_values', 'range': (0, 5000), 'dialog': True,
     'default': {band: window[1] for band, window in panel_spectrum.REDEDGE_P_BANDS.items()},
     'group': 'Reflectance of the panel', 'label': 'Bandwidth of each band (nm)',
     'help': 'Width of the spectral window of each band.'},
    {'key': 'PANEL.spectral_min_coverage', 'kind': 'float', 'default': 0.5, 'dialog': True,
     'group': 'Reflectance of the panel', 'label': 'Smallest covered part of a band window',
     'help': 'Fraction (0 to 1] of the window of a band that must have valid samples in the file.'},
    {'key': 'PANEL.panels_used', 'kind': 'choice', 'choices': list(rct.PANELS_USED), 'default': 'first',
     'labels': {'first': 'First panel only', 'last': 'Last panel only', 'mean': 'Average of the two panels',
                'interpolate_time': 'Both panels, weighted by time', 'nearest_time': 'Nearest panel in time',
                'nearest_illumination': 'Nearest panel in light (uses the DLS)',
                'interpolate_illumination': 'Both panels, weighted by light (uses the DLS)'},
     'group': 'Panel measurement', 'label': 'Panel selection approach',
     'help': ("How the panel captures calibrate the images when there are two (before and after the flight). Only used "
                  "when there is a second panel capture.\n"
                  "# Choices (the name that is saved is in brackets)\n"
                  "- First panel only (first): only the first panel calibrates the images; the second is only compared "
                  "with it.\n"
                  "- Last panel only (last): only the second panel, the one after the flight, calibrates the images; the "
                  "first is only compared with it.\n"
                  "- Average of the two panels (mean).\n"
                  "- Both panels, weighted by time (interpolate_time): each image is weighted by its time between the two "
                  "panel captures (times from EXIF:DateTimeOriginal).\n"
                  "- Nearest panel in time (nearest_time): each image uses the panel captured closest in time to it.\n"
                  "- Nearest panel in light (nearest_illumination): each image uses the panel whose light was the most "
                  "like its own, compared with the DLS irradiance of all the bands (their level and how it is spread over "
                  "the bands).\n"
                  "- Both panels, weighted by light (interpolate_illumination): the same comparison, but both panels are "
                  "used, each weighted by 1 / (difference of the light) squared: a panel with the light of the image counts "
                  "alone, and the weight changes smoothly with the light, so there is no step between neighbouring images "
                  "that would use different panels.\n"
                  "The two choices by the light use the DLS data of the images and of both panel captures: choosing one "
                  "turns on 'Correct illumination changes with the DLS' (DLS page), and turning that off takes the choice "
                  "back to the first panel.\n"
                  "This choice changes the result: it decides which panel data are used.\n"
                  "With the DLS correction each panel is used with its own DLS irradiance, so the result does not "
                  "depend on a 'reference panel' of the DLS: only the panels that are used matter.")},
    {'key': 'PANEL.percent_inner_area', 'kind': 'float', 'default': 0.7, 'group': 'Panel measurement',
     'label': 'Inner area of the panel', 'help': ("The size of the square that is measured on the panel, as a fraction of the "
                  "side of the panel: above 0 and up to 1. With 0.7 it is a square in the middle of the panel with 70 % "
                  "of its width and of its height (49 % of its area); with 1 it is about the whole panel.\n"
                  "# Why not all of it\n"
                  "The edge of the panel can include its border, a shadow or the ground, which would change the radiance "
                  "that is measured. A smaller square in the middle avoids that, as long as it still has enough pixels.\n"
                  "# Where it is used\n"
                  "- The region that is found from the QR code is drawn with this size. A band that has no accepted region "
                  "is measured on it.\n"
                  "- A band whose region was accepted or drawn in Check / draw panel regions is measured on that region, "
                  "exactly as it was saved: this value does not change it.\n"
                  "- The tool has the same value: change it there to see the region that it gives before you accept it.")},
    {'key': 'PANEL.saturation_threshold', 'kind': 'int', 'default': 65000, 'group': 'Panel measurement',
     'label': 'Saturation threshold', 'help': ("Pixels above this value are left out of the panel radiance: they are saturated and do not measure "
                  "the light correctly.\n"
                  "The raw values of the camera go up to 65535 (16 bits); 65000 leaves out the pixels at the limit.")},
    {'key': 'PANEL.manual_regions_file', 'kind': 'file', 'default': '', 'group': 'Reference panel regions',
     'label': 'Panel regions drawn by hand (optional)',
     'help': ("File (.json) with the panel regions that were accepted or drawn in Check / draw panel regions. "
                  "Empty: panel_regions.json in the results folder.\n"
                  "The file does not have to exist yet: Browse... lets you choose the folder and name a new one, and the "
                  "tool creates it. While the file is the panel_regions.json of the results folder it follows that folder "
                  "when you change it; a file of another name or folder stays where you put it.\n"
                  "A band with a region in the file uses it instead of the region found from the QR code; the bands "
                  "without one are detected automatically.\n"
                  "The buttons below check the inputs and open the tool that accepts or draws the regions.")},
    {'key': 'CORRECTION.use_dls', 'group': 'Correction with the DLS', 'kind': 'bool', 'default': False,
     'label': 'Correct illumination changes with the DLS',
     'help': ("Multiply the reflectance of each band (Panchro included) by the DLS ratio E_panel / E_image.\n"
                  "Off: only the panel is used, so changes of the light between the panel photo and an image are not "
                  "corrected.\n"
                  "On: every band of the image and of the panel capture needs a valid irradiance "
                  "(XMP:HorizontalIrradiance); otherwise the run stops with an error.")},
    {'key': 'CHECKS.band_ratio_tolerance', 'group': 'Checks of the DLS data (limits)', 'kind': 'float', 'default': rct.DLS_CHECKS['band_ratio_tolerance'],
     'label': 'Band ratio tolerance', 'help': ("Largest departure accepted (0.03 = 3 %) of the DLS ratio of a band from the median of the ratios of "
                  "all the bands of the same capture.\n"
                  "The light changes all the bands together, so within one capture the ratio E_panel / E_image should "
                  "be almost the same in every band. A band that departs more than this points to a problem with the "
                  "irradiance of that band in that capture.")},
    {'key': 'CHECKS.ratio_range_min', 'group': 'Checks of the DLS data (limits)', 'kind': 'float', 'default': rct.DLS_CHECKS['ratio_range'][0],
     'label': 'Smallest plausible DLS ratio', 'help': ("Ratios below this are flagged. A ratio of 0.5 means that the DLS measured twice as much light at the "
                  "image as at the panel.\n"
                  "That is hardly credible within one flight. It happens, for example, if the panel was photographed in "
                  "shade or under cloud and the image was taken in full sun, or if the irradiance of the image is "
                  "wrong.\n"
                  "The correction is still applied; the flag asks you to look at that capture.")},
    {'key': 'CHECKS.ratio_range_max', 'group': 'Checks of the DLS data (limits)', 'kind': 'float', 'default': rct.DLS_CHECKS['ratio_range'][1],
     'label': 'Largest plausible DLS ratio', 'help': ("Ratios above this are flagged. A ratio of 2 means that the DLS measured half as much light at the "
                  "image as at the panel.\n"
                  "It is credible when the panel was photographed in full sun and the image was taken under heavy cloud "
                  "or in shade. It also happens when the DLS was covered or pointing away from the sun at that moment, "
                  "and then the irradiance of the image is wrong.\n"
                  "The correction is still applied; the flag asks you to look at that capture.")},
    {'key': 'CHECKS.max_sun_angle_discrepancy_deg', 'group': 'Checks of the DLS data (limits)', 'kind': 'float',
     'default': rct.DLS_CHECKS['max_sun_angle_discrepancy_deg'],
     'label': 'Largest sun angle discrepancy (deg)',
     'help': ("Largest difference accepted, in degrees, between two measurements of the angle of the sun from the "
                  "vertical (the solar zenith).\n"
                  "# How it works\n"
                  "- The DLS measures where the direct sunlight comes from, in its own frame "
                  "(XMP:EstimatedDirectLightVector). If the DLS is level, the angle between that direction and the "
                  "vertical is the solar zenith.\n"
                  "- The camera also writes the solar elevation (XMP:SolarElevation). The solar zenith is 90 degrees "
                  "minus it, and it does not depend on the DLS.\n"
                  "- The two agree when the DLS is level. If the aircraft was rolled or pitched, the DLS was tilted and "
                  "they differ: the tilt is at least as large as the difference.\n"
                  "A tilted DLS measures the irradiance on a tilted surface instead of a horizontal one, so the DLS "
                  "ratio of that capture is less reliable. The check is made on the panel capture and on the image "
                  "capture.")},
    {'key': 'CHECKS.jump_threshold', 'group': 'Checks of the DLS data (limits)', 'kind': 'float', 'default': rct.DLS_CHECKS['jump_threshold'],
     'label': 'Irradiance jump threshold', 'help': ("Change (0.1 = 10 %) of the irradiance of a band between consecutive captures that is reported.\n"
                  "- A spike: the irradiance jumps and comes back, for example a shaded or tilted DLS in one capture.\n"
                  "- A step: the change stays, for example the sun going behind a cloud.")},
    {'key': 'CHECKS.panel_drift_tolerance', 'group': 'Checks of the DLS data (limits)', 'kind': 'float', 'default': rct.DLS_CHECKS['panel_drift_tolerance'],
     'label': 'Panel drift tolerance', 'help': ("Compares the two panel captures, band by band. Largest change accepted (0.05 = 5 %) in each of "
                  "these:\n"
                  "- The panel factor: the relation between the reflectance of the panel and what the camera read from "
                  "it. It changes if the light was not the same in the two photos, or if the panel was shaded, dirty or "
                  "held differently.\n"
                  "- The agreement between the DLS and the camera: the change of the DLS irradiance between the two "
                  "photos, divided by the change of the irradiance that the camera sees on the panel. It is 1 if the "
                  "DLS followed the light that the panel received, and it is not when the DLS was tilted or shaded in "
                  "one of the photos.\n"
                  "Only used with a second panel capture. A flag does not change the result; whether the second panel "
                  "also calibrates the images is chosen in 'Panel selection approach'.")},
    {'key': 'CHECKS.panel_illumination_tolerance', 'group': 'Checks of the DLS data (limits)', 'kind': 'float',
     'default': rct.DLS_CHECKS['panel_illumination_tolerance'], 'label': 'Light of the panel: largest difference',
     'help': ("Only used when the panels are chosen by their light (nearest_illumination or interpolate_illumination). "
                  "Largest difference accepted (0.15 = about 15 %) between the DLS irradiance of an image and that of the "
                  "panel that is the most like it. A capture beyond it is flagged: no panel was photographed in light "
                  "like its own, and another panel capture under that light would help. A flag does not change the "
                  "result.")},
    {'key': 'CHECKS.max_camera_tilt_deg', 'group': 'Acquisition geometry (limits)', 'kind': 'float', 'default': rct.GEOMETRY_CHECKS['max_camera_tilt_deg'],
     'label': 'Largest camera tilt (deg)',
     'help': ("Largest angle accepted between the optical axis of the camera and the vertical, in degrees, in the "
                  "panel capture and in each image capture. It is worked out from XMP:Roll and XMP:Pitch.\n"
                  "A camera that does not look straight down sees the ground at an angle, and surfaces that are not "
                  "diffuse (soil, leaves) look different from it.")},
    {'key': 'CHECKS.panel_heading_tolerance_deg', 'group': 'Acquisition geometry (limits)', 'kind': 'float', 'default': rct.GEOMETRY_CHECKS['panel_heading_tolerance_deg'],
     'label': 'Panel: largest angle from facing the sun or away (deg)',
     'help': ("How far from the line of the sun the heading of the camera may be in a panel capture, in degrees. "
                  "The angle between the heading and the sun is 0 when the camera faces the sun and 180 when it has its "
                  "back to it, and the check accepts the angles within this tolerance of 0 or of 180.\n"
                  "With 30, the angles from 0 to 30 and from 150 to 180 are accepted.")},
    {'key': 'CHECKS.image_heading_tolerance_deg', 'group': 'Acquisition geometry (limits)', 'kind': 'float', 'default': rct.GEOMETRY_CHECKS['image_heading_tolerance_deg'],
     'label': 'Images: largest angle from the sun at the side (deg)',
     'help': ("How far from 90 degrees the angle between the heading of the camera and the sun may be in the images "
                  "to correct, in degrees. 90 is the sun at the side of the camera.\n"
                  "With 30, the angles from 60 to 120 are accepted.")},
    {'key': 'CHECKS.heading_offset_deg', 'group': 'Acquisition geometry (limits)', 'kind': 'float', 'default': rct.GEOMETRY_CHECKS['heading_offset_deg'],
     'label': 'Correction of the yaw (deg)',
     'help': ("Added to XMP:Yaw before it is compared with the direction of the sun, in degrees.\n"
                  "The direction of the sun is measured from true north. If the yaw of your camera is measured from "
                  "magnetic north, put here the magnetic declination of the place (east is positive). If you do not "
                  "know it, leave 0 and use wide limits.")},
]

FIELD_KEYS = [field['key'] for field in FIELDS]


class ConfigError(ValueError):
    """ Raised when the configuration is not valid """


def default_config():
    """ A copy of the default value of every key """
    return json.loads(json.dumps({field['key']: field['default'] for field in FIELDS}))


def configuration_from_json(data):
    """
    The values of a configuration in what a json file holds: the values themselves (a configuration or a profile) or what a
    run wrote with them: radiometric_result.json ('config') or the provenance record of a capture (<capture>_provenance.json,
    'parameters' / 'config'), so that the settings of a past run can be used again
    """
    if isinstance(data, dict):
        if isinstance(data.get('config'), dict):
            return data['config']
        parameters = data.get('parameters')
        if isinstance(parameters, dict) and isinstance(parameters.get('config'), dict):
            return parameters['config']
    return data


def load_config(path):
    """
    Default configuration with the values of a json file on top (a configuration, a profile, or the result or provenance file
    of a run: see configuration_from_json); unknown keys are an error
    """
    saved = configuration_from_json(json.loads(Path(path).read_text(encoding='utf-8')))
    return merge_config(saved)


def merge_config(values):
    if not isinstance(values, dict):
        raise ConfigError("A configuration is a json object with the keys of the configuration")
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


CONFIG_FOLDER = environment.CONFIG_FOLDER                 # in the project folder: the configuration, the profiles, the own presets
DEFAULT_CONFIG_NAME = environment.DEFAULT_CONFIG_NAME


def default_config_path(root='.'):
    """
    The configuration file that the window and the tools use when none is given: config/radiometric_inputs.json in the
    project folder; the file that earlier versions kept in the project folder itself is used while the new one does not exist
    """
    root = Path(root)
    path = root / CONFIG_FOLDER / DEFAULT_CONFIG_NAME
    legacy = root / DEFAULT_CONFIG_NAME
    return legacy if not path.is_file() and legacy.is_file() else path


def exiftool_executable(config, root='.'):
    """ ExifTool to use: the configured file, else the environment variable exiftoolpath, else None (system PATH) """
    if config['INPUTS.exiftool_path']:
        return str(resolve_path(root, config['INPUTS.exiftool_path']))
    return os.environ.get('exiftoolpath')


def find_exiftool(config, root='.'):
    """
    The ExifTool that will be used and where it was looked for, in this order: INPUTS.exiftool_path, the environment
    variable exiftoolpath, the system PATH. Returns (path, source); the path is None if it is not there
    """
    return environment.locate_exiftool(config['INPUTS.exiftool_path'], root)


exiftool_version = environment.exiftool_version               # the version that 'exiftool -ver' prints, None if it cannot be run
pyexiftool_installed = environment.pyexiftool_installed       # whether the Python package PyExifTool (import exiftool) is there


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


def validate_config(config, root='.', regions_file_may_be_missing=False):
    """
    Return the list of problems of a configuration (empty if it is valid). Paths are resolved from root
    regions_file_may_be_missing: the file of the panel regions (PANEL.manual_regions_file) does not have to exist yet, for the
    tool that creates it; a run needs it
    """
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
    if config['PANEL.panels_used'] in rct.PANELS_BY_ILLUMINATION and not config['CORRECTION.use_dls']:
        problems.append(f"PANEL.panels_used is '{config['PANEL.panels_used']}', which chooses the panel by the light read "
                        "by the DLS, but CORRECTION.use_dls is off: turn the DLS correction on or choose another way to "
                        "use the panels")
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
                'CHECKS.panel_drift_tolerance', 'CHECKS.panel_illumination_tolerance', 'CHECKS.ratio_range_min',
                'CHECKS.ratio_range_max'):
        if config[key] <= 0:
            problems.append(f"{key} must be positive")
    if config['CHECKS.ratio_range_min'] >= config['CHECKS.ratio_range_max']:
        problems.append("CHECKS.ratio_range_min must be smaller than CHECKS.ratio_range_max")
    for key in ('CHECKS.max_camera_tilt_deg', 'CHECKS.panel_heading_tolerance_deg', 'CHECKS.image_heading_tolerance_deg'):
        if not 0 < config[key] <= 90:
            problems.append(f"{key} must be between 0 and 90 degrees")
    if not -180 <= config['CHECKS.heading_offset_deg'] <= 180:
        problems.append("CHECKS.heading_offset_deg must be between -180 and 180 degrees")

    images = resolve_path(root, config['INPUTS.images_folder'])
    if not images.is_dir():
        problems.append(f"Images folder not found: {images}")
    if not config['INPUTS.results_folder'].strip():
        problems.append("INPUTS.results_folder is empty")
    exiftool_path, exiftool_source = find_exiftool(config, root)
    if exiftool_path is None:
        if config['INPUTS.exiftool_path'].strip():
            problems.append(f"ExifTool not found: {resolve_path(root, config['INPUTS.exiftool_path'])}")
        else:
            problems.append(f"ExifTool was not found in {exiftool_source}: install it (exiftool.org), or choose it in "
                            f"INPUTS.exiftool_path")
    if not pyexiftool_installed():
        problems.append("The Python package pyexiftool is not installed (pip install pyexiftool)")
    if config['PANEL.manual_regions_file']:
        regions_file = resolve_path(root, config['PANEL.manual_regions_file'])
        if not regions_file.is_file():
            if not regions_file_may_be_missing:
                problems.append(f"Panel regions file not found: {regions_file} (draw and accept the regions with Check / "
                                "draw panel regions, or empty the field to find the panel from its QR code)")
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
    limits, geometry_limits = dict(rct.DLS_CHECKS), dict(rct.GEOMETRY_CHECKS)
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
                               'panel_drift_tolerance': config['CHECKS.panel_drift_tolerance'],
                               'panel_illumination_tolerance': config['CHECKS.panel_illumination_tolerance']})
        rct.GEOMETRY_CHECKS.update({key: config['CHECKS.' + key] for key in geometry_limits})
        _run_stages(config, root, results_folder, result, progress)
    except Exception as error:   # the result must always say what happened
        result['errors'].append(f"{type(error).__name__}: {error}")
        result['traceback'] = traceback.format_exc()
        result['status'] = 'failed'
    finally:
        rct.DLS_CHECKS.clear()
        rct.DLS_CHECKS.update(limits)
        rct.GEOMETRY_CHECKS.clear()
        rct.GEOMETRY_CHECKS.update(geometry_limits)
    if results_folder is not None:
        result['result_file'] = str(results_folder / RESULT_FILE_NAME)
        _write_result(results_folder / RESULT_FILE_NAME, result)
    return result


def _light(images, name):
    """ {band: DLS irradiance} of a capture; MetadataError that names the capture if a band has none """
    try:
        return {band: image.dls_irradiance() for band, image in images.items()}
    except rct.MetadataError as error:
        raise rct.MetadataError(f"{name}: {error}") from error


def _second_panel_weight(mode, second_present, image, first_panel, second_panel, names):
    """
    Weight of the second panel for one image (see rct.panel_weight)
    Inputs: the mode, whether there is a second panel, and the capture, the first panel and the second panel as
            {band: rct.Image}; names are those of the three captures
    Outputs: (weight, distances), where distances is {'first': ..., 'second': ...}, how different the light of the image is
             from that of each panel (see rct.illumination_distance), for the modes by illumination, and None for the others
    """
    if not second_present or mode == 'first':
        return 0.0, None
    if mode in rct.PANELS_BY_ILLUMINATION:
        light = _light(image, names[0])
        distances = (rct.illumination_distance(light, _light(first_panel, names[1])),
                     rct.illumination_distance(light, _light(second_panel, names[2])))
        return rct.panel_weight(mode, distances=distances), {'first': distances[0], 'second': distances[1]}
    if mode not in ('interpolate_time', 'nearest_time'):
        return rct.panel_weight(mode), None
    band = next(iter(image))
    return rct.panel_weight(mode, rct.capture_time(image[band].metadata, names[0]),
                            rct.capture_time(first_panel[band].metadata, names[1]),
                            rct.capture_time(second_panel[band].metadata, names[2])), None


def _read_capture(images_folder, name, bands, exiftool_path, pattern=None, ignored=()):
    files = rct.read_band_files(rct.find_capture_paths(images_folder, name, pattern), bands, exiftool_path, ignored)
    return {band: rct.Image(*files[band]) for band in bands}


def compact_reflectance_info(info):
    """ What the reports say about where the reflectance of the panel came from, without what the configuration says """
    compact = {key: value for key, value in info.items() if key not in ('format', 'min_coverage', 'bands')}
    if 'bands' in info:
        compact['bands'] = {band: {key: values[key] for key in ('samples', 'coverage')}
                            for band, values in info['bands'].items()}
    return compact


def _run_stages(config, root, results_folder, result, progress):
    images_folder = resolve_path(root, config['INPUTS.images_folder'])
    exiftool_path = exiftool_executable(config, root)
    ignored = capture_names(config['INPUTS.ignored_bands'])
    levels, reflectance_info = panel_reflectance_levels(config, root)
    reflectance_info = compact_reflectance_info(reflectance_info)
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
        result['panel'][label] = {'capture': name, 'bands': {band: rct._jsonable(rct.compact_panel_audit(image.panel_audit))
                                                              for band, image in objects.items()}}
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            geometry = rct.check_panel_geometry(objects)
        result['panel'][label]['geometry'] = rct._jsonable(geometry['values'])
        result['flags'].extend(f"{name}: {flag}" for flag in geometry['flags'])
    if 'after' in panels:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            comparison = rct.compare_panel_captures(panels['before'], panels['after'])
        result['panel']['comparison'] = rct._jsonable(comparison)
        result['flags'].extend(comparison['flags'])

    # REFLECTANCE and DLS_CHECKS of each capture
    series = {band: [] for band in bands}
    events, checked = [], 0                      # the flags of the captures, to be counted by criterion (group_flag_events)
    parameters = {'config': config, 'panel_reflectance_levels': levels, 'panel_reflectance_source': reflectance_info}
    for number, name in enumerate(names, start=1):
        _progress(progress, 'REFLECTANCE', f"Correcting {name}", number - 1, len(names))
        entry = {'outputs': [], 'errors': []}
        result['captures'][name] = entry
        written, flags, details = [], [], []
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                objects = _read_capture(images_folder, name, bands, exiftool_path, config['INPUTS.file_pattern'], ignored)
                ratios = rct.capture_dls_ratios(panels['before'], objects) if use_dls else {}
                second = panels.get('after')
                weight, distances = _second_panel_weight(config['PANEL.panels_used'], second is not None, objects,
                                                         panels['before'], second, (name, panel_name, panel_after_name))
                entry['panels'] = {'used': config['PANEL.panels_used'] if second is not None else 'first',
                                   'weight_of_the_second_panel': weight}
                if distances:
                    entry['panels']['light_difference'] = distances
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
                    written.append(str(output_path))
                    entry['outputs'].append(output_path.name)           # the files are in the results folder
                checks = rct.check_dls_capture(panels['before'], objects, ratios or None)
                geometry = rct.check_image_geometry(objects)
                checks['geometry'] = {'panel': result['panel']['before']['geometry'], 'image': geometry['values']}
                checks['flags'] = list(checks['flags']) + geometry['flags']
                checks['unavailable'] = list(checks['unavailable']) + geometry['unavailable']
                flags, details = list(checks['flags']), checks['flag_details'] + geometry['flag_details']
                if distances:
                    checks['panel_light_difference'] = distances
                    nearest, tolerance = min(distances.values()), rct.DLS_CHECKS['panel_illumination_tolerance']
                    if nearest > tolerance:
                        flags.append(f"Light of the capture differs {100 * nearest:.0f} % from that of the nearest panel "
                                     f"capture (limit {100 * tolerance:g} %)")
                        details.append({'criterion': 'panel_illumination', 'scope': 'image', 'band': None,
                                        'value': nearest, 'limit': tolerance})
                checked += 1
                entry['geometry'] = rct._jsonable(geometry['values'])
                entry['dls'] = rct._jsonable({key: checks[key] for key in ('median_ratio',) if key in checks})
                provenance = rct.write_provenance(results_folder / (name + '_provenance.json'), records,
                                                  parameters=parameters, checks=checks)
                entry['provenance'] = Path(provenance).name
            for band in bands:
                try:
                    series[band].append((name, objects[band].dls_irradiance()))
                except rct.MetadataError:
                    pass
            result['outputs'].extend(written)
        except Exception as error:
            entry['errors'].append(f"{type(error).__name__}: {error}")
            result['errors'].append(f"{name}: {error}")
        for number, flag in enumerate(flags):
            detail = details[number] if number < len(details) else None
            if detail is None:
                result['flags'].append(f"{name}: {flag}")
            elif detail['scope'] == 'panel':                         # the same in every capture: once, with the panel
                flag = f"{panel_name}: {flag}"
                if flag not in result['flags']:
                    result['flags'].append(flag)
            else:
                events.append((name, detail))

    groups = group_flag_events(events, checked, names)

    _progress(progress, 'DLS_CHECKS', "Checking the irradiance along the captures", len(names), len(names))
    result['irradiance_changes'] = irradiance_series_changes(series, config['CHECKS.jump_threshold'])
    groups += group_irradiance_changes(result['irradiance_changes'], config['CHECKS.jump_threshold'], len(names), names)
    result['flag_groups'] = [{key: value for key, value in group.items() if key != 'message'} for group in groups]
    result['flags'].extend(group['message'] for group in groups)

    _progress(progress, 'PROVENANCE', "Saving the result file", len(names), len(names))
    produced = any(entry['outputs'] for entry in result['captures'].values())
    failed = any(entry['errors'] for entry in result['captures'].values())
    result['status'] = 'failed' if not produced else ('completed_with_errors' if failed else 'completed')
    result['summary'] = build_summary(result, config)


# The flags of the captures are counted by criterion: 'DLS tilt over 10 deg in 386 of 409 captures (...): IMG_0010 to IMG_0044, ...'
FLAG_CRITERIA = (                                # (criterion, label from the limit, name of the value, unit, decimals)
    ('camera_tilt', lambda limit: f"Camera tilt over {limit:g} deg", 'tilt', ' deg', 1),
    ('heading', lambda limit: f"Camera heading more than {limit:g} deg from the sun at the side (90 deg)",
     'heading', ' deg from the sun', 0),
    ('dls_tilt', lambda limit: f"DLS tilt over {limit:g} deg", 'tilt of at least', ' deg', 1),
    ('ratio_range', lambda limit: f"DLS ratio outside {limit[0]:g} to {limit[1]:g}", 'ratio', '', 3),
    ('band_ratio_departure', lambda limit: f"DLS ratio of a band more than {100 * limit:g} % from the median of the bands",
     'departure', ' %', 1),
    ('panel_illumination', lambda limit: f"Light more than {100 * limit:g} % different from the nearest panel capture",
     'difference', ' %', 1))


def capture_list_text(names, order, texts=None):
    """
    The captures of a flag as text, all of them, in the order of the run (order: the names of all the captures): three or
    more neighbours are written as 'first to last'. texts: {capture: text written after its name}; the captures are then
    written one by one
    """
    position = {name: number for number, name in enumerate(order)}
    names = sorted(set(names), key=position.__getitem__)
    items, index = [], 0
    while index < len(names):
        end = index
        while not texts and end + 1 < len(names) and position[names[end + 1]] == position[names[end]] + 1:
            end += 1
        if end - index >= 2:
            items.append(f"{names[index]} to {names[end]}")
        else:
            items.extend(name + (f" ({texts[name]})" if texts else '') for name in names[index:end + 1])
        index = end + 1
    return ', '.join(items)


def group_flag_events(events, total, order):
    """
    Count the flags of the captures by criterion
    Inputs: [(capture, detail)] with the 'flag_details' of rct.check_dls_capture and rct.check_image_geometry; the number
            of captures that were checked and the names of all the captures in the order of the run
    Outputs: one dictionary per criterion that was flagged, in the order of FLAG_CRITERIA: {'criterion', 'label', 'count',
             'of', 'captures' (all of them), 'bands', 'min', 'max' (of the values) and 'message'}, the text for the log
             with the count, the range of the values and the captures (see capture_list_text)
    """
    groups = []
    for criterion, label, value_name, unit, digits in FLAG_CRITERIA:
        found = [(capture, detail) for capture, detail in events if detail['criterion'] == criterion]
        if not found:
            continue
        captures = sorted({capture for capture, _ in found}, key=order.index)
        scale = 100 if criterion in ('band_ratio_departure', 'panel_illumination') else 1
        values = [abs(detail['value']) * scale for _, detail in found]
        bands = sorted({detail['band'] for _, detail in found if detail['band']})
        low, high = min(values), max(values)
        values_text = (f"{low:.{digits}f}{unit}" if round(low, digits) == round(high, digits)
                       else f"{low:.{digits}f} to {high:.{digits}f}{unit}")
        label_text = label(found[0][1]['limit'])
        message = (f"{label_text} in {len(captures)} of {total} capture{'s' if total != 1 else ''} "
                   f"({value_name} {values_text}{'; bands ' + ', '.join(bands) if bands else ''}): "
                   f"{capture_list_text(captures, order)}")
        groups.append({'criterion': criterion, 'label': label_text, 'count': len(captures), 'of': total,
                       'captures': captures, 'bands': bands, 'min': low, 'max': high, 'message': message})
    return groups


def group_irradiance_changes(changes, threshold, total, order):
    """
    The spikes and steps of the irradiance (see irradiance_series_changes) counted by kind over the bands, in the form of
    group_flag_events; each capture says the largest change and in how many bands it was found
    """
    groups = []
    for kind in ('spike', 'step'):
        found = {}
        for events in changes.values():
            for event in events:
                if event['kind'] == kind:
                    found.setdefault(event['capture'], []).append(event['change'])
        if not found:
            continue
        texts = {capture: f"{100 * max(values, key=abs):+.1f} %, {len(values)} band{'s' if len(values) > 1 else ''}"
                 for capture, values in found.items()}
        captures = sorted(found, key=order.index)
        label = f"Irradiance {kind} of more than {100 * threshold:g} % from the previous capture"
        message = (f"{label} in {len(captures)} of {total} capture{'s' if total != 1 else ''}: "
                   f"{capture_list_text(captures, order, texts=texts)}")
        groups.append({'criterion': f'irradiance_{kind}', 'label': label, 'count': len(captures), 'of': total,
                       'captures': captures, 'changes': {capture: found[capture] for capture in captures},
                       'message': message})
    return groups


def _statistics(values):
    """ {'n', 'min', 'median', 'max'} of the numbers that are not None, or None if there are none """
    numbers = [value for value in values if value is not None]
    if not numbers:
        return None
    return {'n': len(numbers), 'min': min(numbers), 'median': statistics.median(numbers), 'max': max(numbers)}


def build_summary(result, config):
    """
    What the run found, in numbers, for the end of the log and the result file: how many captures were corrected, the
    geometry and the DLS of the panel captures, and the minimum, median and maximum over the corrected captures of the tilt of
    the camera, the angle between its heading and the sun, the elevation of the sun, the tilt of the DLS and the DLS ratio,
    each with how many captures are outside the limits of the configuration
    """
    done = [entry for entry in result['captures'].values() if entry.get('outputs') and not entry.get('errors')]
    tilt_limit, tolerance = config['CHECKS.max_camera_tilt_deg'], config['CHECKS.image_heading_tolerance_deg']
    dls_limit = config['CHECKS.max_sun_angle_discrepancy_deg']
    low, high = config['CHECKS.ratio_range_min'], config['CHECKS.ratio_range_max']

    def column(group, key):
        return [entry[group].get(key) for entry in done if entry.get(group)]
    images = {}
    for name, values, outside in (
            ('camera_tilt_deg', column('geometry', 'camera_tilt_deg'), lambda value: value > tilt_limit),
            ('heading_to_sun_deg', column('geometry', 'heading_to_sun_deg'), lambda value: abs(value - 90) > tolerance),
            ('sun_elevation_deg', column('geometry', 'sun_elevation_deg'), None),
            ('dls_tilt_deg', column('geometry', 'dls_tilt_deg'), lambda value: value > dls_limit),
            ('dls_ratio', column('dls', 'median_ratio'), lambda value: not low <= value <= high)):
        stats = _statistics(values)
        if stats:
            stats['outside'] = None if outside is None else sum(outside(value) for value in values if value is not None)
            images[name] = stats
    panels = {label: dict(result['panel'][label]['geometry'], capture=result['panel'][label]['capture'])
              for label in ('before', 'after') if result['panel'].get(label, {}).get('geometry')}
    weights = [entry['panels']['weight_of_the_second_panel'] for entry in done if entry.get('panels')]
    mode = config['PANEL.panels_used']
    panel_weights = None
    if weights and mode != 'first' and 'after' in result['panel']:
        panel_weights = {'mode': mode, 'first_only': sum(weight == 0 for weight in weights),
                         'second_only': sum(weight == 1 for weight in weights),
                         'both': sum(0 < weight < 1 for weight in weights)}
    return {'captures': len(result['captures']), 'corrected': len(done), 'files': len(result['outputs']),
            'flags': len(result['flags']), 'panel_weights': panel_weights,
            'irradiance_changes': sum(len(events) for events in result.get('irradiance_changes', {}).values()),
            'limits': {'camera_tilt_deg': tilt_limit, 'heading_tolerance_deg': tolerance, 'dls_tilt_deg': dls_limit,
                       'dls_ratio': [low, high]},
            'panels': panels, 'images': rct._jsonable(images)}


def summary_lines(result):
    """ The summary of a run (see build_summary) as lines of text for the log; [] if the run has none """
    summary = result.get('summary')
    if not summary:
        return []

    def number(value, digits=0):
        return 'unknown' if value is None else f"{value:.{digits}f}"

    lines = [f"Summary: {summary['corrected']} of {summary['captures']} capture(s) corrected, {summary['files']} file(s) "
             f"written, {summary['flags']} flag(s)."]
    for panel in summary['panels'].values():
        lines.append(f"Panel {panel['capture']}: camera tilt {number(panel.get('camera_tilt_deg'), 1)} deg; heading "
                     f"{number(panel.get('heading_to_sun_deg'))} deg from the sun (0 = facing it, 180 = the back to it); "
                     f"DLS tilt {number(panel.get('dls_tilt_deg'), 1)} deg; sun elevation "
                     f"{number(panel.get('sun_elevation_deg'))} deg.")
    used = summary.get('panel_weights')
    if used:
        lines.append(f"Panels ({used['mode']}): the first panel alone in {used['first_only']} capture(s), the second alone in "
                     f"{used['second_only']}, both in {used['both']}.")
    images, limits = summary['images'], summary['limits']
    descriptions = (
        ('camera_tilt_deg', 1, "camera tilt (deg)", f"limit {limits['camera_tilt_deg']:g}"),
        ('heading_to_sun_deg', 0, "camera heading from the sun (deg; 90 = the sun at the side)",
         f"limit {limits['heading_tolerance_deg']:g} from 90"),
        ('sun_elevation_deg', 0, "sun elevation (deg)", None),
        ('dls_tilt_deg', 1, "DLS tilt (deg, at least)", f"limit {limits['dls_tilt_deg']:g}"),
        ('dls_ratio', 3, "DLS ratio, panel / image", f"range {limits['dls_ratio'][0]:g} to {limits['dls_ratio'][1]:g}"))
    for key, digits, label, limit in descriptions:
        stats = images.get(key)
        if not stats:
            continue
        line = (f"Images ({stats['n']}): {label}: min {number(stats['min'], digits)}, median "
                f"{number(stats['median'], digits)}, max {number(stats['max'], digits)}")
        if limit:
            line += f"; {stats['outside']} outside the {limit}"
        lines.append(line + '.')
    if summary.get('irradiance_changes'):
        lines.append(f"Irradiance along the captures: {summary['irradiance_changes']} spike(s) or step(s) found.")
    return lines


def _write_result(path, result):
    """ Write the result atomically, so that a reader never finds a partial file """
    path = Path(path)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(rct._jsonable(result), indent=2), encoding='utf-8')
    os.replace(temporary, path)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Radiometric correction of Micasense RedEdge-P captures")
    parser.add_argument('--config', help="json file with the values to change from the defaults (also the "
                                         "radiometric_result.json or a provenance file of a past run, to repeat it)")
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
            values = configuration_from_json(json.loads(Path(arguments.config).read_text(encoding='utf-8')))
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
    for line in summary_lines(result):
        print(line)
    if 'result_file' in result:
        print(f"RADIOMETRIC_RESULT_JSON={Path(result['result_file']).resolve()}")
    print(f"Status: {result['status']}")
    return 0 if result['status'] == 'completed' else 1


if __name__ == '__main__':
    sys.exit(main())
