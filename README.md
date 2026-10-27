# Radiometric Correction for Micasense RedEdge-P

This Python module provides a simplified alternative for converting raw images from the Micasense RedEdge-P sensor into radiance or reflectance, using the reference panel and, optionally, the Downwelling Light Sensor (DLS) data to correct changes in illumination.

**What has been tested.** The program is written so that it can be used with other MicaSense cameras (RedEdge-M/MX, Altum, Altum-PT, dual-camera systems): the bands, the file names and the bands that are left out are configurable, and the bands can be read from the images themselves (see "Other cameras and other numbers of bands"). For the moment, however, it has been tested **only with images of the MicaSense RedEdge-P**. The other MicaSense models, and cameras of other makers (for example Sentera), have not been tested, and the presets of other models come from the documentation of the manufacturer, not from images. Use "Detect from capture" and compare the result with your own data before relying on them.

All functions are derived from the original [Micasense image processing repository](https://github.com/micasense/imageprocessing) but have been streamlined to mitigate issues caused by Exif metadata inconsistencies. These issues were encountered while pre-processing images captured with the camera firmware v1.4.5 in preparation for geometric correction using photogrammetry software.

For processing, it is only required the path and metadata for the images and for the reference panel. Images metadata can be extracted for example using [ExifTool](https://exiftool.org/). The only caution needed here is to maintain the same field labels adopted in the original Exif information.

**ExifTool must be installed before running.** The metadata (bands, panel QR code, DLS irradiance) is read with ExifTool, a separate program, through the Python package ```PyExifTool``` (```pip install PyExifTool```; both are needed; see "Installing on a new computer"). On Windows, download the "Windows Executable" from [exiftool.org](https://exiftool.org/), rename ```exiftool(-k).exe``` to ```exiftool.exe``` and keep it next to its ```exiftool_files``` folder. The program is found in this order: the ```INPUTS.exiftool_path``` field of the configuration (the "ExifTool executable" field of the window), the environment variable ```exiftoolpath```, the system PATH (it is enough that ```exiftool -ver``` works in a terminal). The field is only needed when it is not found by itself. The window looks for ExifTool and pyexiftool when it opens and whenever the field changes, and says in the help column what it found (the program, how it was found, its version, whether the package is installed); ```Check inputs``` and the runner report them as a problem when they are missing, and ```radiometric_workflow.find_exiftool``` gives the same answer from Python.

An example of how the pre-processing would work in practice is presented below. It processes the example images in this repository (```images/raw```): the capture ```IMG_0005``` shows the reference panel and ```IMG_0010``` is the capture to be corrected.

First, lets import the required libraries and module.

```
# Standard library imports
import os
from pathlib import Path

# Third-party library imports
import tifffile

# Module for radiometric correction
#  Set the root directory for your working environment first
root_path = Path.cwd()
images_folder = 'images/raw'
results_folder = 'results/reflectance'
# Import module
import radiometric_correction_tools as rct

# Path to ExifTool - None uses the one found in the system PATH
exiftool_path = os.environ.get('exiftoolpath')
```

Next, it is necessary to provide some information required for processing, including: (i) name of the captures with the reference panel and with the images to be transformed from raw Digital Numbers (DNs) to reflectance, (ii) reflectance levels for different bands in the reference panel, a value provided by Micasense with each reference panel, and (iii) whether the DLS should be used. The files of each capture are matched to the bands using the band name written in their metadata (```XMP:BandName```), not the order of the files: in the example images the file ```_3``` is the Red band and ```_6``` is the Panchro band. Each result keeps the name of its source file, so ```IMG_0010_3.tif``` is the Red band both before and after the correction.

```
# Name of the captures: the reference panel and the images to be corrected
reference_panel_name = 'IMG_0005'
image_to_process_name = 'IMG_0010'

# Reflectance level for the different bands in the reference panel -
# these values are measured and provided by Micasense, marked in the reference panel,
# but they can also be requested via the Micasense technical support.
# obs.: the names must be the band names written by the camera (XMP:BandName), the order does not matter.
reflectance_level_reference_panel = { "Blue": 0.477, "Green": 0.478, "Panchro": 0.479, \
                                      "Red": 0.478, "Red edge": 0.478, "NIR": 0.477 }

# Use the Downwelling Light Sensor (DLS) to correct changes in illumination between the panel
# capture and each image? (False = panel only). All bands, Panchro included, use the irradiance written in their files.
use_dls = False

# Find the files of each capture and match them to the bands using the metadata (not the file order)
bands = list( reflectance_level_reference_panel.keys() )
reference_panel_files = rct.read_band_files( rct.find_capture_paths(root_path / images_folder, reference_panel_name),
                                             expected_bands=bands, exiftool_path=exiftool_path )
image_files = rct.read_band_files( rct.find_capture_paths(root_path / images_folder, image_to_process_name),
                                   expected_bands=bands, exiftool_path=exiftool_path )
```

Finally, lets iterate over the different bands saving images after transforming them from DNs to radiance and at the end to reflectance.

```
# Images of both captures (the panel is detected inside the loop; the DLS ratios only need the metadata)
reference_panel_objs = { band: rct.Image( *reference_panel_files[band] ) for band in bands }
image_objs = { band: rct.Image( *image_files[band] ) for band in bands }

# DLS ratios (E_panel / E_image) of all the bands. A missing or invalid irradiance in any band, Panchro included,
# raises an error: nothing is estimated from other bands
dls_ratios = rct.capture_dls_ratios( reference_panel_objs, image_objs ) if use_dls else {}

band_records = []
for band in bands: # Iterate over bands

    # Reference panel image (path and metadata of the file of this band)
    reference_panel_obj = reference_panel_objs[band]

    # Identify automatically the coordinates for the reference panel in the unidistorted image and
    # calculate the factor used to convert radiance to reflectance
    panel_bounds, radiance_to_reflectance_factor, _, _ = \
            reference_panel_obj.detect_reference_panel( reflectance_level_reference_panel,
                                                        percent_inner_area=0.7, sat_threshold=65000 )
    print( reference_panel_obj.panel_audit )  # what was used to calibrate this band

    # Image to be processed
    image_obj = image_objs[band]

    # Reflectance image
    dls_ratio, dls_ratio_source = dls_ratios.get( band, (None, None) )
    image_reflectance = image_obj.digital_numbers_to_reflectance( radiance_to_reflectance_factor,
                                                                  dls_ratio=dls_ratio,
                                                                  dls_ratio_source=dls_ratio_source )

    ##
    # Save reflectance image
    output_path = root_path / results_folder / Path(image_files[band][0]).name
    tifffile.imwrite( output_path, image_reflectance )  # keeps the float values (no rescaling to integers)
    band_records.append( rct.band_record(output_path, reference_panel_obj, image_obj, image_reflectance) )

# Checks of the DLS data of the capture (the messages are also issued as warnings)
checks = rct.check_dls_capture( reference_panel_objs, image_objs, dls_ratios or None )
print( checks['flags'] )

# Record of what was used to produce the results, saved with them
rct.write_provenance( root_path / results_folder / (image_to_process_name + '_provenance.json'), band_records,
                      parameters={ 'reflectance_level_reference_panel': reflectance_level_reference_panel,
                                   'percent_inner_area': 0.7, 'sat_threshold': 65000, 'use_dls': use_dls },
                      checks=checks )
```

In summary, with the function ```rct.Image(image, metadata)``` an object with the image and associated information is created, while the function ```self.detect_reference_panel(reflectance_level_reference_panel, percent_inner_area, sat_threshold)``` detects the reference panel automatically on undistorted radiance images and calculate the conversion factor necessay to transform radiance to reflectance. The final step includes transforming the target image to radiance using the function ```self.digital_numbers_to_radiance()``` and multiplying the output by the conversion factor calculated using the reference panel, which is what ```self.digital_numbers_to_reflectance(factor)``` does. The functions ```rct.find_capture_paths()``` and ```rct.read_band_files()``` list the files of a capture and match them to the bands.

## Checks and calibration record

- Metadata are checked when an image is created. If a field is missing or invalid (non-numeric, wrong number of values, non-positive exposure, ISO and so on) a ```rct.MetadataError``` names the file and the field.
- ```detect_reference_panel()``` raises a clear error if the QR code is not found, if the band has no reflectance level or if all panel pixels are saturated. Saturated pixels (above ```sat_threshold```) are left out of the panel mean.
- After ```detect_reference_panel()```, ```reference_panel_obj.panel_audit``` holds what was used to calibrate the band: panel region, number of pixels and of saturated pixels, mean and standard deviation of the radiance, reflectance level and the final factor.
- ```rct.band_record()``` and ```rct.write_provenance()``` save a json file with the results (```IMG_0010_provenance.json``` in the example). What is the same in all the bands is written once, in ```capture```: the ids of the image capture and of the panel capture and the attitude of the camera as written in the file (```Roll```, ```Pitch```, ```Yaw```, in degrees). For each band: the panel and image files, the panel audit above (without what the configuration says: the file, the band, the saturation threshold and the inner area), the DLS irradiances of the panel and of the image (they are recorded even when the DLS is not used) and, when the DLS is used, its ratio; the second panel only when it was used. Then the checks of the capture (each flag once, in ```checks.flags```), the method, the parameters of the run (the complete configuration, so that the limits of the checks are in it, and the reflectance of the panel), the date and the software used (hash of ```radiometric_correction_tools.py```, python, numpy and opencv versions). ```rct.compact_band_records()``` and ```rct.compact_checks()``` do the compaction; ```rct.band_record()``` itself keeps everything.

## Optional DLS correction

**What the DLS ratio is.** The DLS (downwelling light sensor) measures, for each band, the irradiance on a horizontal surface, ```E``` (```XMP:HorizontalIrradiance```). The DLS ratio of a band is ```E_panel / E_image```: the irradiance measured when the panel was photographed divided by the irradiance measured when the image was taken. The panel calibrates the camera for the light of that moment, so an image taken in more light than the panel (ratio below 1) would look too bright, and multiplying by the ratio corrects it. A ratio of 0.885 means that the DLS measured 13 % more light at the image than at the panel, so the reflectance is lowered by 11.5 %. The ratio is 1 at the panel capture, and the scale to SI units (```XMP:IrradianceScaleToSIUnits```), when it is written, cancels in it. The same explanation is shown in the DLS page of the window.

**What data each image uses.** The DLS correction needs no reference data other than what is written in files: the irradiance of an image comes from the metadata of that image only (```XMP:HorizontalIrradiance```, per band), and the irradiance of the panel comes from the metadata of the panel capture (or captures) that is used. There is no flight-wide reference, no averaging with neighbouring images and no smoothing of the irradiance along the flight. The checks that compare neighbouring captures (spikes and steps) only raise flags; they never change a result.

**How the solar zenith is obtained.** The solar elevation is read as it is written in the file (```XMP:SolarElevation```, in radians; it is not recomputed from position and time) and the solar zenith is 90 degrees minus the elevation. The DLS also writes the direction of the direct sunlight in its own frame (```XMP:EstimatedDirectLightVector```). If the DLS were level, the angle between that direction and the axis of the DLS (the arccosine of minus the z component of the unit vector) would be equal to the zenith; the tilt of the DLS is at least as large as the difference between the two (or the two are inconsistent). It was checked only against the example panel capture (0.2 degrees, with a level DLS).

With ```use_dls = True``` the reflectance of each image is multiplied by ```E_panel / E_image```, the ratio between the horizontal irradiance measured by the DLS (```XMP:HorizontalIrradiance```) at the panel capture and at the image capture, for the same band:

```
reflectance(image) = radiance(image) * panel_reflectance / panel_radiance * E_panel / E_image
```

- At the panel capture the ratio is 1, so the result reproduces the panel-only conversion. Panel-only is the default and never reads the DLS fields.
- It is a relative correction: it assumes that the panel and the DLS saw the same illumination when the panel was imaged. ```XMP:IrradianceScaleToSIUnits``` is applied when it exists; it cancels in the ratio, so nothing is assumed when it is absent (it must be in both captures or in none).
- The Panchro band is corrected like the others, with the irradiance written in its own file. In the example images its ratio (0.8842) agrees with those of the multispectral bands (0.883 to 0.888). How the camera derives that value is not documented here. A missing or invalid irradiance in any band, Panchro included, raises an error: no ratio is estimated from other bands.
- This correction does not account for the DLS orientation or shading, so check the data as described below before relying on it.

### Checks of the DLS data

The limits are in ```rct.DLS_CHECKS```; they are first guesses, not validated values, so adjust them to your data. Each check issues a ```RuntimeWarning```, and the messages are saved in the provenance record.

- ```rct.check_dls_capture(panel_images, images)```, for one capture:
  - the DLS ratio of every band must be close to the median of the bands (3 %);
  - the ratio must be inside 0.5 to 2 (also warned by ```rct.dls_irradiance_ratio()```);
  - the angle between the sun direction seen by the DLS (```XMP:EstimatedDirectLightVector```) and the solar zenith (from ```XMP:SolarElevation```) is compared. With a level DLS they are equal, so a difference above 10 degrees means that the DLS was tilted by at least that much, or that the angles are inconsistent. The conventions were checked only against the example panel capture (0.2 degrees); in the example, the image capture shows about 15 degrees and is flagged;
  - the DLS irradiance divided by the irradiance estimated from the panel (pi x radiance / reflectance) is reported per band. It differs between bands (about 14 % in the example, probably a calibration difference between camera and DLS), so it is not judged on one capture.
- ```rct.compare_panel_captures(before, after)```, for a panel photographed twice (for example before and after the flight): the panel factor of each band, and the DLS irradiance divided by the panel estimate, must not change more than 5 %. The first change shows that the light or the panel changed, the second that the DLS did not follow the light seen by the camera.
- ```rct.classify_irradiance_changes(irradiances)```, for the irradiance of one band along the flight in acquisition order: a *spike* is a single capture that jumps and comes back (a shaded or tilted DLS), a *step* is a change that persists (the sun going behind a cloud).

### Geometry of the acquisition

How the camera was pointed in each capture. The limits are in ```rct.GEOMETRY_CHECKS``` and in the configuration (```CHECKS.max_camera_tilt_deg```, ```CHECKS.panel_heading_tolerance_deg```, ```CHECKS.image_heading_tolerance_deg``` and ```CHECKS.heading_offset_deg```); they are first guesses. They only raise flags (```RuntimeWarning```), never change the result, and what is missing in the metadata is reported as unavailable instead of failing.

- **Camera tilt** (```rct.camera_tilt_deg()```): the angle between the optical axis of the camera and the vertical, from ```XMP:Roll``` and ```XMP:Pitch``` (written in degrees). Limit: 5 degrees, for the panel capture and for each image capture.
- **Heading and the sun** (```rct.heading_to_sun_deg()```): the angle between the heading of the camera (```XMP:Yaw```, in degrees, plus ```CHECKS.heading_offset_deg```: the direction of the top of the image) and the direction of the sun (```XMP:SolarAzimuth```, in radians, from north clockwise), from 0 to 180 without a side: 0 is facing the sun, 90 the sun at the side, 180 the back to the sun. The panel is expected close to 0 or to 180 (within 30 degrees) and the images to correct close to 90 (within 30 degrees). ```XMP:SolarAzimuth``` was checked against the position of the sun calculated from the GPS position and the time of the example (the time written by the camera is UTC); whether ```XMP:Yaw``` is referred to true or to magnetic north is not known, which is what the correction of the yaw is for.
- It is the heading of the camera that counts, not the direction of the flight lines. If the aircraft keeps its heading, the angle is the same on every line; if its nose follows the lines of a crossed pattern, one set of lines differs by 90 degrees from the other, and the sun cannot be at the side in both.
- ```rct.check_panel_geometry(panel_images)``` and ```rct.check_image_geometry(images)``` do the checks (with ```rct.measure_geometry()```, the median over the bands). The flags of a panel capture are written once in the result, not for every capture; the values of the panel and of each capture are saved in the ```geometry``` of the checks of the provenance record.
- In the example the panel capture shows a heading 87 degrees from the sun and is flagged (it was photographed with the sun at the side, not facing it or with the back to it) and the image capture shows 83 degrees, which is accepted.

### The flags of a run are counted by criterion

A criterion that fails in many captures is written once, with how many captures failed and which ones, not once per capture. In the example flight (409 captures) the line is, for instance:

```
DLS tilt over 10 deg in 386 of 409 captures (tilt of at least 10.1 to 22.0 deg): IMG_0010 to IMG_0044, IMG_0047, IMG_0050 to IMG_0100, ...
```

- Every capture is listed, in the order of the run (there is no limit to the length of the list); three or more neighbours are written as ```first to last``` to keep it short. The ```flag_groups``` of ```radiometric_result.json``` have the same captures in a list (```criterion```, ```label```, ```count```, ```of```, ```captures```, the range of the values and, for the criteria that are checked in each band, ```bands```).
- The criteria that are counted are the tilt of the camera, its heading against the sun, the tilt of the DLS, the DLS ratio outside its plausible range and the ratio of a band that departs from the other bands (```wf.FLAG_CRITERIA```), and the spikes and steps of the irradiance (one line for each kind, with the largest change and the number of bands in which it was found for each capture). The result of a check carries ```flag_details``` for this (the criterion, the scope, the band, the value and the limit of each flag, in the order of ```flags```); ```wf.group_flag_events()``` and ```wf.group_irradiance_changes()``` do the counting, ```wf.capture_list_text()``` writes the list.
- What is not repeated per capture stays as before: the flags of the panel captures (heading, tilt, the DLS of the panel and the comparison of two panels) are written once with the name of the panel capture. The provenance record of each capture still has all of its own flags, with their numbers.

### The summary at the end of a run

At the end of the log, and in the command line output, the run says what it found, with the values of the whole run: how many captures were corrected, the geometry and the DLS of the panel captures and, over the corrected captures, the minimum, median and maximum of the tilt of the camera, the angle between its heading and the sun, the elevation of the sun, the tilt of the DLS and the DLS ratio, each with how many captures are outside the limits, and how many spikes or steps of the irradiance were found. The numbers are also in ```summary``` of ```radiometric_result.json``` (```wf.build_summary()``` and ```wf.summary_lines()```).

## Two panel captures (before and after the flight)

A second panel capture (```INPUTS.reference_panel_capture_after```) is always compared with the first one (the panel factor of every band, and the DLS irradiance relative to the irradiance estimated from the panel, must not change more than ```CHECKS.panel_drift_tolerance```). ```PANEL.panels_used``` says whether it also calibrates the images:

- ```first``` (default): only the first panel is used; the second is only compared.
- ```last```: only the second panel, the one after the flight, calibrates the images; the first is only compared with it (it needs a second panel capture).
- ```mean```: the two panels weigh the same (0.5 each).
- ```interpolate_time```: each image is weighted by its time between the two panel captures, from ```EXIF:DateTimeOriginal``` (0 at the first panel, 1 at the second; an image outside them uses the nearest panel; two panels taken at the same time weigh 0.5 each). It needs that tag in the images of the capture and of both panels.
- ```nearest_time```: each image uses, entirely, the panel that was captured closest in time to it (the first one if both are equally close or were taken at the same time). It needs the same time tags as ```interpolate_time```.
- ```nearest_illumination```: each image uses, entirely, the panel whose light was the most like its own (the first one if both are equally like it). The light of a capture is its DLS irradiance in every band (```XMP:HorizontalIrradiance```, which is the level of the light and how it is spread over the bands), and how different two lights are is ```rct.illumination_distance()```: the root mean square over the bands of the natural logarithm of the ratio of the irradiances (0.1 is about 10 %; 0 is the same light). It needs the DLS data in the images and in both panel captures, and ```CORRECTION.use_dls``` on: the two choices by illumination ("Nearest panel in light (uses the DLS)" and "Both panels, weighted by light (uses the DLS)") are in the list of the window, and choosing one turns "Correct illumination changes with the DLS" on (the status line says so); turning the correction off while one of them is chosen goes back to ```first``` and says so, and the check of the inputs and a run refuse them without the correction. The names in the list are descriptions of the values that are saved (```first```, ```last```, ```mean```, ```interpolate_time```, ```nearest_time```, ```nearest_illumination```, ```interpolate_illumination```).
- ```interpolate_illumination```: the same comparison, but both panels are used, each weighted by 1 / distance squared (```rct.ILLUMINATION_POWER```): ```w = d1^2 / (d1^2 + d2^2)``` is the weight of the second panel, where d1 and d2 are the distances of the light of the image to that of the first and of the second panel. A panel with the light of the image counts alone, equal distances weigh 0.5, and the weight changes smoothly with the light, so neighbouring images do not jump between two panels that disagree (the step that ```nearest_illumination``` can make where the choice changes, as large as the disagreement of the two panels).

The capture is compared as a whole, so the same weight is used in every band, which keeps the relation between the bands. In both modes the distances of each capture are written in the result file (```light_difference``` of the capture) and in its provenance record (```checks.panel_light_difference```), and a capture whose light is more different than ```CHECKS.panel_illumination_tolerance``` (0.15, about 15 %) from the nearest panel is flagged, counted like the other flags ("Light more than 15 % different from the nearest panel capture in N of M captures: ..."): there is no panel in light like its own and another panel capture would help. The summary at the end of the run says how many captures used the first panel alone, the second alone and both.

In the example flight (a first panel at 13:26 UTC and a last one at 13:48 UTC) the irradiance rises by 32 to 35 % between the two, and most of the 409 captures are close to the light of the last panel; with ```nearest_illumination``` the first panel is chosen for about 40 of them (the first two captures, a few at ```IMG_0087```-```IMG_0092``` and a stretch from ```IMG_0256``` with a drop of the irradiance), and ```nearest_time``` would have chosen it for the whole first half of the flight.

Without the DLS the panel factors are averaged with these weights. With the DLS what is averaged is the factor times the irradiance of each panel, which is a constant of the camera that does not depend on the light (the two panels should agree on it), and the result is expressed for the irradiance of the first panel, which the DLS ratio refers to: ```factor = ((1 - w) f1 E1 + w f2 E2) / E1```. The weight and the factor that was used are written in the provenance record of each band (```second_panel_weight```, ```effective_factor```) and the weight of each capture in the result file. The regions of both panel captures are drawn and accepted in the same tool, and both must be accepted before saving.

With the DLS the choice of "the reference panel" does not exist as such: the product factor x irradiance of a panel is the same constant wherever the panel was photographed, so what matters is which panels contribute to it (```first```, the nearest, or a mean), and the result is the same whatever panel the irradiance ratio is written relative to. The two panels agree on that constant when the DLS followed the light seen by the camera; ```CHECKS.panel_drift_tolerance``` flags it when they do not. MicaSense advises taking the panel image that best represents the average illumination of the flight when only one is used.

How Metashape combines several panel images, and what its sun sensor option is referenced to, is not described in its manual (2.1) or in the forum threads of Agisoft that were read for this project; the photogrammetric workflow gives all the panel captures of the chunk to ```chunk.calibrateReflectance(use_reflectance_panels, use_sun_sensor)``` and cannot choose among them. Do not assume that it averages or follows the time: test it with panels of known different factors before relying on one behaviour.

## Installing on a new computer

What the program needs, and how to see whether the computer has it:

1. **Python 3.10 or newer, with Tkinter** (the window). The Python of python.org and conda have it (python.org: keep "tcl/tk and IDLE" selected; Linux: ```sudo apt install python3-tk```). Windows 10 or 11 for the launchers; the program itself is plain Python.
2. **The Python packages** of ```requirements.txt```: numpy, opencv-python, pillow, tifffile, matplotlib, scikit-image, shapely, pyzbar and PyExifTool (the versions in the file are the ones it was tested with: Python 3.10.20 on Windows, numpy 2.2.6, opencv 4.12.0, pillow 12.0.0, tifffile 2025.5.10, matplotlib 3.10.8, scikit-image 0.25.2, shapely 2.1.2, pyzbar 0.1.9, PyExifTool 0.5.6). Install them in a **dedicated environment**, not in the system Python (the program never installs anything: when a package is missing it says which and how, and stops). In a terminal, in the project folder:
   - conda: ```conda env create -f environment.yml``` (the environment is called ```radiometric```; the file was checked by solving it with conda-forge, but the environment was not created and used from scratch on another computer);
   - or a virtual environment: ```python -m venv .venv```, then ```.venv\Scripts\python.exe -m pip install -r requirements.txt```;
   - then choose the ```python.exe``` of that environment (```launchers/select_python.cmd```, the dialog at the start of the window or the Python environment section of the Inputs page). If the Python that was chosen already is a dedicated environment, ```python -m pip install -r requirements.txt``` with it is enough.
   - pyzbar reads the QR code of the panel; the Windows package brings the zbar library it needs (and asks for the Microsoft Visual C++ Redistributable of 2013), macOS needs ```brew install zbar``` and Linux ```sudo apt install libzbar0```.
3. **ExifTool**, the program (not a Python package): see the paragraph at the top. Without it the window opens but nothing can be read from the images.
4. **Choose the Python** (Windows launchers): double-click ```launchers/select_python.cmd```. A window like the one of the photogrammetric workflow asks for the ```python.exe``` (Browse...), checks that it is Python 3.10 or newer with Tkinter, saves it in ```config/python_environment.json``` and then lists what else is missing. The other launchers use that file; the first time, with no file, they open the same window by themselves, and when the saved Python no longer exists they ask again. To change the Python, run ```select_python.cmd``` again, or use the window (next point). The file is plain json (```{"python": "C:\\...\\python.exe"}```) and is not kept in git.

5. **The window asks for the Python every time it starts** ("Select Python Environment", ```radiometric_app/python_selection.py```, the same dialog as the photogrammetric workflow). It proposes the saved Python (or the one that is running) and you confirm it, paste another path or browse. "Confirm and open window" tries the Python first, in a subprocess (Python 3.10 or newer, and a Tk window can be created), saves it, and, if it is not the Python that is running, starts the window again with it (with its own folder and, for a conda environment, ```Library\bin``` first in the PATH, and without the ```PYTHONHOME``` and ```PYTHONPATH``` of the other one) and closes this one. A Python that does not work is explained and not saved, so the one that works is never replaced. Cancel closes everything. The packages and ExifTool of the Python that is confirmed are checked next (see below). ```--skip-python-dialog```, or the environment variable ```RADIOMETRIC_PYTHON_CONFIRMED=1``` (which the window that was started again has), skip the dialog, for example to run ```python -m radiometric_app window``` from a script; the panel tool and the check never ask.

**The check of the computer** (```radiometric_app/environment.py```, only the standard library, so it works before anything is installed) lists the Python and its path, Tkinter, each package with its version and ExifTool with its version and how it was found (the ```INPUTS.exiftool_path``` of the configuration, ```exiftoolpath```, the PATH), and what to install for each thing that is missing, in one ```pip install``` command for the packages. It is used in four places:

- Before the window or the panel tool opens (```python -m radiometric_app window```, which is what the launchers run): when something is missing a message box says what and what to install, and nothing is opened; when only ExifTool is missing the message asks whether to open the program anyway. A package that is installed but cannot be loaded (a damaged installation, a missing DLL) is named in the message instead of a traceback that nobody sees without a console.
- The button "Check environment" on the "Run and results" page lists the same in the log (with every package imported, so a broken one shows).
- ```launchers/check_environment.cmd``` or ```python -m radiometric_app check [--deep] [--json] [--quiet] [--root folder]```: the same list in a terminal, exit code 0 when all is in place and 1 when not (```--deep``` imports every package, ```--json``` gives the result for other programs).
- From Python: ```environment.check_environment()``` returns one dictionary per thing (```name```, ```status``` ok, missing, broken or old, ```version```, ```detail```, ```fix```, ```level``` required or needed), and ```environment.install_text()``` and ```environment.message_text()``` write the instructions.

## Project layout

```
radiometric_correction_tools.py   the library (it keeps its name and its place: other programs import it)
radiometric_app/                  the program: configuration and runner, window, panel regions tool, capture picker, ...
    presets/                      the camera presets that come with the program
tests/                            the tests
launchers/                        launchers for Windows (double-click), the choice of the Python and the check of the computer
config/                           your configuration, profiles, presets and chosen Python (not kept in git)
requirements.txt, environment.yml the Python packages (pip) and the conda environment (see "Installing on a new computer")
images/raw/, results/reflectance/ the example images and their results
```

The commands in this file are run from the project folder (the one with ```radiometric_correction_tools.py```), for example ```python -m radiometric_app.radiometric_window```. ```radiometric_app``` can be copied into another program together with ```radiometric_correction_tools.py``` (the modules of the folder import each other with relative imports).

## Workflow runner (no user interface)

```radiometric_app/radiometric_workflow.py``` runs the whole example above, for all the captures of a folder, from a configuration with fixed keys. It has no user interface, so a window can be built on top of it (```FIELDS``` describes each key: default, kind of value, label and description).

```
python -m radiometric_app.radiometric_workflow --write-default-config config/radiometric_inputs.json   # write the defaults, then edit them
python -m radiometric_app.radiometric_workflow --config config/radiometric_inputs.json                   # run
python -m radiometric_app.radiometric_workflow --request-json request.json                               # run from a request (see below)
```

- The configuration is a flat json file with the keys of ```FIELDS``` (```INPUTS.*``` folders and captures, ```PANEL.*``` reflectance levels and panel measurement, ```CORRECTION.use_dls```, ```CHECKS.*``` limits). Only the keys to change are needed; unknown keys are an error. Relative paths are resolved from ```--root``` (default: the current folder). Labels and descriptions may change in future versions, the keys will not.
- With ```INPUTS.image_captures``` empty, every capture in the images folder except the panel capture(s) is corrected. The panel is detected once and used for all of them. A second panel capture (```INPUTS.reference_panel_capture_after```) is only compared with the first.
- The stages are ```LOAD_CAPTURES```, ```PANEL_CALIBRATION```, ```REFLECTANCE```, ```DLS_CHECKS``` and ```PROVENANCE```. A capture that fails (for example a missing band file, or a missing DLS irradiance when ```CORRECTION.use_dls``` is on) is reported and the others continue.
- Each corrected capture gets its ```<capture>_provenance.json``` (the details: panel audit, DLS irradiances, checks); the whole run gets ```radiometric_result.json``` in the results folder, which says what happened and where to look: the status (```completed```, ```completed_with_errors``` or ```failed```), the errors, the flags of all the checks (those of the panel captures once, with the name of the capture, and those of the captures to correct counted by criterion with the captures listed, see "The flags of a run are counted by criterion"; ```flag_groups``` has the structured form), the complete configuration, the panel calibration (the reflectance of each band and where it came from, and a short audit of the panel of each band), the spikes/steps of the irradiance along the captures and, for each capture, the names of its output files and of its provenance record (in the results folder), the panels used, its errors and a summary of the DLS checks (the median ratio and the sun angle of the panel and of the image). ```outputs``` has the complete path of every output file. The command prints ```RADIOMETRIC_RESULT_JSON=<path>``` and returns 0 only for ```completed``` (1 for errors, 2 for an unreadable request or configuration file).
- A request is ```{"version": 1, "root": "<folder>", "config": {...}}```, so that another program can start the workflow as a separate process (the libraries needed here are usually not available in the Python of a photogrammetry program) and read the result file.
- From Python: ```radiometric_workflow.run(config, root, progress)```, where ```progress``` receives ```{'stage', 'message', 'current', 'total'}```.

## Other cameras and other numbers of bands

**Tested only with the MicaSense RedEdge-P.** What follows describes what the program is prepared for, not what has been verified on other cameras.

Nothing in the processing depends on there being six bands: the bands are a table of the configuration (name, spectral window and panel reflectance of each band), and the same code runs with 3, 5, 6 or 10 of them.

- **Band table** (Reference panel page of the window; ```PANEL.reflectance_by_band```, ```PANEL.band_center_wavelengths```, ```PANEL.band_bandwidths``` and ```INPUTS.ignored_bands``` in the configuration). One row per band; rows can be added, renamed and removed. The names must be those written by the camera in ```XMP:BandName```. Un-ticking *Use* keeps a band that is in the files out of the processing (it is written in ```INPUTS.ignored_bands```): without that, a file with a band that is not in the table is an error, so that a missing or extra band is never silently ignored. The panel reflectance of a band is only needed when the source of the reflectance is ```values```.
- **Detect from capture** reads the table from the images themselves: the name, central wavelength and bandwidth of each band (```XMP:BandName```, ```XMP:CentralWavelength```, ```XMP:WavelengthFWHM```) in the order of the rig, and the name of the camera and its serial number. A band that looks thermal (a center above 2500 nm, or a name such as LWIR) is left out. The panel reflectances that were typed are kept for bands with the same name. This is the most reliable way to get the windows of your own camera: for the example images it gives Red 668 / 14 nm, and the units of one model differ from another (the RedEdge-MX bandwidths of the older RX01 units are 20, 20, 10, 10 and 40 nm). From the command line: ```python radiometric_workflow.py --detect-bands IMG_0005```.
- **Camera presets** are JSON files in the ```radiometric_app/presets``` folder (those that come with the program) and in ```config/presets``` (your own). They fill the table and the file pattern; the reflectance of the panel belongs to the panel, not to the camera, so a preset does not contain it. The folder has the RedEdge-P (the bands of the example images, checked against them by a test), the RedEdge-M/MX (5 bands, the current windows of the MicaSense documentation) and the Altum (5 bands and the thermal band left out); the last two were not checked on images, so use *Detect from capture* to confirm them. Add a file to add a camera (the format is at the top of ```radiometric_app/camera_presets.py```).
- **File names** (```INPUTS.file_pattern```). The files of a capture are found with a name template (not case sensitive): ```{capture}``` stands for the name of the capture (the part shared by the files of its bands), ```{index}``` for the number of the file (optional; the files are ordered by it) and ```*``` for any other text. The default, ```{capture}_{index}.tif```, is the MicaSense naming (```IMG_0010_1.tif```). Another naming scheme only needs another template, for example ```{capture}_MS_*.tif``` for ```DJI_0010_MS_G.tif```. A regular expression with a group named ```capture``` (and, optionally, ```index```) is also accepted when the template cannot describe the names, for example ```(?P<capture>DJI_\d+)_MS_(?P<band>[A-Z]+)\.tif$```. The band of a file always comes from its metadata, never from its name.
- **What the program assumes.** The conversion to radiance is the MicaSense model (vignetting polynomial, black level, row gradient and the three radiometric calibration coefficients, exposure and ISO), and the panel is found from the QR code of a MicaSense panel. A camera that writes the same metadata works with the table above (RedEdge-M/MX/P, Altum and Altum-PT should; only RedEdge-P images were available to test, and a dual-camera 10-band set, whose two cameras may write different capture ids, was not tried). A camera with another radiometric model needs its own version of ```rct.Image``` (the method that gives the radiance and the one that finds the panel); everything else (the band table, the DLS ratios and checks, the panel regions, the provenance and the window) is independent of the camera.

## Panel reflectance from its spectral response file

Instead of typing the reflectance of each band, it can be computed from the spectral response of the panel, a .csv file with the reflectance by wavelength (```PANEL.reflectance_source``` = ```spectral_file```, ```PANEL.spectral_file```). The reflectance of a band is the mean of the curve over its spectral window, ```PANEL.band_center_wavelengths``` +/- ```PANEL.band_bandwidths``` / 2 (the band table of the window; the defaults are those of the RedEdge-P bands). The weighting is uniform, an approximation of the real spectral response of the filters: with the example panel it gives Blue 0.4766, Green 0.4783, Red 0.4785, Red edge 0.4777, NIR 0.4769 and Panchro 0.4775, against the 0.477, 0.478, 0.478, 0.478, 0.477 and 0.479 printed on the panel.

- **Format.** The file is read with the format in the ```PANEL.spectral_*``` keys: delimiter (```auto```, ```comma```, ```semicolon```, ```tab```, ```space``` or ```custom``` with one or more characters), header rows to skip (```auto``` skips the rows at the start that are not numbers, or a number), footer rows to skip, a comment prefix (such as ```#```), lines to exclude (```5, 10-12```), the smallest and largest wavelength used, the wavelength and reflectance columns, the wavelength unit (```nm``` or ```um```), the reflectance unit (```fraction``` or ```percent```), a reflectance multiplier (applied after the unit, for example 0.001) and the decimal mark (point or comma). In the dialog, lines can also be excluded by selecting them in the table.
- **Samples that are not used.** A reflectance outside (0, 1] (the example file has zeros below 400 nm) is not used. A window that reaches beyond the end of the file (the NIR window of the example, up to 870 nm, against a file that ends at 848 nm) is averaged over the part that exists. ```PANEL.spectral_min_coverage``` (0.5 by default) is the smallest part of a window that must have samples; every band below it is listed in the error.
- **In the window.** "Browse..." on the Reference panel page chooses the file and opens a dialog on it, with its format already detected (the file that is already the source keeps the format that was saved with it; "Preview and format..." opens the same dialog for it without choosing a file). The path and the format are applied together, only when one of the two buttons at the bottom is pressed, so cancelling leaves everything as it was. The dialog shows what was read from each line (header rows, rows with errors and unused samples are marked), the curve with the window of each band, and the reflectance, samples and covered part of each band, updated while the options are changed. "Detect format" guesses the delimiter, decimal mark, units and header rows. The windows and the minimum coverage can be edited there. Then "Use the file as the panel reflectance" fills the keys and makes the file the source, or "Copy the values to the typed list" only copies the computed numbers to the typed values. Only the controls of the chosen source can be edited on the page.
- **Record.** The result file and the provenance record keep the reflectance used for each band, the file, its SHA-256, and the samples and covered part of the window of each band (the format and the windows are in the configuration, which is saved too).
- From Python: ```panel_spectrum.load_spectrum(path, format)``` and ```panel_spectrum.band_reflectances(spectrum, windows, min_coverage)```; ```radiometric_workflow.panel_reflectance_levels(config, root)``` gives the levels that the workflow uses.

## Where things are saved

- **Configuration** (```config/radiometric_inputs.json```): the values that differ from the defaults, in the ```config``` folder of the project (or the file given with ```--config```; a file ```radiometric_inputs.json``` in the project folder, where earlier versions kept it, is still used while the new one does not exist). "Save configuration" in the window, and "Run", write it (so the working configuration is always what was used in the last run). It keeps only the differences from the defaults, which is why a profile, that keeps every value, is the safer way to keep a setup for a camera or a flight.
- **Profiles** (```config/profiles/*.json```): "Save profile..." writes every value of the window to a file in the ```config/profiles``` folder of the project (it can be saved anywhere); "Load profile..." replaces all the values with those of a file. A profile is complete, so it does not depend on the defaults. Running does not make a profile: what every run used is written in ```config``` of ```radiometric_result.json``` and in ```parameters.config``` of the provenance record of each capture, next to the results, and "Load profile..." reads either of those files too (```wf.configuration_from_json()```), so the settings of any past run can be brought back (the results folder is then that of the run, which the status line says; check it before running again). The command line takes them as well: ```--config <results folder>/radiometric_result.json``` repeats a run. This follows the profiles of the photogrammetric workflow window, which uses a ```profiles``` folder next to its code and keeps the working values in ```config_user_inputs.json```.
- **Panel regions drawn by hand**: ```panel_regions.json``` in the results folder, or the file in ```PANEL.manual_regions_file```.
- **Results**: the reflectance images, ```<capture>_provenance.json``` and ```radiometric_result.json``` in the results folder.

## Checking, accepting and drawing the panel regions

The panel region of each band can be checked, accepted and, when it is wrong or the QR code cannot be read, drawn with ```radiometric_app/panel_selector.py```. It is opened with "Check / draw panel regions..." on the Reference panel page (after the panel measurement settings, which include the inner area of the panel, with the file where the regions are kept; the panel captures it reads are chosen on the Inputs page), or with ```open_panel_selector.cmd``` (```python -m radiometric_app.panel_selector --config config/radiometric_inputs.json --root <folder>```). The "Check inputs" button next to it checks the folders, the panel capture and the bands first, without leaving the page, and the tool is not opened while there are problems. The file of the regions does not have to exist yet: the tool starts without regions and creates the file (and its folder) with "Save and close"; a run, instead, needs the file to exist. Browse... next to the field is the dialog that names a file, so any folder can be chosen and a new name typed (an existing file is chosen without being replaced), and while the field holds the ```panel_regions.json``` of the results folder it follows the results folder when that is changed (a file of another name or folder stays where it was put).

- **One band at a time.** The bands are in a list at the left (so that any number fits, for example the ten bands of a dual camera; each shows whether it is accepted, to accept or has no region) and the image of the selected band is shown close up around the panel ("Whole image" shows all of it): the region found from the QR code (yellow, dashed), the region that is accepted or saved (green) or proposed (cyan), and below it the pixels, the saturated pixels and the variation (std / mean; lower is more uniform) inside the region. "Accept" (Enter) takes the region as it is shown and goes to the next band that is not accepted, without entering the drawing; pressing it again on an accepted band takes the acceptance back. "Accept all automatic regions" accepts the regions found from the QR code of all the bands that are not accepted yet, so nothing has to be drawn when they are right. The regions saved before are shown as accepted, so the window is also the way to look at the regions that are in the file.
- **The inner area of the panel is in the tool.** ```PANEL.percent_inner_area``` decides the region that is found from the QR code (the central part of the panel that is measured, without its edge), so the tool has it too, next to "Whole image" ("Inner area of the panel", Enter or Apply). Changing it makes the yellow region of every band again. The regions that were accepted as found from the QR code were made with the old value, so they go back to "to accept" (the one that is then proposed is the new one) and the regions that were drawn or edited stay; it is not possible while a region is being drawn (confirm or discard it first). "Save and close" sets ```PANEL.percent_inner_area``` of the window to the value that the tool was left with and saves the configuration; the tool of the command line (```panel_selector.main```) says when the value is not the one of the configuration. The value of the page is only used where a band has no accepted region: a band whose region was accepted or drawn uses that region as it is, whatever the inner area says.
- **One kind of acceptance.** Every band has to be accepted, so the region used for the reference panel is always one that you saw and confirmed: a region found from the QR code is not used silently. "Save and close" is available when all the bands are accepted, and writes the polygon of every band (vertices in pixels of the undistorted image) with where it came from (```accepted_automatic``` if it is the region found from the QR code, ```edited``` if it was moved or drawn) in ```PANEL.manual_regions_file```, or, if it is empty, in ```panel_regions.json``` in the results folder. Nothing is saved if the window is cancelled.
- **Drawing in the same window.** "Edit region" (E) turns the drawing on for the band: the polygon starts from the region that is shown and a click adds a vertex (on an edge it inserts one), dragging moves a vertex and a right click removes it; the wheel zooms and the middle button drags the image (when not editing the left button drags it too). "Confirm edit" (Enter) accepts the polygon for the band and goes to the next band that is not accepted; "Discard edit" (Esc) turns the drawing off without accepting what was changed. "Reset to automatic" (R), "Clear" (C) and Backspace (last vertex) work only while editing; Left/Right change the band (what was not confirmed is lost). The region accepted or saved before is drawn in green for comparison.
- **Captures.** The window opens the panel captures only (the first one and the one after the flight, when there is one). The "Capture" selector (top right) switches between them and shows how many bands of each are accepted. Images are read when a capture is opened, and what was accepted is kept for each capture while switching. "Save and close" needs every band of the panel captures accepted.
- **What each element does** is shown at the bottom of the window while the mouse is over it (buttons, the list of bands, the image and the capture selector); there is no fixed legend. The area has the width of the window and the height of the longest explanation at that width, so pointing at an element never changes the size of the image.
- A band that has a region in the file uses it, the others are detected from the QR code when the workflow runs, and a region from the file does not need a QR code. The panel audit and the provenance record say, for each band, whether the region was ```automatic``` or ```manual``` (from the file) and its ```region_origin```.
- From Python: ```rct.Image.detect_reference_panel(levels, percent_inner_area, manual_bounds=polygon, region_origin=origin)``` and ```panel_selector.select_panel_regions(images, saved, saved_origins=origins)```, which returns the accepted polygons as a dictionary (a ```SelectionResults```, with the ```percent_inner_area``` that the tool was left with) whose items have an ```origins``` attribute. The logic of the window (```PolygonModel```, ```ViewTransform```, ```SelectionSession```, ```CaptureSet```) has no window and is tested without a display.

## Launchers (Windows)

Double-click one of these files in the ```launchers``` folder. They use the Python saved in ```config/python_environment.json``` (chosen with ```select_python.cmd```, which they open by themselves if there is none; the window then asks for it again, see "Installing on a new computer") and the project folder as ```--root```, so the relative paths of the configuration (for example ```images/raw```) are resolved from there. Each one checks that the Python is 3.10 or newer with Tkinter; the packages and ExifTool are checked by the program before it opens (see "Installing on a new computer").

- ```launchers/open_radiometric_window.cmd```: the window, without a console (like the launcher of the photogrammetric workflow); it is ```pythonw -m radiometric_app window --root .```.
- ```launchers/open_radiometric_window_debug.cmd```: the same window with the console kept open, to see messages and error traces while testing.
- ```launchers/open_panel_selector.cmd```: only the dialog to draw the panel regions, using ```config/radiometric_inputs.json``` in the project folder if it exists, otherwise the defaults. Extra arguments are passed on (for example ```--capture IMG_0005```).
- ```launchers/select_python.cmd```: choose the ```python.exe``` (window written in PowerShell, so that it works before any Python is chosen; ```select_python.ps1```) and check it.
- ```launchers/check_environment.cmd```: list the Python, Tkinter, the packages with their versions and ExifTool, and what to install.
- ```launchers/_python.cmd```: used by the others to find the Python; it is not run by itself.

## Window

**Choosing captures and reading the file names.** "Select..." next to the reference panel capture, the second panel capture and "Select captures" opens a window with one picture for each capture of the images folder, so that the capture that shows the panel can be recognised by looking at it (the second panel capture can also be left empty there). A box chooses the band shown (the bands of the table; the position of each band among the files is read from the metadata of the first capture) and the pictures are read in the background, so the window opens at once; the captures to correct are chosen by clicking (several), with "Select all" and "Clear". The names can still be typed. The panel captures are not offered among the captures to correct; the field shows the names chosen, and when it is empty all the other captures of the folder are corrected (the help column says which). The file name pattern is explained in the help column with how it reads the images folder: how many files match, in how many captures, and an example of how the first files are read; "Suggest from files" writes a pattern from the names of the files in the folder (names such as ```IMG_0005_1.tif```: capture name, a separator, the number of the file, the extension). The pattern is a library function too (```suggest_file_pattern```, ```preview_file_pattern```, ```pattern_template```).

```
python -m radiometric_app.radiometric_window [--config config/radiometric_inputs.json] [--root <folder>]
```

**Layout.** Tiles on the left choose the page (Inputs, Reference panel, DLS, Run and results), the fields are in the centre and a column on the right explains them. The column shows the explanation of the field under the mouse pointer or, when the pointer is elsewhere, of the field that has the keyboard focus (Tab moves between fields), and the explanation of the page when there is neither; while the pointer is over the column the explanation stays, so that it can be read and scrolled. The line between the fields and the column can be dragged to change their widths, and the text follows the width of the column. A field that belongs to a part with its own note (for example the limits of the DLS checks) shows that note under its own explanation, and so do the headings. Information that depends on the data is also shown there, under the explanation: how the file name pattern reads the images folder, which captures will be corrected and what the spectral file gives in each band. What is wrong (a pattern that matches nothing, a capture that is not in the folder, a spectral file that cannot be read) is written under the field, in red, and only while it is wrong; the state of the inputs stays next to the buttons that check them. The explanations are the ```label``` and ```help``` of each entry of ```FIELDS``` and ```GROUP_NOTES```, written with a few marks (a line that starts with ```# ``` is a sub-heading, one that starts with ```- ``` is an item of a list, any other line is a paragraph) that the column turns into paragraphs (the lines after the first are indented), lists and sections.

The pages (Inputs, Reference panel, DLS) are built from ```radiometric_workflow.FIELDS```, so a new key appears in the window without changes to it; the pages scroll when they are longer than the window. The bands of the camera are edited in a table (see above). A value that cannot be read is marked in red. "Save configuration" writes only the values that differ from the defaults (```radiometric_inputs.json``` in ```--root``` by default).

Browse... next to the images folder and the results folder opens a window that lists the folder (folders first, then the files with their sizes; Up, Home, Refresh, New Folder..., and an address box that takes a typed path) and shows, at the right, a picture of the image file that is selected, with its size in pixels and the type of its values: the raw images of a flight are recognised by looking at them, and the same goes for the reflectance images of a results folder. The picture is for the eye only (stretched between the 1 % and 99.5 % percentiles of the image, so dark raw images show; the file is not changed) and is read in the background, so a folder with thousands of files stays quick to move in. "Select Folder" takes the folder that is selected in the list, or else the one that is open (```radiometric_app/folder_browser.py```; the other Browse... buttons, for files, still use the file dialog of the system).

The "Python environment" section at the end of the Inputs page shows the Python that is running the window and the one for the next launches (the field, with Browse...). A running window cannot change its own Python, so this is the one that the dialog at the start proposes and the launchers use next: "Check this Python" runs it (in another thread) and lists its version, whether it can open a window and which packages and ExifTool it has, with the ```pip install``` command for that Python for what is missing, and "Use it for the next launches" tries it as the dialog does and saves it in ```config/python_environment.json```; it is not part of the configuration or of the profiles. The last page runs the workflow in a separate thread, so the window stays responsive: "Check inputs" lists every problem, "Check environment" lists the Python, Tkinter, the packages with their versions and ExifTool of the computer and what to install, "Check / draw panel regions..." (on the Inputs page, next to the panel captures and a "Check inputs" button) opens the tool above (and sets ```PANEL.manual_regions_file``` if it was empty), "Run" saves the configuration and shows the progress, the flags of the checks, the errors and the result file, and "Open results folder" opens the folder in the file explorer.

- Load and Save profile (see above) are in the bottom bar. ```radiometric_window.RadiometricFrame``` builds its widgets in a ```ttk.Frame``` (```.widget```), so it can be placed in any window, for example as a page of another program's window. ```radiometric_app/ui_theme.py``` holds the colors and styles, which follow the dark window of the photogrammetric workflow.
- ```ConfigModel``` (the text typed in each field, its conversion and the file) has no window and is tested without a display.

## Tests

```
python -m unittest discover -s tests -t . -v
```

The tests use the example images and need ExifTool (set the environment variable ```exiftoolpath``` if it is not in the PATH).

For more details about the steps required for pre-processing Micasense multispectral images see the tutorials (e.g. ['Tutorial 1'](https://github.com/micasense/imageprocessing/blob/master/MicaSense%20Image%20Processing%20Tutorial%201.ipynb)) provided by Micasense in their repository.
