# Radiometric Correction for Micasense RedEdge-P

This Python module provides a simplified alternative for converting raw images from the Micasense RedEdge-P sensor into radiance or reflectance, using the reference panel and, optionally, the Downwelling Light Sensor (DLS) data to correct changes in illumination.

**What has been tested.** The program is written so that it can be used with other MicaSense cameras (RedEdge-M/MX, Altum, Altum-PT, dual-camera systems): the bands, the file names and the bands that are left out are configurable, and the bands can be read from the images themselves (see "Other cameras and other numbers of bands"). For the moment, however, it has been tested **only with images of the MicaSense RedEdge-P**. The other MicaSense models, and cameras of other makers (for example Sentera), have not been tested, and the presets of other models come from the documentation of the manufacturer, not from images. Use "Detect from capture" and compare the result with your own data before relying on them.

All functions are derived from the original [Micasense image processing repository](https://github.com/micasense/imageprocessing) but have been streamlined to mitigate issues caused by Exif metadata inconsistencies. These issues were encountered while pre-processing images captured with the camera firmware v1.4.5 in preparation for geometric correction using photogrammetry software.

For processing, it is only required the path and metadata for the images and for the reference panel. Images metadata can be extracted for example using [ExifTool](https://exiftool.org/). The only caution needed here is to maintain the same field labels adopted in the original Exif information.

**ExifTool must be installed before running.** The metadata (bands, panel QR code, DLS irradiance) is read with ExifTool, a separate program, through the Python package ```pyexiftool``` (```pip install pyexiftool```; both are needed). On Windows, download the "Windows Executable" from [exiftool.org](https://exiftool.org/), rename ```exiftool(-k).exe``` to ```exiftool.exe``` and keep it next to its ```exiftool_files``` folder. The program is found in this order: the ```INPUTS.exiftool_path``` field of the configuration (the "ExifTool executable" field of the window), the environment variable ```exiftoolpath```, the system PATH (it is enough that ```exiftool -ver``` works in a terminal). The field is only needed when it is not found by itself.

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
- ```rct.band_record()``` and ```rct.write_provenance()``` save a json file with the results (```IMG_0010_provenance.json``` in the example): for each band, the panel and image files and capture ids, the panel audit above, the DLS irradiances and ratio (the irradiances are recorded even when the DLS is not used), plus the method, the parameters of the run, the date and the software used (hash of ```radiometric_correction_tools.py```, python, numpy and opencv versions).

## Optional DLS correction

**What the DLS ratio is.** The DLS (downwelling light sensor) measures, for each band, the irradiance on a horizontal surface, ```E``` (```XMP:HorizontalIrradiance```). The DLS ratio of a band is ```E_panel / E_image```: the irradiance measured when the panel was photographed divided by the irradiance measured when the image was taken. The panel calibrates the camera for the light of that moment, so an image taken in more light than the panel (ratio below 1) would look too bright, and multiplying by the ratio corrects it. A ratio of 0.885 means that the DLS measured 13 % more light at the image than at the panel, so the reflectance is lowered by 11.5 %. The ratio is 1 at the panel capture, and the scale to SI units (```XMP:IrradianceScaleToSIUnits```), when it is written, cancels in it. The same explanation is shown in the DLS page of the window.

**What data each image uses.** The DLS correction needs no reference data other than what is written in files: the irradiance of an image comes from the metadata of that image only (```XMP:HorizontalIrradiance```, per band), and the irradiance of the panel comes from the metadata of the panel capture (or captures) that is used. There is no flight-wide reference, no averaging with neighbouring images and no smoothing of the irradiance along the flight. The checks that compare neighbouring captures (spikes and steps) only raise flags; they never change a result.

**How the solar zenith is obtained.** The solar elevation is read as it is written in the file (```XMP:SolarElevation```, in radians; it is not recomputed from position and time) and the solar zenith is 90 degrees minus the elevation. The DLS also writes the direction of the direct sunlight in its own frame (```XMP:EstimatedDirectLightVector```). If the DLS were level, the angle between that direction and the axis of the DLS (the arccosine of minus the z component of the unit vector) would be equal to the zenith; the absolute difference between the two is at least the tilt of the DLS, or a sign that the two are inconsistent. It was checked only against the example panel capture (0.2 degrees, with a level DLS).

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

## Two panel captures (before and after the flight)

A second panel capture (```INPUTS.reference_panel_capture_after```) is always compared with the first one (the panel factor of every band, and the DLS irradiance relative to the irradiance estimated from the panel, must not change more than ```CHECKS.panel_drift_tolerance```). ```PANEL.panels_used``` says whether it also calibrates the images:

- ```first``` (default): only the first panel is used; the second is only compared.
- ```mean```: the two panels weigh the same (0.5 each).
- ```interpolate_time```: each image is weighted by its time between the two panel captures, from ```EXIF:DateTimeOriginal``` (0 at the first panel, 1 at the second; an image outside them uses the nearest panel; two panels taken at the same time weigh 0.5 each). It needs that tag in the images of the capture and of both panels.
- ```nearest_time```: each image uses, entirely, the panel that was captured closest in time to it (the first one if both are equally close or were taken at the same time). It needs the same time tags as ```interpolate_time```.

Without the DLS the panel factors are averaged with these weights. With the DLS what is averaged is the factor times the irradiance of each panel, which is a constant of the camera that does not depend on the light (the two panels should agree on it), and the result is expressed for the irradiance of the first panel, which the DLS ratio refers to: ```factor = ((1 - w) f1 E1 + w f2 E2) / E1```. The weight and the factor that was used are written in the provenance record of each band (```second_panel_weight```, ```effective_factor```) and the weight of each capture in the result file. The regions of both panel captures are drawn and accepted in the same tool, and both must be accepted before saving.

With the DLS the choice of "the reference panel" does not exist as such: the product factor x irradiance of a panel is the same constant wherever the panel was photographed, so what matters is which panels contribute to it (```first```, the nearest, or a mean), and the result is the same whatever panel the irradiance ratio is written relative to. The two panels agree on that constant when the DLS followed the light seen by the camera; ```CHECKS.panel_drift_tolerance``` flags it when they do not. MicaSense advises taking the panel image that best represents the average illumination of the flight when only one is used.

How Metashape combines several panel images, and what its sun sensor option is referenced to, is not described in its manual (2.1) or in the forum threads of Agisoft that were read for this project; the photogrammetric workflow gives all the panel captures of the chunk to ```chunk.calibrateReflectance(use_reflectance_panels, use_sun_sensor)``` and cannot choose among them. Do not assume that it averages or follows the time: test it with panels of known different factors before relying on one behaviour.

## Workflow runner (no user interface)

```radiometric_workflow.py``` runs the whole example above, for all the captures of a folder, from a configuration with fixed keys. It has no user interface, so a window can be built on top of it (```FIELDS``` describes each key: default, kind of value, label and description).

```
python radiometric_workflow.py --write-default-config radiometric_inputs.json   # write the defaults, then edit them
python radiometric_workflow.py --config radiometric_inputs.json                   # run
python radiometric_workflow.py --request-json request.json                        # run from a request (see below)
```

- The configuration is a flat json file with the keys of ```FIELDS``` (```INPUTS.*``` folders and captures, ```PANEL.*``` reflectance levels and panel measurement, ```CORRECTION.use_dls```, ```CHECKS.*``` limits). Only the keys to change are needed; unknown keys are an error. Relative paths are resolved from ```--root``` (default: the current folder). Labels and descriptions may change in future versions, the keys will not.
- With ```INPUTS.image_captures``` empty, every capture in the images folder except the panel capture(s) is corrected. The panel is detected once and used for all of them. A second panel capture (```INPUTS.reference_panel_capture_after```) is only compared with the first.
- The stages are ```LOAD_CAPTURES```, ```PANEL_CALIBRATION```, ```REFLECTANCE```, ```DLS_CHECKS``` and ```PROVENANCE```. A capture that fails (for example a missing band file, or a missing DLS irradiance when ```CORRECTION.use_dls``` is on) is reported and the others continue.
- Each corrected capture gets its ```<capture>_provenance.json```; the whole run gets ```radiometric_result.json``` in the results folder with the status (```completed```, ```completed_with_errors``` or ```failed```), the errors, the flags of all the checks, the panel calibration and the spikes/steps of the irradiance along the captures. The command prints ```RADIOMETRIC_RESULT_JSON=<path>``` and returns 0 only for ```completed``` (1 for errors, 2 for an unreadable request or configuration file).
- A request is ```{"version": 1, "root": "<folder>", "config": {...}}```, so that another program can start the workflow as a separate process (the libraries needed here are usually not available in the Python of a photogrammetry program) and read the result file.
- From Python: ```radiometric_workflow.run(config, root, progress)```, where ```progress``` receives ```{'stage', 'message', 'current', 'total'}```.

## Other cameras and other numbers of bands

**Tested only with the MicaSense RedEdge-P.** What follows describes what the program is prepared for, not what has been verified on other cameras.

Nothing in the processing depends on there being six bands: the bands are a table of the configuration (name, spectral window and panel reflectance of each band), and the same code runs with 3, 5, 6 or 10 of them.

- **Band table** (Reference panel page of the window; ```PANEL.reflectance_by_band```, ```PANEL.band_center_wavelengths```, ```PANEL.band_bandwidths``` and ```INPUTS.ignored_bands``` in the configuration). One row per band; rows can be added, renamed and removed. The names must be those written by the camera in ```XMP:BandName```. Un-ticking *Use* keeps a band that is in the files out of the processing (it is written in ```INPUTS.ignored_bands```): without that, a file with a band that is not in the table is an error, so that a missing or extra band is never silently ignored. The panel reflectance of a band is only needed when the source of the reflectance is ```values```.
- **Detect from capture** reads the table from the images themselves: the name, central wavelength and bandwidth of each band (```XMP:BandName```, ```XMP:CentralWavelength```, ```XMP:WavelengthFWHM```) in the order of the rig, and the name of the camera and its serial number. A band that looks thermal (a center above 2500 nm, or a name such as LWIR) is left out. The panel reflectances that were typed are kept for bands with the same name. This is the most reliable way to get the windows of your own camera: for the example images it gives Red 668 / 14 nm, and the units of one model differ from another (the RedEdge-MX bandwidths of the older RX01 units are 20, 20, 10, 10 and 40 nm). From the command line: ```python radiometric_workflow.py --detect-bands IMG_0005```.
- **Camera presets** are JSON files in the ```camera_presets``` folder (next to the program, and in the project folder). They fill the table and the file pattern; the reflectance of the panel belongs to the panel, not to the camera, so a preset does not contain it. The folder has the RedEdge-P (the bands of the example images, checked against them by a test), the RedEdge-M/MX (5 bands, the current windows of the MicaSense documentation) and the Altum (5 bands and the thermal band left out); the last two were not checked on images, so use *Detect from capture* to confirm them. Add a file to add a camera (the format is at the top of ```camera_presets.py```).
- **File names** (```INPUTS.file_pattern```). The files of a capture are found with a name template (not case sensitive): ```{capture}``` stands for the name of the capture (the part shared by the files of its bands), ```{index}``` for the number of the file (optional; the files are ordered by it) and ```*``` for any other text. The default, ```{capture}_{index}.tif```, is the MicaSense naming (```IMG_0010_1.tif```). Another naming scheme only needs another template, for example ```{capture}_MS_*.tif``` for ```DJI_0010_MS_G.tif```. A regular expression with a group named ```capture``` (and, optionally, ```index```) is also accepted when the template cannot describe the names, for example ```(?P<capture>DJI_\d+)_MS_(?P<band>[A-Z]+)\.tif$```. The band of a file always comes from its metadata, never from its name.
- **What the program assumes.** The conversion to radiance is the MicaSense model (vignetting polynomial, black level, row gradient and the three radiometric calibration coefficients, exposure and ISO), and the panel is found from the QR code of a MicaSense panel. A camera that writes the same metadata works with the table above (RedEdge-M/MX/P, Altum and Altum-PT should; only RedEdge-P images were available to test, and a dual-camera 10-band set, whose two cameras may write different capture ids, was not tried). A camera with another radiometric model needs its own version of ```rct.Image``` (the method that gives the radiance and the one that finds the panel); everything else (the band table, the DLS ratios and checks, the panel regions, the provenance and the window) is independent of the camera.

## Panel reflectance from its spectral response file

Instead of typing the reflectance of each band, it can be computed from the spectral response of the panel, a .csv file with the reflectance by wavelength (```PANEL.reflectance_source``` = ```spectral_file```, ```PANEL.spectral_file```). The reflectance of a band is the mean of the curve over its spectral window, ```PANEL.band_center_wavelengths``` +/- ```PANEL.band_bandwidths``` / 2 (the band table of the window; the defaults are those of the RedEdge-P bands). The weighting is uniform, an approximation of the real spectral response of the filters: with the example panel it gives Blue 0.4766, Green 0.4783, Red 0.4785, Red edge 0.4777, NIR 0.4769 and Panchro 0.4775, against the 0.477, 0.478, 0.478, 0.478, 0.477 and 0.479 printed on the panel.

- **Format.** The file is read with the format in the ```PANEL.spectral_*``` keys: delimiter (```auto```, ```comma```, ```semicolon```, ```tab```, ```space``` or ```custom``` with one or more characters), header rows to skip (```auto``` skips the rows at the start that are not numbers, or a number), footer rows to skip, a comment prefix (such as ```#```), lines to exclude (```5, 10-12```), the smallest and largest wavelength used, the wavelength and reflectance columns, the wavelength unit (```nm``` or ```um```), the reflectance unit (```fraction``` or ```percent```), a reflectance multiplier (applied after the unit, for example 0.001) and the decimal mark (point or comma). In the dialog, lines can also be excluded by selecting them in the table.
- **Samples that are not used.** A reflectance outside (0, 1] (the example file has zeros below 400 nm) is not used. A window that reaches beyond the end of the file (the NIR window of the example, up to 870 nm, against a file that ends at 848 nm) is averaged over the part that exists. ```PANEL.spectral_min_coverage``` (0.5 by default) is the smallest part of a window that must have samples; every band below it is listed in the error.
- **In the window.** "Import..." on the Reference panel page opens a dialog to choose the file and the format: it shows what was read from each line (header rows, rows with errors and unused samples are marked), the curve with the window of each band, and the reflectance, samples and covered part of each band, updated while the options are changed. "Detect format" guesses the delimiter, decimal mark, units and header rows. The windows and the minimum coverage can be edited there. Then "Use the file as the panel reflectance" fills the keys and makes the file the source, or "Copy the values to the typed list" only copies the computed numbers to the typed values. Only the controls of the chosen source can be edited on the page.
- **Record.** The result file and the provenance record keep the reflectance used for each band, the file, its SHA-256, the format, and the window, samples and covered part of each band.
- From Python: ```panel_spectrum.load_spectrum(path, format)``` and ```panel_spectrum.band_reflectances(spectrum, windows, min_coverage)```; ```radiometric_workflow.panel_reflectance_levels(config, root)``` gives the levels that the workflow uses.

## Where things are saved

- **Configuration** (```radiometric_inputs.json```): the values that differ from the defaults, in the project folder (or the file given with ```--config```). "Save configuration" in the window, and "Run", write it.
- **Profiles** (```profiles/*.json```): "Save profile..." writes every value of the window to a file in the ```profiles``` folder of the project (it can be saved anywhere); "Load profile..." replaces all the values with those of a file. A profile is complete, so it does not depend on the defaults. This follows the profiles of the photogrammetric workflow window, which uses a ```profiles``` folder next to its code and keeps the working values in ```config_user_inputs.json```.
- **Panel regions drawn by hand**: ```panel_regions.json``` in the results folder, or the file in ```PANEL.manual_regions_file```.
- **Results**: the reflectance images, ```<capture>_provenance.json``` and ```radiometric_result.json``` in the results folder.

## Checking, accepting and drawing the panel regions

The panel region of each band can be checked, accepted and, when it is wrong or the QR code cannot be read, drawn with ```panel_selector.py```. It is opened with "Check / draw panel regions..." on the Inputs page (next to the panel captures and the file where the regions are kept), or with ```open_panel_selector.cmd``` (```python panel_selector.py --config radiometric_inputs.json --root <folder>```). The "Check inputs" button next to it checks the folders, the panel capture and the bands first, without leaving the page, and the tool is not opened while there are problems.

- **Overview.** The window opens on one tile per band: the image around the panel, the region found from the QR code (yellow, dashed), the region that is accepted or saved (green) or proposed (cyan), and the pixels, the saturated pixels and the variation (std / mean; lower is more uniform) inside it. "Accept" takes the region of a tile; "Accept all automatic regions" accepts the regions found from the QR code of all the bands that are not accepted yet, so nothing has to be drawn when they are right; "Edit..." opens the band in the editor. The regions saved before are shown as accepted, so the overview is also the way to look at the regions that are in the file.
- **One kind of acceptance.** Every band has to be accepted, so the region used for the reference panel is always one that you saw and confirmed: a region found from the QR code is not used silently. "Save and close" is available when all the bands are accepted, and writes the polygon of every band (vertices in pixels of the undistorted image) with where it came from (```accepted_automatic``` if it is the region found from the QR code, ```edited``` if it was moved or drawn) in ```PANEL.manual_regions_file```, or, if it is empty, in ```panel_regions.json``` in the results folder. Nothing is saved if the window is cancelled.
- **Editor.** The polygon starts from the region that is shown and can be edited: a click adds a vertex (on an edge it inserts one), dragging moves a vertex, a right click removes it, the wheel zooms and the middle button drags the image. "Confirm" (Enter) accepts the polygon for the band and goes to the next band that is not accepted (or back to the overview); "Back to overview" (Esc) leaves the band without accepting what was changed; "Reset to automatic" (R), "Clear" (C), Backspace (last vertex) and Left/Right (other bands) are also available. The region accepted or saved before is drawn in green for comparison.
- **Captures.** The window opens the panel captures only (the first one and the one after the flight, when there is one). The "Capture" selector (top right) switches between them and shows how many bands of each are accepted. Images are read when a capture is opened, and what was accepted is kept for each capture while switching. "Save and close" needs every band of the panel captures accepted.
- **What each element does** is shown at the bottom of the window while the mouse is over it (buttons, images, tiles and the capture selector); there is no fixed legend. The area has the width of the window and the height of the longest explanation at that width, so pointing at an element never changes the size of the image.
- A band that has a region in the file uses it, the others are detected from the QR code when the workflow runs, and a region from the file does not need a QR code. The panel audit and the provenance record say, for each band, whether the region was ```automatic``` or ```manual``` (from the file) and its ```region_origin```.
- From Python: ```rct.Image.detect_reference_panel(levels, percent_inner_area, manual_bounds=polygon, region_origin=origin)``` and ```panel_selector.select_panel_regions(images, saved, saved_origins=origins)```, which returns the accepted polygons as a dictionary with an ```origins``` attribute. The logic of the window (```PolygonModel```, ```ViewTransform```, ```SelectionSession```, ```render_tile```) has no window and is tested without a display.

## Launchers (Windows)

Double-click one of these files in the project folder. They use the Python 3.10 environment at the top of each file (```PYTHON_DIR```; edit it if yours is elsewhere), and the project folder as ```--root```, so the relative paths of the configuration (for example ```images/raw```) are resolved from there.

- ```open_radiometric_window.cmd```: the window, without a console (like the launcher of the photogrammetric workflow).
- ```open_radiometric_window_debug.cmd```: the same window with the console kept open, to see messages and error traces while testing.
- ```open_panel_selector.cmd```: only the dialog to draw the panel regions, using ```radiometric_inputs.json``` in the project folder if it exists, otherwise the defaults. Extra arguments are passed on (for example ```--capture IMG_0005```).

## Window

**Choosing captures and reading the file names.** The reference panel capture and the second panel capture are chosen in a list of the captures found in the images folder (or typed); "Select..." next to "Select captures" opens a list in which any number of them can be selected (the panel captures are not offered; the field then shows the names, and when it is empty all the other captures of the folder are corrected, which the line under the field spells out). Under the file name pattern the window shows how the pattern reads the names (```{capture}_{index}.tif```), how many files of the folder it matches, how many captures and files per capture it finds, and an example of how the first files are read; "Suggest from files" writes a pattern from the names of the files in the folder (names such as ```IMG_0005_1.tif```: capture name, a separator, the number of the file, the extension). The pattern is a library function too (```suggest_file_pattern```, ```preview_file_pattern```, ```pattern_template```).

```
python radiometric_window.py [--config radiometric_inputs.json] [--root <folder>]
```

**Layout.** Large tiles on the left choose the page (Inputs, Reference panel, DLS, Run and results), the fields are in the centre and a column on the right explains them: it shows the explanation of the field under the mouse pointer or, when the pointer is elsewhere, of the field that has the keyboard focus (Tab moves between fields), and the explanation of the page when there is neither. A field that belongs to a part with its own note (for example the limits of the DLS checks) shows that note under its own explanation, and so do the headings. The explanations are the ```label``` and ```help``` of each entry of ```FIELDS``` and ```GROUP_NOTES```; results and messages (the line under the captures, the preview of the file name pattern, the state of the inputs) stay next to their fields.

The pages (Inputs, Reference panel, DLS) are built from ```radiometric_workflow.FIELDS```, so a new key appears in the window without changes to it; the pages scroll when they are longer than the window. The bands of the camera are edited in a table (see above). A value that cannot be read is marked in red. "Save configuration" writes only the values that differ from the defaults (```radiometric_inputs.json``` in ```--root``` by default).

The last page runs the workflow in a separate thread, so the window stays responsive: "Check inputs" lists every problem, "Check / draw panel regions..." (on the Inputs page, next to the panel captures and a "Check inputs" button) opens the tool above (and sets ```PANEL.manual_regions_file``` if it was empty), "Run" saves the configuration and shows the progress, the flags of the checks, the errors and the result file, and "Open results folder" opens the folder in the file explorer.

- Load and Save profile (see above) are in the bottom bar. ```radiometric_window.RadiometricFrame``` builds its widgets in a ```ttk.Frame``` (```.widget```), so it can be placed in any window, for example as a page of another program's window. ```ui_theme.py``` holds the colors and styles, which follow the dark window of the photogrammetric workflow.
- ```ConfigModel``` (the text typed in each field, its conversion and the file) has no window and is tested without a display.

## Tests

```
python -m unittest test_radiometric_correction_tools test_radiometric_workflow test_panel_selector test_panel_spectrum test_panel_spectrum_dialog test_camera_presets test_ui_widgets test_radiometric_window -v
```

The tests use the example images and need ExifTool (set the environment variable ```exiftoolpath``` if it is not in the PATH).

For more details about the steps required for pre-processing Micasense multispectral images see the tutorials (e.g. ['Tutorial 1'](https://github.com/micasense/imageprocessing/blob/master/MicaSense%20Image%20Processing%20Tutorial%201.ipynb)) provided by Micasense in their repository.
