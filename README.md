# Radiometric Correction for Micasense RedEdge-P

This Python module provides a simplified alternative for converting raw images from the Micasense RedEdge-P sensor into radiance or reflectance, using the reference panel and, optionally, the Downwelling Light Sensor (DLS) data to correct changes in illumination.

All functions are derived from the original [Micasense image processing repository](https://github.com/micasense/imageprocessing) but have been streamlined to mitigate issues caused by Exif metadata inconsistencies. These issues were encountered while pre-processing images captured with the camera firmware v1.4.5 in preparation for geometric correction using photogrammetry software.

For processing, it is only required the path and metadata for the images and for the reference panel. Images metadata can be extracted for example using [ExifTool](https://exiftool.org/). The only caution needed here is to maintain the same field labels adopted in the original Exif information.

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

## Drawing the panel region by hand

When the QR code is not found, or the automatic region is not on the panel, the region can be drawn for each band with ```panel_selector.py```:

```
python panel_selector.py --config radiometric_inputs.json --root <folder>
```

The bands of the panel capture are shown one after another. The polygon starts from the region found from the QR code (also drawn as a yellow dashed line) or from a region saved before, and it can be edited: a click adds a vertex (on an edge it inserts one), dragging moves a vertex, a right click removes it, the wheel zooms and the middle button drags the image. The pixels, the saturated pixels and the variation (std / mean) inside the region are shown while editing; a lower variation means a more uniform panel. Keys: Enter confirms, S uses the automatic detection for the band, R goes back to the automatic region, C clears, Backspace removes the last vertex, Left/Right change the band, Escape cancels.

- "Save and close" is available when every band was confirmed or set to the automatic detection. The regions are saved (vertices in pixels of the undistorted image) in ```PANEL.manual_regions_file``` or, if it is empty, in ```panel_regions.json``` in the results folder; set the key to that file to use it. Nothing is saved if the dialog is cancelled.
- A band with a region in the file uses it, the others are detected from the QR code, and a manual region does not need a QR code. The panel audit and the provenance record say, for each band, whether the region was ```automatic``` or ```manual```.
- From Python: ```rct.Image.detect_reference_panel(levels, percent_inner_area, manual_bounds=polygon)``` and ```panel_selector.select_panel_regions(images)```. The logic of the dialog (```PolygonModel```, ```ViewTransform```, ```SelectionSession```) has no window and is tested without a display.

## Tests

```
python -m unittest test_radiometric_correction_tools test_radiometric_workflow test_panel_selector -v
```

The tests use the example images and need ExifTool (set the environment variable ```exiftoolpath``` if it is not in the PATH).

For more details about the steps required for pre-processing Micasense multispectral images see the tutorials (e.g. ['Tutorial 1'](https://github.com/micasense/imageprocessing/blob/master/MicaSense%20Image%20Processing%20Tutorial%201.ipynb)) provided by Micasense in their repository.
