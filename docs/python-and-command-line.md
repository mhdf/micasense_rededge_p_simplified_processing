# Python library and workflow runner

[Back to the README](../README.md)

For processing, it is only required the path and metadata for the images and for the reference panel. Images metadata can be extracted for example using [ExifTool](https://exiftool.org/). The only caution needed here is to maintain the same field labels adopted in the original Exif information.

## Example: raw images to reflectance with the Python library

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
- Each corrected capture gets its ```<capture>_provenance.json``` (the details: panel audit, DLS irradiances, checks); the whole run gets ```radiometric_result.json``` in the results folder, which says what happened and where to look: the status (```completed```, ```completed_with_errors``` or ```failed```), the errors, the flags of all the checks (those of the panel captures once, with the name of the capture, and those of the captures to correct counted by criterion with the captures listed, see [the flags of a run](dls-correction.md#the-flags-of-a-run-are-counted-by-criterion); ```flag_groups``` has the structured form), the complete configuration, the panel calibration (the reflectance of each band and where it came from, and a short audit of the panel of each band), the spikes/steps of the irradiance along the captures and, for each capture, the names of its output files and of its provenance record (in the results folder), the panels used, its errors and a summary of the DLS checks (the median ratio and the sun angle of the panel and of the image). ```outputs``` has the complete path of every output file. The command prints ```RADIOMETRIC_RESULT_JSON=<path>``` and returns 0 only for ```completed``` (1 for errors, 2 for an unreadable request or configuration file).
- A request is ```{"version": 1, "root": "<folder>", "config": {...}}```, so that another program can start the workflow as a separate process (the libraries needed here are usually not available in the Python of a photogrammetry program) and read the result file.
- From Python: ```radiometric_workflow.run(config, root, progress)```, where ```progress``` receives ```{'stage', 'message', 'current', 'total'}```.
