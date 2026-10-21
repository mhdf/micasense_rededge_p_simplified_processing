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
# capture and each image? (False = panel only). The Panchro band is always converted with the panel only.
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
band_records = []
for band in bands: # Iterate over bands

    # Reference panel image (path and metadata of the file of this band)
    reference_panel_obj = rct.Image( *reference_panel_files[band] )

    # Identify automatically the coordinates for the reference panel in the unidistorted image and
    # calculate the factor used to convert radiance to reflectance
    panel_bounds, radiance_to_reflectance_factor, _, _ = \
            reference_panel_obj.detect_reference_panel( reflectance_level_reference_panel,
                                                        percent_inner_area=0.7, sat_threshold=65000 )
    print( reference_panel_obj.panel_audit )  # what was used to calibrate this band

    # Image to be processed
    image_obj = rct.Image( *image_files[band] )

    # Reflectance image
    dls_panel = reference_panel_obj if (use_dls and band != 'Panchro') else None
    image_reflectance = image_obj.digital_numbers_to_reflectance( radiance_to_reflectance_factor,
                                                                  panel_image=dls_panel )

    ##
    # Save reflectance image
    output_path = root_path / results_folder / Path(image_files[band][0]).name
    tifffile.imwrite( output_path, image_reflectance )  # keeps the float values (no rescaling to integers)
    band_records.append( rct.band_record(output_path, reference_panel_obj, image_obj, image_reflectance) )

# Record of what was used to produce the results, saved with them
rct.write_provenance( root_path / results_folder / (image_to_process_name + '_provenance.json'), band_records,
                      parameters={ 'reflectance_level_reference_panel': reflectance_level_reference_panel,
                                   'percent_inner_area': 0.7, 'sat_threshold': 65000, 'use_dls': use_dls } )
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
- The Panchro band is not corrected, because no DLS channel is assumed to match it (```rct.dls_irradiance_ratio()``` raises an error for it).
- This correction does not account for the DLS orientation or shading, so inspect the irradiance values of your flight before relying on it.

## Tests

```
python -m unittest test_radiometric_correction_tools -v
```

The tests use the example images and need ExifTool (set the environment variable ```exiftoolpath``` if it is not in the PATH).

For more details about the steps required for pre-processing Micasense multispectral images see the tutorials (e.g. ['Tutorial 1'](https://github.com/micasense/imageprocessing/blob/master/MicaSense%20Image%20Processing%20Tutorial%201.ipynb)) provided by Micasense in their repository.
