# Radiometric Correction for Micasense RedEdge-P

This Python module provides a simplified alternative for converting raw images from the Micasense RedEdge-P sensor into radiance or reflectance, using (only) reference panel data (no Downwelling Light Sensor -- DLS data considered).

All functions are derived from the original [Micasense image processing repository](https://github.com/micasense/imageprocessing) but have been streamlined to mitigate issues caused by Exif metadata inconsistencies. These issues were encountered while pre-processing images captured with the camera firmware v1.4.5 in preparation for geometric correction using photogrammetry software.

For processing, it is only required the path and metadata for the images and for the reference panel. Images metadata can be extracted for example using [ExifTool](https://exiftool.org/). The only caution needed here is to maintain the same field labels adopted in the original Exif information.

An example of how the pre-processing would work in practice is presented below.

First, lets import the required libraries and module.

```
# Standard library imports
import os
import glob

# Third-party library imports
from pathlib import Path
from skimage import io

# Module for radiometric correction
#  Set the root directory for your working environment first
root_path = Path.cwd()
# Import module
import radiometric_correction_tools as rct

# Handle optional exiftool
try:
    import exiftool
except ImportError:
    exiftool = None
exiftoolPath = os.environ.get('exiftoolpath')
```

Next, it is necessary to provide some information required for processing, including: (i) name and paths for the images with the reference panel, (ii) name and paths for the images to be transformed from raw Digital Numbers (DNs) to reflectance and (iii) reflectance levels for different bands in the reference panel, a value provided by Micasense with each reference panel.

```
# Name and paths for the image with the reference panel
reference_panel_name = 'IMG_0005'
reference_panel_paths = glob.glob( str( root_path / images_folder / (reference_panel_name + '*.tif')), 
                                   recursive=True )[0]

# Name and paths for the image to be corrected
image_to_process_name = 'IMG_0010'
image_paths = glob.glob( str(root_path / images_folder / (image_to_process_name + '*.tif')), 
                         recursive=True )[0]

# Reflectance level for the different bands in the reference panel - 
# these valause are measured and provided by Micasense, marked in the reference panel, 
# but they can also be requested via the Micasense technical support.
# obs.: be sure that the sequence of the bands here match the sequence registered by the camera, 
# for example, band 1 should be blue, etc.
reflectance_level_reference_panel = { "Blue": 0.477, "Green": 0.478, "Panchro": 0.479, \
                                      "Red": 0.478, "Red edge": 0.478, "NIR": 0.477 }
```

Finally, lets iterate over the different bands saving images after transforming them from DNs to radiance and at the end to reflectance.

```
#
for ii, band in enumerate( list(reflectance_level_reference_panel.keys()) ): # Iterate over bands

    ##
    # Reference panel metadata
    with exiftool.helper.ExifToolHelper() as et:
        reference_panel_metadata  = et.get_metadata( reference_panel_paths[ii], params=['-b', '-F'] )[0]
 
    # Reference panel image
    reference_panel_obj = rct.Image( reference_panel_paths[ii], reference_panel_metadata )

    # Identify automatically the coordinates for the reference panel in the unidistorted image and 
	# calculate the factor used to convert radiance to reflectance
    panel_bounds, radiance_to_reflectance_factor, _, _ = \
            reference_panel_obj.detect_reference_panel( reflectance_level_reference_panel,
                                                        percent_inner_area=0.7, sat_threshold=65000 )

    ##  
    # Image metadata
    with exiftool.helper.ExifToolHelper() as et:
        image_metadata = et.get_metadata(image_paths[ii], params=['-b', '-F'])[0]
    
    # Get image to be processed
    image_obj = rct.Image( image_paths[ii], image_metadata )

    # Reflectance image
    image_reflectance = image_obj.digital_numbers_to_radiance() * radiance_to_reflectance_factor

    ##
    # Save reflectance image
    plugin = 'tifffile'  # alternative: 'imageio'
    io.imsave( root_path / results_folder / (image_to_process_name + '_' + str(band) + '.tif'),
               image_reflectance,
               check_contrast=False,
               plugin=plugin ) # maintain resolution
```

In summary, with the function ```rct.Image(image, metadata)``` an object with the image and associated information is created, while the function ```self.detect_reference_panel(reflectance_level_reference_panel, percent_inner_area, sat_threshold)``` detects the reference panel automatically on undistorted radiance images and calculate the conversion factor necessay to transform radiance to reflectance. The final step includes transforming the target image to radiance using the function ```self.digital_numbers_to_radiance()``` and multiplying the output by the conversion factor calculated using the reference panel.

For more details about the steps required for pre-processing Micasense multispectral images see the tutorials (e.g. ['Tutorial 1'](https://github.com/micasense/imageprocessing/blob/master/MicaSense%20Image%20Processing%20Tutorial%201.ipynb)) provided by Micasense in their repository.
