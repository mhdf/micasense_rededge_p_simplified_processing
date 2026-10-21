# %%
"""
RedEdge Image Class for Radiometric Correction

Simplified from the Micasense RedEdge Image Processing Library to avoid issues with missing 
Exif metadata in RedEdge-P images (frameware v1.3.1 and v1.4.5)

For the original/complete library, see:
https://github.com/micasense/imageprocessing

The Micasense imagerocessing Library is licensed under the MIT License
"""

# Standard Library Imports
import os
from pathlib import Path

# Third-Party Imports
import cv2
import matplotlib.pyplot as plt
# import matplotlib.patches as patches
import numpy as np
from pyzbar.pyzbar import decode, ZBarSymbol
from shapely.geometry import Point as PointS, LinearRing as LinearR, LineString as LineS
from shapely.geometry.polygon import Polygon as PolyS
from skimage import measure


# Define the camera matrix
def cv2_camera_matrix(principal_point, focal_length, focal_plane_resolution_px_per_mm): 
    center_x = principal_point[0] * focal_plane_resolution_px_per_mm[0]
    center_y = principal_point[1] * focal_plane_resolution_px_per_mm[1]

    # set up camera matrix for cv2
    cam_mat = np.zeros((3, 3))
    cam_mat[0, 0] = focal_length * focal_plane_resolution_px_per_mm[0]
    cam_mat[1, 1] = focal_length * focal_plane_resolution_px_per_mm[1]
    cam_mat[2, 2] = 1.0
    cam_mat[0, 2] = center_x
    cam_mat[1, 2] = center_y

    # set up distortion coefficients for cv2
    return cam_mat


# Metadata validation helpers - report the file and the field when metadata are missing or invalid
class MetadataError(ValueError):
    """ Raised when image metadata are missing or invalid, naming the file and the field """


def _metadata_value(metadata, key, image_path):
    if key not in metadata or metadata[key] is None:
        raise MetadataError(f"{image_path}: missing metadata field '{key}'")
    return metadata[key]


def _metadata_floats(metadata, key, image_path, count=None):
    """ Return a field as a list of finite floats, accepting a list or a string separated by commas/spaces """
    value = _metadata_value(metadata, key, image_path)
    if isinstance(value, str):
        value = value.replace(',', ' ').split()
    elif np.isscalar(value):
        value = [value]
    try:
        numbers = [float(val) for val in value]
    except (TypeError, ValueError):
        raise MetadataError(f"{image_path}: metadata field '{key}' is not numeric: {metadata[key]!r}")
    if len(numbers) == 0 or not np.all(np.isfinite(numbers)):
        raise MetadataError(f"{image_path}: metadata field '{key}' has no finite values: {metadata[key]!r}")
    if count is not None and len(numbers) != count:
        raise MetadataError(f"{image_path}: metadata field '{key}' has {len(numbers)} values, expected {count}")
    return numbers


def _metadata_positive(metadata, key, image_path):
    """ Return a field (unchanged) after checking that it is a single positive finite number """
    value = _metadata_value(metadata, key, image_path)
    number = _metadata_floats(metadata, key, image_path, count=1)[0]
    if number <= 0:
        raise MetadataError(f"{image_path}: metadata field '{key}' must be positive, got {value!r}")
    return value


# Map the files of one capture to their bands using the band name in the metadata (not the file order)
def map_band_files(image_paths, metadata_list, expected_bands=None):
    """
    Inputs: image paths of one capture, their metadata (same order) and, optionally, the expected band names
    Outputs: dictionary {band name: (image path, metadata)}, with exactly one file per band
    """
    if len(image_paths) != len(metadata_list):
        raise ValueError("image_paths and metadata_list must have the same length")
    band_files = {}
    for image_path, metadata in zip(image_paths, metadata_list):
        band_name = _metadata_value(metadata, 'XMP:BandName', image_path)
        if band_name in band_files:
            raise MetadataError(f"Band '{band_name}' appears twice: {band_files[band_name][0]} and {image_path}")
        band_files[band_name] = (image_path, metadata)
    capture_ids = { metadata.get('XMP:CaptureId') for _, metadata in band_files.values() }
    if len(capture_ids) > 1:
        raise MetadataError(f"Files belong to different captures (XMP:CaptureId): {sorted(map(str, capture_ids))}")
    if expected_bands is not None:
        missing = [band for band in expected_bands if band not in band_files]
        unexpected = [band for band in band_files if band not in expected_bands]
        if missing or unexpected:
            raise MetadataError(f"Band mismatch - missing: {missing}, unexpected: {unexpected}")
    return band_files


def find_capture_paths(images_folder, capture_name):
    """ Return the sorted image paths of a capture, e.g. 'IMG_0005' -> IMG_0005_1.tif, IMG_0005_2.tif, ... """
    paths = sorted(str(path) for path in Path(images_folder).glob(capture_name + '_*.tif'))
    if not paths:
        raise FileNotFoundError(f"No images named '{capture_name}_*.tif' in {images_folder}")
    return paths


def read_band_files(image_paths, expected_bands=None, exiftool_path=None):
    """ Read the metadata with ExifTool and map the files of one capture to their bands (see map_band_files) """
    import exiftool
    with exiftool.ExifToolHelper(executable=exiftool_path) as et:
        metadata_list = [ et.get_metadata(image_path, params=['-b', '-F'])[0] for image_path in image_paths ]
    return map_band_files(image_paths, metadata_list, expected_bands)


#
class Image(object):

    """
    Create an image object from the path for a micasense rededge image and corresponding metadata
    """

    def __init__(self, image_path, image_metadata):

        # Import image and store general information
        self.image_path = image_path
        self.metadata = image_metadata
        self.image = plt.imread( image_path )
        self.size = _metadata_positive(self.metadata, 'EXIF:ImageWidth', image_path), \
                    _metadata_positive(self.metadata, 'EXIF:ImageHeight', image_path)
        self.bits_per_pixel = _metadata_positive(self.metadata, 'EXIF:BitsPerSample', image_path)
        self.band_name = _metadata_value(self.metadata, 'XMP:BandName', image_path)

        # Parameters for camera matrix calculation
        self.principal_point = _metadata_floats(self.metadata, 'XMP:PrincipalPoint', image_path, count=2)
        self.focal_plane_resolution_px_per_mm = [ _metadata_positive(self.metadata, 'EXIF:FocalPlaneXResolution', image_path) ]*2
        self.focal_length = _metadata_positive(self.metadata, 'EXIF:FocalLength', image_path)
        self.distortion_parameters = np.array(
            _metadata_floats(self.metadata, 'XMP:PerspectiveDistortion', image_path, count=5) )[[0, 1, 3, 4, 2]]
        self.camera_matrix = cv2_camera_matrix(self.principal_point, self.focal_length , self.focal_plane_resolution_px_per_mm)

        # Pixel-wise polynomial 2D vignetting coefficients, according to functions provided by Micasense (vignette_map)
        self.k = _metadata_floats(self.metadata, 'XMP:VignettingPolynomial2D', image_path) # pol2DCoeff
        self.e = _metadata_floats(self.metadata, 'XMP:VignettingPolynomial2DName', image_path) # pol2DExponents
        if len(self.e) != 2 * len(self.k):
            raise MetadataError(f"{image_path}: 'XMP:VignettingPolynomial2DName' needs {2 * len(self.k)} exponents, "
                                f"found {len(self.e)}")

        # Parameters for radiometric calibration - radiometric sensitivity
        self.a1, self.a2, self.a3 = _metadata_floats(self.metadata, 'XMP:RadiometricCalibration', image_path, count=3)
        if self.a1 <= 0:
            raise MetadataError(f"{image_path}: first 'XMP:RadiometricCalibration' coefficient must be positive")

        # Dark current pixel values
        self.black_levels = np.array( _metadata_floats(self.metadata, 'EXIF:BlackLevel', image_path) )

        # Exposure time & gain (gain = ISO/100)
        self.exposure_time = float( _metadata_positive(self.metadata, 'EXIF:ExposureTime', image_path) )
        self.gain          = float( _metadata_positive(self.metadata, 'EXIF:ISOSpeed', image_path) ) / 100.0


    # Undistort the image
    def undistorted(self): 
        """ return the undistorted image from input image """
        # get the optimal new camera matrix
        new_cam_mat, _ = cv2.getOptimalNewCameraMatrix( self.camera_matrix,
                                                        self.distortion_parameters,
                                                        self.size,
                                                        1 )
        map1, map2 = cv2.initUndistortRectifyMap( self.camera_matrix,
                                                  self.distortion_parameters,
                                                  np.eye(3),
                                                  new_cam_mat,
                                                  self.size,
                                                  cv2.CV_32F)  # cv2.CV_32F for 32 bit floats
        # Compute the undistorted 16-bit image
        image_undistorted = cv2.remap(self.image, map1, map2, cv2.INTER_LINEAR)
        return image_undistorted


    # Calculate region statistics
    def region_stats(self, region, sat_threshold=None):
        """
        Provide regional statistics for an image over a region
        Inputs: image is any image ndarray, region is a skimage shape
        Outputs: mean, std, count, and saturated count tuple for the region
        """
        rev_panel_pts = np.fliplr( region )  # image coords are reversed
        w, h = self.image.shape[0:2]
        mask = measure.grid_points_in_poly((w, h), rev_panel_pts)
        num_pixels = mask.sum()
        panel_pixels = self.image[mask]
        stdev = panel_pixels.std()
        mean_value = panel_pixels.mean()
        saturated_count = 0
        if sat_threshold is not None:
            saturated_count = (panel_pixels > sat_threshold).sum()
        return mean_value, stdev, num_pixels, saturated_count


    # Convert DNs to radiance
    def digital_numbers_to_radiance(self):
        
        # Pixel-wise polynomial 2D vignetting coefficients, according to functions provided by Micasense (vignette_map)
        k = self.k
        e = self.e

        # Dark current pixel values
        dark_level = self.black_levels.mean()
        
        # Coordinate grid across image, seem swapped because of transposed vignette
        x, y = np.meshgrid( np.arange(self.size[1]), np.arange(self.size[0]) )

        # Vignette
        # meshgrid returns transposed arrays
        x = x.T
        y = y.T
        ##
        xv = x.T / self.size[1]
        yv = y.T / self.size[0]
        p2 = np.zeros_like(xv, dtype=float)
        for valT, c in enumerate(k):
            ex  = e[2 * valT]
            ey  = e[2 * valT + 1]
            p2 += c * xv ** ex * yv ** ey
        V = (1. / p2).T  #
        # plt.imshow( V )

        # apply image correction methods to raw image
        # step 1 - row gradient correction, vignette & radiometric calibration:
        
        # row gradient correction
        R = 1.0 / (1.0 + self.a2 * y / self.exposure_time - self.a3 * y)
        
        # subtract the dark level and adjust for vignette and row gradient
        # LT = VT  * RT * ( (imgT/normFactT) - (dark_level/normFactT) )
        L = V * R * ( self.image - dark_level )
        
        # Floor any negative radiance's to zero (can happen due to noise around black_level)
        L[L < 0] = 0
        # L = np.round(L).astype(np.uint16)
        # plt.imshow( L )
        
        # Apply the radiometric calibration - i.e. scale by the gain-exposure product and
        # multiply with the radiometric calibration coefficient
        # need to normalize by 2^16 for 16 bit images
        # because coefficients are scaled to work with input values of max 1.0
        dn_maximum     = float(2**self.bits_per_pixel)
        image_radiance = ( L.astype(float) / (self.gain * self.exposure_time) * self.a1 ) / dn_maximum
        # plt.imshow( image_radiance )

        #
        return image_radiance


    # Automaticaly detect the reference panel and return radiance to reflectance conversion factor
    def detect_reference_panel(self, reflectance_level_reference_panel, percent_inner_area, sat_threshold=65000):
        
        # Undistorted image
        image_undistorted = self.undistorted()
        # plt.imshow( image_undistorted ) 

        # Convert from uint16 to uint8
        gray_undistorted = (image_undistorted // 256).astype(np.uint8)
        # plt.imshow( gray )

        # Detect and decode barcodes
        barcode = decode(gray_undistorted, symbols=[ZBarSymbol.QRCODE])
        if( len(barcode) > 0 ):
            barcode = barcode[0]
            barcode_source = 'undistorted'
        else: # if barcode is not detected in the undistroted image, use the QR code from the original image
            gray = (self.image // 256).astype(np.uint8)
            barcode = decode(gray, symbols=[ZBarSymbol.QRCODE])[0]
            barcode_source = 'original'
        barcode_data = barcode.data.decode("utf-8")

        # Get bounds of the barcode
        barcode_bounds = []
        for point in barcode.polygon:
            barcode_bounds.append([point.x, point.y])
        barcode_bounds = np.asarray(barcode_bounds, np.int32)

        # Measures describing panel size and position
        if int(barcode_data[2:4]) < 3:
            # use the actual panel measures here - we use units of [mm]
            # the panel is 154.4 x 152.4 mm , vs. the 84 x 84 mm for the QR code
            # it is left 143.20 mm from the QR code 
            # use the inner 50% square of the panel
            s = 76.2
            p = 42
            T = np.array([-143.2, 0])
        elif (int(barcode_data[2:4]) >= 3) and (int(barcode_data[2:4]) < 6):
            s = 50
            p = 45
            T = np.array([-145.8, 0])
        elif int(barcode_data[2:4]) >= 6:
            # use the actual panel measures here - we use units of [mm]
            # the panel is 100 x 100 mm , vs. the 91 x 91 mm for the QR code
            # it is down 125.94 mm from the QR code 
            # use the inner 50% square of the panel
            p = 41
            s = 50
            T = np.array([0, -130.84])

        #
        reference_panel_pts = np.asarray([[-s, s], [s, s], [s, -s], [-s, -s]], dtype=np.float32) * percent_inner_area + T
        reference_qr_pts    = np.asarray([[-p, p], [p, p], [p, -p], [-p, -p]], dtype=np.float32)
        bounds = []
        costs  = []
        for rotation in range(0, 4):
            qr_points = np.roll(reference_qr_pts, rotation, axis=0)

            src = np.asarray([tuple(row) for row in qr_points[:]], np.float32)
            dst = np.asarray([tuple(row) for row in barcode_bounds[:]], np.float32)  

            # we determine the homography from the 4 corner points
            warp_matrix = cv2.getPerspectiveTransform(src, dst)

            pts = np.asarray([reference_panel_pts], 'float32')
            panel_bounds = cv2.convexHull(cv2.perspectiveTransform(pts, warp_matrix), clockwise=False)
            panel_bounds = np.squeeze(panel_bounds)  # remove nested lists

            #
            for i, point in enumerate(panel_bounds):
                mean, std, _, _ = self.region_stats( region=panel_bounds, 
                                                     sat_threshold=sat_threshold )
                bounds.append( panel_bounds.astype(np.int32) )
                costs.append(std / mean)
        
        idx = costs.index(min(costs))
        panel_bounds = bounds[idx]

        # # plot image to check
        # fig, ax = plt.subplots()
        # ax.imshow( image_undistorted )
        # polygon = patches.Polygon(panel_bounds, closed=True, edgecolor='r', facecolor='none')
        # ax.add_patch(polygon)

        ##
        # Calculate radiance for reference panel image
        image_panel_radiance = self.digital_numbers_to_radiance()
        # plt.imshow( image_panel_radiance  )

        # Create an empty mask for panel region
        panel_mask = np.zeros_like(self.image, dtype=np.uint8)

        # Fill the polygon in the mask
        cv2.fillPoly(panel_mask, [panel_bounds], 255)
        # plt.imshow( panel_mask  )

        # Extract pixel values inside the polygon
        masked_values_average = image_panel_radiance[ panel_mask == 255 ].mean()
        # print('Mean Radiance in panel region: {:1.4f} W/m^2/nm/sr'.format( masked_values_average ))

        # Specific calibrated reflectance level for the band - provided by Micasense
        panel_reflectance_level = reflectance_level_reference_panel[ self.band_name ]

        # Calculate radiance to reflectance conversion factor
        radiance_to_reflectance_factor = panel_reflectance_level / masked_values_average

        return panel_bounds, radiance_to_reflectance_factor, barcode_data, barcode_source

