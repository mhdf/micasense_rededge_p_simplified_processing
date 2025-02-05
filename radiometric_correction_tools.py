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
        self.size = self.metadata['EXIF:ImageWidth'], self.metadata[ 'EXIF:ImageHeight' ]
        self.bits_per_pixel = self.metadata[ 'EXIF:BitsPerSample' ]
        self.band_name = self.metadata[ 'XMP:BandName' ]

        # Parameters for camera matrix calculation
        self.principal_point = [ float(val) for val in self.metadata[ 'XMP:PrincipalPoint' ].split(',') ]
        self.focal_plane_resolution_px_per_mm = [ self.metadata[ 'EXIF:FocalPlaneXResolution' ] ]*2
        self.focal_length = self.metadata[ 'EXIF:FocalLength' ]
        self.distortion_parameters = np.array([ float(val) for val in \
                                                self.metadata[ 'XMP:PerspectiveDistortion' ] ])[[0, 1, 3, 4, 2]]
        self.camera_matrix = cv2_camera_matrix(self.principal_point, self.focal_length , self.focal_plane_resolution_px_per_mm)
        
        # Pixel-wise polynomial 2D vignetting coefficients, according to functions provided by Micasense (vignette_map)
        self.k = list( map(float, self.metadata[ 'XMP:VignettingPolynomial2D' ].split(',')) ) # pol2DCoeff      
        self.e = list( map(float, self.metadata[ 'XMP:VignettingPolynomial2DName' ].split(',')) ) # pol2DExponents

        # Parameters for radiometric calibration - radiometric sensitivity
        self.a1, self.a2, self.a3 = [ float(val) for val in self.metadata[ 'XMP:RadiometricCalibration' ] ]

        # Dark current pixel values
        self.black_levels = np.array( [float(val) for val in self.metadata[ 'EXIF:BlackLevel' ].split(' ')] )

        # Exposure time & gain (gain = ISO/100)
        self.exposure_time = float( self.metadata[ 'EXIF:ExposureTime' ])
        self.gain          = float( self.metadata[ 'EXIF:ISOSpeed' ] ) / 100.0


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

