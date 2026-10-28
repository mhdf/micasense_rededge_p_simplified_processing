# How the reflectance is calculated

[Back to the README](../README.md)

<div align="justify">

&emsp;&emsp;Each pixel goes through two conversions: from the raw value written by the camera to radiance, the light that leaves the surface towards the camera, and from radiance to reflectance, the fraction of the light that the surface reflects. The first conversion uses only the metadata of the image; the second uses the reference panel and, optionally, the DLS. Both follow the method of MicaSense ([imageprocessing](https://github.com/micasense/imageprocessing), `Image.radiance()` and `Panel`).

## From the raw values to radiance

&emsp;&emsp;The raw value of a pixel is a digital number (DN), proportional to the light that reached the sensor during the exposure, plus an offset and distorted by the optics. `Image.digital_numbers_to_radiance()` corrects it in four steps, each with values written in the metadata of the image:

| Step | What it corrects | Metadata |
|---|---|---|
| 1. Dark level | The value that the sensor gives without light, subtracted from every pixel | `EXIF:BlackLevel` (mean of the values) |
| 2. Vignetting | The lens lets less light reach the edges of the sensor than its centre; each pixel is multiplied by 1 / a polynomial of its position (1 at the centre, up to about 2 in the corners) | `XMP:VignettingPolynomial2D`, `XMP:VignettingPolynomial2DName` (RedEdge-P, Altum-PT) or `XMP:VignettingCenter`, `XMP:VignettingPolynomial` (RedEdge-M/MX, Altum) |
| 3. Row gradient | The sensor is read row by row, so the rows further down are multiplied by a small correction | `XMP:RadiometricCalibration` (a2, a3), `EXIF:ExposureTime` |
| 4. Scale to radiance | Divided by gain x exposure time and by 2 to the bits per pixel, multiplied by the calibration coefficient a1 | `EXIF:ISOSpeed` (gain = ISO / 100), `EXIF:ExposureTime`, `EXIF:BitsPerSample`, `XMP:RadiometricCalibration` (a1) |

```
radiance = vignetting(x, y) x row_gradient(row) x (DN - dark level) / (gain x exposure time) x a1 / 2^bits
```

&emsp;&emsp;In the vignetting polynomial, `x` is the position across the width of the image (0 at the left, 1 at the right) and `y` down its height (0 at the top, 1 at the bottom). The radiance is in W/m²/sr/nm. Negative values, which come from noise around the dark level, become 0.

&emsp;&emsp;Older cameras (RedEdge-M/MX, Altum) write a radial model instead: the polynomial is of `r`, the distance in pixels from the centre of the vignetting, `1 + c1 r + c2 r² + ...`. The program uses the model that the image has. The radial model was checked on the RedEdge-MX sample captures published by MicaSense (the same radiance as MicaSense's code), not on flights of that camera. As in MicaSense's code, an exposure of 1/6329 s written by some old RedEdge firmware is read as 0.274 ms, and the DLS irradiance of the RedEdge and RedEdge-M/MX before firmware 5.1.7 (Altum before 1.2.3) is not used.

## From radiance to reflectance

&emsp;&emsp;The reference panel has a known reflectance in each band. Its mean radiance, measured in the panel region of the panel capture (the inner part of the panel, without saturated pixels), gives the factor that converts radiance into reflectance for the light of that moment:

```
panel factor = panel reflectance / mean radiance of the panel
reflectance  = radiance x panel factor                      (panel only)
reflectance  = radiance x panel factor x E_panel / E_image  (with the DLS)
```

&emsp;&emsp;`E_panel` and `E_image` are the irradiances estimated by the DLS at the panel capture and at the image (see [Optional DLS correction](dls-correction.md)). With two panel captures, each panel gives its own factor (and, with the DLS, its own ratio), and the results are averaged with the weights of the [panel selection approach](reference-panel.md#how-the-panels-are-combined). The factor applied to each band of each capture is saved in its provenance record (`factor_applied`).

## Geometry of the images

&emsp;&emsp;The lens also distorts the geometry of the image (straight lines look curved towards the edges). The panel is found and measured on the undistorted image, where its QR code and its square shape are reliable. The output images keep the original, distorted geometry of the camera, like the raw images, because photogrammetry software (Metashape, for example) corrects the distortion itself when it builds the orthomosaic.

</div>
