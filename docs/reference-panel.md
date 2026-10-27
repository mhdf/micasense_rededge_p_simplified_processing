# The reference panel: reflectance and two panel captures

[Back to the README](../README.md)

<div align="justify">

## Panel reflectance from its spectral response file

&emsp;&emsp;Instead of typing the reflectance of each band, it can be computed from the spectral response of the panel: a .csv file with the reflectance by wavelength (`PANEL.reflectance_source` = `spectral_file`, `PANEL.spectral_file`). MicaSense provides this file for its panels (together with the values printed on the panel; it can also be requested from their technical support). The reflectance of a band is the mean of the curve over its spectral window, the band center +/- half of its bandwidth (from the band table of the UI). The weighting is uniform, an approximation of the real spectral response of the filters. With the example panel it gives:

| Band | Blue | Green | Red | Red edge | NIR | Panchro |
|---|---|---|---|---|---|---|
| Computed from the file | 0.4766 | 0.4783 | 0.4785 | 0.4777 | 0.4769 | 0.4775 |
| Printed on the panel | 0.477 | 0.478 | 0.478 | 0.478 | 0.477 | 0.479 |

- **Format.** The file is read with the format given in the `PANEL.spectral_*` keys: delimiter, header and footer rows to skip, comment prefix, lines to exclude, range of wavelengths, columns, units (`nm` or `um`; `fraction` or `percent`), a multiplier and the decimal mark. The UI detects them (see below).
- **Samples that are not used.** A reflectance outside (0, 1] (the example file has zeros below 400 nm) is not used. A band window that goes beyond the end of the file (the NIR window of the example, up to 870 nm, against a file that ends at 848 nm) is averaged over the part that exists, as long as at least `PANEL.spectral_min_coverage` (0.5 by default) of it has samples; otherwise the band is listed in the error.
- **In the UI.** "Browse..." on the Reference panel page chooses the file and opens a dialog with its format already detected ("Detect format" guesses it again). The dialog shows what was read from each line, the curve with the window of each band, and the reflectance, samples and covered part of each band, updated as the options change. "Use the file as the panel reflectance" makes the file the source; "Copy the values to the typed list" only copies the computed numbers. Cancelling leaves everything as it was, and "Preview and format..." reopens the dialog for the file already chosen.
- **Record.** The result file and the provenance record keep the reflectance used for each band, the file with its **SHA-256** and the samples and covered part of the window of each band. SHA-256 is a checksum: a 64-character fingerprint calculated from the content of the file, which changes if the file changes by even one byte, so the record shows exactly which file was used.
- From Python: `panel_spectrum.load_spectrum(path, format)` and `panel_spectrum.band_reflectances(spectrum, windows, min_coverage)`; `radiometric_workflow.panel_reflectance_levels(config, root)` gives the levels that the workflow uses.

![The dialog that imports the spectral response of the panel: what was read from each line, the curve with the window of each band and the reflectance of each band](screenshots/import_spectrum.png)

## Two panel captures (before and after the flight)

&emsp;&emsp;A second panel capture (`INPUTS.reference_panel_capture_after`) is always compared with the first: the panel factor of every band, and the DLS irradiance relative to the irradiance estimated from the panel, must not change by more than `CHECKS.panel_drift_tolerance` (5 %), or a flag is raised. `PANEL.panels_used` (the "Panel selection approach" of the UI) says whether the second capture also calibrates the images:

- `first` (default): only the first panel calibrates the images; the second is only compared.
- `last`: only the second panel, the one after the flight, calibrates the images; the first is only compared.
- `mean`: the two panels weigh the same (0.5 each).
- `interpolate_time`: each image is weighted by its time between the two panel captures (`EXIF:DateTimeOriginal`): 0 at the first panel, 1 at the second; an image outside them uses the nearest panel.
- `nearest_time`: each image uses, entirely, the panel captured closest in time to it (the first one if both are equally close).
- `nearest_illumination` (needs the DLS): each image uses, entirely, the panel whose light was the most like its own (the first one if both are equally like it).
- `interpolate_illumination` (needs the DLS): both panels are used, weighted by how much their light resembles that of the image: the weight of the second panel is `w = d1^2 / (d1^2 + d2^2)`, where `d1` and `d2` are the illumination distances (see below) from the image to the first and to the second panel. A panel with the light of the image counts alone, equal distances weigh 0.5, and the weight changes smoothly, so neighbouring images do not jump between two panels that disagree (the step that `nearest_illumination` can make where its choice changes, as large as the disagreement of the two panels).

&emsp;&emsp;The example images include a second panel capture, `IMG_0422`, taken after the flight, to try these options.

### The illumination distance

&emsp;&emsp;The light of a capture is its DLS irradiance in each band (`XMP:HorizontalIrradiance`), which carries both the level of the light and how it is spread over the bands. The distance between the light of two captures is calculated band by band: the natural logarithm of the ratio of the irradiances, then the root mean square over the bands:

```
distance = sqrt( mean over the bands of ( ln(E_capture / E_panel) )^2 )
```

&emsp;&emsp;A distance of 0 means the same light, and 0.1 is about a 10 % difference (if the irradiance of every band is 10 % higher than at the panel, each logarithm is ln(1.1) = 0.095 and so is the distance). The capture is compared as a whole, so the same weight is used in every band, which keeps the relation between the bands.

&emsp;&emsp;The options by illumination need the DLS data in the images and in both panel captures, and `CORRECTION.use_dls` on. In the UI, choosing one of them ("Nearest panel in light (uses the DLS)" or "Both panels, weighted by light (uses the DLS)") turns the DLS correction on, and turning it off again goes back to `first`; the check of the inputs and a run refuse them without the correction. The distances of each capture are written in the result file (`light_difference`) and in its provenance record (`checks.panel_light_difference`). A capture whose light differs by more than `CHECKS.panel_illumination_tolerance` (0.15, about 15 %) from the nearest panel is flagged, because no panel was taken in light like its own and another panel capture would help. The summary at the end of a run says how many captures used the first panel alone, the second alone and both.

### How the panels are combined

&emsp;&emsp;Without the DLS, the panel factors are averaged with these weights. With the DLS, what is averaged is the factor times the irradiance of each panel, expressed for the irradiance of the first panel, which the DLS ratio refers to: `factor = ((1 - w) f1 E1 + w f2 E2) / E1`, where `f1`, `f2` are the factors and `E1`, `E2` the irradiances of the two panels and `w` the weight of the second. The weight and the factor used are written in the provenance record of each band (`second_panel_weight`, `effective_factor`) and the weight of each capture in the result file. The regions of both panel captures are accepted in the same tool, and both must be accepted before saving.

### With the DLS, which panel is "the reference"?

&emsp;&emsp;A panel capture calibrates the camera for the light of that moment. Without the DLS this is only right while the light stays as it was, so the choice of panel (or of a mix of both) matters directly. With the DLS, each image is also scaled by the change in irradiance since the panel capture, so what a panel provides is the product of its factor and the irradiance measured at that moment, `f x E`. This product is a constant of the camera that does not depend on the light, so it should be the same for either panel, and it does not matter which panel the irradiance ratio is written relative to. What matters is which panels contribute to the constant: the first, the nearest or a mean of both.

&emsp;&emsp;The two panels agree on the constant when the DLS followed the light seen by the camera; when they do not (for example, a cloud or a tilted DLS), `CHECKS.panel_drift_tolerance` raises a flag. When only one panel capture is used, MicaSense advises taking the one that best represents the average illumination of the flight.

### The example flight

&emsp;&emsp;The example images are three captures of a longer flight of 411 captures, which is not included in the repository: the first panel (`IMG_0005`, 13:26 UTC), one capture to correct (`IMG_0010`) and the last panel (`IMG_0422`, 13:48 UTC). Over the whole flight (409 captures in between) the irradiance rises by 32 to 35 % between the two panels, and most captures are closer in light to the last panel. With `nearest_illumination` the first panel is chosen for about 40 of them (the first two captures, a few around `IMG_0087` to `IMG_0092` and a stretch from `IMG_0256`, where the irradiance drops), and `nearest_time` would have chosen it for the whole first half of the flight. These figures come from the complete flight and cannot be reproduced with the example images alone.

### Metashape

&emsp;&emsp;How Metashape combines several panel images, and what its sun sensor option is referenced to, is not described in its manual (2.1) or in the forum threads of Agisoft that were read for this project. A Metashape workflow gives all the panel captures of the chunk to `chunk.calibrateReflectance(use_reflectance_panels, use_sun_sensor)` and cannot choose among them, so do not assume that it averages or follows the time: test it with panels of known different factors before relying on one behaviour.

</div>
