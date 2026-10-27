# Radiometric Correction for Micasense Multispectral Cameras

This Python module provides a simplified alternative for converting raw images from Micasense multispectral cameras into radiance or reflectance, using the reference panel and, optionally, the Downwelling Light Sensor (DLS) data to correct changes in illumination. It was written for the RedEdge-P and tested only with it (see [What has been tested](#what-has-been-tested)). It can be used from a window (with launchers for Windows), from the command line (the workflow runner) or as a Python library.

**Contents**

- [What has been tested](#what-has-been-tested)
- [Quick start](#quick-start)
- [Documentation](#documentation)
  - [Installing and launching](docs/installation.md)
  - [The window](docs/window.md)
  - [The reference panel: reflectance and two panel captures](docs/reference-panel.md)
  - [Checking, accepting and drawing the panel regions](docs/panel-regions.md)
  - [Optional DLS correction](docs/dls-correction.md)
  - [Python library and workflow runner](docs/python-and-command-line.md)
  - [Other cameras and other numbers of bands](docs/other-cameras.md)
  - [Development notes](docs/development.md)
- [Background and credits](#background-and-credits)

<p align="center">
  <img src="docs/workflow_tools.png" width="900"
       alt="The tools of the workflow: the user interface (the set-up tools, the four pages of the window with their sections, and the saved settings: the profiles and the saved configuration) and the alternative ways to use the tools without the window, the workflow runner and the Python library.">
</p>

## What has been tested

The program is written so that it can be used with other MicaSense cameras (RedEdge-M/MX, Altum, Altum-PT, dual-camera systems): the bands, the file names and the bands that are left out are configurable, and the bands can be read from the images themselves (see [Other cameras](docs/other-cameras.md)). For the moment, however, it has been tested **only with images of the MicaSense RedEdge-P**. The other MicaSense models, and cameras of other makers (for example Sentera), have not been tested, and the presets of other models come from the documentation of the manufacturer, not from images. Use "Detect from capture" and compare the result with your own data before relying on them.

## Quick start

1. **Install** Python 3.10 or newer with Tkinter, in a dedicated environment, the packages of ```requirements.txt``` (or ```conda env create -f environment.yml```) and ExifTool, a separate program that reads the metadata of the images. The details, and what to check on a new computer, are in [Installing and launching](docs/installation.md).
2. **Open the window**: double-click ```open_radiometric_window.cmd``` (Windows) or run ```python -m radiometric_app window --root .``` in the project folder. It asks which Python to use, checks the packages and ExifTool, and opens.
3. **Correct the example images** in ```images/raw``` (```IMG_0005``` shows the reference panel and ```IMG_0010``` is the capture to correct). Go through the pages of the window, which are described in [the window](docs/window.md):
   - *Inputs*: the images folder, the results folder, the capture that shows the panel and the captures to correct.
   - *Reference panel*: the reflectance of the panel (typed, or computed from its spectral response file; the one of the example panel is in ```reference_panel_spectral_response```), then "Check / draw panel regions..." to accept the region of the panel in each band.
   - *DLS* (optional): corrects the changes in illumination between the panel capture and each image.
   - *Run and results*: "Check inputs", then "Run".
4. **Read the results** in the results folder: a reflectance image (float values) for each band, with the name of its source file (```IMG_0010_3.tif``` is the Red band before and after the correction), a ```<capture>_provenance.json``` for each capture with what was used to make it, and ```radiometric_result.json``` with the status of the run, the flags of the checks and the settings.

<p align="center">
  <img src="docs/screenshots/window_inputs.png" width="800" alt="The Inputs page of the window">
</p>

The same steps run without the window, with the [workflow runner](docs/python-and-command-line.md#workflow-runner-no-user-interface) (a command line that reads a JSON configuration) or with the [Python library](docs/python-and-command-line.md#example-raw-images-to-reflectance-with-the-python-library).

## Documentation

| Page | What is in it |
|---|---|
| [Installing and launching](docs/installation.md) | What the program needs (Python, packages, ExifTool), choosing the Python, the check of the computer and the Windows launchers |
| [The window](docs/window.md) | The four pages with screenshots, the saved settings (configuration and profiles) and where things are saved |
| [The reference panel](docs/reference-panel.md) | The reflectance of the panel, typed or computed from its spectral response file; one or two panel captures and how they are combined (by time or by illumination) |
| [Panel regions](docs/panel-regions.md) | The tool that checks, accepts or draws the region of the panel in each band |
| [Optional DLS correction](docs/dls-correction.md) | What the DLS ratio is, the checks of the DLS data and of the geometry, how the flags of a run are counted and the summary at its end |
| [Python library and workflow runner](docs/python-and-command-line.md) | The example in Python, the checks and the calibration record, and the command line |
| [Other cameras](docs/other-cameras.md) | The band table, detecting the bands from a capture, camera presets, file names, and what the program assumes about the camera |
| [Development notes](docs/development.md) | Project layout, how the window is built and the tests |

## Background and credits

All functions are derived from the original [Micasense image processing repository](https://github.com/micasense/imageprocessing) but have been streamlined to mitigate issues caused by Exif metadata inconsistencies. These issues were encountered while pre-processing images captured with the camera firmware v1.4.5 in preparation for geometric correction using photogrammetry software.

For more details about the steps required for pre-processing Micasense multispectral images see the tutorials (e.g. ['Tutorial 1'](https://github.com/micasense/imageprocessing/blob/master/MicaSense%20Image%20Processing%20Tutorial%201.ipynb)) provided by Micasense in their repository.
