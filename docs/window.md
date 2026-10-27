# The window

[Back to the README](../README.md)

The window has four pages, chosen with the tiles on the left, a column on the right that explains the field under the pointer, and a bottom bar with the saved settings. Open it with ```open_radiometric_window.cmd``` (see [Installing and launching](installation.md#launchers-windows)) or, from the project folder:

```
python -m radiometric_app.radiometric_window [--config config/radiometric_inputs.json] [--root <folder>]
```

The pictures below were taken on the example images, with the DLS on.

## Layout

Tiles on the left choose the page (Inputs, Reference panel, DLS, Run and results), the fields are in the centre and a column on the right explains them. The column shows the explanation of the field under the mouse pointer or, when the pointer is elsewhere, of the field that has the keyboard focus (Tab moves between fields), and the explanation of the page when there is neither; while the pointer is over the column the explanation stays, so that it can be read and scrolled. The line between the fields and the column can be dragged to change their widths, and the text follows the width of the column. A field that belongs to a part with its own note (for example the limits of the DLS checks) shows that note under its own explanation, and so do the headings. Information that depends on the data is also shown there, under the explanation: how the file name pattern reads the images folder, which captures will be corrected and what the spectral file gives in each band. What is wrong (a pattern that matches nothing, a capture that is not in the folder, a spectral file that cannot be read) is written under the field, in red, and only while it is wrong; the state of the inputs stays next to the buttons that check them. A value that cannot be read is marked in red, and the pages scroll when they are longer than the window.

## Page 1: Inputs

Where the images are, which captures show the reference panel and which captures are corrected.

![The Inputs page](screenshots/window_inputs.png)

- **Folders and tools**: the images folder, the results folder and the ExifTool executable (only needed when it is not found by itself, see [ExifTool](installation.md#exiftool)). Browse... next to the two folders opens a window that lists the folder (folders first, then the files with their sizes; Up, Home, Refresh, New Folder..., and an address box that takes a typed path) and shows, at the right, a picture of the image file that is selected, with its size in pixels and the type of its values: the raw images of a flight are recognised by looking at them, and the same goes for the reflectance images of a results folder. The picture is for the eye only (stretched between the 1 % and 99.5 % percentiles of the image, so dark raw images show; the file is not changed) and is read in the background, so a folder with thousands of files stays quick to move in. "Select Folder" takes the folder that is selected in the list, or else the one that is open (```radiometric_app/folder_browser.py```; the other Browse... buttons, for files, still use the file dialog of the system).
- **Reference panel**: the capture that shows the panel and, optionally, a second one taken after the flight (see [two panel captures](reference-panel.md#two-panel-captures-before-and-after-the-flight)). "Select..." opens a window with one picture for each capture of the images folder, so that the capture that shows the panel can be recognised by looking at it (the second panel capture can also be left empty there). A box chooses the band shown (the bands of the table; the position of each band among the files is read from the metadata of the first capture) and the pictures are read in the background, so the window opens at once. The names can still be typed.
- **Images to correct**: "Select..." opens the same window to choose the captures to correct by clicking (several), with "Select all" and "Clear". The panel captures are not offered among them; the field shows the names chosen, and when it is empty all the other captures of the folder are corrected (the help column says which). The file name pattern is explained in the help column with how it reads the images folder: how many files match, in how many captures, and an example of how the first files are read; "Suggest from files" writes a pattern from the names of the files in the folder (names such as ```IMG_0005_1.tif```: capture name, a separator, the number of the file, the extension). The pattern is a library function too (```suggest_file_pattern```, ```preview_file_pattern```, ```pattern_template```); other naming schemes are described in [Other cameras](other-cameras.md).
- **Python environment**: shows the Python that is running the window and the one for the next launches (the field, with Browse...). A running window cannot change its own Python, so this is the one that the dialog at the start proposes and the launchers use next (see [Choosing the Python](installation.md#choosing-the-python)): "Check this Python" runs it (in another thread) and lists its version, whether it can open a window and which packages and ExifTool it has, with the ```pip install``` command for that Python for what is missing, and "Use it for the next launches" tries it as the dialog does and saves it in ```config/python_environment.json```; it is not part of the configuration or of the profiles.

| The folder browser, with the picture of the selected image | The capture picker |
|---|---|
| ![The folder browser](screenshots/folder_browser.png) | ![The capture picker](screenshots/capture_picker.png) |

## Page 2: Reference panel

The bands of the camera, the known reflectance of the reference panel and how the panel is measured in its image.

![The Reference panel page](screenshots/window_reference_panel.png)

- **Bands of the camera**: the table of the bands (name, center and bandwidth, panel reflectance), with the buttons to add a band, to detect the bands from a capture and to fill them from a camera preset. See [Other cameras](other-cameras.md).
- **Reflectance of the panel**: where the value of each band comes from, the values typed in the table or computed from the spectral response file of the panel ("Browse..." chooses it and opens the import dialog; "Preview and format..." opens it for the file that is already chosen). See [Panel reflectance from its spectral response file](reference-panel.md#panel-reflectance-from-its-spectral-response-file).
- **Panel measurement**: the panel selection approach (which panel capture calibrates the images, see [two panel captures](reference-panel.md#two-panel-captures-before-and-after-the-flight)), the inner area of the panel that is measured and the saturation threshold.
- **Reference panel regions**: "Check inputs" tells whether the folders, the panel capture and the bands are right, and "Check / draw panel regions..." opens the tool in which the region of the panel in each band is checked, accepted or drawn (see [the panel regions tool](panel-regions.md)). The regions are kept in the file of the field above it (```PANEL.manual_regions_file```), by default ```panel_regions.json``` in the results folder (the tool fills the field in with that file when it was empty).

## Page 3: DLS

The correction with the DLS, which is optional, and the limits of the checks of its data and of the geometry of the acquisition.

![The DLS page, with the help of the field that has the focus](screenshots/window_dls.png)

- **Correction with the DLS**: the box that turns the correction on. See [Optional DLS correction](dls-correction.md).
- **Checks of the DLS data (limits)**: the limits that raise a flag when the DLS data should not be trusted (the flags never change a result). See [Checks of the DLS data](dls-correction.md#checks-of-the-dls-data).
- **Acquisition geometry (limits)**: the limits for the tilt of the camera and its heading relative to the sun. See [Geometry of the acquisition](dls-correction.md#geometry-of-the-acquisition).

## Page 4: Run and results

The last page runs the workflow in a separate thread, so the window stays responsive.

![The Run and results page after a run with the DLS](screenshots/window_run.png)

- **Check inputs** lists every problem, without running anything.
- **Check environment** lists the Python, Tkinter, the packages with their versions and ExifTool of the computer and what to install (see [The check of the computer](installation.md#the-check-of-the-computer)).
- **Run** saves the configuration and shows the progress, the flags of the checks (counted by criterion, see [the flags of a run](dls-correction.md#the-flags-of-a-run-are-counted-by-criterion)), the errors, a summary and the result file.
- **Open results folder** opens the folder in the file explorer.

## Saved settings and where things are saved

The bottom bar is the same on every page. It has three buttons ("Load profile...", "Save profile..." and "Update saved configuration") and, at the left, the configuration file in use. What is saved, and where:

- **Configuration** (```config/radiometric_inputs.json```): the values that differ from the defaults, in the ```config``` folder of the project (or the file given with ```--config```; a file ```radiometric_inputs.json``` in the project folder, where earlier versions kept it, is still used while the new one does not exist). "Update saved configuration" (it replaces the saved configuration with what is in the window; it is not tied to a project or to a results folder) and "Run" write it, so the saved configuration is always what was used in the last run, and the window opens with it the next time. It keeps only the differences from the defaults, which is why a profile, that keeps every value, is the safer way to keep a setup for a camera or a flight.
- **Profiles** (```config/profiles/*.json```): "Save profile..." writes every value of the window to a file in the ```config/profiles``` folder of the project (it can be saved anywhere); "Load profile..." replaces all the values with those of a file. A profile is complete, so it does not depend on the defaults. Running does not make a profile: what every run used is written in ```config``` of ```radiometric_result.json``` and in ```parameters.config``` of the provenance record of each capture, next to the results, and "Load profile..." reads either of those files too (```wf.configuration_from_json()```), so the settings of any past run can be brought back (the results folder is then that of the run, which the status line says; check it before running again). The command line takes them as well: ```--config <results folder>/radiometric_result.json``` repeats a run.
- **Panel regions drawn by hand**: ```panel_regions.json``` in the results folder, or the file in ```PANEL.manual_regions_file```.
- **Results**: the reflectance images, ```<capture>_provenance.json``` and ```radiometric_result.json``` in the results folder.
- **Python for the next launches**: ```config/python_environment.json```, not part of the configuration or of the profiles.
