"""
The radiometric correction program: the configuration and the runner (radiometric_workflow), the window
(radiometric_window), the tools that it opens (panel_selector, capture_picker, panel_spectrum_dialog) and what they share
(ui_theme, ui_widgets, camera_presets, panel_spectrum). radiometric_correction_tools, the library that does the
radiometric work, stays next to this folder, so that it keeps the name that other programs import.

Run from the project folder (the one with radiometric_correction_tools.py):

    python -m radiometric_app.radiometric_window [--config file] [--root folder]
    python -m radiometric_app.radiometric_workflow --config config/radiometric_inputs.json
    python -m radiometric_app.panel_selector [--config file] [--root folder]
"""
