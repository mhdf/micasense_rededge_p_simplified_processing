"""
Tests for ui_widgets.py: the rows of the band table without a display, the widgets with a hidden Tk root (skipped if
Tk cannot open one)

Run from the project folder with:  python -m unittest tests.test_ui_widgets -v
"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from radiometric_app import ui_widgets as uw


def row(name, center='475', width='32', reflectance='0.477', process=True):
    return {'name': name, 'center': center, 'width': width, 'reflectance': reflectance, 'process': process}


class RowsFromModelTests(unittest.TestCase):

    def test_rows_follow_the_order_of_the_bands_and_end_with_the_ignored_ones(self):
        rows = uw.rows_from_model({'Blue': '0.477', 'Green': '0.478'}, {'Blue': '475', 'Green': '560'},
                                  {'Blue': '32', 'Green': '27'}, 'LWIR, Thermal')
        self.assertEqual([r['name'] for r in rows], ['Blue', 'Green', 'LWIR', 'Thermal'])
        self.assertEqual(rows[0], row('Blue'))
        self.assertEqual([r['process'] for r in rows], [True, True, False, False])
        self.assertEqual(rows[2], row('LWIR', '', '', '', False))

    def test_a_band_missing_from_one_dictionary_still_gets_a_row_with_an_empty_value(self):
        rows = uw.rows_from_model({'Blue': '0.5', 'Odd': '0.4'}, {'Blue': '475'}, {'Blue': '32'})
        self.assertEqual([r['name'] for r in rows], ['Blue', 'Odd'])
        self.assertEqual(rows[1]['center'], '')

    def test_an_ignored_band_that_is_also_processed_is_not_repeated(self):
        rows = uw.rows_from_model({'Blue': '1'}, {'Blue': '475'}, {'Blue': '32'}, 'Blue, LWIR')
        self.assertEqual([r['name'] for r in rows], ['Blue', 'LWIR'])

    def test_rows_go_back_to_the_model(self):
        rows = [row('Blue'), row('Green', '560', '27', '0.478'), row('LWIR', '', '', '', False), row('  ', '1', '1', '1')]
        reflectance, centers, widths, ignored = uw.model_from_rows(rows)
        self.assertEqual(list(reflectance), ['Blue', 'Green'])
        self.assertEqual(centers, {'Blue': '475', 'Green': '560'})
        self.assertEqual(widths['Green'], '27')
        self.assertEqual(ignored, 'LWIR')                       # the row without a name is not written

    def test_the_round_trip_keeps_the_rows(self):
        rows = [row('Blue'), row('Green', '560', '27', '0.478'), row('LWIR', '', '', '', False)]
        self.assertEqual(uw.rows_from_model(*uw.model_from_rows(rows)), rows)

    def test_split_names(self):
        self.assertEqual(uw.split_names(' A , B,, C '), ['A', 'B', 'C'])
        self.assertEqual(uw.split_names(''), [])


class ValidateRowsTests(unittest.TestCase):

    def fields(self, rows, check=True):
        return [(number, field) for number, field, _ in uw.validate_rows(rows, check) if number is not None]

    def test_valid_rows_have_no_problems(self):
        self.assertEqual(uw.validate_rows([row('Blue'), row('LWIR', '', '', '', False)]), [])

    def test_names_must_exist_and_be_unique(self):
        self.assertEqual(self.fields([row('')]), [(0, 'name')])
        self.assertEqual(self.fields([row('Blue'), row('Blue', '560')]), [(0, 'name'), (1, 'name')])
        self.assertEqual(self.fields([row(' Blue'), row('Blue ')]), [(0, 'name'), (1, 'name')])

    def test_the_window_must_be_a_positive_number_for_a_band_that_is_used(self):
        self.assertEqual(self.fields([row('A', 'x', '0')]), [(0, 'center'), (0, 'width')])
        self.assertEqual(self.fields([row('A', '-1', 'nan')]), [(0, 'center'), (0, 'width')])
        self.assertEqual(self.fields([row('A', '', '', '', False)]), [])             # not used: nothing to check

    def test_the_reflectance_is_checked_only_when_it_is_used(self):
        self.assertEqual(self.fields([row('A', reflectance='')]), [(0, 'reflectance')])
        self.assertEqual(self.fields([row('A', reflectance='1.2')]), [(0, 'reflectance')])
        self.assertEqual(self.fields([row('A', reflectance='0')]), [(0, 'reflectance')])
        self.assertEqual(self.fields([row('A', reflectance='')], check=False), [])
        self.assertEqual(self.fields([row('A', reflectance='1')]), [])

    def test_at_least_one_band_must_be_used(self):
        problems = uw.validate_rows([row('A', process=False)])
        self.assertEqual(problems, [(None, 'process', 'at least one band must be used')])
        self.assertEqual(uw.validate_rows([]), [])                                     # an empty table has no rows to judge


class WidgetTests(unittest.TestCase):

    def setUp(self):
        import tkinter as tk
        try:
            self.root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"no display for Tk: {error}")
        self.root.withdraw()
        from radiometric_app.ui_theme import apply_dark_theme
        apply_dark_theme(self.root)

    def tearDown(self):
        try:
            self.root.destroy()
        except Exception:
            pass

    # --- ScrollablePage ---------------------------------------------------------------------------------------

    def page(self, labels, height=200):
        from tkinter import ttk
        self.root.attributes('-alpha', 0.0)             # shown (so that it gets a size) but invisible
        self.root.deiconify()
        page = uw.ScrollablePage(self.root)
        page.widget.pack(fill='both', expand=True)
        for number in range(labels):
            ttk.Label(page.body, text=f"line {number}").pack(anchor='w', pady=6)
        self.root.geometry(f'400x{height}')
        self.root.update()
        return page

    def test_a_page_taller_than_the_window_gets_a_scroll_bar_and_scrolls(self):
        page = self.page(40)
        self.assertTrue(page.scrolls())
        self.assertEqual(page.canvas.yview()[0], 0.0)
        page._on_wheel(SimpleNamespace(delta=-120))
        self.root.update()
        self.assertGreater(page.canvas.yview()[0], 0.0)
        page._on_wheel(SimpleNamespace(delta=120))
        self.root.update()
        self.assertEqual(page.canvas.yview()[0], 0.0)

    def test_a_short_page_has_no_scroll_bar_and_does_not_move(self):
        page = self.page(2, height=400)
        self.assertFalse(page.scrolls())
        page._on_wheel(SimpleNamespace(delta=-120))
        self.root.update()
        self.assertEqual(page.canvas.yview()[0], 0.0)

    def test_the_content_follows_the_width_of_the_page(self):
        page = self.page(3)
        self.root.geometry('600x400')
        self.root.update()
        self.assertEqual(page.body.winfo_width(), page.canvas.winfo_width())

    def test_the_scroll_bar_goes_away_when_the_window_grows(self):
        page = self.page(12, height=150)
        self.assertTrue(page.scrolls())
        self.root.geometry('400x900')
        self.root.update()
        self.assertFalse(page.scrolls())

    # --- BandTable ----------------------------------------------------------------------------------------------

    def table(self, rows=None):
        self.changes = 0

        def changed():
            self.changes += 1
        table = uw.BandTable(self.root, changed)
        table.widget.pack()
        table.set_rows(rows if rows is not None else [row('Blue'), row('Green', '560', '27', '0.478'),
                                                       row('LWIR', '', '', '', False)])
        self.root.update()
        self.changes = 0
        return table

    def test_rows_are_set_and_read_back(self):
        rows = [row('Blue'), row('LWIR', '', '', '', False)]
        table = self.table(rows)
        self.assertEqual(table.get_rows(), rows)

    def test_setting_the_rows_notifies_once_and_an_edit_notifies_once(self):
        table = self.table()
        table.set_rows([row('A'), row('B', '560')])
        self.assertEqual(self.changes, 1)
        table.rows[0]['vars']['center'].set('480')
        self.assertEqual(self.changes, 2)
        self.assertEqual(table.get_rows()[0]['center'], '480')

    def test_rows_are_added_and_removed(self):
        table = self.table()
        table.add_row({'name': 'NIR', 'center': '842', 'width': '57', 'reflectance': '0.477'})
        self.assertEqual([r['name'] for r in table.get_rows()], ['Blue', 'Green', 'LWIR', 'NIR'])
        self.assertEqual(self.changes, 1)
        table.remove_at(1)
        self.assertEqual([r['name'] for r in table.get_rows()], ['Blue', 'LWIR', 'NIR'])
        self.assertEqual(self.changes, 2)
        self.assertEqual(len(table.reflectance_entries()), 3)
        table.set_rows([])
        self.assertEqual(table.get_rows(), [])

    def test_a_new_row_is_empty_and_marked_invalid_until_it_is_filled(self):
        table = self.table([row('Blue')])
        new = table.add_row()
        self.root.update()
        self.assertEqual(str(new['widgets']['name'].cget('style')), 'Invalid.TEntry')
        new['vars']['name'].set('Green')
        new['vars']['center'].set('560')
        new['vars']['width'].set('27')
        new['vars']['reflectance'].set('0.478')
        self.assertEqual(table.problems(), [])
        for field in ('name', 'center', 'width', 'reflectance'):
            self.assertEqual(str(new['widgets'][field].cget('style')), 'TEntry')

    def test_invalid_entries_are_marked(self):
        table = self.table([row('Blue'), row('Blue', 'abc')])
        styles = [(number, field, str(r['widgets'][field].cget('style')))
                  for number, r in enumerate(table.rows) for field in ('name', 'center')]
        self.assertIn((0, 'name', 'Invalid.TEntry'), styles)
        self.assertIn((1, 'center', 'Invalid.TEntry'), styles)
        self.assertIn((0, 'center', 'TEntry'), styles)

    def test_a_band_that_is_not_used_cannot_have_its_values_edited(self):
        table = self.table()
        lwir = table.rows[2]['widgets']
        self.assertTrue(lwir['center'].instate(['disabled']))
        self.assertTrue(lwir['reflectance'].instate(['disabled']))
        self.assertFalse(lwir['name'].instate(['disabled']))
        table.rows[2]['vars']['process'].set(True)
        self.assertFalse(lwir['center'].instate(['disabled']))
        self.assertEqual(str(lwir['center'].cget('style')), 'Invalid.TEntry')           # now it needs a value

    def test_the_reflectance_is_edited_only_when_it_is_the_source(self):
        table = self.table([row('A', reflectance='')])
        self.assertEqual(len(table.problems()), 1)
        table.set_reflectance_state(False)
        self.assertEqual(table.problems(), [])
        self.assertTrue(all(entry.instate(['disabled']) for entry in table.reflectance_entries()))
        self.assertFalse(table.rows[0]['widgets']['center'].instate(['disabled']))
        table.set_reflectance_state(True)
        self.assertFalse(any(entry.instate(['disabled']) for entry in table.reflectance_entries()))


@unittest.skipUnless(sys.platform == 'win32', 'the title bar icon is a Windows matter')
class TitleIconTests(unittest.TestCase):

    def setUp(self):
        import tkinter as tk
        try:
            self.root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"no display for Tk: {error}")
        self.root.withdraw()

    def tearDown(self):
        try:
            self.root.destroy()
        except Exception:
            pass

    @staticmethod
    def frame_state(window):
        """ (icon registered on the class of the title-bar window, has the frame without the icon) """
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.WinDLL('user32', use_last_error=True)
        user32.GetParent.argtypes = (wintypes.HWND,)
        user32.GetParent.restype = wintypes.HWND
        user32.GetClassLongPtrW.argtypes = (wintypes.HWND, ctypes.c_int)
        user32.GetClassLongPtrW.restype = ctypes.c_void_p
        user32.GetWindowLongW.argtypes = (wintypes.HWND, ctypes.c_int)
        user32.GetWindowLongW.restype = ctypes.c_long
        wrapper = user32.GetParent(window.winfo_id())
        return bool(user32.GetClassLongPtrW(wrapper, -14)), bool(user32.GetWindowLongW(wrapper, -20) & 1)

    def test_the_icon_of_the_title_bar_is_hidden_in_a_window_and_in_a_dialog(self):
        import tkinter as tk
        from radiometric_app.ui_theme import hide_title_icon
        self.assertFalse(self.frame_state(self.root)[1])                         # the normal frame at the start
        self.assertTrue(hide_title_icon(self.root))
        self.assertEqual(self.frame_state(self.root), (False, True))            # no icon, frame without the icon slot
        dialog = tk.Toplevel(self.root)
        self.assertTrue(hide_title_icon(dialog))
        self.assertEqual(self.frame_state(dialog), (False, True))

    def test_it_can_be_repeated_and_never_raises(self):
        from radiometric_app.ui_theme import hide_title_icon
        self.assertTrue(hide_title_icon(self.root))
        self.assertTrue(hide_title_icon(self.root))
        self.root.destroy()
        self.assertFalse(hide_title_icon(self.root))                             # a window that is gone: no error

    def test_the_windows_of_the_program_hide_it(self):
        from radiometric_app import radiometric_window as rw
        import tempfile
        from radiometric_app.ui_theme import apply_dark_theme
        apply_dark_theme(self.root)
        with tempfile.TemporaryDirectory() as folder:
            frame = rw.RadiometricFrame(self.root, Path(folder) / 'inputs.json', folder)
            frame.widget.pack()
            from radiometric_app import panel_spectrum_dialog as psd
            import tkinter as tk
            dialog_window = tk.Toplevel(self.root)
            psd.ImportDialog(dialog_window, psd.ImportSession())
            self.assertEqual(self.frame_state(dialog_window), (False, True))


class ThemeColorsTests(unittest.TestCase):
    """ No state of a field may show a light box: the corners, the arrow and the check box have to stay dark """

    def setUp(self):
        import tkinter as tk
        try:
            self.root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"no display for Tk: {error}")
        self.root.withdraw()
        from radiometric_app.ui_theme import apply_dark_theme
        apply_dark_theme(self.root)
        from tkinter import ttk
        self.style = ttk.Style(self.root)

    def tearDown(self):
        self.root.destroy()

    def luminance(self, color):
        red, green, blue = (value // 256 for value in self.root.winfo_rgb(color))
        return 0.299 * red + 0.587 * green + 0.114 * blue

    def test_the_fields_are_dark_in_every_state(self):
        states = ([], ['active'], ['pressed'], ['focus'], ['readonly'], ['disabled'], ['focus', 'active'], ['selected'])
        options = {'TCombobox': ('background', 'fieldbackground', 'arrowcolor'),
                   'TEntry': ('background', 'fieldbackground'),
                   'Invalid.TEntry': ('background', 'fieldbackground'),
                   'TSpinbox': ('background', 'fieldbackground', 'arrowcolor'),
                   'TButton': ('background',),
                   'TCheckbutton': ('background', 'indicatorbackground')}
        for style, names in options.items():
            for state in states:
                for name in names:
                    color = self.style.lookup(style, name, state)
                    if name == 'arrowcolor':
                        self.assertGreater(self.luminance(color), 90, f"{style} {name} {state}: the arrow is lost")
                    else:
                        self.assertLess(self.luminance(color), 130, f"{style} {name} {state}: {color} is light")

    def test_the_check_mark_can_be_seen(self):
        for state in ([], ['selected']):
            mark = self.style.lookup('TCheckbutton', 'indicatorforeground', state)
            box = self.style.lookup('TCheckbutton', 'indicatorbackground', state)
            self.assertGreater(self.luminance(mark), self.luminance(box) + 60)


if __name__ == '__main__':
    unittest.main()
