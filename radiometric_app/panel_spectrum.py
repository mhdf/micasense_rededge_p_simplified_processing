"""
Reflectance of the calibrated reference panel by band, from its spectral response (a .csv file)

Micasense supplies, with each panel, the reflectance by wavelength. The reflectance of a band is the mean of that curve
over the spectral window of the band (center wavelength +/- half of the bandwidth): the weighting is uniform, which is
an approximation of the real spectral response of the filters. For the example panel it reproduces the values
printed on the panel (0.477 to 0.479) within 0.3 %.

This module has no user interface: it reads the file with the format chosen by the user (delimiter, header rows,
columns, units, decimal mark), keeps what was read from each row for a preview, and computes the values of the bands.
"""

# Standard Library Imports
import hashlib
from pathlib import Path

# Third-Party Imports
import numpy as np

DELIMITERS = {'auto': None, 'comma': ',', 'semicolon': ';', 'tab': '\t', 'space': ' ', 'custom': None}
WAVELENGTH_UNITS = {'nm': 1.0, 'um': 1000.0}
REFLECTANCE_UNITS = {'fraction': 1.0, 'percent': 0.01}
DECIMAL_MARKS = ('.', ',')

# What is read from the file. delimiter_char is the separator when the delimiter is 'custom'; skip_rows are the header
# rows ('auto': the rows at the start that are not numbers); skip_footer the last rows; lines that start with
# comment_prefix are ignored; exclude_rows is a selection of lines such as '5, 10-12'; rows with a wavelength
# outside wavelength_min/wavelength_max (nm, None for no limit) are not used; reflectance_scale multiplies the
# reflectance after the unit (for example 0.001 for per mille)
DEFAULT_FORMAT = {'delimiter': 'auto', 'delimiter_char': '', 'skip_rows': 'auto', 'skip_footer': 0,
                  'comment_prefix': '', 'exclude_rows': '', 'wavelength_column': 1, 'reflectance_column': 2,
                  'wavelength_unit': 'nm', 'reflectance_unit': 'fraction', 'reflectance_scale': 1.0,
                  'wavelength_min': None, 'wavelength_max': None, 'decimal': '.'}

# Spectral windows of the bands of the RedEdge-P: (center wavelength, bandwidth) in nm
REDEDGE_P_BANDS = {'Blue': (475.0, 32.0), 'Green': (560.0, 27.0), 'Panchro': (634.5, 463.0),
                   'Red': (668.0, 14.0), 'Red edge': (717.0, 12.0), 'NIR': (842.0, 57.0)}


_integrate = getattr(np, 'trapezoid', None) or np.trapz    # trapz was renamed in numpy 2


class SpectrumError(ValueError):
    """ Raised when the file cannot be read with the chosen format, or does not cover a band """


def parse_row_selection(text):
    """ Line numbers of a selection such as '5, 10-12' (empty text: none); SpectrumError if it cannot be read """
    selected = set()
    for part in str(text or '').replace(';', ',').split(','):
        part = part.strip()
        if not part:
            continue
        try:
            first, _, last = part.partition('-')
            low = int(first)
            high = int(last) if last.strip() else low
        except ValueError:
            raise SpectrumError(f"'{part}' is not a line number or a range such as 10-12")
        if low < 1 or high < low:
            raise SpectrumError(f"'{part}' is not a valid range of lines (from 1, in increasing order)")
        selected.update(range(low, high + 1))
    return selected


def format_row_selection(numbers):
    """ The text of a set of line numbers, with ranges: {5, 10, 11, 12} -> '5, 10-12' """
    ordered, parts = sorted(set(numbers)), []
    start = previous = None
    for number in ordered:
        if start is None:
            start = previous = number
        elif number == previous + 1:
            previous = number
        else:
            parts.append(str(start) if start == previous else f"{start}-{previous}")
            start = previous = number
    if start is not None:
        parts.append(str(start) if start == previous else f"{start}-{previous}")
    return ', '.join(parts)


def validate_format(fmt):
    """ Return the list of problems of a format dictionary (empty if valid) """
    fmt = dict(DEFAULT_FORMAT, **fmt)
    problems = []
    if fmt['delimiter'] not in DELIMITERS:
        problems.append(f"delimiter must be one of {list(DELIMITERS)}")
    skip = fmt['skip_rows']
    if not (skip == 'auto' or (isinstance(skip, int) and not isinstance(skip, bool) and skip >= 0)):
        problems.append("skip_rows must be 'auto' or a whole number from 0")
    for key in ('wavelength_column', 'reflectance_column'):
        value = fmt[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            problems.append(f"{key} must be a whole number from 1")
    if fmt['wavelength_column'] == fmt['reflectance_column']:
        problems.append("the wavelength and reflectance columns must be different")
    if fmt['wavelength_unit'] not in WAVELENGTH_UNITS:
        problems.append(f"wavelength_unit must be one of {list(WAVELENGTH_UNITS)}")
    if fmt['reflectance_unit'] not in REFLECTANCE_UNITS:
        problems.append(f"reflectance_unit must be one of {list(REFLECTANCE_UNITS)}")
    if fmt['decimal'] not in DECIMAL_MARKS:
        problems.append(f"decimal must be one of {list(DECIMAL_MARKS)}")
    if fmt['decimal'] == ',' and fmt['delimiter'] == 'comma':
        problems.append("the decimal mark and the delimiter cannot both be a comma")
    if fmt['delimiter'] == 'custom':
        if not isinstance(fmt['delimiter_char'], str) or not fmt['delimiter_char']:
            problems.append("a custom delimiter needs at least one character")
        elif fmt['delimiter_char'] == fmt['decimal']:
            problems.append("the decimal mark and the delimiter cannot be the same character")
    footer = fmt['skip_footer']
    if not isinstance(footer, int) or isinstance(footer, bool) or footer < 0:
        problems.append("skip_footer must be a whole number from 0")
    if not isinstance(fmt['comment_prefix'], str):
        problems.append("comment_prefix must be text")
    try:
        parse_row_selection(fmt['exclude_rows'])
    except SpectrumError as error:
        problems.append(f"exclude_rows: {error}")
    for key in ('wavelength_min', 'wavelength_max'):
        value = fmt[key]
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value)):
            problems.append(f"{key} must be a number or empty")
    if not problems and None not in (fmt['wavelength_min'], fmt['wavelength_max']) \
            and fmt['wavelength_min'] >= fmt['wavelength_max']:
        problems.append("wavelength_min must be smaller than wavelength_max")
    scale = fmt['reflectance_scale']
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not np.isfinite(scale) or scale <= 0:
        problems.append("reflectance_scale must be a number above 0")
    return problems


def read_text_lines(path):
    """ Lines of the file (UTF-8, with or without a byte order mark); undecodable characters are replaced """
    return Path(path).read_text(encoding='utf-8-sig', errors='replace').splitlines()


def resolve_delimiter(lines, delimiter='auto', custom=''):
    """
    The characters that separate the columns ('auto': the one that gives the same number of columns in most lines;
    'custom': the given text)
    """
    if delimiter == 'custom':
        return custom
    if delimiter != 'auto':
        return DELIMITERS[delimiter]
    sample = [line for line in lines if line.strip()][:50]
    # tab and semicolon first: a comma may be the decimal mark. A character is the delimiter if most of the lines have
    # the same number of columns (2 or more), so that one odd line does not hide it
    for name in ('tab', 'semicolon', 'comma'):
        counts = [len(line.split(DELIMITERS[name])) for line in sample]
        if counts:
            common = max(set(counts), key=counts.count)
            if common >= 2 and counts.count(common) >= 0.6 * len(counts):
                return DELIMITERS[name]
    return ' '


def delimiter_of(lines, fmt):
    """ The delimiter that the format gives for these lines (comment lines do not take part in the detection) """
    fmt = dict(DEFAULT_FORMAT, **fmt)
    comment = fmt['comment_prefix']
    kept = [line for line in lines if not (comment and line.lstrip().startswith(comment))]
    return resolve_delimiter(kept, fmt['delimiter'], fmt['delimiter_char'])


def split_line(line, delimiter):
    """ Fields of a line; for the space, runs of blanks count as one separator """
    if delimiter == ' ':
        return line.split()
    return [field.strip() for field in line.split(delimiter)]


def _number(text, decimal):
    return float(text.strip().replace(decimal, '.') if decimal != '.' else text.strip())


def parse_rows(lines, fmt):
    """
    Read every line with the format
    Outputs: list of dictionaries {'line', 'raw', 'fields', 'wavelength' (nm), 'reflectance' (fraction), 'status'},
             status being 'ok' (a sample), 'header', 'footer', 'comment', 'excluded', 'blank',
             'outside range' (a sample outside wavelength_min/wavelength_max) or 'error: reason'
    """
    fmt = dict(DEFAULT_FORMAT, **fmt)
    problems = validate_format(fmt)
    if problems:
        raise SpectrumError('; '.join(problems))
    delimiter = delimiter_of(lines, fmt)
    wavelength_index, reflectance_index = fmt['wavelength_column'] - 1, fmt['reflectance_column'] - 1
    skip, excluded = fmt['skip_rows'], parse_row_selection(fmt['exclude_rows'])
    comment = fmt['comment_prefix']
    scale = REFLECTANCE_UNITS[fmt['reflectance_unit']] * fmt['reflectance_scale']
    rows, data_started = [], False
    for number, raw in enumerate(lines, start=1):
        row = {'line': number, 'raw': raw, 'fields': [], 'wavelength': None, 'reflectance': None, 'status': 'ok'}
        rows.append(row)
        if not raw.strip():
            row['status'] = 'blank'
            continue
        if comment and raw.lstrip().startswith(comment):
            row['status'] = 'comment'
            continue
        row['fields'] = split_line(raw, delimiter)
        if number in excluded:
            row['status'] = 'excluded'
            continue
        if skip != 'auto' and number <= skip:
            row['status'] = 'header'
            continue
        try:
            fields = row['fields']
            if len(fields) <= max(wavelength_index, reflectance_index):
                raise ValueError(f"only {len(fields)} column(s)")
            wavelength = _number(fields[wavelength_index], fmt['decimal']) * WAVELENGTH_UNITS[fmt['wavelength_unit']]
            reflectance = _number(fields[reflectance_index], fmt['decimal']) * scale
            if not (np.isfinite(wavelength) and np.isfinite(reflectance)):
                raise ValueError("not a finite number")
            row['wavelength'], row['reflectance'] = wavelength, reflectance
            data_started = True
            if (fmt['wavelength_min'] is not None and wavelength < fmt['wavelength_min']) or \
                    (fmt['wavelength_max'] is not None and wavelength > fmt['wavelength_max']):
                row['status'] = 'outside range'
        except ValueError as error:
            if skip == 'auto' and not data_started:
                row['status'] = 'header'
            else:
                row['status'] = f"error: {error}"
    # the last rows of the file that are not blank
    footer = fmt['skip_footer']
    if footer:
        for row in [row for row in rows if row['raw'].strip()][-footer:]:
            row['status'] = 'footer'
    return rows


def load_spectrum(path, fmt, min_samples=2):
    """
    Read the spectral response file
    Outputs: dictionary {'wavelength' (nm, increasing), 'reflectance' (fraction), 'rows', 'delimiter', 'warnings',
             'file', 'sha256'}; SpectrumError (naming the lines) if the file cannot be read with the format
    """
    path = Path(path)
    lines = read_text_lines(path)
    rows = parse_rows(lines, fmt)
    errors = [f"line {row['line']}: {row['status'][7:]}" for row in rows if row['status'].startswith('error')]
    if errors:
        more = f" (and {len(errors) - 5} more)" if len(errors) > 5 else ''
        raise SpectrumError(f"{path.name}: " + '; '.join(errors[:5]) + more)
    data = [(row['wavelength'], row['reflectance']) for row in rows if row['status'] == 'ok']
    if len(data) < min_samples:
        raise SpectrumError(f"{path.name}: {len(data)} data row(s) found, at least {min_samples} are needed")
    wavelength, reflectance = (np.array(column, dtype=float) for column in zip(*data))
    warnings_found = []
    if np.any(np.diff(wavelength) <= 0):
        order = np.argsort(wavelength, kind='stable')
        wavelength, reflectance = wavelength[order], reflectance[order]
        warnings_found.append("the wavelengths were not increasing and were sorted")
    if np.any(np.diff(wavelength) == 0):
        raise SpectrumError(f"{path.name}: the same wavelength appears more than once")
    invalid = int(np.sum((reflectance <= 0) | (reflectance > 1)))
    if invalid:
        warnings_found.append(f"{invalid} sample(s) with a reflectance outside (0, 1] are not used")
    return {'wavelength': wavelength, 'reflectance': reflectance, 'rows': rows,
            'delimiter': delimiter_of(lines, fmt),
            'warnings': warnings_found, 'file': str(path),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def band_reflectances(spectrum, bands, min_coverage=0.5):
    """
    Mean reflectance of the curve over the spectral window of each band
    Inputs: spectrum from load_spectrum, {band: (center wavelength, bandwidth)} in nm and the smallest fraction of the
            window that must have valid samples (a window that reaches beyond the end of the file is averaged over the
            part that exists)
    Outputs: {band: {'reflectance', 'window_nm', 'samples', 'coverage'}}; SpectrumError listing every band that is
             not covered
    """
    wavelength, reflectance = spectrum['wavelength'], spectrum['reflectance']
    valid = (reflectance > 0) & (reflectance <= 1)
    step = float(np.median(np.diff(wavelength))) if len(wavelength) > 1 else 1.0
    result, problems = {}, []
    for band, (center, width) in bands.items():
        low, high = center - width / 2, center + width / 2
        inside = valid & (wavelength >= low) & (wavelength <= high)
        count = int(inside.sum())
        # samples that the window would have if the file's own grid of wavelengths continued across it
        first = wavelength[0]
        expected = max(1, int(np.floor((high - first) / step + 1e-9) - np.ceil((low - first) / step - 1e-9)) + 1)
        coverage = min(1.0, count / expected)
        if count == 0 or coverage < min_coverage:
            problems.append(f"Band '{band}': the spectral file has valid samples for {coverage:.0%} of its window "
                            f"{low:g}-{high:g} nm (at least {min_coverage:.0%} is needed)")
            continue
        w, r = wavelength[inside], reflectance[inside]
        value = float(r[0]) if count == 1 or w[-1] == w[0] else float(_integrate(r, w) / (w[-1] - w[0]))
        result[band] = {'reflectance': value, 'window_nm': [low, high], 'samples': count, 'coverage': coverage}
    if problems:
        raise SpectrumError('; '.join(problems))
    return result


def detect_format(path):
    """
    Guess the format of a file: delimiter, decimal mark, units and whether it has header rows
    Outputs: (format dictionary, list of notes that say what was guessed); columns 1 and 2 are assumed
    """
    lines = read_text_lines(path)
    fmt = dict(DEFAULT_FORMAT)
    notes = []
    if any(line.startswith('#') for line in lines[:20]):
        fmt['comment_prefix'] = '#'
        notes.append("lines that start with # are ignored")
        lines = [line for line in lines if not line.startswith('#')]
    delimiter = resolve_delimiter(lines)
    fmt['delimiter'] = {value: name for name, value in DELIMITERS.items() if name != 'auto'}[delimiter]
    notes.append(f"delimiter: {fmt['delimiter']}")
    sample = [line for line in lines if line.strip()][:50]
    if delimiter != ',' and any(',' in field for line in sample for field in split_line(line, delimiter)):
        fmt['decimal'] = ','
        notes.append("decimal mark: comma")
    rows = parse_rows(lines, fmt)
    headers = sum(1 for row in rows if row['status'] == 'header' and row['raw'].strip())
    if headers:
        notes.append(f"{headers} header row(s) will be skipped automatically")
    data = [row for row in rows if row['status'] == 'ok']
    if data:
        wavelengths = np.array([row['wavelength'] for row in data])
        reflectances = np.array([row['reflectance'] for row in data])
        if np.median(wavelengths) < 10:                      # 0.475 um
            fmt['wavelength_unit'] = 'um'
            notes.append("wavelength unit: um (values below 10)")
        if reflectances.max() > 1.5:
            fmt['reflectance_unit'] = 'percent'
            notes.append("reflectance unit: percent (values above 1)")
        if fmt['wavelength_unit'] == 'um' or fmt['reflectance_unit'] == 'percent':
            rows = parse_rows(lines, fmt)
    else:
        notes.append("no numeric rows were found in columns 1 and 2")
    return fmt, notes
