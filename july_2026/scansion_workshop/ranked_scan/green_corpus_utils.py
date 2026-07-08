"""
Shared helpers for iterating the green-flagged (human-vetted) rows across
scansion_tool/*/*.csv and comparing generated scansion against them. Used
by assess_versions.py and (indirectly, via the same conventions) the other
comparison scripts in this directory.
"""

import glob
import os
import re

FOLDERS = ['to_complete', 'in_progress', 'completed_unvetted', 'gold']
DEFAULT_TARGET = 10
BD_HF_PREFIXES = ('BD_', 'HF_')


def target_for_filename(filename):
    return 8 if filename.startswith(BD_HF_PREFIXES) else DEFAULT_TARGET


def normalize(s):
    s = re.sub(r'[xX]', '', s or '')
    return re.sub(r'\s+', ' ', s).strip()


def syllable_count(s):
    return len(normalize(s).replace(' ', ''))


def regenerate(module, text, target, reference=None):
    """
    reference, if given, is passed straight through to module.scan() --
    used to score against an explicit (e.g. train-split) reference table
    instead of whatever the module would otherwise lazily load from disk.
    """
    if not text or not str(text).strip():
        return ''
    if reference is not None:
        stresses, _ = module.scan(text, target, reference=reference)
    else:
        stresses, _ = module.scan(text, target)
    return ' '.join(stresses)


def iter_all_rows(scansion_tool_dir):
    """
    Yields a dict per row (green AND non-green) across scansion_tool/*/*.csv:
    folder, file, line_number, target, og_text, text, scansion, confidence
    (float or None if blank/missing), is_green. Malformed rows (a
    pre-existing data issue -- see regenerate_non_green.py's history --
    where a stray embedded header gives a row more fields than the CSV
    header) are skipped entirely.
    """
    import csv

    for folder in FOLDERS:
        for path in sorted(glob.glob(os.path.join(scansion_tool_dir, folder, '*.csv'))):
            filename = os.path.basename(path)
            target = target_for_filename(filename)

            with open(path, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if None in row:
                        continue
                    conf_raw = (row.get('OXFORD_SCANSION_CONFIDENCE') or '').strip()
                    yield {
                        'folder': folder,
                        'file': filename,
                        'line_number': row.get('LINE_NUMBER', ''),
                        'target': target,
                        'og_text': row.get('OG_OXFORD_TEXT', '') or row.get('OXFORD_TEXT', ''),
                        'text': row.get('OXFORD_TEXT', ''),
                        'scansion': row.get('OXFORD_SCANSION', ''),
                        'confidence': float(conf_raw) if conf_raw else None,
                        'is_green': (row.get('SCANSION_FLAG_COLOR') or '').strip().lower() == 'green',
                    }


def iter_green_rows(scansion_tool_dir):
    """
    Yields a dict per green-flagged row: folder, file, line_number, target,
    og_text, text, green_scansion.
    """
    for row in iter_all_rows(scansion_tool_dir):
        if not row['is_green']:
            continue
        yield {
            'folder': row['folder'],
            'file': row['file'],
            'line_number': row['line_number'],
            'target': row['target'],
            'og_text': row['og_text'],
            'text': row['text'],
            'green_scansion': row['scansion'],
        }
