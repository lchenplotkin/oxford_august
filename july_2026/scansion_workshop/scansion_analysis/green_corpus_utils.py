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


def regenerate(module, text, target):
    if not text or not str(text).strip():
        return ''
    stresses, _ = module.scan(text, target)
    return ' '.join(stresses)


def iter_green_rows(scansion_tool_dir):
    """
    Yields a dict per green-flagged row: folder, file, line_number, target,
    og_text, text, green_scansion.
    """
    import csv

    for folder in FOLDERS:
        for path in sorted(glob.glob(os.path.join(scansion_tool_dir, folder, '*.csv'))):
            filename = os.path.basename(path)
            target = target_for_filename(filename)

            with open(path, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    flag = (row.get('SCANSION_FLAG_COLOR') or '').strip().lower()
                    if flag != 'green':
                        continue
                    yield {
                        'folder': folder,
                        'file': filename,
                        'line_number': row.get('LINE_NUMBER', ''),
                        'target': target,
                        'og_text': row.get('OG_OXFORD_TEXT', '') or row.get('OXFORD_TEXT', ''),
                        'text': row.get('OXFORD_TEXT', ''),
                        'green_scansion': row.get('OXFORD_SCANSION', ''),
                    }
