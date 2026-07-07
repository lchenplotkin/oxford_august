"""
Regenerates OXFORD_SCANSION/RIVERSIDE_SCANSION for every row in dataset/*_gui_complete.csv
using the current scansion.py, and compares the result against what's already stored there.

BD (Book of the Duchess) and HF (House of Fame) are octosyllabic, so they're scanned with
target=8 instead of the usual 10. Comparisons ignore 'x' (elided-syllable) characters on
both sides, since those aren't part of the alternating-meter search target itself.

Usage: python3 verify_scansion.py
Writes a summary to stdout and a full mismatch listing to scansion_verification_mismatches.csv.
"""

import csv
import os
import re
import glob
from scansion import scan

DATASET_DIR = 'dataset'
DEFAULT_TARGET_SYLLABLES = 10
TARGET_BY_FILE = {
    'BD_gui_complete.csv': 8,
    'HF_gui_complete.csv': 8,
}


def regenerate(text, target):
    if not text or not str(text).strip():
        return '', 0
    stresses, num_sybs = scan(text, target)
    return ' '.join(stresses), num_sybs


def normalize_ignore_x(s):
    """Drop elided-syllable 'x' markers and collapse any whitespace that leaves behind."""
    s = re.sub(r'[xX]', '', s)
    return re.sub(r'\s+', ' ', s).strip()


def main():
    total_rows = 0
    oxford_match = 0
    oxford_mismatch = 0
    riverside_match = 0
    riverside_mismatch = 0
    mismatches = []

    for path in sorted(glob.glob(os.path.join(DATASET_DIR, '*_gui_complete.csv'))):
        filename = os.path.basename(path)
        target = TARGET_BY_FILE.get(filename, DEFAULT_TARGET_SYLLABLES)
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_rows += 1
                line_number = row.get('LINE_NUMBER', '')

                ox_text = row.get('OXFORD_TEXT', '')
                ox_existing = row.get('OXFORD_SCANSION', '')
                ox_new, ox_syllables_new = regenerate(ox_text, target)
                if normalize_ignore_x(ox_new) == normalize_ignore_x(ox_existing):
                    oxford_match += 1
                else:
                    oxford_mismatch += 1
                    mismatches.append({
                        'file': filename,
                        'line_number': line_number,
                        'edition': 'OXFORD',
                        'text': ox_text,
                        'existing_scansion': ox_existing,
                        'regenerated_scansion': ox_new,
                        'existing_syllables': row.get('OXFORD_SYLLABLES', ''),
                        'regenerated_syllables': ox_syllables_new,
                    })

                riv_text = row.get('RIVERSIDE_TEXT', '')
                riv_existing = row.get('RIVERSIDE_SCANSION', '')
                riv_new, riv_syllables_new = regenerate(riv_text, target)
                if normalize_ignore_x(riv_new) == normalize_ignore_x(riv_existing):
                    riverside_match += 1
                else:
                    riverside_mismatch += 1
                    mismatches.append({
                        'file': filename,
                        'line_number': line_number,
                        'edition': 'RIVERSIDE',
                        'text': riv_text,
                        'existing_scansion': riv_existing,
                        'regenerated_scansion': riv_new,
                        'existing_syllables': row.get('RIVERSIDE_SYLLABLES', ''),
                        'regenerated_syllables': riv_syllables_new,
                    })

    print(f"Total rows checked: {total_rows}")
    print(f"OXFORD:    {oxford_match} match, {oxford_mismatch} mismatch ({oxford_mismatch / total_rows * 100:.2f}%)")
    print(f"RIVERSIDE: {riverside_match} match, {riverside_mismatch} mismatch ({riverside_mismatch / total_rows * 100:.2f}%)")

    out_path = 'scansion_verification_mismatches.csv'
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['file', 'line_number', 'edition', 'text', 'existing_scansion',
                      'regenerated_scansion', 'existing_syllables', 'regenerated_syllables']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in mismatches:
            writer.writerow(m)
    print(f"\nFull mismatch listing written to {out_path}")


if __name__ == '__main__':
    main()
