"""
Same check as verify_scansion.py, but using the simpler alternate scansion
algorithm in scansion_alt.py (pure alternating u/S search, no elided-syllable
'x' notation, no ELISION_FOLLOWERS/variable-syllable-word tables).

Usage: python3 verify_scansion_alt.py
"""

import csv
import os
import re
import glob
from scansion_alt import try_all_scans

DATASET_DIR = 'dataset'
TARGET_SYLLABLES = 10
VAR_DICT = {}  # no variable-stress-pattern CSV exists in the repo for this algorithm


def regenerate(text):
    if not text or not str(text).strip():
        return '', 0
    words = re.findall(r"\b[\w']+\b", text)
    if not words:
        return '', 0
    result, status = try_all_scans(words, VAR_DICT, TARGET_SYLLABLES)
    if not result:
        return '', 0
    stresses = [s for _, s in result]
    total = sum(len(s) for s in stresses)
    return ' '.join(stresses), total


def main():
    total_rows = 0
    oxford_match = 0
    oxford_mismatch = 0
    riverside_match = 0
    riverside_mismatch = 0
    mismatches = []

    for path in sorted(glob.glob(os.path.join(DATASET_DIR, '*_gui_complete.csv'))):
        filename = os.path.basename(path)
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_rows += 1
                line_number = row.get('LINE_NUMBER', '')

                ox_text = row.get('OXFORD_TEXT', '')
                ox_existing = row.get('OXFORD_SCANSION', '')
                ox_new, ox_syllables_new = regenerate(ox_text)
                if ox_new == ox_existing:
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
                riv_new, riv_syllables_new = regenerate(riv_text)
                if riv_new == riv_existing:
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

    out_path = 'scansion_alt_verification_mismatches.csv'
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
