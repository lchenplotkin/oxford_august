"""
Three-way comparison across every green-flagged (human-vetted) row in
scansion_tool/: green (trusted) vs. current scansion.py vs. scansion_v2.py
(the candidate fixes). Reports match rates for both against green, and a
fixed/regressed/still-wrong breakdown between v1 and v2.

Usage: python3 compare_v1_v2_green.py
"""

import csv
import os
import re
import sys
import glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import scansion as v1
import scansion_v2 as v2

SCANSION_TOOL_DIR = os.path.join(ROOT, 'scansion_tool')
FOLDERS = ['to_complete', 'in_progress', 'completed_unvetted', 'gold']
DEFAULT_TARGET = 10
BD_HF_PREFIXES = ('BD_', 'HF_')


def target_for_filename(filename):
    return 8 if filename.startswith(BD_HF_PREFIXES) else DEFAULT_TARGET


def regenerate(module, text, target):
    if not text or not str(text).strip():
        return ''
    stresses, _ = module.scan(text, target)
    return ' '.join(stresses)


def normalize(s):
    s = re.sub(r'[xX]', '', s)
    return re.sub(r'\s+', ' ', s).strip()


def main():
    total_green = 0
    v1_match = 0
    v2_match = 0
    fixed = []       # v1 wrong, v2 right
    regressed = []   # v1 right, v2 wrong
    still_wrong = [] # both wrong (and differ from each other or not)

    for folder in FOLDERS:
        for path in sorted(glob.glob(os.path.join(SCANSION_TOOL_DIR, folder, '*.csv'))):
            filename = os.path.basename(path)
            target = target_for_filename(filename)

            with open(path, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    flag = (row.get('SCANSION_FLAG_COLOR') or '').strip().lower()
                    if flag != 'green':
                        continue
                    total_green += 1

                    text = row.get('OXFORD_TEXT', '')
                    green_scansion = row.get('OXFORD_SCANSION', '')
                    green_norm = normalize(green_scansion)

                    v1_scansion = regenerate(v1, text, target)
                    v2_scansion = regenerate(v2, text, target)
                    v1_ok = normalize(v1_scansion) == green_norm
                    v2_ok = normalize(v2_scansion) == green_norm

                    if v1_ok:
                        v1_match += 1
                    if v2_ok:
                        v2_match += 1

                    record = {
                        'folder': folder,
                        'file': filename,
                        'line_number': row.get('LINE_NUMBER', ''),
                        'oxford_text': text,
                        'green_scansion': green_scansion,
                        'v1_scansion': v1_scansion,
                        'v2_scansion': v2_scansion,
                    }

                    if not v1_ok and v2_ok:
                        fixed.append(record)
                    elif v1_ok and not v2_ok:
                        regressed.append(record)
                    elif not v1_ok and not v2_ok:
                        still_wrong.append(record)

    print(f"Total green-flagged rows: {total_green}")
    print(f"scansion.py    match: {v1_match} ({v1_match/total_green*100:.2f}%)")
    print(f"scansion_v2.py match: {v2_match} ({v2_match/total_green*100:.2f}%)")
    print()
    print(f"Fixed by v2 (v1 wrong -> v2 right):     {len(fixed)}")
    print(f"Regressed by v2 (v1 right -> v2 wrong): {len(regressed)}")
    print(f"Still wrong in both:                    {len(still_wrong)}")

    out_dir = os.path.dirname(os.path.abspath(__file__))
    for name, records in [('fixed_by_v2.csv', fixed), ('regressed_by_v2.csv', regressed), ('still_wrong.csv', still_wrong)]:
        path = os.path.join(out_dir, name)
        with open(path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['folder', 'file', 'line_number', 'oxford_text', 'green_scansion', 'v1_scansion', 'v2_scansion']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                writer.writerow(r)
        print(f"Wrote {len(records)} rows to {path}")


if __name__ == '__main__':
    main()
