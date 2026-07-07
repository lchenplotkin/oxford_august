"""
Compares scansion_v2.py (first-valid-match search) against
ranked_scan_v1.py (collects every valid match and ranks by green-flagged
word/pattern frequency) across every green-flagged row in scansion_tool/.

For comparing *all* ranked_scan_vN.py versions against each other, use
assess_versions.py instead -- this script is kept for a quick sanity check
of the ranking idea against the pre-ranking baseline specifically.

Usage: python3 compare_ranked.py
"""

import csv
import os
import re
import sys
import glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'ranked_scan'))
import scansion_v2 as v2
import ranked_scan_v1 as ranked

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
    total = 0
    v2_match = 0
    r_match = 0
    fixed = []
    regressed = []

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
                    total += 1

                    text = row.get('OXFORD_TEXT', '')
                    green_scansion = row.get('OXFORD_SCANSION', '')
                    green_norm = normalize(green_scansion)

                    v2_scansion = regenerate(v2, text, target)
                    r_scansion = regenerate(ranked, text, target)
                    v2_ok = normalize(v2_scansion) == green_norm
                    r_ok = normalize(r_scansion) == green_norm

                    if v2_ok:
                        v2_match += 1
                    if r_ok:
                        r_match += 1

                    record = {
                        'folder': folder,
                        'file': filename,
                        'line_number': row.get('LINE_NUMBER', ''),
                        'oxford_text': text,
                        'green_scansion': green_scansion,
                        'v2_scansion': v2_scansion,
                        'ranked_scansion': r_scansion,
                    }

                    if not v2_ok and r_ok:
                        fixed.append(record)
                    elif v2_ok and not r_ok:
                        regressed.append(record)

    print(f"Total green-flagged rows: {total}")
    print(f"scansion_v2.py match: {v2_match} ({v2_match/total*100:.2f}%)")
    print(f"ranked_scan_v1.py match: {r_match} ({r_match/total*100:.2f}%)")
    print()
    print(f"Fixed by ranking (v2 wrong -> ranked right):     {len(fixed)}")
    print(f"Regressed by ranking (v2 right -> ranked wrong): {len(regressed)}")
    print(f"Net: {len(fixed) - len(regressed)}")

    out_dir = os.path.dirname(os.path.abspath(__file__))
    for name, records in [('fixed_by_ranking.csv', fixed), ('regressed_by_ranking.csv', regressed)]:
        path = os.path.join(out_dir, name)
        with open(path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['folder', 'file', 'line_number', 'oxford_text', 'green_scansion', 'v2_scansion', 'ranked_scansion']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                writer.writerow(r)
        print(f"Wrote {len(records)} rows to {path}")


if __name__ == '__main__':
    main()
