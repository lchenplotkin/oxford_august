"""
Builds a per-line comparison CSV across every green-flagged (human-vetted)
row in scansion_tool/: original Oxford text, cleaned Oxford text, the gold
scansion, and what scansion.py (v1) and scansion_v2.py (v2) each generate
for it, plus syllable counts and match flags for each version against gold
(x's stripped before comparing/counting, since those mark elided/silent
syllables rather than sounded ones).

Usage: python3 make_v1_v2_green_csv.py
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


def syllable_count(s):
    return len(normalize(s).replace(' ', ''))


def main():
    rows_out = []

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

                    og_text = row.get('OG_OXFORD_TEXT', '') or row.get('OXFORD_TEXT', '')
                    text = row.get('OXFORD_TEXT', '')
                    green_scansion = row.get('OXFORD_SCANSION', '')

                    v1_scansion = regenerate(v1, text, target)
                    v2_scansion = regenerate(v2, text, target)

                    green_norm = normalize(green_scansion)
                    v1_norm = normalize(v1_scansion)
                    v2_norm = normalize(v2_scansion)

                    rows_out.append({
                        'OG_OXFORD_TEXT': og_text,
                        'OXFORD_TEXT': text,
                        'GREEN_SCANSION': green_scansion,
                        'V1_SCANSION': v1_scansion,
                        'V2_SCANSION': v2_scansion,
                        'GREEN_SYLLABLES': syllable_count(green_scansion),
                        'V1_SYLLABLES': syllable_count(v1_scansion),
                        'V2_SYLLABLES': syllable_count(v2_scansion),
                        'V1_MATCHES_GREEN': int(v1_norm == green_norm),
                        'V2_MATCHES_GREEN': int(v2_norm == green_norm),
                    })

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v1_v2_green_comparison.csv')
    fieldnames = ['OG_OXFORD_TEXT', 'OXFORD_TEXT', 'GREEN_SCANSION', 'V1_SCANSION', 'V2_SCANSION',
                  'GREEN_SYLLABLES', 'V1_SYLLABLES', 'V2_SYLLABLES',
                  'V1_MATCHES_GREEN', 'V2_MATCHES_GREEN']
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows_out:
            writer.writerow(r)

    total = len(rows_out)
    v1_match = sum(r['V1_MATCHES_GREEN'] for r in rows_out)
    v2_match = sum(r['V2_MATCHES_GREEN'] for r in rows_out)
    print(f"Wrote {total} rows to {out_path}")
    print(f"v1 matches green: {v1_match} ({v1_match/total*100:.2f}%)")
    print(f"v2 matches green: {v2_match} ({v2_match/total*100:.2f}%)")


if __name__ == '__main__':
    main()
