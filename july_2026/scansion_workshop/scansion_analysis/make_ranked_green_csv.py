"""
Builds a per-line comparison CSV across every green-flagged (human-vetted)
row in scansion_tool/: the gold scansion vs. what ranked_scan_v1.py
generates for it, syllable counts for each (x's stripped, since those mark
elided/silent syllables rather than sounded ones), whether they match, and
the ranked scansion's confidence score (line_confidence()).

For comparing all ranked_scan_vN.py versions at once, use
assess_versions.py instead -- this script only ever looks at v1.

Usage: python3 make_ranked_green_csv.py
"""

import csv
import os
import re
import sys
import glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'ranked_scan'))
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

                    ranked_scansion = regenerate(ranked, text, target)
                    confidence = ranked.line_confidence(text, ranked_scansion)

                    green_norm = normalize(green_scansion)
                    ranked_norm = normalize(ranked_scansion)

                    rows_out.append({
                        'OG_OXFORD_TEXT': og_text,
                        'OXFORD_TEXT': text,
                        'GREEN_SCANSION': green_scansion,
                        'RANKED_SCANSION': ranked_scansion,
                        'GREEN_SYLLABLES': syllable_count(green_scansion),
                        'RANKED_SYLLABLES': syllable_count(ranked_scansion),
                        'RANKED_MATCHES_GREEN': int(ranked_norm == green_norm),
                        'RANKED_CONFIDENCE': f"{confidence:.2f}",
                    })

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ranked_green_comparison.csv')
    fieldnames = ['OG_OXFORD_TEXT', 'OXFORD_TEXT', 'GREEN_SCANSION', 'RANKED_SCANSION',
                  'GREEN_SYLLABLES', 'RANKED_SYLLABLES',
                  'RANKED_MATCHES_GREEN', 'RANKED_CONFIDENCE']
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows_out:
            writer.writerow(r)

    total = len(rows_out)
    ranked_match = sum(r['RANKED_MATCHES_GREEN'] for r in rows_out)
    print(f"Wrote {total} rows to {out_path}")
    print(f"ranked matches green: {ranked_match} ({ranked_match/total*100:.2f}%)")


if __name__ == '__main__':
    main()
