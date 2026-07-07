"""
Regenerates OXFORD_SCANSION for every green-flagged (human-vetted) row across
scansion_tool/{to_complete,in_progress,completed_unvetted,gold} using the
current scansion.py, and writes every disagreement to a CSV. Since green rows
are trusted-correct, these disagreements point at real gaps in scansion.py.

Config knobs below: ONLY_FILES restricts which file(s) to check (None = all),
IGNORE_X strips elided-syllable 'x' markers from both sides before comparing.

Usage: python3 compare_green_flags.py
"""

import csv
import os
import re
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scansion import scan

SCANSION_TOOL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scansion_tool')
FOLDERS = ['to_complete', 'in_progress', 'completed_unvetted', 'gold']
DEFAULT_TARGET = 10
BD_HF_PREFIXES = ('BD_', 'HF_')

ONLY_FILES = None
IGNORE_X = True


def target_for_filename(filename):
    return 8 if filename.startswith(BD_HF_PREFIXES) else DEFAULT_TARGET


def regenerate(text, target):
    if not text or not str(text).strip():
        return '', 0
    stresses, num_sybs = scan(text, target)
    return ' '.join(stresses), num_sybs


def normalize(s):
    if IGNORE_X:
        s = re.sub(r'[xX]', '', s)
    return re.sub(r'\s+', ' ', s).strip()


def main():
    total_green = 0
    disagreements = []

    for folder in FOLDERS:
        for path in sorted(glob.glob(os.path.join(SCANSION_TOOL_DIR, folder, '*.csv'))):
            filename = os.path.basename(path)
            if ONLY_FILES is not None and filename not in ONLY_FILES:
                continue
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
                    generated_scansion, generated_syllables = regenerate(text, target)

                    if normalize(generated_scansion) != normalize(green_scansion):
                        blank_generated = 1 if generated_scansion.strip() == '' else 0

                        green_syllables = row.get('OXFORD_SYLLABLES', '')
                        try:
                            feminine = 1 if int(green_syllables) == target + 1 else 0
                        except (TypeError, ValueError):
                            feminine = 0

                        disagreements.append({
                            'source_folder': folder,
                            'file': filename,
                            'line_number': row.get('LINE_NUMBER', ''),
                            'target_syllables': target,
                            'oxford_text': text,
                            'green_scansion': green_scansion,
                            'generated_scansion': generated_scansion,
                            'green_syllables': green_syllables,
                            'generated_syllables': generated_syllables,
                            'BLANK_GENERATED': blank_generated,
                            'FEMININE': feminine,
                        })

    print(f"Total green-flagged rows checked: {total_green}")
    print(f"Disagreements: {len(disagreements)} ({len(disagreements) / total_green * 100:.2f}%)")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'green_flag_disagreements.csv')
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['source_folder', 'file', 'line_number', 'target_syllables', 'oxford_text',
                      'green_scansion', 'generated_scansion', 'green_syllables', 'generated_syllables',
                      'BLANK_GENERATED', 'FEMININE']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for d in disagreements:
            writer.writerow(d)
    print(f"\nDisagreements written to {out_path}")


if __name__ == '__main__':
    main()
