"""
Builds a word -> {scansion_pattern: frequency} JSON file, same schema as
scansion_tool/scansion_word_patterns.json (used for GUI autocomplete), but
sourced only from green-flagged (human-vetted) lines across scansion_tool/,
rather than the full (mostly machine-generated) dataset/combined.csv. This
is the reference table used to rank candidate scansions in every
ranked_scan/ranked_scan_vN.py.

Usage: python3 generate_green_word_patterns.py
"""

import csv
import json
import os
import re
import glob
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCANSION_TOOL_DIR = os.path.join(ROOT, 'scansion_tool')
FOLDERS = ['to_complete', 'in_progress', 'completed_unvetted', 'gold']
OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'green_word_patterns.json')

VALID_PATTERN = re.compile(r"^[SuUxX]+$")


def main():
    patterns = defaultdict(lambda: defaultdict(int))
    rows_used = 0

    for folder in FOLDERS:
        for path in sorted(glob.glob(os.path.join(SCANSION_TOOL_DIR, folder, '*.csv'))):
            with open(path, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    flag = (row.get('SCANSION_FLAG_COLOR') or '').strip().lower()
                    if flag != 'green':
                        continue

                    words = row.get('OXFORD_TEXT', '').split()
                    scans = row.get('OXFORD_SCANSION', '').split()
                    if not words or len(words) != len(scans):
                        # Skip rows where per-word tokenization doesn't line
                        # up 1:1 (elided words sometimes collapse spacing in
                        # the gold annotation) -- we'd rather drop the row
                        # than mis-attribute a pattern to the wrong word.
                        continue

                    rows_used += 1
                    for word, scan in zip(words, scans):
                        if not VALID_PATTERN.match(scan):
                            continue
                        key = re.sub(r'[^a-zA-Z]', '', word).lower()
                        if not key:
                            continue
                        norm = "".join(c.upper() if c.lower() == "s" else c.lower() for c in scan)
                        patterns[key][norm] += 1

    out = {word: dict(sorted(p.items(), key=lambda kv: -kv[1])) for word, p in patterns.items()}

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"Used {rows_used} green-flagged rows (with clean 1:1 word/token alignment)")
    print(f"Wrote {len(out)} words to {OUTPUT}")


if __name__ == '__main__':
    main()
