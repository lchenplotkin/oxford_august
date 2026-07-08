"""
Builds a word -> {scansion_pattern: frequency} table, same schema as
scansion_tool/scansion_word_patterns.json (used for GUI autocomplete), but
sourced only from green-flagged (human-vetted) lines, rather than the full
(mostly machine-generated) dataset/combined.csv. This is the reference
table used to rank candidate scansions in every ranked_scan/ranked_scan_vN.py.

build_patterns(rows) does the actual counting and is reusable on any subset
of green rows (e.g. assess_versions.py's train split, so evaluation never
happens on the same lines the reference was built from). main() is the
"build from everything and write green_word_patterns.json to disk" entry
point used by update_non_green_with_latest.py for real deployment.

Usage: python3 generate_green_word_patterns.py
"""

import json
import os
import re
from collections import defaultdict

from green_corpus_utils import iter_green_rows

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCANSION_TOOL_DIR = os.path.join(ROOT, 'scansion_tool')
OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'green_word_patterns.json')

VALID_PATTERN = re.compile(r"^[SuUxX]+$")


def build_patterns(rows):
    """
    rows: iterable of dicts with 'text' and 'green_scansion' string keys
    (the shape iter_green_rows() yields). Returns (patterns_dict, rows_used).
    """
    patterns = defaultdict(lambda: defaultdict(int))
    rows_used = 0

    for row in rows:
        words = row['text'].split()
        scans = row['green_scansion'].split()
        if not words or len(words) != len(scans):
            # Skip rows where per-word tokenization doesn't line up 1:1
            # (elided words sometimes collapse spacing in the gold
            # annotation) -- we'd rather drop the row than mis-attribute a
            # pattern to the wrong word.
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
    return out, rows_used


def main():
    rows = list(iter_green_rows(SCANSION_TOOL_DIR))
    out, rows_used = build_patterns(rows)

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"Used {rows_used} green-flagged rows (with clean 1:1 word/token alignment)")
    print(f"Wrote {len(out)} words to {OUTPUT}")
    return out


if __name__ == '__main__':
    main()
