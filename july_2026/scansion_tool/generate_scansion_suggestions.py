"""
Builds a word -> {scansion_pattern: frequency} JSON suggestion file from the
Oxford dataset, for use as autocomplete suggestions in oxford_scansion_gui.html.

Usage:
    python3 generate_scansion_suggestions.py

Reads ../dataset/combined.csv and writes scansion_word_patterns.json next to
this script.
"""

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

DATASET = Path(__file__).resolve().parent.parent / "dataset" / "combined.csv"
OUTPUT = Path(__file__).resolve().parent / "scansion_word_patterns.json"

VALID_PATTERN = re.compile(r"^[SuUxX]+$")


def main():
    patterns = defaultdict(lambda: defaultdict(int))

    with open(DATASET, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            words = row.get("OXFORD_TEXT", "").split()
            scans = row.get("OXFORD_SCANSION", "").split()
            for word, scan in zip(words, scans):
                if not VALID_PATTERN.match(scan):
                    continue
                key = word.lower()
                # normalize to canonical case: S uppercase, u/x lowercase
                norm = "".join(c.upper() if c.lower() == "s" else c.lower() for c in scan)
                patterns[key][norm] += 1

    out = {word: dict(sorted(p.items(), key=lambda kv: -kv[1])) for word, p in patterns.items()}

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(out)} words to {OUTPUT}")


if __name__ == "__main__":
    main()
