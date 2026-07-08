"""
Regenerates OXFORD_SCANSION for every row in scansion_tool/*/*.csv that is
NOT green-flagged (i.e. not yet human-vetted), using the highest-numbered
ranked_scan_vN.py in ranked_scan/ (override with --version), then propagates
those same regenerated values into the matching rows of dataset/combined.csv
(matched by OUTPUT_FILENAME + LINE_NUMBER).

Green-flagged rows are never touched, in either location -- those are the
trusted, human-vetted ground truth this whole project is built on.

Every row (green and non-green) also gets an OXFORD_SCANSION_CONFIDENCE
column, scored against whatever OXFORD_SCANSION ends up in that row -- the
freshly regenerated one for non-green rows, or the existing human one for
green rows. A blank scansion scores 0.

Also refreshes green_word_patterns.json first (same as assess_versions.py),
and writes scansion_tool/confidence_calibration.json -- the
accuracy-at-each-confidence-threshold curve for the version used here,
against the green-flagged set -- so oxford_scansion_gui.html's "Accuracy
below X%" filter stays in sync with whichever version actually produced the
confidence numbers sitting in the data.

Only OXFORD_SCANSION is regenerated. OXFORD_SYLLABLES is treated as the
target syllable count (the input to scan()), not a derived value, and is
left alone -- it's frequently 11 for a 10-syllable feminine-ending line's
*target*, which the module then hits via the +1 feminine-ending branch.
RIVERSIDE_TEXT/RIVERSIDE_SCANSION are untouched; this scansion program only
ever operates on OXFORD_TEXT.

Usage:
    python3 update_non_green_with_latest.py --dry-run   # report only, no writes
    python3 update_non_green_with_latest.py              # write for real, latest version
    python3 update_non_green_with_latest.py --version 2  # pin to a specific version
"""

import argparse
import csv
import json
import os

import generate_green_word_patterns
from assess_versions import discover_versions, load_version_module, RANKED_SCAN_DIR
from confidence_calibration import accuracy_table
from green_corpus_utils import iter_green_rows, normalize, regenerate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCANSION_TOOL_DIR = os.path.join(ROOT, 'scansion_tool')
DATASET_PATH = os.path.join(os.path.dirname(ROOT), 'dataset', 'combined.csv')
FOLDERS = ['to_complete', 'in_progress', 'completed_unvetted', 'gold']
DEFAULT_TARGET = 10
BD_HF_PREFIXES = ('BD_', 'HF_')
CONFIDENCE_COLUMN = 'OXFORD_SCANSION_CONFIDENCE'
CALIBRATION_PATH = os.path.join(SCANSION_TOOL_DIR, 'confidence_calibration.json')


def target_for_row(row, filename):
    raw = (row.get('OXFORD_SYLLABLES') or '').strip()
    if raw.isdigit():
        return int(raw)
    return 8 if filename.startswith(BD_HF_PREFIXES) else DEFAULT_TARGET


def regenerate_scansion_tool(module, dry_run):
    """
    Rewrites OXFORD_SCANSION for non-green rows, and OXFORD_SCANSION_CONFIDENCE
    for every row, in every scansion_tool CSV. Returns a dict of
    (OUTPUT_FILENAME, LINE_NUMBER) -> (OXFORD_SCANSION, confidence) for
    every row, for propagation into dataset/combined.csv.
    """
    import glob

    changed_lookup = {}
    files_touched = 0
    rows_regenerated = 0
    rows_considered = 0
    rows_total = 0
    rows_malformed = 0

    for folder in FOLDERS:
        for path in sorted(glob.glob(os.path.join(SCANSION_TOOL_DIR, folder, '*.csv'))):
            filename = os.path.basename(path)

            with open(path, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                fieldnames = list(reader.fieldnames)
                rows = list(reader)

            if CONFIDENCE_COLUMN not in fieldnames:
                fieldnames.append(CONFIDENCE_COLUMN)

            for row in rows:
                rows_total += 1
                if None in row:
                    # Malformed row (more fields than the header -- some
                    # rows in this dataset have a stray embedded header
                    # concatenated onto them, a pre-existing data issue).
                    # Leave it alone rather than guess at realignment.
                    rows_malformed += 1
                    continue

                text = row.get('OXFORD_TEXT', '')
                flag = (row.get('SCANSION_FLAG_COLOR') or '').strip().lower()

                if flag != 'green':
                    rows_considered += 1
                    target = target_for_row(row, filename)
                    new_scansion = regenerate(module, text, target)
                    if new_scansion != row.get('OXFORD_SCANSION', ''):
                        rows_regenerated += 1
                    row['OXFORD_SCANSION'] = new_scansion

                scansion = row.get('OXFORD_SCANSION', '')
                confidence = module.line_confidence(text, scansion)
                row[CONFIDENCE_COLUMN] = f"{confidence:.2f}"

                key = (row.get('OUTPUT_FILENAME', ''), row.get('LINE_NUMBER', ''))
                changed_lookup[key] = (scansion, row[CONFIDENCE_COLUMN])

            files_touched += 1
            if not dry_run:
                with open(path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                    writer.writeheader()
                    writer.writerows(rows)

    print(f"scansion_tool: {rows_total} rows across {files_touched} files -- "
          f"{rows_considered} non-green rows re-scanned ({rows_regenerated} actually changed), "
          f"{CONFIDENCE_COLUMN} " + ("computed" if dry_run else "written") + " for all rows"
          + (f" ({rows_malformed} malformed rows skipped, left as-is)" if rows_malformed else ""))
    return changed_lookup


def propagate_to_dataset(changed_lookup, dry_run):
    if not os.path.exists(DATASET_PATH):
        print(f"dataset/combined.csv not found at {DATASET_PATH}, skipping")
        return

    with open(DATASET_PATH, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    if CONFIDENCE_COLUMN not in fieldnames:
        fieldnames.append(CONFIDENCE_COLUMN)

    rows_matched = 0
    rows_unmatched = 0
    rows_malformed = 0
    for row in rows:
        if None in row:
            rows_malformed += 1
            continue
        key = (row.get('OUTPUT_FILENAME', ''), row.get('LINE_NUMBER', ''))
        if key not in changed_lookup:
            rows_unmatched += 1
            continue
        rows_matched += 1
        new_scansion, confidence = changed_lookup[key]
        row['OXFORD_SCANSION'] = new_scansion
        row[CONFIDENCE_COLUMN] = confidence

    print(f"dataset/combined.csv: {rows_matched} rows matched to scansion_tool and "
          + ("would be " if dry_run else "") + "synced (scansion + confidence), "
          f"{rows_unmatched} rows had no matching scansion_tool key (left untouched)"
          + (f", {rows_malformed} malformed rows skipped, left as-is" if rows_malformed else ""))

    if not dry_run:
        with open(DATASET_PATH, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)


def export_calibration_curve(module, version, dry_run):
    """
    Scores `module` against every green-flagged line, builds an
    accuracy-at-each-integer-confidence-threshold curve, and writes it to
    CALIBRATION_PATH for the GUI's "Accuracy below X%" filter to consume.
    """
    pairs = []
    for line in iter_green_rows(SCANSION_TOOL_DIR):
        scansion = regenerate(module, line['text'], line['target'])
        confidence = module.line_confidence(line['text'], scansion)
        is_match = int(normalize(scansion) == normalize(line['green_scansion']))
        pairs.append((confidence, is_match))

    curve = [
        {"confidence": t, "accuracy": (None if acc is None else round(acc * 100, 2)), "n": n, "matches": matches}
        for t, acc, n, matches in accuracy_table(pairs, list(range(0, 101)))
    ]
    payload = {"version": version, "n_green_lines": len(pairs), "curve": curve}

    print(f"confidence_calibration.json: built from {len(pairs)} green-flagged lines scored with v{version}"
          + (" (dry run, not written)" if dry_run else f" -> {CALIBRATION_PATH}"))
    if not dry_run:
        with open(CALIBRATION_PATH, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--version', type=int, default=None,
                         help='which ranked_scan_vN.py to use (default: highest version number found)')
    parser.add_argument('--dry-run', action='store_true',
                         help='report what would change without writing any files')
    args = parser.parse_args()

    versions = discover_versions()
    if not versions:
        print(f"No ranked_scan_vN.py files found in {RANKED_SCAN_DIR}")
        return

    version = args.version if args.version is not None else max(versions)
    if version not in versions:
        print(f"ranked_scan_v{version}.py not found. Available: {sorted(versions)}")
        return

    print("Refreshing green_word_patterns.json from current green-flagged data...")
    generate_green_word_patterns.main()
    print()

    print(f"Using ranked_scan_v{version}.py" + (" (dry run)" if args.dry_run else ""))
    module = load_version_module(version, versions[version])

    changed_lookup = regenerate_scansion_tool(module, args.dry_run)
    propagate_to_dataset(changed_lookup, args.dry_run)
    print()
    export_calibration_curve(module, version, args.dry_run)


if __name__ == '__main__':
    main()
