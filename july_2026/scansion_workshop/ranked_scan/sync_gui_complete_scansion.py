"""
Propagates the latest ranked_scan_vN scansion (as computed against
scansion_tool/*/*.csv, respecting green-flagged rows) into every
dataset/*_gui_complete.csv file, matched by (OUTPUT_FILENAME, LINE_NUMBER).

This mirrors update_non_green_with_latest.py's propagation into
dataset/combined.csv, but targets the per-text _gui_complete.csv files
that scan_adjs.py / 2adjs.py actually read. Those had drifted out of sync
with combined.csv, since only combined.csv was ever kept in sync with
scansion_tool's regenerated output.

scansion_tool/*/*.csv itself is never modified by this script (it's read
in dry-run mode) -- it stays the single source of truth, kept current by
update_non_green_with_latest.py.

Usage:
    python3 sync_gui_complete_scansion.py --dry-run
    python3 sync_gui_complete_scansion.py
    python3 sync_gui_complete_scansion.py --version 7
"""

import argparse
import csv
import glob
import os

from assess_versions import discover_versions, load_version_module, RANKED_SCAN_DIR
from update_non_green_with_latest import regenerate_scansion_tool, CONFIDENCE_COLUMN

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(os.path.dirname(ROOT), 'dataset')


def propagate_to_gui_complete(changed_lookup, dry_run):
    paths = sorted(glob.glob(os.path.join(DATASET_DIR, '*_gui_complete.csv')))
    total_matched = 0
    total_unmatched = 0
    total_malformed = 0

    for path in paths:
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames)
            rows = list(reader)

        if CONFIDENCE_COLUMN not in fieldnames:
            fieldnames.append(CONFIDENCE_COLUMN)

        matched = unmatched = malformed = 0
        for row in rows:
            if None in row:
                malformed += 1
                continue
            key = (row.get('OUTPUT_FILENAME', ''), row.get('LINE_NUMBER', ''))
            if key not in changed_lookup:
                unmatched += 1
                continue
            matched += 1
            new_scansion, confidence = changed_lookup[key]
            row['OXFORD_SCANSION'] = new_scansion
            row[CONFIDENCE_COLUMN] = confidence

        total_matched += matched
        total_unmatched += unmatched
        total_malformed += malformed

        print(f"{os.path.basename(path)}: {matched} matched, {unmatched} unmatched"
              + (f", {malformed} malformed skipped" if malformed else ""))

        if not dry_run:
            # lineterminator='\n': these files were LF-terminated; csv's default
            # \r\n would otherwise touch every line and swamp the real diff.
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore', lineterminator='\n')
                writer.writeheader()
                writer.writerows(rows)

    print()
    print(f"Total: {total_matched} matched, {total_unmatched} unmatched"
          + (f", {total_malformed} malformed skipped" if total_malformed else ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--version', type=int, default=None,
                         help='which ranked_scan_vN.py to use (default: highest version number found)')
    parser.add_argument('--dry-run', action='store_true',
                         help='report what would change without writing dataset/*_gui_complete.csv')
    args = parser.parse_args()

    versions = discover_versions()
    if not versions:
        print(f"No ranked_scan_vN.py files found in {RANKED_SCAN_DIR}")
        return

    version = args.version if args.version is not None else max(versions)
    if version not in versions:
        print(f"ranked_scan_v{version}.py not found. Available: {sorted(versions)}")
        return

    print(f"Using ranked_scan_v{version}.py" + (" (dry run)" if args.dry_run else ""))
    module = load_version_module(version, versions[version])

    # dry_run=True: only compute changed_lookup, never write scansion_tool/*.csv.
    changed_lookup = regenerate_scansion_tool(module, dry_run=True)
    print()
    propagate_to_gui_complete(changed_lookup, args.dry_run)


if __name__ == '__main__':
    main()
