"""
Given a target accuracy rate (default 95%), finds the lowest confidence
threshold T such that "accept every line with RANKED_CONFIDENCE >= T" would
have hit that accuracy rate on the green-flagged (human-vetted) set -- i.e.
the confidence level above which we can expect that accuracy.

Data source: ranked_green_comparison.csv (RANKED_CONFIDENCE + RANKED_MATCHES_GREEN
columns), built by make_ranked_green_csv.py. This is an empirical calibration
against the one sample we have ground truth for, not a guarantee -- see the
CAVEATS printed at the bottom of the output.

"Lowest threshold that meets the target" is used rather than "the threshold
with the single highest accuracy" because it maximizes coverage (how many
non-green lines would pass the filter) subject to the accuracy floor -- a
tighter/higher threshold that also clears the target just throws away lines
for no benefit.

The underlying math (threshold_for_target_accuracy, accuracy_table) lives in
ranked_scan/confidence_calibration.py, shared with assess_versions.py.

Usage:
    python3 find_confidence_threshold.py             # target 95%
    python3 find_confidence_threshold.py --target 90
    python3 find_confidence_threshold.py --target 95 --data path/to/other_comparison.csv
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ranked_scan'))
from confidence_calibration import (
    load_confidence_match_pairs,
    threshold_for_target_accuracy,
    accuracy_table,
    best_achievable_accuracy,
)

DEFAULT_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ranked_green_comparison.csv')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--target', type=float, default=95,
                         help='target accuracy rate as a percentage, e.g. 95 (default: 95)')
    parser.add_argument('--data', default=DEFAULT_DATA,
                         help=f'path to a *_green_comparison.csv with RANKED_CONFIDENCE + RANKED_MATCHES_GREEN columns (default: {DEFAULT_DATA})')
    args = parser.parse_args()

    target = args.target / 100.0
    pairs = load_confidence_match_pairs(args.data)
    if not pairs:
        print(f"No usable rows found in {args.data}")
        return

    overall_acc = sum(m for _, m in pairs) / len(pairs)
    print(f"Loaded {len(pairs)} green-flagged lines from {args.data}")
    print(f"Overall accuracy (no filtering): {overall_acc*100:.2f}%\n")

    result = threshold_for_target_accuracy(pairs, target)
    if result is None:
        best = best_achievable_accuracy(pairs)
        print(f"No confidence threshold reaches {target*100:.0f}% accuracy on this data.")
        print(f"Highest achievable accuracy is at confidence >= {best[0]:.2f}: "
              f"{best[1]*100:.2f}% ({best[2]}/{best[3]} lines).")
        return

    threshold, achieved, n_at, n_total = result
    print(f"Target accuracy: {target*100:.0f}%")
    print(f"Confidence threshold: >= {threshold:.2f}")
    print(f"Achieved accuracy at that threshold: {achieved*100:.2f}% ({n_at}/{n_total} lines, "
          f"{n_at/n_total*100:.1f}% coverage of the green set)")

    print("\nAccuracy at nearby round thresholds, for context:")
    print(f"{'threshold':>10s} {'accuracy':>10s} {'n':>8s} {'coverage':>10s}")
    for t, acc, count, _matches in accuracy_table(pairs, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]):
        if acc is None:
            print(f"{t:>10.0f} {'--':>10s} {count:>8d} {count/len(pairs)*100:>9.1f}%")
        else:
            print(f"{t:>10.0f} {acc*100:>9.2f}% {count:>8d} {count/len(pairs)*100:>9.1f}%")

    print("""
CAVEATS:
- This is an empirical calibration against the green-flagged set only (the
  same lines the reference frequency table was built from), not a held-out
  test. Treat it as a rough guide, not a statistical guarantee, especially
  near the top of the confidence range where n gets small.
- The confidence score measures how closely a line's chosen reading matches
  typical green-flagged word/pattern frequencies, not correctness directly
  -- see the note in ranked_green_comparison.csv's average confidence
  split (matched vs. unmatched lines) for how well that proxy tracks
  actual accuracy.""")


if __name__ == '__main__':
    main()
