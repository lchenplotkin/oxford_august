"""
Shared calibration logic: given a list of (confidence, is_match) pairs for
some set of scored lines, find the lowest confidence threshold whose
"confidence >= threshold" subset hits a target accuracy, or build a table
of accuracy at a list of thresholds. Used by find_confidence_threshold.py
(single version, CLI) and assess_versions.py (all versions, comparison +
plots), so there's exactly one implementation of the underlying math.
"""

import bisect
import csv


def load_confidence_match_pairs(path, confidence_col='RANKED_CONFIDENCE', match_col='RANKED_MATCHES_GREEN'):
    pairs = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            conf = row.get(confidence_col, '')
            match = row.get(match_col, '')
            if conf == '' or match == '':
                continue
            pairs.append((float(conf), int(match)))
    return pairs


def _suffix_match_counts(pairs_sorted):
    """pairs_sorted must already be sorted ascending by confidence."""
    n = len(pairs_sorted)
    suffix = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + pairs_sorted[i][1]
    return suffix


def threshold_for_target_accuracy(pairs, target):
    """
    Returns (threshold, achieved_accuracy, n_at_or_above, n_total) for the
    lowest confidence threshold whose "confidence >= threshold" subset hits
    `target` accuracy (0-1), or None if no threshold (including 0, i.e. the
    whole set) achieves it.
    """
    if not pairs:
        return None

    pairs_sorted = sorted(pairs, key=lambda p: p[0])
    confidences = [p[0] for p in pairs_sorted]
    n = len(pairs_sorted)
    suffix_match_count = _suffix_match_counts(pairs_sorted)

    for v in sorted(set(confidences)):
        i = bisect.bisect_left(confidences, v)
        count = n - i
        acc = suffix_match_count[i] / count
        if acc >= target:
            return v, acc, count, n

    return None


def best_achievable_accuracy(pairs):
    """(threshold, accuracy, n_at_or_above, n_total) for whichever threshold has the highest accuracy."""
    if not pairs:
        return None
    pairs_sorted = sorted(pairs, key=lambda p: p[0])
    confidences = [p[0] for p in pairs_sorted]
    n = len(pairs_sorted)
    suffix_match_count = _suffix_match_counts(pairs_sorted)

    best = None
    for v in sorted(set(confidences)):
        i = bisect.bisect_left(confidences, v)
        count = n - i
        acc = suffix_match_count[i] / count
        if best is None or acc > best[1]:
            best = (v, acc, count, n)
    return best


def accuracy_table(pairs, thresholds):
    """
    List of (threshold, accuracy_or_None, n_at_or_above, matches_at_or_above)
    for each threshold given. The raw match count (not just the rounded
    accuracy percentage) is included so callers can derive the accuracy of
    the *complementary* below-threshold group by subtraction without
    compounding rounding error -- accuracy alone, rounded to 2 decimals,
    isn't precise enough for that once the below-threshold group is small.
    """
    pairs_sorted = sorted(pairs, key=lambda p: p[0])
    confidences = [p[0] for p in pairs_sorted]
    n = len(pairs_sorted)
    suffix_match_count = _suffix_match_counts(pairs_sorted)

    rows = []
    for t in thresholds:
        i = bisect.bisect_left(confidences, t)
        count = n - i
        matches = suffix_match_count[i]
        if count == 0:
            rows.append((t, None, 0, 0))
            continue
        acc = matches / count
        rows.append((t, acc, count, matches))
    return rows
