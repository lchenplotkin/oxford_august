"""
Assesses every ranked_scan_vN.py version against the current green-flagged
(human-vetted) data, end to end:

1. Regenerates green_word_patterns.json from whatever green-flagged data
   exists right now (so tweaks + newly-flagged lines are always reflected).
2. Runs every ranked_scan/ranked_scan_vN.py against every green-flagged
   line and writes version_comparison.csv: one row per line, with the gold
   scansion plus each version's scansion/syllables/match/confidence.
3. Plots an accuracy-vs-confidence histogram per version (10-point-wide
   confidence buckets) to version_reports/accuracy_by_confidence_vN.png.
4. Reports two recommendations:
   - Best raw accuracy: the version that matches green the most often,
     no filtering.
   - Best calibration: among versions that can reach --target-accuracy
     (default 90%) at all, the one needing the LOWEST confidence threshold
     to get there -- i.e. the version whose confidence score is doing the
     most work, so the fewest lines get filtered out to hit the target.

Usage:
    python3 assess_versions.py                    # target accuracy 90%
    python3 assess_versions.py --target-accuracy 95
"""

import argparse
import glob
import importlib.util
import os
import re
import sys
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import generate_green_word_patterns
from confidence_calibration import threshold_for_target_accuracy, best_achievable_accuracy
from green_corpus_utils import iter_green_rows, normalize, syllable_count, regenerate

ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(ANALYSIS_DIR)
RANKED_SCAN_DIR = os.path.join(ROOT, 'ranked_scan')
SCANSION_TOOL_DIR = os.path.join(ROOT, 'scansion_tool')
REPORTS_DIR = os.path.join(ANALYSIS_DIR, 'version_reports')
COMPARISON_CSV = os.path.join(ANALYSIS_DIR, 'version_comparison.csv')

VERSION_RE = re.compile(r'^ranked_scan_v(\d+)\.py$')


def discover_versions():
    """{version_number: path}, sorted ascending by version number."""
    versions = {}
    for path in sorted(glob.glob(os.path.join(RANKED_SCAN_DIR, 'ranked_scan_v*.py'))):
        m = VERSION_RE.match(os.path.basename(path))
        if not m:
            continue
        versions[int(m.group(1))] = path
    return dict(sorted(versions.items()))


def load_version_module(version, path):
    name = f'ranked_scan_v{version}'
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_versions_against_green(modules):
    """
    Returns (rows_out, per_version_pairs):
      rows_out: list of dicts for the comparison CSV.
      per_version_pairs: {version: [(confidence, is_match), ...]}
    """
    rows_out = []
    per_version_pairs = {v: [] for v in modules}

    for line in iter_green_rows(SCANSION_TOOL_DIR):
        row = {
            'OG_OXFORD_TEXT': line['og_text'],
            'OXFORD_TEXT': line['text'],
            'GREEN_SCANSION': line['green_scansion'],
            'GREEN_SYLLABLES': syllable_count(line['green_scansion']),
        }
        green_norm = normalize(line['green_scansion'])

        for v, mod in modules.items():
            scansion = regenerate(mod, line['text'], line['target'])
            confidence = mod.line_confidence(line['text'], scansion)
            is_match = int(normalize(scansion) == green_norm)

            row[f'V{v}_SCANSION'] = scansion
            row[f'V{v}_SYLLABLES'] = syllable_count(scansion)
            row[f'V{v}_MATCHES_GREEN'] = is_match
            row[f'V{v}_CONFIDENCE'] = f"{confidence:.2f}"

            per_version_pairs[v].append((confidence, is_match))

        rows_out.append(row)

    return rows_out, per_version_pairs


def write_comparison_csv(rows_out, versions):
    fieldnames = ['OG_OXFORD_TEXT', 'OXFORD_TEXT', 'GREEN_SCANSION', 'GREEN_SYLLABLES']
    for v in versions:
        fieldnames += [f'V{v}_SCANSION', f'V{v}_SYLLABLES', f'V{v}_MATCHES_GREEN', f'V{v}_CONFIDENCE']

    with open(COMPARISON_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Wrote {len(rows_out)} rows to {COMPARISON_CSV}")


def plot_accuracy_histogram(version, pairs):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    bucket_edges = list(range(0, 101, 10))  # 0,10,...,100
    bucket_labels = [f"{lo}-{hi}" for lo, hi in zip(bucket_edges[:-1], bucket_edges[1:])]
    bucket_matches = [0] * (len(bucket_edges) - 1)
    bucket_counts = [0] * (len(bucket_edges) - 1)

    for conf, is_match in pairs:
        idx = min(int(conf // 10), len(bucket_counts) - 1)
        bucket_counts[idx] += 1
        bucket_matches[idx] += is_match

    accuracies = [
        (bucket_matches[i] / bucket_counts[i] * 100) if bucket_counts[i] else 0
        for i in range(len(bucket_counts))
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(bucket_labels, accuracies, color='#4C78A8')
    for bar, count in zip(bars, bucket_counts):
        if count:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"n={count}", ha='center', va='bottom', fontsize=8)
    ax.set_xlabel('Confidence bucket')
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 105)
    ax.set_title(f'ranked_scan_v{version}: accuracy by confidence bucket')
    fig.tight_layout()

    out_path = os.path.join(REPORTS_DIR, f'accuracy_by_confidence_v{version}.png')
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--target-accuracy', type=float, default=90,
                         help='target accuracy rate as a percentage for the calibration recommendation (default: 90)')
    args = parser.parse_args()
    target = args.target_accuracy / 100.0

    print("Step 1/4: regenerating green_word_patterns.json from current green-flagged data...")
    generate_green_word_patterns.main()
    print()

    versions = discover_versions()
    if not versions:
        print(f"No ranked_scan_vN.py files found in {RANKED_SCAN_DIR}")
        return
    print(f"Step 2/4: found {len(versions)} version(s): "
          + ", ".join(f"v{v}" for v in versions))
    modules = {v: load_version_module(v, path) for v, path in versions.items()}

    print("Running every version against every green-flagged line...")
    rows_out, per_version_pairs = run_versions_against_green(modules)
    write_comparison_csv(rows_out, versions)
    print()

    print("Step 3/4: plotting accuracy-vs-confidence histograms...")
    for v, pairs in per_version_pairs.items():
        out_path = plot_accuracy_histogram(v, pairs)
        print(f"  v{v}: {out_path}")
    print()

    print("Step 4/4: scoring versions...")
    total = len(rows_out)
    raw_accuracy = {v: sum(m for _, m in pairs) / total for v, pairs in per_version_pairs.items()}
    calibration = {v: threshold_for_target_accuracy(pairs, target) for v, pairs in per_version_pairs.items()}

    print(f"\n{'version':>10s} {'raw accuracy':>14s} {'threshold for ' + str(int(target*100)) + '% acc':>22s} {'coverage at threshold':>22s}")
    for v in versions:
        acc_str = f"{raw_accuracy[v]*100:.2f}%"
        cal = calibration[v]
        if cal is None:
            thr_str = "unreachable"
            cov_str = "--"
        else:
            thr_str = f">= {cal[0]:.2f}"
            cov_str = f"{cal[2]}/{cal[3]} ({cal[2]/cal[3]*100:.1f}%)"
        print(f"{'v'+str(v):>10s} {acc_str:>14s} {thr_str:>22s} {cov_str:>22s}")

    best_raw = max(raw_accuracy, key=raw_accuracy.get)
    reachable = {v: cal[0] for v, cal in calibration.items() if cal is not None}

    print(f"\nBest raw accuracy: v{best_raw} ({raw_accuracy[best_raw]*100:.2f}% match rate, no filtering)")
    if reachable:
        best_calibration = min(reachable, key=reachable.get)
        cal = calibration[best_calibration]
        print(f"Best calibration for {target*100:.0f}% target: v{best_calibration} "
              f"(confidence >= {cal[0]:.2f} reaches {cal[1]*100:.2f}% accuracy, "
              f"covering {cal[2]}/{cal[3]} lines = {cal[2]/cal[3]*100:.1f}% of the green set)")
    else:
        print(f"No version reaches {target*100:.0f}% accuracy at any confidence threshold.")
        for v, pairs in per_version_pairs.items():
            best = best_achievable_accuracy(pairs)
            if best:
                print(f"  v{v}: best achievable is {best[1]*100:.2f}% at confidence >= {best[0]:.2f} "
                      f"({best[2]}/{best[3]} lines)")


if __name__ == '__main__':
    main()
