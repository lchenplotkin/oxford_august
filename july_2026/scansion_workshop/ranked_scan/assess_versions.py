"""
Assesses every ranked_scan_vN.py version on a held-out split of the
green-flagged (human-vetted) data, end to end:

1. Randomly splits all green-flagged lines into a train set (--train-pct,
   default 50%) and a test set (the rest), reshuffled fresh each run unless
   --seed is fixed (it defaults to a fixed seed, so reruns are comparable
   unless the underlying green-flagged data itself changed).
2. Builds the word/pattern reference table from the TRAIN set only, in
   memory -- this never touches the shared green_word_patterns.json file
   on disk (that file is for real deployment, via
   update_non_green_with_latest.py, and should always reflect ALL
   green-flagged data, not a 50% slice of it).
3. Runs every ranked_scan/ranked_scan_vN.py against the TEST set only,
   scored against that same train-only reference, and writes
   version_comparison.csv: one row per held-out test line, with the gold
   scansion plus each version's scansion/syllables/match/confidence.
4. Plots an accuracy-vs-confidence histogram per version (10-point-wide
   confidence buckets) to version_reports/accuracy_by_confidence_vN.png.
5. Reports two recommendations, both computed on the held-out test set:
   - Best raw accuracy: the version that matches green the most often,
     no filtering.
   - Best calibration: among versions that can reach --target-accuracy
     (default 90%) at all, the one needing the LOWEST confidence threshold
     to get there -- i.e. the version whose confidence score is doing the
     most work, so the fewest lines get filtered out to hit the target.

Why the split matters: green_word_patterns.json is built directly from
green-flagged lines, so scoring a version against those same lines is
partly grading it on lines it (or rather, the reference table) has already
"seen" -- a word that only occurs once in the green set will always score
100% on its own line, regardless of whether the rule that produced it is
actually any good. Held-out evaluation is a fairer read on how a version
would do on the much larger non-green set it'll actually be used on.

Usage:
    python3 assess_versions.py                          # 50/50 split, target accuracy 90%
    python3 assess_versions.py --train-pct 70            # train on 70%, test on the rest
    python3 assess_versions.py --target-accuracy 95
    python3 assess_versions.py --seed 7                  # different random split
"""

import argparse
import glob
import importlib.util
import os
import random
import re
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import generate_green_word_patterns
from confidence_calibration import threshold_for_target_accuracy, best_achievable_accuracy
from green_corpus_utils import iter_green_rows, normalize, syllable_count, regenerate

RANKED_SCAN_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(RANKED_SCAN_DIR)
SCANSION_TOOL_DIR = os.path.join(ROOT, 'scansion_tool')
REPORTS_DIR = os.path.join(RANKED_SCAN_DIR, 'version_reports')
COMPARISON_CSV = os.path.join(RANKED_SCAN_DIR, 'version_comparison.csv')

VERSION_RE = re.compile(r'^ranked_scan_v(\d+)\.py$')
DEFAULT_SEED = 42


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


def train_test_split(rows, train_pct, seed):
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    split_at = round(len(shuffled) * train_pct / 100.0)
    return shuffled[:split_at], shuffled[split_at:]


def collapse_reference(raw):
    """
    build_patterns() returns raw (uncollapsed) pattern counts, same schema
    as scansion_word_patterns.json. Every ranked_scan_vN.py's own
    load_reference() collapses x's out of pattern keys before scoring
    (silent vs. sounded final -e isn't a different pattern for ranking
    purposes) -- do the same collapsing here so a reference we hand
    directly to scan()/line_confidence() (bypassing load_reference()
    entirely) is in the format score_pattern() expects.
    """
    collapsed = {}
    for word, pattern_counts in raw.items():
        bucket = collapsed.setdefault(word, {})
        for pattern, count in pattern_counts.items():
            stripped = re.sub(r'[xX]', '', pattern)
            if not stripped:
                continue
            bucket[stripped] = bucket.get(stripped, 0) + count
    return collapsed


def run_versions_against_test_set(modules, test_rows, train_reference):
    """
    Returns (rows_out, per_version_pairs, per_version_answered):
      rows_out: list of dicts for the comparison CSV (test lines only).
      per_version_pairs: {version: [(confidence, is_match), ...]}
      per_version_answered: {version: [bool, ...]} -- False where the
        version output a blank scansion (abstained or failed outright).
    """
    rows_out = []
    per_version_pairs = {v: [] for v in modules}
    per_version_answered = {v: [] for v in modules}

    for line in test_rows:
        row = {
            'OG_OXFORD_TEXT': line['og_text'],
            'OXFORD_TEXT': line['text'],
            'GREEN_SCANSION': line['green_scansion'],
            'GREEN_SYLLABLES': syllable_count(line['green_scansion']),
        }
        green_norm = normalize(line['green_scansion'])

        for v, mod in modules.items():
            scansion = regenerate(mod, line['text'], line['target'], reference=train_reference)
            confidence = mod.line_confidence(line['text'], scansion, reference=train_reference)
            is_match = int(normalize(scansion) == green_norm)

            row[f'V{v}_SCANSION'] = scansion
            row[f'V{v}_SYLLABLES'] = syllable_count(scansion)
            row[f'V{v}_MATCHES_GREEN'] = is_match
            row[f'V{v}_CONFIDENCE'] = f"{confidence:.2f}"

            per_version_pairs[v].append((confidence, is_match))
            per_version_answered[v].append(bool(scansion.strip()))

        rows_out.append(row)

    return rows_out, per_version_pairs, per_version_answered


def write_comparison_csv(rows_out, versions):
    fieldnames = ['OG_OXFORD_TEXT', 'OXFORD_TEXT', 'GREEN_SCANSION', 'GREEN_SYLLABLES']
    for v in versions:
        fieldnames += [f'V{v}_SCANSION', f'V{v}_SYLLABLES', f'V{v}_MATCHES_GREEN', f'V{v}_CONFIDENCE']

    with open(COMPARISON_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Wrote {len(rows_out)} held-out test rows to {COMPARISON_CSV}")


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
    ax.set_title(f'ranked_scan_v{version}: accuracy by confidence bucket (held-out test set)')
    fig.tight_layout()

    out_path = os.path.join(REPORTS_DIR, f'accuracy_by_confidence_v{version}.png')
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--train-pct', type=float, default=50,
                         help='%% of green-flagged lines used to build the reference table; the rest are held out for testing (default: 50)')
    parser.add_argument('--target-accuracy', type=float, default=90,
                         help='target accuracy rate as a percentage for the calibration recommendation (default: 90)')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                         help=f'random seed for the train/test split, so reruns are comparable (default: {DEFAULT_SEED})')
    args = parser.parse_args()
    target = args.target_accuracy / 100.0

    all_rows = list(iter_green_rows(SCANSION_TOOL_DIR))
    train_rows, test_rows = train_test_split(all_rows, args.train_pct, args.seed)
    print(f"Step 1/4: split {len(all_rows)} green-flagged lines -> "
          f"{len(train_rows)} train ({args.train_pct:.0f}%) / {len(test_rows)} test "
          f"(seed={args.seed})")
    train_reference, train_rows_used = generate_green_word_patterns.build_patterns(train_rows)
    train_reference = collapse_reference(train_reference)
    print(f"Built reference table from {train_rows_used} train lines "
          f"({len(train_reference)} distinct words). Not written to disk.")
    print()

    versions = discover_versions()
    if not versions:
        print(f"No ranked_scan_vN.py files found in {RANKED_SCAN_DIR}")
        return
    print(f"Step 2/4: found {len(versions)} version(s): "
          + ", ".join(f"v{v}" for v in versions))
    modules = {v: load_version_module(v, path) for v, path in versions.items()}

    print("Running every version against the held-out test set...")
    rows_out, per_version_pairs, per_version_answered = run_versions_against_test_set(modules, test_rows, train_reference)
    write_comparison_csv(rows_out, versions)
    print()

    print("Step 3/4: plotting accuracy-vs-confidence histograms...")
    for v, pairs in per_version_pairs.items():
        out_path = plot_accuracy_histogram(v, pairs)
        print(f"  v{v}: {out_path}")
    print()

    print("Step 4/4: scoring versions (all on the held-out test set)...")
    total = len(rows_out)
    raw_accuracy = {v: sum(m for _, m in pairs) / total for v, pairs in per_version_pairs.items()}
    calibration = {v: threshold_for_target_accuracy(pairs, target) for v, pairs in per_version_pairs.items()}

    # Abstention-aware view: a version that deliberately outputs blank
    # scansion for lines it can't vouch for (v6+) looks worse on raw
    # accuracy (every blank counts as a miss there), so also report how
    # often it answers at all, and how accurate it is when it does.
    answered_stats = {}
    for v, answered_flags in per_version_answered.items():
        n_answered = sum(answered_flags)
        n_correct = sum(m for (_, m), a in zip(per_version_pairs[v], answered_flags) if a)
        answered_stats[v] = (n_answered, n_correct / n_answered if n_answered else 0.0)

    print(f"\n{'version':>10s} {'raw accuracy':>14s} {'answered':>16s} {'prec. on answered':>18s} {'threshold for ' + str(int(target*100)) + '% acc':>22s} {'coverage at threshold':>22s}")
    for v in versions:
        acc_str = f"{raw_accuracy[v]*100:.2f}%"
        n_answered, answered_prec = answered_stats[v]
        ans_str = f"{n_answered}/{total} ({n_answered/total*100:.1f}%)"
        prec_str = f"{answered_prec*100:.2f}%"
        cal = calibration[v]
        if cal is None:
            thr_str = "unreachable"
            cov_str = "--"
        else:
            thr_str = f">= {cal[0]:.2f}"
            cov_str = f"{cal[2]}/{cal[3]} ({cal[2]/cal[3]*100:.1f}%)"
        print(f"{'v'+str(v):>10s} {acc_str:>14s} {ans_str:>16s} {prec_str:>18s} {thr_str:>22s} {cov_str:>22s}")

    best_raw = max(raw_accuracy, key=raw_accuracy.get)
    reachable = {v: cal[0] for v, cal in calibration.items() if cal is not None}

    print(f"\nBest raw accuracy: v{best_raw} ({raw_accuracy[best_raw]*100:.2f}% match rate, no filtering, on held-out test set)")
    if reachable:
        best_calibration = min(reachable, key=reachable.get)
        cal = calibration[best_calibration]
        print(f"Best calibration for {target*100:.0f}% target: v{best_calibration} "
              f"(confidence >= {cal[0]:.2f} reaches {cal[1]*100:.2f}% accuracy, "
              f"covering {cal[2]}/{cal[3]} lines = {cal[2]/cal[3]*100:.1f}% of the held-out test set)")
    else:
        print(f"No version reaches {target*100:.0f}% accuracy at any confidence threshold on the held-out test set.")
        for v, pairs in per_version_pairs.items():
            best = best_achievable_accuracy(pairs)
            if best:
                print(f"  v{v}: best achievable is {best[1]*100:.2f}% at confidence >= {best[0]:.2f} "
                      f"({best[2]}/{best[3]} lines)")


if __name__ == '__main__':
    main()
