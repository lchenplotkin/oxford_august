"""
Recommends which NOT-yet-green-flagged lines to scan next, prioritizing
lines that would teach green_word_patterns.json the most -- instead of
just picking lines at random (or in text order), which tends to re-confirm
words/patterns you've already scanned plenty of and under-cover rare ones.

How "productive" is defined: every word gets a "need" score --
max(0, --min-desired-count minus how many green-flagged occurrences it
already has (all patterns of that word summed)). A word never seen at all
starts at the full --min-desired-count; a word already seen that many
times or more has need 0 and stops driving recommendations. A candidate
line's value is the sum of its words' current need.

This is a *greedy weighted set-cover*, not just "sort by novelty and take
the top N": after each line is picked, every word it contains has its need
reduced by 1 (as if this scan will supply one more example), so a second
line hitting the same rare word again is worth less next time around, and
recommending 50 lines that all lean on the same handful of rare words is
avoided almost automatically. (Implemented as lazy-greedy over a heap --
gains only ever go down as things get picked, so a stale heap entry is
always safe to recompute-and-recheck rather than trust.)

Ties are common -- gain is a small integer, so it's normal for hundreds of
candidate lines across many different works to land on the exact same
value. Whichever ones "win" a tie only matters for --count cutoff purposes,
but the underlying candidate order (folders in a fixed order, files sorted
alphabetically, rows in file order) would otherwise decide those ties --
systematically favoring alphabetically-early files for no principled
reason. Candidates are shuffled once up front (with --seed, fixed by
default so reruns are reproducible) specifically so tie-breaks spread
across the whole dataset instead of always resolving toward whichever
files happen to sort first.

Each recommended line also shows its current OXFORD_SCANSION_CONFIDENCE
(from the last update_non_green_with_latest.py run) and current generated
scansion, for context -- low confidence there usually (not always) lines
up with high need, since both stem from the same lack of data.

Usage:
    python3 find_productive_lines.py                       # top 100
    python3 find_productive_lines.py --count 200
    python3 find_productive_lines.py --min-desired-count 5  # want more examples per word before it's "enough"
    python3 find_productive_lines.py --max-confidence 50    # only consider lines the model is already unsure about
    python3 find_productive_lines.py --seed 7               # different tie-break shuffle
"""

import argparse
import heapq
import os
import random
import re
from collections import defaultdict
import csv

import generate_green_word_patterns
from green_corpus_utils import iter_all_rows

RANKED_SCAN_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(RANKED_SCAN_DIR)
SCANSION_TOOL_DIR = os.path.join(ROOT, 'scansion_tool')
OUTPUT_CSV = os.path.join(RANKED_SCAN_DIR, 'productive_lines.csv')
DEFAULT_SEED = 42


def word_tokens(text):
    return [w for w in (re.sub(r'[^a-zA-Z]', '', tok).lower() for tok in (text or '').split()) if w]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--count', type=int, default=500,
                         help='how many lines to recommend (default: 100)')
    parser.add_argument('--min-desired-count', type=int, default=10,
                         help='how many green-flagged examples of a word we want before it stops driving recommendations (default: 3)')
    parser.add_argument('--max-confidence', type=float, default=None,
                         help='optional: only consider candidate lines whose current confidence is below this (e.g. 50), to focus on lines the model is already unsure about')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                         help=f'random seed for shuffling tie-break order across the dataset (default: {DEFAULT_SEED})')
    args = parser.parse_args()

    all_rows = list(iter_all_rows(SCANSION_TOOL_DIR))
    green_rows = [r for r in all_rows if r['is_green']]
    candidates = [r for r in all_rows if not r['is_green']]
    if args.max_confidence is not None:
        candidates = [r for r in candidates
                      if r['confidence'] is None or r['confidence'] < args.max_confidence]

    # Shuffle before anything else touches candidate order. Gain ties are
    # extremely common (small integers), and without this, ties always
    # resolve toward whichever files happen to sort first alphabetically --
    # this makes tie-breaks spread across the whole dataset instead.
    random.Random(args.seed).shuffle(candidates)

    print(f"{len(green_rows)} green-flagged rows (reference), "
          f"{len(candidates)} candidate not-yet-green rows"
          + (f" with confidence < {args.max_confidence}" if args.max_confidence is not None else "")
          + f" (seed={args.seed})")

    patterns, _ = generate_green_word_patterns.build_patterns(
        [{'text': r['text'], 'green_scansion': r['scansion']} for r in green_rows]
    )
    ref_count = {w: sum(c.values()) for w, c in patterns.items()}

    cand_words = [word_tokens(r['text']) for r in candidates]

    need = defaultdict(lambda: args.min_desired_count)
    for w, c in ref_count.items():
        need[w] = max(0, args.min_desired_count - c)

    def gain(i):
        return sum(need[w] for w in cand_words[i])

    heap = [(-gain(i), i) for i in range(len(candidates))]
    heapq.heapify(heap)

    selected = []
    while heap and len(selected) < args.count:
        _, i = heapq.heappop(heap)
        g = gain(i)
        if g <= 0:
            continue  # fully covered by earlier picks (or never needed) -- drop, don't recommend
        if not heap or g >= -heap[0][0]:
            contributing = sorted({w for w in cand_words[i] if need[w] > 0},
                                   key=lambda w: (-need[w], w))
            selected.append((candidates[i], g, contributing))
            for w in cand_words[i]:
                if need[w] > 0:
                    need[w] -= 1
        else:
            heapq.heappush(heap, (-g, i))

    print(f"Selected {len(selected)} productive lines "
          f"(ran out of positive-gain candidates)" if len(selected) < args.count else
          f"Selected {len(selected)} productive lines")

    fieldnames = ['rank', 'folder', 'file', 'line_number', 'oxford_text',
                  'current_confidence', 'current_scansion', 'gain', 'needed_words']
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, (row, g, contributing) in enumerate(selected, 1):
            needed_words_str = ', '.join(f"{w}({ref_count.get(w, 0)})" for w in contributing)
            writer.writerow({
                'rank': rank,
                'folder': row['folder'],
                'file': row['file'],
                'line_number': row['line_number'],
                'oxford_text': row['text'],
                'current_confidence': '' if row['confidence'] is None else f"{row['confidence']:.2f}",
                'current_scansion': row['scansion'],
                'gain': g,
                'needed_words': needed_words_str,
            })

    print(f"Wrote {OUTPUT_CSV}\n")
    print(f"{'rank':>4s}  {'gain':>4s}  {'conf':>6s}  {'file':25s}  text / needed words")
    for rank, (row, g, contributing) in enumerate(selected[:20], 1):
        conf_str = '--' if row['confidence'] is None else f"{row['confidence']:.1f}"
        print(f"{rank:4d}  {g:4d}  {conf_str:>6s}  {row['file']:25s}  {row['text']}")
        print(f"{'':4s}  {'':4s}  {'':6s}  {'':25s}  needs: {', '.join(contributing[:8])}"
              + (', ...' if len(contributing) > 8 else ''))
    if len(selected) > 20:
        print(f"... and {len(selected) - 20} more in {OUTPUT_CSV}")


if __name__ == '__main__':
    main()
