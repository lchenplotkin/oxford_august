"""
Finds words whose green-attested syllable count is UNREACHABLE by the
latest ranked_scan version's analyze_word() -- i.e. no candidate the
search enumerates can ever scan that word the way the human did, no matter
how candidates get ranked or vetoed. These are mechanical word-list gaps;
the output is ready-to-paste MINED_SYLLABLE_WORDS entries (the dict
ranked_scan_v7.py introduced).

Methodology / leakage warning: by default this mines the TRAIN half of the
same seed-42 split assess_versions.py scores against, so the held-out test
set stays a fair judge of whether mined entries generalize. Once you've
frozen a version and want the best possible deployed word list, re-run
with --all-green to mine every green line -- but never paste --all-green
output into a version and then trust assess_versions.py's numbers for it,
because the test split's own lines helped build that list.

Each emitted entry's count options are ordered by attestation frequency
(most-attested first, current rules' reachable counts appended as
fallbacks), so the greedy default-order search tries the human-preferred
reading first. Words already covered by VARIABLE_SYLLABLE_WORDS (or
MINED_SYLLABLE_WORDS) are reported separately for manual review instead of
re-emitted. Function words / apostrophe fragments whose miscounts are
elision-context artifacts (me/the/ne attested 0 = fully elided) are
excluded -- a whole-word 0-syllable option would let any line silently
drop them.

Usage:
    python3 mine_word_list_gaps.py                 # train split only (safe for assess_versions.py)
    python3 mine_word_list_gaps.py --all-green     # everything (deployment only, see above)
    python3 mine_word_list_gaps.py --version 4     # mine against a specific version
    python3 mine_word_list_gaps.py --seed 7 --train-pct 70
"""

import argparse
import glob
import importlib.util
import os
import random
import re
from collections import Counter, defaultdict
from itertools import product

from green_corpus_utils import iter_green_rows

RANKED_SCAN_DIR = os.path.dirname(os.path.abspath(__file__))
SCANSION_TOOL_DIR = os.path.join(os.path.dirname(RANKED_SCAN_DIR), 'scansion_tool')
VERSION_RE = re.compile(r'^ranked_scan_v(\d+)\.py$')
DEFAULT_SEED = 42

# Miscounts on these are elision-context / apostrophe-fragment artifacts,
# not word-list problems (see docstring).
STOPLIST = {'me', 'the', 'ne', 'that', 'th', 'm', 'yn', 'n', 't', 's'}


def latest_version():
    versions = {}
    for path in glob.glob(os.path.join(RANKED_SCAN_DIR, 'ranked_scan_v*.py')):
        m = VERSION_RE.match(os.path.basename(path))
        if m:
            versions[int(m.group(1))] = path
    return max(versions), versions[max(versions)]


def load_version(version):
    path = os.path.join(RANKED_SCAN_DIR, f'ranked_scan_v{version}.py')
    spec = importlib.util.spec_from_file_location(f'ranked_scan_v{version}', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def train_split(rows, train_pct, seed):
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    return shuffled[:round(len(shuffled) * train_pct / 100.0)]


def reachable_counts(mod, word, prev_word, next_word):
    """All word-total syllable counts analyze_word's cluster options allow."""
    cluster_options = mod.analyze_word(word, prev_word, next_word)
    return {sum(combo) for combo in product(*cluster_options)}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--version', type=int, default=None,
                        help='ranked_scan version to mine against (default: latest)')
    parser.add_argument('--all-green', action='store_true',
                        help='mine ALL green lines instead of the train split (deployment only -- see docstring)')
    parser.add_argument('--train-pct', type=float, default=50,
                        help='train split size, must match assess_versions.py (default: 50)')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help=f'split seed, must match assess_versions.py (default: {DEFAULT_SEED})')
    args = parser.parse_args()

    if args.version is None:
        version, _ = latest_version()
    else:
        version = args.version
    mod = load_version(version)

    all_rows = list(iter_green_rows(SCANSION_TOOL_DIR))
    if args.all_green:
        rows = all_rows
        print(f"# Mining ALL {len(rows)} green lines against ranked_scan_v{version} "
              f"-- do NOT trust assess_versions.py numbers for a version built from this")
    else:
        rows = train_split(all_rows, args.train_pct, args.seed)
        print(f"# Mining {len(rows)} train lines (seed={args.seed}, {args.train_pct:.0f}%) "
              f"against ranked_scan_v{version}; test split untouched")

    word_attested = defaultdict(Counter)
    word_reachable = defaultdict(set)
    unreachable_words = set()
    unreachable_occurrences = 0

    for line in rows:
        text, scansion = line['text'], line['green_scansion']
        if not text or not scansion:
            continue
        words = mod.minimal_clean(text).split()
        tokens = scansion.split(' ')
        if len(words) != len(tokens):
            continue
        for i, (word, tok) in enumerate(zip(words, tokens)):
            attested = len(re.sub(r'[^SuU]', '', tok))
            prev_word = words[i - 1] if i > 0 else None
            next_word = words[i + 1] if i < len(words) - 1 else None
            w = word.lower()
            try:
                reach = reachable_counts(mod, word, prev_word, next_word)
            except Exception:
                continue
            word_attested[w][attested] += 1
            word_reachable[w] |= reach
            if attested not in reach:
                unreachable_words.add(w)
                unreachable_occurrences += 1

    already_covered = []
    entries = {}
    for w in sorted(unreachable_words):
        if w in STOPLIST:
            continue
        if mod.get_variable_word_syllables(w, mod.simplify_word(w)) is not None:
            already_covered.append(w)
            continue
        ordered = [c for c, _ in word_attested[w].most_common() if c > 0]
        for c in sorted(word_reachable[w]):
            if c not in ordered and c > 0:
                ordered.append(c)
        if ordered:
            entries[w] = ordered

    print(f"# {unreachable_occurrences} unreachable word-occurrences, "
          f"{len(entries)} new entries:\n")
    for w, counts in entries.items():
        att = dict(word_attested[w])
        print(f'\t"{w}": (["{w}"], {counts}),  # attested {att}')

    if already_covered:
        print(f"\n# Already have VARIABLE/MINED entries but still miss attested counts")
        print(f"# (needs manual review, likely a wrong existing entry):")
        for w in already_covered:
            print(f"#   {w}: attested {dict(word_attested[w])}, "
                  f"entry gives {mod.get_variable_word_syllables(w, mod.simplify_word(w))}")


if __name__ == '__main__':
    main()
