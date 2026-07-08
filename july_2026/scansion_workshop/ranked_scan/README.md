# ranked_scan

Everything for the ranked scansion pipeline lives here: the versioned model
files, the two scripts you run, their shared dependencies, and the reports
those scripts produce. This is the one folder you need for day-to-day work;
`../scansion_analysis/` holds earlier one-off investigation scripts that
are no longer part of the main workflow.

## Files

- **`ranked_scan_v1.py`, `ranked_scan_v2.py`, ...** — the ranking program
  itself. Each version is a full, self-contained copy (same convention as
  `scansion.py` → `scansion_v2.py`): when you want to try a tweak, copy the
  latest `ranked_scan_vN.py` to `ranked_scan_v{N+1}.py` and edit the copy.
  Never edit an existing version in place — the whole point of versioning
  is being able to compare the new idea against everything that came
  before it.
- **`assess_versions.py`** — run this after adding a new version, or after
  flagging more lines green, to see whether it actually helped.
- **`update_non_green_with_latest.py`** — run this to make the highest
  version the one actually producing scansion for un-vetted lines.
- **`find_productive_lines.py`** — run this before a manual-scanning
  session to get a list of not-yet-green lines worth flagging next,
  prioritized so you're not repeatedly confirming words you've already
  scanned plenty of. See "Deciding what to scan next" below.
- **`mine_word_list_gaps.py`** — run this after flagging more lines green
  to find words the current version *mechanically cannot* scan the way
  you did (the human-attested syllable count is unreachable by every
  candidate the search enumerates, so no ranking or abstention change can
  ever fix them). Prints ready-to-paste `MINED_SYLLABLE_WORDS` entries
  (the dict `ranked_scan_v7.py` introduced) for the next version. By
  default it mines only the train half of the same seed-42 split
  `assess_versions.py` scores against, so held-out assessment of the
  resulting version stays honest; `--all-green` mines everything, for a
  deployment-only word list (see the leakage warning in the script's
  docstring before trusting any assessment of a version built that way).
- **`generate_green_word_patterns.py`, `confidence_calibration.py`,
  `green_corpus_utils.py`** — shared code the scripts above depend on.
  You shouldn't need to touch these directly.
- **`green_word_patterns.json`** — word → {stress pattern: count}, built
  from ALL green-flagged lines. This is the reference table every
  `ranked_scan_vN.py` ranks candidate scansions against **in real use**
  (via `update_non_green_with_latest.py`). It is deliberately *not* what
  `assess_versions.py` scores versions against — see "Train/test split"
  below.
- **`version_comparison.csv`, `version_reports/*.png`** — output of
  `assess_versions.py` (see below).

## Workflow: trying a tweak

1. Copy the current latest version to a new one and make your change:
   ```
   cp ranked_scan_v1.py ranked_scan_v2.py
   ```
2. Run the assessment:
   ```
   python3 assess_versions.py
   python3 assess_versions.py --target-accuracy 95   # or whatever bar matters to you
   python3 assess_versions.py --train-pct 70          # train on 70% instead of 50%
   ```
   This randomly splits the green-flagged lines into a train set
   (`--train-pct`, default 50%) and a held-out test set (the rest — see
   "Train/test split" below), builds the reference table from the train
   set only, runs *every* `ranked_scan_vN.py` found in this folder against
   the test set, and writes/prints:
   - `version_comparison.csv` — one row per held-out test line, with the
     gold scansion plus each version's scansion, syllable count, match
     flag, and confidence score, side by side.
   - `version_reports/accuracy_by_confidence_v{N}.png` — a histogram of
     accuracy per confidence bucket (0–10, 10–20, ..., 90–100), so you can
     see at a glance whether confidence is tracking correctness for that
     version.
   - A recommendation for **best raw accuracy** (highest match rate against
     green, no filtering) and **best calibration** (the version needing
     the *lowest* confidence threshold to hit `--target-accuracy`, i.e.
     whose confidence score does the most useful work — see "Confidence
     and calibration" below).
   - Since v6, versions may **abstain** — deliberately output a blank
     scansion for lines they can't vouch for (see "Abstention" below).
     Raw accuracy counts every blank as a miss, so the table also shows
     **answered** (how many test lines got a non-blank scansion) and
     **prec. on answered** (accuracy over just those). For an abstaining
     version, judge it on those two columns, not raw accuracy.
3. If the new version wins on the metric you care about, keep it. If not,
   delete the file — nothing else references a version until you run the
   scripts above.

## Train/test split

`assess_versions.py` never scores a version on the same lines its
reference table was built from. Early on it did exactly that (reference
built from *all* green lines, scored against those same lines), which
inflates accuracy — a word that occurs only once in the green set will
always score 100% on its own line, no matter how good the rule that
produced it actually is. Now: green-flagged lines are shuffled (with
`--seed`, default fixed so reruns are comparable) and split into train
(`--train-pct`, default 50%) and test; the reference table is built from
train only, kept in memory, and never written to `green_word_patterns.json`
on disk; every version is scored only against the test lines. All versions
in a single run share the same split, so comparisons between them stay
fair.

`green_word_patterns.json` on disk is untouched by `assess_versions.py` --
it's only ever (re)written by `update_non_green_with_latest.py`, from ALL
green-flagged data, for real deployment. Held-out accuracy during
assessment will typically run a bit below what you'd see if you (wrongly)
tested on the full set the live reference table was built from.

## Workflow: shipping a version

Once you've picked a version (usually just "whichever is highest," since
that should be your best one — the two scripts don't rename or reorder
files for you):

```
python3 update_non_green_with_latest.py --dry-run   # see what would change first
python3 update_non_green_with_latest.py             # write for real
```

This regenerates `OXFORD_SCANSION` for every row in `scansion_tool/*/*.csv`
that isn't green-flagged, using the highest-numbered `ranked_scan_vN.py`
(pass `--version N` to pin a specific one instead), and:

- Never touches green-flagged rows — those are human-vetted ground truth.
- Writes an `OXFORD_SCANSION_CONFIDENCE` column for *every* row (green rows
  get scored against their existing human scansion; a blank scansion
  scores 0).
- Propagates both `OXFORD_SCANSION` and the confidence column into the
  matching rows of `dataset/combined.csv` (matched by `OUTPUT_FILENAME` +
  `LINE_NUMBER`; ~40 rows there have a pre-existing malformed-header
  issue and are skipped rather than guessed at).
- Writes `scansion_tool/confidence_calibration.json` — the
  accuracy-at-each-confidence-threshold curve for whichever version you
  just used, so `oxford_scansion_gui.html`'s "Accuracy below X%" filter
  stays in sync with the confidence numbers actually sitting in the data.

## Deciding what to scan next

Manually flagging lines green in order (or at random) tends to
over-confirm words you've already scanned plenty of and under-cover rare
ones, since common words dominate any random sample. Run this before a
scanning session instead:

```
python3 find_productive_lines.py                       # top 100
python3 find_productive_lines.py --count 200
python3 find_productive_lines.py --max-confidence 50    # also focus on lines the model is unsure about
```

It scores every not-yet-green line by how much its words still "need" more
green-flagged examples (a word gets a full `--min-desired-count`, default
3, worth of need if unseen, less if it already has some, zero once it's
had enough), picks lines greedily by that score, and reduces the need for
every word a picked line contains before scoring the rest -- so it won't
recommend ten different lines that all just repeat the same rare word.
Writes `productive_lines.csv` (also prints the top 20) with each line's
current confidence, current generated scansion, and which words earned it
a spot, so you can jump straight into the GUI armed with a worklist instead
of reading straight through a whole poem hoping to stumble onto unusual
vocabulary.

## Abstention (v6+)

A wrong-but-present scansion silently pollutes the data; a blank one reads
as 0 syllables / 0 confidence and surfaces in the GUI's "Scansion
Failures" worklist, where it gets found and hand-scanned. So from v6 on,
`scan()` returns a blank scansion — the same shape as a total search
failure — whenever it can't vouch for its own pick:

- **Method disagreement**: v6 picks greedily (first valid candidate in
  linguistic-default order, v4's rule) but also computes what pure
  corpus-frequency ranking (v2's rule) would have chosen from the same
  candidates. Held out, the greedy pick is right 84% of the time when the
  two methods agree and only 43% when they disagree, so disagreement
  triggers a blank.
- **Low corpus confidence**: even with agreement, a chosen reading whose
  `score_pattern()` falls below `ABSTAIN_MIN_CONFIDENCE` (see the constant
  in the version file for the measured coverage/precision trade-off at
  each threshold) is blanked.

Why a veto rather than a blended score: arithmetic blends of the greedy
and corpus scores were swept thoroughly (v5's `order_weight`; evidence-
filtered and per-word-weighted variants) and all capped within one test
line of pure greedy. 58.5% of reference-table words have exactly one green
occurrence, so the corpus score is too noisy to out-*pick* the linguistic
defaults — but strong enough to *flag* picks it clearly contradicts.

## Confidence and calibration

Confidence is **not** a correctness probability — it's how closely a
line's chosen reading matches typical green-flagged word/pattern
frequencies (each word's score is the % of that word's green occurrences
sharing this exact stress pattern, x's stripped, averaged over the line).
It's a useful proxy (empirically, higher confidence does mean higher
accuracy) but not a guarantee.

Two different calibration curves exist, for two different purposes, and
they're not directly comparable:
- `assess_versions.py`'s numbers (and `version_reports/*.png`) are
  held-out, per the train/test split above — a fair read on a version's
  real-world accuracy.
- `scansion_tool/confidence_calibration.json` (written by
  `update_non_green_with_latest.py`, read by the GUI) is built and scored
  against ALL green-flagged data, no split — appropriate there because it
  needs to describe the actual deployed reference table
  (`green_word_patterns.json`, also built from all green data), not a
  hypothetical held-out version of it.

"Best calibration" in `assess_versions.py`'s output means: among versions
that can reach the target accuracy at all, which one needs the *lowest*
confidence threshold to get there. Lower is better because it means fewer
lines get filtered out to hit the same accuracy bar — the version's
confidence score is doing more of the work, not just being conservative.

## GUI

`scansion_tool/oxford_scansion_gui.html`'s "Accuracy below X%" worklist
filter reads `scansion_tool/confidence_calibration.json` (written by
`update_non_green_with_latest.py`) and resolves your target accuracy to a
confidence threshold client-side, showing it in parentheses next to the
input. If you run `update_non_green_with_latest.py` with a different
version, re-open the GUI (or reload) to pick up the new calibration curve.
