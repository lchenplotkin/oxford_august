"""
final_e_endings.py - How often does ranked_scan_v9 render a final -e as
stressed (S), unstressed (u), or silent (x), per dataset file?

Runs ranked_scan_v9.scan() fresh over OXFORD_TEXT of every row of every
dataset/*_gui_complete.csv (combined.csv skipped -- it duplicates them),
following update_non_green_with_latest.py's conventions: OXFORD_SYLLABLES
is the target syllable count when present, otherwise 8 for the
octosyllabic texts (BD, HF, Thopas) and 10 for everything else. v9
abstains on low-confidence lines (blank scansion); those lines are
counted but contribute no final -e data.

A "final -e word" here is v9's own is_final_e definition: the word's
last vowel cluster is a lone "e", word-final, after a consonant, and the
word has more than one vowel cluster -- so "trouthe" and "grene" count,
but "the"/"he" (one cluster) and "seye" (final cluster "eye") don't.
The -e's rendering is the last character of the word's scansion token:
x = not pronounced, u = unstressed, S = stressed. Line-final words are
also split out, since a sounded -e there is just the ordinary feminine
ending and would otherwise swamp the (more interesting) internal rates.

Words on v9's VARIABLE_SYLLABLE_WORDS list (noble, temple, viage...) are
scanned as one opaque unit with no per-cluster x markers, so their token
characters can't be mapped onto the final -e; they're tallied in the
variable_excluded column instead of guessed at (~0.4% of final -e words).

Outputs (final_e_analysis_output/):
  final_e_by_file.csv  per-file S/u/x counts + rates, internal vs line-final
  final_e_by_word.csv  per-word S/u/x counts across the whole dataset

Usage:
	python3 final_e_endings.py
"""

import csv
import glob
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'scansion_workshop', 'ranked_scan'))
import ranked_scan_v9 as rs

DATASET_DIR = os.path.join(HERE, 'dataset')
OUTPUT_DIR = os.path.join(HERE, 'final_e_analysis_output')
DEFAULT_TARGET = 10
SHORT_VERSE_PREFIXES = ('bd_', 'hf_', 'thop')  # octosyllabic texts


def has_final_e(word: str) -> bool:
	"""v9's is_final_e condition, applied to a whole word."""
	w = rs.minimal_clean(word.lower())
	clusters = rs.vowel_clusters(w)
	return (len(clusters) > 1
			and clusters[-1][1] == 'e'
			and clusters[-1][2] == ''
			and rs.ends_with_consonant_e(w))


def target_for_row(row, csv_name: str) -> int:
	raw = (row.get('OXFORD_SYLLABLES') or '').strip()
	if raw.isdigit() and int(raw) > 0:
		return int(raw)
	return 8 if csv_name.lower().startswith(SHORT_VERSE_PREFIXES) else DEFAULT_TARGET


def main():
	os.makedirs(OUTPUT_DIR, exist_ok=True)
	reference = rs.load_reference()

	file_stats = {}
	word_stats = defaultdict(Counter)

	paths = sorted(glob.glob(os.path.join(DATASET_DIR, '*_gui_complete.csv')))
	for path in paths:
		csv_name = os.path.basename(path)
		text_name = csv_name.replace('_gui_complete.csv', '')
		stats = file_stats[text_name] = Counter()

		with open(path, newline='', encoding='utf-8') as f:
			for row in csv.DictReader(f):
				text = (row.get('OXFORD_TEXT') or '').strip()
				if not text:
					continue
				stats['lines_total'] += 1

				tokens, total_syllables = rs.scan(text, target_for_row(row, csv_name), reference)
				if total_syllables == 0:
					stats['lines_abstained'] += 1
					continue
				stats['lines_scanned'] += 1

				words = rs.minimal_clean(text).split()
				last = len(words) - 1
				for i, (word, tok) in enumerate(zip(words, tokens)):
					if not tok or not has_final_e(word):
						continue
					# VARIABLE_SYLLABLE_WORDS are scanned as one opaque unit
					# with no per-cluster x markers ("noble" as 1 syllable is
					# just "S", "viage" as 2 is "uS"), so the token's characters
					# can't be mapped onto the final -e. Count them separately
					# instead of guessing.
					w = rs.minimal_clean(word.lower())
					if rs.get_variable_word_syllables(w, rs.simplify_word(w)) is not None:
						stats['variable_excluded'] += 1
						continue
					rendering = tok[-1]
					if rendering not in 'Sux':
						continue
					stats[rendering] += 1
					stats[('final_' if i == last else 'internal_') + rendering] += 1
					word_stats[rs.minimal_clean(word).lower()][rendering] += 1

		print(f"{text_name}: {stats['lines_scanned']}/{stats['lines_total']} lines scanned, "
			  f"{stats['S'] + stats['u'] + stats['x']} final -e words "
			  f"(S {stats['S']}, u {stats['u']}, x {stats['x']})")

	# --- per-file CSV
	by_file_path = os.path.join(OUTPUT_DIR, 'final_e_by_file.csv')
	fields = ['file', 'lines_total', 'lines_scanned', 'lines_abstained',
			  'final_e_words', 'S', 'u', 'x', 'pct_S', 'pct_u', 'pct_x',
			  'internal_S', 'internal_u', 'internal_x',
			  'line_final_S', 'line_final_u', 'line_final_x',
			  'variable_excluded']
	with open(by_file_path, 'w', newline='', encoding='utf-8') as f:
		writer = csv.writer(f)
		writer.writerow(fields)
		grand = Counter()
		for text_name, stats in file_stats.items():
			grand.update(stats)
		for text_name in list(file_stats) + ['TOTAL']:
			stats = grand if text_name == 'TOTAL' else file_stats[text_name]
			n = stats['S'] + stats['u'] + stats['x']
			writer.writerow([
				text_name, stats['lines_total'], stats['lines_scanned'],
				stats['lines_abstained'], n, stats['S'], stats['u'], stats['x'],
				round(100 * stats['S'] / n, 2) if n else '',
				round(100 * stats['u'] / n, 2) if n else '',
				round(100 * stats['x'] / n, 2) if n else '',
				stats['internal_S'], stats['internal_u'], stats['internal_x'],
				stats['final_S'], stats['final_u'], stats['final_x'],
				stats['variable_excluded'],
			])

	# --- per-word CSV, most frequent first
	by_word_path = os.path.join(OUTPUT_DIR, 'final_e_by_word.csv')
	with open(by_word_path, 'w', newline='', encoding='utf-8') as f:
		writer = csv.writer(f)
		writer.writerow(['word', 'total', 'S', 'u', 'x', 'pct_S', 'pct_u', 'pct_x'])
		for word, counts in sorted(word_stats.items(),
								   key=lambda kv: -sum(kv[1].values())):
			n = sum(counts.values())
			writer.writerow([word, n, counts['S'], counts['u'], counts['x'],
							 round(100 * counts['S'] / n, 2),
							 round(100 * counts['u'] / n, 2),
							 round(100 * counts['x'] / n, 2)])

	n = grand['S'] + grand['u'] + grand['x']
	print(f"\nTOTAL: {grand['lines_scanned']}/{grand['lines_total']} lines scanned "
		  f"({grand['lines_abstained']} abstained/failed)")
	if n:
		print(f"final -e words: {n} -- S {grand['S']} ({100*grand['S']/n:.1f}%), "
			  f"u {grand['u']} ({100*grand['u']/n:.1f}%), "
			  f"x {grand['x']} ({100*grand['x']/n:.1f}%)")
	print(f"\nwrote {by_file_path}\nwrote {by_word_path}")


if __name__ == '__main__':
	main()
