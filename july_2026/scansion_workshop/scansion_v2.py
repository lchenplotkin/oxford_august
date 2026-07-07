"""
scansion_v2.py - scansion.py plus six validated (or explicitly accepted)
fixes, chosen empirically by testing each candidate change in isolation
against the green-flagged (human-vetted) lines in scansion_tool/ (1,534 of
them when this file was started; the vetted set has since grown as more
lines get flagged -- rerun scansion_analysis/compare_v1_v2_green.py for the
current total). See scansion_analysis/compare_variants.py and
compare_variants2.py for the harnesses, scansion_analysis/compare_green_flags.py
for the original gap analysis, and scansion_analysis/still_wrong.csv for the
categorized round-2 gap analysis. As of the last rerun (2,012 green lines):
scansion.py 1670/2012 (83.00%), this file 1730/2012 (85.98%).

KEPT:
1. Feminine-ending check was overfit to one word shape (consonant + "en").
   Of 101 vetted feminine-ending lines, only 1 satisfied that extra check --
   the other 100 end in all sorts of unstressed syllables (-rie, -nes, -ere,
   -eth, -ges, -ond, -nce, -age, -oun, ...). Dropped the word-shape
   restriction; kept the actual "last syllable is unstressed" requirement.
   Alone: +50 net (90 fixed, 40 regressed).

2. Two missing VARIABLE_SYLLABLE_WORDS entries the vetted data called for:
   "every" (can syncopate to 2 syllables, "ev'ry") and "eyen" (occasionally
   1 syllable in context). Alone: +4 net (4 fixed, 0 regressed).

3. Word-initial "ye" was being treated as a 2-letter vowel diphthong (like
   "ie"/"eye"/"aye") whenever a follower consonant came after it, so "yes",
   "yerne", "yerde", "yede" etc. were scanned as 2 syllables instead of 1.
   Added "ye" to ONE_IF_INITIAL (the list that already exists for exactly
   this glide-vs-vowel problem with "yi"/"ya"). Restricted to true
   word-initial position only -- see TRIED AND DROPPED below for why.
   Alone: +2 net (6 fixed, 4 regressed).

4. "Proign(e)" (Progne, the swallow of the Procne/Philomela myth) has no
   vowel-cluster rule that gives it 2 syllables ("oi" isn't in any of the
   TWO_SYLLABLES/TWO_IF_FOLLOWED lists), so it was always scanned as 1.
   Added a VARIABLE_SYLLABLE_WORDS entry, same pattern as Troilus/Caliope.
   Alone: +1 net (1 fixed, 0 regressed).

5. "me" added to ELISION_EXCEPTIONS, so it can never be silent/elided (like
   "he"/"be"/"we"/"the"/"ne" already there). Tested in isolation this is a
   net WASH on the green set -- +0 (2 fixed: "that me thoughte", "For me
   were lever"; 2 regressed: "that me were levest", "of me were nought"),
   meaning "me" genuinely is elidable in at least some lines and this is
   overriding correct behavior in those. Kept anyway (not because the
   metric asked for it) as a deliberate stance rather than a data-driven
   one -- see scansion_analysis/variants2/variant_i_me_never_elided.py for
   the isolated test.

6. The line-final word can no longer fully elide (no all-"x" last word,
   i.e. it must contribute at least one sounded syllable). This closes a
   real gap: words like "she" (never added to ELISION_EXCEPTIONS) have a
   [0,1] option on their only vowel cluster whenever nothing follows them
   in the line, which previously let the search zero out an entire
   line-final word if that happened to satisfy the target count/meter --
   a line can't actually end on silence. Net 0 on the 2,012-line green set
   (this exact defect never happened to decide a green line's outcome),
   but flips 9 lines across the full ~33k-line corpus, all of them
   correctly (e.g. "...be it he or she" no longer scans "she" as silent).
   Deliberately kept despite the 0 net on the measured sample, since it's
   a correctness constraint rather than a fitted rule.

TRIED AND DROPPED (all made things worse or were a net no-op -- kept here as
a record so they aren't retried without re-checking):

- Preferring a common modal preterite's final -e as pronounced-over-silent
  (hadde/wolde/sholde/coude/moste). The green-flag data suggested this
  should help, but tested alone it's net NEGATIVE: -20 (7 fixed, 27
  regressed). These words need per-context flexibility that a blanket
  default can't provide.

- Making an exact-syllable-count match always win over a feminine-ending
  match, instead of returning whichever is found first. This seems more
  "correct" in principle, but tested on top of fix 1 alone it cuts the net
  gain from +50 down to +4 (6 fixed, 2 regressed) -- most of fix 1's
  benefit actually comes from letting the (fairly arbitrary) combinatorial
  search accept the first valid alternating reading it finds, feminine or
  not, rather than us second-guessing which "should" win. This says
  something bigger about the algorithm: for ambiguous lines with multiple
  valid alternating parses, there's no real preference model -- iteration
  order effectively decides -- so tightening the search doesn't reliably
  help.

- Making "ye" always 1 syllable regardless of word position (not just
  word-initial). This fixes two more lines ("foryeten", "foryeve", where
  the leading "y" of the second half of a compound is still the glide) for
  a raw +6 net on the green set. But it does this by breaking the
  distinction between glide-initial "ye" (yes, yet, yerne -- always 1) and
  vowel "y" immediately before a suffix "e" in words like "cryen", "twyes",
  "hyed" (stem+suffix, where "y" is a vowel and "e" is its own syllable --
  always 2). That second pattern is a common, productive Middle English
  inflection (verb infinitives/plurals in -yen/-yed/-yes: crien, plyed,
  alweyes...), so a positionless rule trades a couple of real fixes for a
  systematic risk across a much larger and more frequent word class than
  the green sample shows. Restricting to true word-initial position (KEPT
  fix 3 above) avoids this because glide "y" only ever starts a word or a
  free morpheme, never a bound suffix -- so it doesn't accidentally catch
  "cryen" etc. See scansion_analysis/variants2/variant_g2_ye_always_one.py.

- Restricting the "medial e can be silent before d/r/s/n/th/st" rule (the
  SET_TO... cascade branch with `end in ['d','r','s','n','th','st']`) to
  only fire on a word's true final cluster, instead of any interior one.
  The hypothesis was that this rule over-applies to interior vowels (e.g.
  the medial "e" in "matere", "yfere", which shouldn't be elidable) and was
  causing spurious combinatorial matches. Tested in isolation: exactly 0
  net change across all 1,534 green lines -- not a single line's output
  changed. The over-broad medial case never actually decides an outcome in
  practice (some other flexibility elsewhere in the line always ends up
  mattering more), so it isn't worth the added complexity. See
  scansion_analysis/variants2/variant_d1_final_cluster_only.py.

- Giving diphthong-final "unfollowed" words (deye, seye, preye -- "eye"/
  "aye"/"ye" as the last cluster with nothing after it in the word) a
  [2,1] syllable option, mirroring how consonant+e endings get [0,1]. The
  motivation was real ("deye" sometimes needs 2 syllables and had no way
  to get there), but the rule is far too broad: "ye" itself (the extremely
  common pronoun "you-plural") is also diphthong-final-unfollowed, and
  giving it a 2-syllable option is always wrong. Net: -158 (1 fixed, 159
  regressed) -- badly overfit to one word at the expense of the most
  common word it matches. If this is worth pursuing, it needs a per-word
  VARIABLE_SYLLABLE_WORDS entry for "deye"/"preye"/"seye" specifically,
  not a general rule. See
  scansion_analysis/variants2/variant_d2_diphthong_e_unfollowed.py.

- Building a data-derived closed-class function-word list (279 words,
  mined from the corpus's own OXFORD_TAGGING POS annotations: pronouns,
  prepositions, conjunctions, articles, grammatical adjectives) and using
  it as a tie-break -- among multiple valid alternating parses of a line,
  prefer whichever one stresses the fewest such words, instead of
  whichever the combinatorial search happens to reach first. This directly
  targets the ambiguity already flagged above ("no real preference model
  -- iteration order effectively decides"). It does fix 39 lines, but
  regresses 109: apparently real Chaucerian stress puts metrical ictus on
  pronouns/prepositions/articles far more often than a blanket "closed
  class = unstressed" prior predicts (line-initial position, contrastive
  emphasis, etc.), so this heuristic isn't a good enough proxy for actual
  stress. See scansion_analysis/variants2/variant_f_funcword_tiebreak.py.
"""

import re
import argparse
import csv
from typing import List, Dict, Tuple, Optional
from itertools import product

def minimal_clean(text: str) -> str:
	"""Basic cleaning of text while preserving letters and spaces."""
	retval = re.sub(r"[^a-zA-Z\s]", "", text)
	return retval if not re.match(r"^\s*$", retval) else text

def vowel_clusters(word: str) -> List[Tuple[str, str, str]]:
	"""
	Identify vowel clusters in a word with shared consonants between them.
	Returns a list of tuples (prev_consonant, vowel_cluster, next_consonant)
	where consonants are shared between adjacent clusters.
	"""
	clusters = re.findall(r"(^|[^aeiouy]+)([aeiouy]+)([^aeiouy]+$)?", word.lower())
	if not clusters:
		return []

	processed = []
	prev_end_consonant = ""

	for i, (start, vowels, end) in enumerate(clusters):
		if i == 0:
			prev_consonant = start
		else:
			prev_consonant = prev_end_consonant

		if i == len(clusters) - 1:
			next_consonant = end if end is not None else ""
		else:
			if clusters[i+1][0]:
				next_consonant = clusters[i+1][0]
				prev_end_consonant = clusters[i+1][0]
			else:
				next_consonant = ""
				prev_end_consonant = ""

		processed.append((prev_consonant, vowels, next_consonant))

	return processed

def simplify_word(word: str) -> str:
	"""Remove duplicate characters from word to match dictionary forms."""
	simplified = []
	for i, c in enumerate(word):
		if i == 0 or c != word[i-1]:
			simplified.append(c)
	return ''.join(simplified)

def ends_with_consonant_e(word: str) -> bool:
	"""Check if word ends with consonant + e pattern."""
	return bool(re.search(r'[bcdfghjklmnpqrstvwxyz]e$', word.lower()))

# Variable syllable words with all possible syllable counts
VARIABLE_SYLLABLE_WORDS = {
	"borow": (["borow", "borw"], [2, 1]),
	"cleped": (["cleped", "clepd"], [2, 1]),
	"ever": (["ever", "evr"], [2, 1]),
	"fetheres": (["fetheres", "fethres"], [3, 2]),
	"foles": (["foles", "fols"], [2, 1]),
	"hereth": (["hereth", "herth"], [2, 1]),
	"heven": (["heven", "hevn"], [2, 1]),
	"livest": (["livest", "livst"], [2, 1]),
	"memorie": (["memorie", "memore", "memori"], [3, 2]),
	"never": (["never", "nevr"], [2, 1]),
	"other": (["other", "othr"], [2, 1]),
	"reputacion": (["reputacion", "reputacn"], [4, 3]),
	"someres": (["someres", "somres"], [3, 2]),
	"through": (["through", "thrugh"], [1, 1]),  # Usually 1 syllable
	"thorugh" : (["thorugh", "thrugh"], [2, 1]),
	"tokenes": (["tokenes", "tokns"], [3, 2, 1]),
	"yvel": (["yvel", "yvl"], [2, 1]),
	"troilus": (["troilus", "trolus"], [3, 2]),
	"preyeth": (["preyeth", "preyth"], [2, 1]),  # Added based on your note
	"proign": (["proign", "proigne"], [2, 1]),
	"every": (["every", "evry"], [2, 3]),
	"eyen": (["eyen", "eyn"], [2, 1]),
}

# Special syllable counting rules
SET_TO_ONE_ELSE_TWO = {
	"eau": ["beautee", "beaute", "reaume"],
	"oe": ["boef", "moeving", "moevere", "moebles", "moeved"],
	"oie": ["joie"],
}

SET_TO_TWO_ELSE_ONE = {
	"ue": ["puella", "cruel", "cruelte"],
}

TWO_IF_ENDS_IN_EUS = ["eu"]

TWO_SYLLABLES = ["iou", "ea", "oye", "io", "oya", "eou", "uie", "uou", "eyi",
				 "eiau", "aiu", "iau", "eia", "euou", "eiu", "ayei", "aya",
				 "oa", "ayey", "ae", "ao", "iyo", "oue", "iye", "eiou",
				 "oiou", "uye", "iey", "iai",'ia']

TWO_IF_FOLLOWED = ["ie", "eye", "ye", "aye", "eie", "aie", "eo"]
FOLLOWERS = ["d", "th", "r", "n", "st", "s", "nt"]

ONE_IF_INITIAL = ["yi", "ya", "iu", "ye"]
ONE_IF_FOLLOWING_Q = ["ua", "uee"]

ELISION_FOLLOWERS = ["have", "haven", "haveth", "havest", "had", "hadde",
					"hadden", "his", "her", "him", "hers", "hide", "hir",
					"hire", "hires", "hirs", "han"]

ELISION_EXCEPTIONS = ["he", "be", "we", "the", "ne", "noble","double","temple", "me"]

def get_variable_word_syllables(word: str, simplified_word: str) -> Optional[List[int]]:
	"""Check if word is a variable syllable word and return possible syllable counts."""
	for base, (forms, counts) in VARIABLE_SYLLABLE_WORDS.items():
		if word in forms or simplified_word in forms:
			return counts
	return None

def analyze_word(word: str, prev_word: str = None, next_word: str = None) -> List[List[int]]:
	"""
	Analyze a word to determine possible syllable counts for each vowel cluster.
	"""
	original_word = word
	word = minimal_clean(word.lower())
	simplified_word = simplify_word(word)
	clusters = vowel_clusters(word)
	syllable_counts = []

	# Check for variable syllable words first
	var_counts = get_variable_word_syllables(word, simplified_word)
	if var_counts:
		syllable_counts = [var_counts]
		return syllable_counts

	for i, (start, vowels, end) in enumerate(clusters):
		is_initial = start == "" and i == 0
		is_final = next_word is None
		is_followed = end in FOLLOWERS
		for follower in FOLLOWERS:
			if end.startswith(follower):
				is_followed = True
		is_vowel_ending = end == ""
		next_word_vowel = next_word and re.match(r"^[aeiou].*", next_word.lower())
		after_q = start.endswith("q")
		if next_word:
			next_word_start = next_word.startswith("w")
		else:
			next_word_start = False

		# Check if this is the final -e in a consonant+e ending
		is_final_e = (i == len(clusters) - 1 and len(clusters)>1 and vowels == "e" and
					  is_vowel_ending and ends_with_consonant_e(word))

		# Single vowel cases

		if len(vowels) == 1:
			if word in ELISION_EXCEPTIONS:
				counts = [1]
			elif is_final_e:
				# Final -e after consonant: can be silent (0) or unstressed (1)
				counts = [0, 1]
			elif is_vowel_ending and next_word_vowel and vowels == "e":
				counts = [0, 1]
			elif is_vowel_ending and next_word_vowel:
				counts = [1, 0]
			elif is_vowel_ending and next_word in ELISION_FOLLOWERS and vowels == "e":
				counts = [0, 1]
			elif is_vowel_ending and next_word_start and vowels == "e":
				counts = [0, 1]
			elif is_vowel_ending and next_word in ELISION_FOLLOWERS:
				counts = [1, 0]
			elif vowels == "e" and is_vowel_ending and end == "":
				counts = [0, 1]
			elif vowels == "e" and len(clusters)>1 and end in ['d','r','s','n','th','st']:
				counts = [1, 0]
			else:
				counts = [1]
		# Special multi-vowel cases
		elif vowels in SET_TO_ONE_ELSE_TWO.keys():
			wordstrip = word.rstrip('e') + "e"
			simplified_strip = simplify_word(wordstrip)
			if (word in SET_TO_ONE_ELSE_TWO[vowels] or
				simplified_word in SET_TO_ONE_ELSE_TWO[vowels] or
				wordstrip in SET_TO_ONE_ELSE_TWO[vowels] or
				simplified_strip in SET_TO_ONE_ELSE_TWO[vowels]):
				counts = [1]
			else:
				counts = [2]
		elif vowels in SET_TO_TWO_ELSE_ONE.keys():
			wordstrip = word.rstrip('e') + "e"
			simplified_strip = simplify_word(wordstrip)
			if (word in SET_TO_TWO_ELSE_ONE[vowels] or
				simplified_word in SET_TO_TWO_ELSE_ONE[vowels] or
				wordstrip in SET_TO_TWO_ELSE_ONE[vowels] or
				simplified_strip in SET_TO_TWO_ELSE_ONE[vowels]):
				counts = [2]
			else:
				counts = [1]
		elif vowels in TWO_IF_ENDS_IN_EUS:
			if word.endswith('eus'):
				counts = [2]
			else:
				counts = [1]
		elif vowels in TWO_SYLLABLES and (next_word_vowel or next_word in ELISION_FOLLOWERS):
			counts = [2,1]
		elif vowels in TWO_SYLLABLES:
			counts = [2]
		elif vowels in ONE_IF_FOLLOWING_Q and after_q:
			counts = [1]
		elif vowels in ONE_IF_FOLLOWING_Q:
			counts = [2]
		elif vowels in ONE_IF_INITIAL and is_initial:
			counts = [1]
		elif vowels in ONE_IF_INITIAL:
			counts = [2]
		elif vowels in TWO_IF_FOLLOWED and is_followed:
			counts = [2]
		else:
			counts = [1]

		syllable_counts.append(counts)

	return syllable_counts

def determine_stress_pattern(next_stress: bool, syllable_counts: List[List[int]], word: str) -> Tuple[bool, List[str]]:
	"""
	Convert syllable counts to stress patterns, ensuring final -e is never stressed
	unless it's the first syllable of the word.
	"""
	stress_patterns = []
	word_clean = minimal_clean(word.lower())

	for i, counts in enumerate(syllable_counts):
		count = counts[0]  # Take first option initially
		is_final_cluster = i == len(syllable_counts) - 1
		is_first_cluster = i == 0
		is_final_e = (is_final_cluster and ends_with_consonant_e(word_clean) and
					  count > 0)  # Only if the -e is actually pronounced

		if count == 0:
			stress_patterns.append("x")
		else:
			stress = ''
			for j in range(count):
				# Final -e can only be stressed if it's the first syllable of the word
				if is_final_e and j == count - 1 and not is_first_cluster:
					stress += "u"  # Final -e is unstressed unless it's first syllable
				else:
					if next_stress:
						stress += "S"
					else:
						stress += "u"
				next_stress = not next_stress
			stress_patterns.append(stress)

	return next_stress, stress_patterns

def is_pattern_alternating(pattern: str) -> bool:
	"""Check if stress pattern is strictly alternating (ignoring spaces and x)."""
	clean_pattern = re.sub(r'[^Su]', '', pattern)
	if len(clean_pattern) <= 1:
		return True

	for i in range(1, len(clean_pattern)):
		if clean_pattern[i] == clean_pattern[i-1]:
			return False
	return True

def generate_syllable_combinations(word_counts: List[List[List[int]]]) -> List[List[int]]:
	"""Generate all possible syllable count combinations for a word."""
	cluster_options = []
	for cluster in word_counts:
		cluster_options.append(cluster)

	combinations = []
	for combo in product(*cluster_options):
		combinations.append([c for c in combo])

	return combinations

def try_scansion_combination(words: List[str], line_counts: List[List[List[int]]],
						   target: int, start_stressed: bool) -> Optional[Dict]:
	"""Try a specific combination of syllable counts and stress pattern."""
	word_options = []
	for word_counts in line_counts:
		word_options.append(generate_syllable_combinations(word_counts))

	# Try all combinations
	for line_combo in product(*word_options):
		total = sum(sum(cluster) for cluster in line_combo)
		# Allow exactly target OR 11 syllables with unstressed final syllable
		if total != target and total != target + 1:
			continue

		# The line-final word can't fully elide -- it needs at least one
		# sounded syllable (no all-x last word).
		if sum(line_combo[-1]) == 0:
			continue

		# Test this combination
		adjusted_counts = []
		for word_combo in line_combo:
			word_counts = []
			for cluster_count in word_combo:
				word_counts.append([cluster_count])
			adjusted_counts.append(word_counts)

		# Generate stress pattern
		next_stress = start_stressed
		stress_pattern = []
		valid = True

		for i, (word, syllable_counts) in enumerate(zip(words, adjusted_counts)):
			next_stress, stresses = determine_stress_pattern(next_stress, syllable_counts, word)
			stress_pattern.extend(stresses)
			stress_pattern.append(' ')

		pattern_str = "".join(stress_pattern).strip()

		# Check if pattern is alternating
		# Check if pattern is alternating
		if is_pattern_alternating(pattern_str):
			if total == target:
				return {
					"adjusted_counts": adjusted_counts,
					"stress_pattern": pattern_str,
					"total_syllables": total,
					"start_stressed": start_stressed
				}
			elif total == target + 1:
				# VARIANT A: only require last syllable unstressed (dropped consonant+en check)
				clean = re.sub(r'[^Su]', '', pattern_str)
				if clean and clean[-1] == 'u':
					return {
						"adjusted_counts": adjusted_counts,
						"stress_pattern": pattern_str,
						"total_syllables": total,
						"start_stressed": start_stressed
					}

	return None

def adjust_line_to_target(line: str, line_counts: List[List[List[int]]], target: int = 10) -> Optional[Dict]:
	"""
	Adjust syllable counts to try to reach the target syllable count with alternating meter.
	"""
	words = minimal_clean(line).split()

	# Try normal pattern (starting unstressed, target 10)
	result = try_scansion_combination(words, line_counts, target, False)
	if result:
		return result

	# Try flipped pattern (starting stressed, target 9)
	flipped_target = target - 1
	result = try_scansion_combination(words, line_counts, flipped_target, True)
	if result:
		return result

	# If neither works, return None (invalid)
	return None

def scan(line: str, target_syllables: int = 10) -> Dict[str, any]:
	"""
	Perform scansion on a single line of text.
	"""
	words = minimal_clean(line).split()

	# First pass - get initial syllable counts
	initial_counts = []
	for i, word in enumerate(words):
		prev_word = words[i-1] if i > 0 else None
		next_word = words[i+1] if i < len(words)-1 else None
		syllable_counts = analyze_word(word, prev_word, next_word)
		initial_counts.append(syllable_counts)

	# Adjust counts to try to reach target with alternating meter
	adjustment_result = adjust_line_to_target(line, initial_counts, target_syllables)

	if not adjustment_result:
		# If we can't find a valid alternating pattern, return error
		stress_pattern = []
		for i in range(len(words)):
			stress_pattern.append('')
		return stress_pattern, 0

	adjusted_counts = adjustment_result["adjusted_counts"]

	# Generate final analysis
	next_stress = adjustment_result["start_stressed"]
	analysis = []
	stress_pattern = []
	total_syllables = 0

	for i, (word, syllable_counts) in enumerate(zip(words, adjusted_counts)):
		next_stress, stresses = determine_stress_pattern(next_stress, syllable_counts, word)

		word_info = {
			"word": word,
			"syllable_counts": syllable_counts,
			"stresses": stresses,
			"total_syllables": sum(c[0] for c in syllable_counts)
		}

		analysis.append(word_info)
		stress_pattern.append(''.join(stresses))
		total_syllables += word_info["total_syllables"]

	return stress_pattern, total_syllables
