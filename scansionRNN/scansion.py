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

ONE_IF_INITIAL = ["yi", "ya", "iu"]
ONE_IF_FOLLOWING_Q = ["ua", "uee"]

ELISION_FOLLOWERS = ["have", "haven", "haveth", "havest", "had", "hadde",
					"hadden", "his", "her", "him", "hers", "hide", "hir",
					"hire", "hires", "hirs", "han"]

ELISION_EXCEPTIONS = ["he", "be", "we", "the", "ne", "noble","double","temple"]

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
				# Allow feminine ending only if:
				# 1. Last syllable is unstressed
				# 2. Last word ends in consonant + "en" (not vowel + "en", not "ien")
				clean = re.sub(r'[^Su]', '', pattern_str)
				last_word = words[-1].lower()
				if (
					clean and clean[-1] == 'u' and
					re.search(r'[^aeiou]en$', last_word)  # consonant + en at end
				):
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
