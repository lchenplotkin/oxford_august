"""
ranked_scan_v7.py - ranked_scan_v6.py plus 92 mined MINED_SYLLABLE_WORDS
entries (see that dict below for the mining methodology -- short version:
words whose green-attested syllable count was UNREACHABLE by any candidate
the previous versions could enumerate, mined from the seed-42 train split
only, so the held-out test split stays a fair judge). Selection and
abstention are unchanged from v6, whose docstring follows.

ranked_scan_v6.py - ranked_scan_v4.py plus ABSTENTION: same greedy
selection (first valid alternating candidate in linguistic-default
enumeration order -- v4's whole trick, and still the best picker), but the
corpus-frequency ranking that v4 dropped comes back as a *veto*, not a
picker. scan() now returns a blank scansion (same shape as a total search
failure) instead of a low-quality guess when either:

1. The greedy pick and the corpus-ranked pick DISAGREE. v6 internally
   also computes what pure corpus ranking (v2's selection rule) would
   have chosen from the same candidate set; on the held-out split
   (seed=42, 50/50), v4's accuracy is 84.1% on the 96.3% of lines where
   the two methods agree, but only 42.9% on the 3.7% where they
   disagree -- disagreement between the two knowledge sources
   (hand-tuned linguistic defaults vs. green-corpus frequencies) is the
   single strongest per-line failure signal available.
2. The corpus confidence (score_pattern) of the chosen reading is below
   ABSTAIN_MIN_CONFIDENCE. Low corpus support catches the failures where
   both methods agree on the same wrong answer.

Why abstain instead of guessing: a wrong-but-present scansion silently
pollutes the data, while a blank line reads as 0 syllables / 0 confidence
and therefore surfaces in the GUI's "Scansion Failures" worklist, where it
gets found and hand-scanned.

Why a veto instead of a blended score: arithmetic blends of the two
scores were swept thoroughly first (v5's order_weight over 0.0-1.0; the
corpus term filtered to words with >=3/6/10/20/50 green occurrences at
each weight; a per-word evidence-weighted blend) and everything capped at
82.64% vs. pure-greedy's 82.57% -- one line out of 1515. Root cause:
58.5% of words in the reference table have exactly one green occurrence
(79.5% have <=3), so the corpus score is too noisy to out-PICK the
linguistic defaults -- but it's still good enough to FLAG the picks it
clearly contradicts.

Selection semantics are identical to v4 on every line v6 answers: the
greedy winner here ("first valid candidate wins") is exactly v4's
"score = 100 - attempt*multiplier, keep the max" (that score strictly
decreases with attempt order, so max == first), just written directly.

Syllable-counting rules are v2/v3/v4/v5's plus MINED_SYLLABLE_WORDS (v7).
"""

import re
import os
import json
import argparse
import csv
from typing import List, Dict, Tuple, Optional
from itertools import product

REFERENCE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'green_word_patterns.json')

# Abstain (return a blank scansion) when the chosen reading's corpus
# confidence (score_pattern, 0-100) is below this -- even if the greedy and
# corpus-ranked picks agree. Held-out sweep (seed=42, 50/50 split), with the
# agreement veto also active:
#   threshold 0  -> answers 96.3% of lines at 84.1% precision
#   threshold 30 -> answers 91.6% at 87.3% (69% of blanked lines truly wrong)
#   threshold 40 -> answers 83.6% at 89.3%
#   threshold 50 -> answers 67.9% at 91.6%
# 30 is the efficiency sweet spot: below it the blanked set stops being
# mostly-wrong lines and starts wasting manual review on correct ones.
# Deployed confidence runs a bit higher than held-out (the full reference
# table has seen more words), so real coverage will beat these numbers.
ABSTAIN_MIN_CONFIDENCE = 30

_DEFAULT_REFERENCE = None


def load_reference(path: str = REFERENCE_PATH) -> Dict[str, Dict[str, int]]:
	"""Load the word -> {x-stripped pattern: count} reference table."""
	with open(path, encoding='utf-8') as f:
		raw = json.load(f)
	# Collapse patterns that only differ in their x's (e.g. "Sx" and "S")
	# into one bucket, since a silent-vs-sounded final -e isn't a different
	# stress pattern for ranking purposes.
	collapsed: Dict[str, Dict[str, int]] = {}
	for word, pattern_counts in raw.items():
		bucket = collapsed.setdefault(word, {})
		for pattern, count in pattern_counts.items():
			stripped = re.sub(r'[xX]', '', pattern)
			bucket[stripped] = bucket.get(stripped, 0) + count
	return collapsed


def unseen_pattern_prior(word: str, reference: Dict[str, Dict[str, int]]) -> float:
	"""
	NOT YET USED in score_pattern() below -- a placeholder for later.

	Right now, a word/pattern combo that was never seen for that word in
	the reference scores a flat 0, same as a word we have zero data for at
	all. That's wrong in one specific way: a word seen only once (with some
	other pattern) shouldn't be treated as equally confident evidence
	against a new pattern as a word seen 100 times and never with it. This
	returns a score based purely on how little data we have for the word
	(higher = less confident = a new pattern should be penalized less),
	e.g. 100/(total_occurrences + 1). Once we're ready to use it, the
	unseen-combo branch in score_pattern() should call this instead of
	returning 0.
	"""
	word_patterns = reference.get(word.lower())
	total = sum(word_patterns.values()) if word_patterns else 0
	return 100.0 / (total + 1)


def score_pattern(pattern_str: str, words: List[str], reference: Dict[str, Dict[str, int]]) -> float:
	"""
	Score a candidate line reading: the sum, over each word, of the
	percentage of that word's green-flagged occurrences that share its
	(x-stripped) stress pattern here, divided by the number of words in the
	line. Unattested word/pattern combos and fully-elided words (pattern is
	all x's) contribute 0 to the sum but still count toward the word total,
	so a line with more unscoreable words ends up with a lower average.

	Dividing by word count doesn't change which candidate wins for a given
	line (every candidate for the same line has the same word count, so
	it's a constant scaling factor within that comparison) -- it only makes
	the resulting number comparable *across* lines of different lengths, so
	it can be used as a per-line confidence score.
	"""
	tokens = pattern_str.split(' ')
	if not words:
		return 0.0
	score = 0.0
	for word, tok in zip(words, tokens):
		w1 = minimal_clean(word).lower()
		stripped = re.sub(r'[xX]', '', tok)
		if not stripped:
			continue  # word is fully elided here -- nothing to score
		word_patterns = reference.get(w1)
		if not word_patterns:
			continue  # no data for this word at all -- 0 for now
		total = sum(word_patterns.values())
		count = word_patterns.get(stripped, 0)
		if count == 0:
			continue  # unattested combo -- 0 for now, see unseen_pattern_prior()
		score += 100.0 * count / total
	return min(max(score / len(words),0),100)

def line_confidence(text: str, scansion_str: str, reference: Dict[str, Dict[str, int]] = None) -> float:
	"""
	Confidence score for an arbitrary (already-produced) scansion of a
	line -- not just one this module generated itself. Used to score
	existing/green scansions the same way as freshly-ranked ones, for
	export. A missing scansion, or one whose token count doesn't line up
	1:1 with the line's words, scores 0.
	"""
	global _DEFAULT_REFERENCE
	if reference is None:
		if _DEFAULT_REFERENCE is None:
			_DEFAULT_REFERENCE = load_reference()
		reference = _DEFAULT_REFERENCE

	if not scansion_str or not str(scansion_str).strip():
		return 0.0

	words = minimal_clean(text).split()
	tokens = scansion_str.split(' ')
	if not words or len(words) != len(tokens):
		return 0.0

	return score_pattern(scansion_str, words, reference)

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
	"troye": (["troye", "troy"], [1]),
	"deye": (["deye"], [2, 1]),
}

# Mined from the seed-42 TRAIN split (test split never consulted): words
# whose green-attested syllable count was unreachable by v4/v6's
# analyze_word() -- no enumerated candidate could ever scan them the way
# the human did, so no ranking or abstention change can fix them. Count
# options are ordered by train-split attestation frequency (most-attested
# first), with the old rules' reachable counts appended as fallbacks.
# Recurring clusters: -ius/-ious/-ien/-ian synizesis (aurelius, curious,
# magicien...), -ie/-ye/-ies diphthongs attested at 1 where the general
# rules force 2 (thries, stories, pye, joye...), -ied/-eyed contraction
# (cried, payed, astonied...), medial syncope (sorweful, william...).
# Comments show each word's attested {count: occurrences} in train.
MINED_SYLLABLE_WORDS = {
	"abraham": (["abraham"], [2, 3]),  # attested {2: 1}
	"adayes": (["adayes"], [2, 3]),  # attested {2: 1}
	"agren": (["agren"], [3, 2]),  # attested {3: 1}
	"alayes": (["alayes"], [2, 3]),  # attested {2: 1}
	"anoyeth": (["anoyeth"], [2, 3]),  # attested {2: 1}
	"apothecaries": (["apothecaries"], [5, 6]),  # attested {5: 1}
	"astonied": (["astonied"], [3, 4]),  # attested {3: 1}
	"astonieth": (["astonieth"], [3, 4]),  # attested {3: 1}
	"aurelius": (["aurelius"], [3, 4]),  # attested {3: 4}
	"avise": (["avise"], [2, 1, 3]),  # attested {2: 1, 1: 1}
	"aweye": (["aweye"], [2, 3]),  # attested {3: 1, 2: 3}
	"baye": (["baye"], [2, 1]),  # attested {2: 1}
	"beautee": (["beautee"], [2, 3]),  # attested {2: 1, 3: 1}
	"bisier": (["bisier"], [2, 3]),  # attested {2: 1}
	"bisieth": (["bisieth"], [2, 3]),  # attested {2: 1}
	"bodies": (["bodies"], [2, 3]),  # attested {2: 1}
	"burie": (["burie"], [1, 2]),  # attested {1: 1}
	"carie": (["carie"], [1, 2]),  # attested {1: 1}
	"carien": (["carien"], [2, 3]),  # attested {2: 1}
	"contrarien": (["contrarien"], [3, 4]),  # attested {3: 1}
	"cotearmure": (["cotearmure"], [3, 4, 5]),  # attested {3: 1}
	"coye": (["coye"], [1, 2]),  # attested {1: 1}
	"cried": (["cried"], [1, 2]),  # attested {1: 1}
	"crisyda": (["crisyda"], [3, 4]),  # attested {3: 1, 4: 1}
	"curious": (["curious"], [2, 3]),  # attested {2: 2}
	"damascien": (["damascien"], [3, 4]),  # attested {3: 1}
	"descensories": (["descensories"], [4, 5]),  # attested {4: 1}
	"destroyeth": (["destroyeth"], [2, 3]),  # attested {2: 1}
	"devel": (["devel"], [1, 2]),  # attested {1: 1}
	"doing": (["doing"], [2, 1]),  # attested {2: 1}
	"esculapius": (["esculapius"], [4, 5]),  # attested {4: 1}
	"evel": (["evel"], [1, 2]),  # attested {1: 1}
	"famulier": (["famulier"], [3, 4]),  # attested {3: 1}
	"farewel": (["farewel"], [2, 3]),  # attested {2: 1}
	"foryeven": (["foryeven"], [3, 4]),  # attested {3: 1}
	"frye": (["frye"], [1, 2]),  # attested {1: 1}
	"furies": (["furies"], [2, 3]),  # attested {2: 1}
	"fustian": (["fustian"], [2, 3]),  # attested {2: 1}
	"galien": (["galien"], [2, 3]),  # attested {2: 1}
	"harlotries": (["harlotries"], [3, 4]),  # attested {3: 1}
	"hye": (["hye"], [2, 1]),  # attested {2: 3, 1: 1}
	"joye": (["joye"], [1, 2]),  # attested {1: 5, 2: 2}
	"libye": (["libye"], [2, 3]),  # attested {2: 1}
	"magicien": (["magicien"], [3, 4]),  # attested {3: 1}
	"melancolious": (["melancolious"], [3, 5]),  # attested {3: 1}
	"muchel": (["muchel"], [2, 1]),  # attested {1: 1, 2: 2}
	"naciouns": (["naciouns"], [4, 2, 3]),  # attested {4: 1}
	"noble": (["noble"], [1, 2]),  # attested {1: 2, 2: 1}
	"novelries": (["novelries"], [3, 4]),  # attested {3: 1}
	"omnia": (["omnia"], [2, 3]),  # attested {2: 1}
	"oratories": (["oratories"], [4, 5]),  # attested {4: 1}
	"pacient": (["pacient"], [3, 2]),  # attested {3: 1, 2: 1}
	"paied": (["paied"], [1, 2]),  # attested {1: 1}
	"pavement": (["pavement"], [2, 3]),  # attested {2: 1}
	"payed": (["payed"], [1, 2]),  # attested {1: 1}
	"pecunial": (["pecunial"], [3, 4]),  # attested {3: 1}
	"perrie": (["perrie"], [1, 2]),  # attested {1: 1}
	"pleyen": (["pleyen"], [1, 2]),  # attested {1: 2, 2: 1}
	"preciously": (["preciously"], [3, 4]),  # attested {3: 1}
	"prioresse": (["prioresse"], [2, 3, 4]),  # attested {2: 1}
	"purveied": (["purveied"], [2, 3]),  # attested {2: 1}
	"pye": (["pye"], [1, 2]),  # attested {1: 1}
	"religious": (["religious"], [3, 4]),  # attested {3: 1}
	"riban": (["riban"], [1, 2]),  # attested {1: 1}
	"royalliche": (["royalliche"], [2, 3, 4]),  # attested {2: 1}
	"sergeant": (["sergeant"], [2, 3]),  # attested {2: 2}
	"sessiouns": (["sessiouns"], [4, 3]),  # attested {4: 1}
	"sevene": (["sevene"], [1, 2, 3]),  # attested {1: 1}
	"simplicius": (["simplicius"], [3, 4]),  # attested {3: 1}
	"sorow": (["sorow"], [2, 1]),  # attested {2: 1, 1: 1}
	"sorowful": (["sorowful"], [2, 3]),  # attested {2: 1}
	"sorweful": (["sorweful"], [2, 3]),  # attested {2: 2}
	"special": (["special"], [2, 3]),  # attested {2: 1}
	"spies": (["spies"], [1, 2]),  # attested {1: 1}
	"squyer": (["squyer"], [1, 2]),  # attested {1: 1}
	"storie": (["storie"], [1, 2]),  # attested {1: 1}
	"stories": (["stories"], [2, 3]),  # attested {2: 1}
	"studie": (["studie"], [2, 1]),  # attested {2: 3, 1: 1}
	"studieth": (["studieth"], [2, 3]),  # attested {2: 1}
	"superstitious": (["superstitious"], [4, 5]),  # attested {4: 1}
	"tarien": (["tarien"], [3, 2]),  # attested {3: 1, 2: 1}
	"temple": (["temple"], [1, 2]),  # attested {1: 3, 2: 1}
	"temporel": (["temporel"], [2, 3]),  # attested {2: 1}
	"thries": (["thries"], [1, 2]),  # attested {1: 2}
	"thryes": (["thryes"], [1, 2]),  # attested {1: 1}
	"varien": (["varien"], [2, 3]),  # attested {2: 1}
	"vertuous": (["vertuous"], [2, 3]),  # attested {2: 1}
	"viage": (["viage"], [2, 1, 3]),  # attested {2: 2, 1: 1}
	"vigilies": (["vigilies"], [3, 4]),  # attested {3: 1}
	"warien": (["warien"], [2, 3]),  # attested {2: 1}
	"weyeden": (["weyeden"], [2, 3]),  # attested {2: 1}
	"william": (["william"], [2, 3]),  # attested {2: 1}
}
VARIABLE_SYLLABLE_WORDS.update(MINED_SYLLABLE_WORDS)

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
			elif vowels == "e" and len(clusters)>1 and end == 'n':
				counts = [1]
			elif vowels == "e" and len(clusters)>1 and end in ['d','r','s','th','st']:
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
						   target: int, start_stressed: bool,
						   reference: Dict[str, Dict[str, int]]) -> Optional[Dict]:
	"""
	Enumerate every valid alternating stress pattern for this target/start
	and pick the FIRST one found (greedy, v4's selection rule -- candidate
	order comes from analyze_word()'s linguistically-preferred defaults, so
	first-found means most-default reading). Alongside it, also track the
	corpus-ranked winner (highest score_pattern(), v2's selection rule) from
	the same candidates, returned as "corpus_stress_pattern" so scan() can
	abstain when the two methods disagree.
	"""
	word_options = []
	for word_counts in line_counts:
		word_options.append(generate_syllable_combinations(word_counts))

	best_exact = None
	best_feminine = None
	corpus_exact = None
	corpus_exact_score = None
	corpus_feminine = None
	corpus_feminine_score = None

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

		for i, (word, syllable_counts) in enumerate(zip(words, adjusted_counts)):
			next_stress, stresses = determine_stress_pattern(next_stress, syllable_counts, word)
			stress_pattern.extend(stresses)
			stress_pattern.append(' ')

		pattern_str = "".join(stress_pattern).strip()

		if not is_pattern_alternating(pattern_str):
			continue

		if total == target:
			if best_exact is None:
				best_exact = {
					"adjusted_counts": adjusted_counts,
					"stress_pattern": pattern_str,
					"total_syllables": total,
					"start_stressed": start_stressed
				}
			corpus_score = score_pattern(pattern_str, words, reference)
			if corpus_exact_score is None or corpus_score > corpus_exact_score:
				corpus_exact_score = corpus_score
				corpus_exact = pattern_str
		elif total == target + 1:
			# VARIANT A: only require last syllable unstressed (dropped consonant+en check)
			clean = re.sub(r'[^Su]', '', pattern_str)
			if clean and clean[-1] == 'u':
				if best_feminine is None:
					best_feminine = {
						"adjusted_counts": adjusted_counts,
						"stress_pattern": pattern_str,
						"total_syllables": total,
						"start_stressed": start_stressed
					}
				corpus_score = score_pattern(pattern_str, words, reference)
				if corpus_feminine_score is None or corpus_score > corpus_feminine_score:
					corpus_feminine_score = corpus_score
					corpus_feminine = pattern_str

	# Both selection rules prefer an exact-target reading over a feminine
	# one, so the agreement check always compares within the same category.
	if best_exact:
		best_exact["corpus_stress_pattern"] = corpus_exact
		return best_exact
	if best_feminine:
		best_feminine["corpus_stress_pattern"] = corpus_feminine
		return best_feminine
	return None

def adjust_line_to_target(line: str, line_counts: List[List[List[int]]], target: int = 10,
						   reference: Dict[str, Dict[str, int]] = None) -> Optional[Dict]:
	"""
	Adjust syllable counts to try to reach the target syllable count with alternating meter.
	"""
	words = minimal_clean(line).split()

	# Try normal pattern (starting unstressed, target 10)
	result = try_scansion_combination(words, line_counts, target, False, reference)
	if result:
		return result

	# Try flipped pattern (starting stressed, target 9)
	flipped_target = target - 1
	result = try_scansion_combination(words, line_counts, flipped_target, True, reference)
	if result:
		return result

	# If neither works, return None (invalid)
	return None

def scan(line: str, target_syllables: int = 10, reference: Dict[str, Dict[str, int]] = None) -> Dict[str, any]:
	"""
	Perform scansion on a single line of text. `reference` is the
	word -> {pattern: count} table used to rank candidates; if not passed,
	it's loaded once from REFERENCE_PATH and cached.
	"""
	global _DEFAULT_REFERENCE
	if reference is None:
		if _DEFAULT_REFERENCE is None:
			_DEFAULT_REFERENCE = load_reference()
		reference = _DEFAULT_REFERENCE

	words = minimal_clean(line).split()

	# First pass - get initial syllable counts
	initial_counts = []
	for i, word in enumerate(words):
		prev_word = words[i-1] if i > 0 else None
		next_word = words[i+1] if i < len(words)-1 else None
		syllable_counts = analyze_word(word, prev_word, next_word)
		initial_counts.append(syllable_counts)

	# Adjust counts to try to reach target with alternating meter
	adjustment_result = adjust_line_to_target(line, initial_counts, target_syllables, reference)

	if not adjustment_result:
		# If we can't find a valid alternating pattern, return error
		stress_pattern = []
		for i in range(len(words)):
			stress_pattern.append('')
		return stress_pattern, 0

	# --- Abstention: blank out readings the corpus evidence won't vouch for
	# (see module docstring). Stress patterns are compared x-stripped, the
	# same normalization ranking itself uses -- silent-vs-sounded final -e
	# isn't a disagreement.
	def _norm(p):
		return re.sub(r'\s+', ' ', re.sub(r'[xX]', '', p or '')).strip()

	chosen_pattern = adjustment_result["stress_pattern"]
	corpus_pattern = adjustment_result.get("corpus_stress_pattern")
	confidence = score_pattern(chosen_pattern, words, reference)
	if _norm(chosen_pattern) != _norm(corpus_pattern) or confidence < ABSTAIN_MIN_CONFIDENCE:
		return [''] * len(words), 0

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
