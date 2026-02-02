import csv
import os
import re
import pandas as pd
from collections import defaultdict, Counter

# Configuration
base_csv_dir = '../gow_csvs'
form_csv = 'complete_verbs_pretpres.csv'

ELISION_FOLLOWERS = ["have", "haven", "haveth", "havest", "had", "hadde",
					"hadden", "his", "her", "him", "hers", "hide", "hir",
					"hire", "hires", "hirs", "han"]

# Load verb classifications
verbs_dict = {}
with open(form_csv, 'r', encoding='utf-8') as csvfile:
	reader = csv.DictReader(csvfile)
	for row in reader:
		if 'headword' in row and 'classification' in row:
			verbs_dict[row['headword'].lower()] = row['classification'].lower()

def is_strong(verb):
	return verbs_dict.get(verb, '') == 'strong'

def is_weak(verb):
	return verbs_dict.get(verb, '') == 'weak'

def is_pretpres(verb):
	return verbs_dict.get(verb, '') == 'pretpres'

def clean_tag(tag):
	"""Remove digits before % in a tag (e.g., v2%pt_1 -> v%pt_1)."""
	if tag.startswith('ger') and '%' not in tag:
		return 'ger'
	return re.sub(r'\d+(?=%)', '', tag)

def is_elided(word, next_word, next_tag):
	"""Check if word is elided before next word"""
	if next_word[0] in 'aeiou' or (next_word[0] == 'h' and next_tag.startswith('pron')) or next_word in ELISION_FOLLOWERS:
		return True
	return False

def classify_ending_detailed(word):
	"""Classify word ending into specific categories: -ede, -ete, -de, -te, -e, -en, -d, -t, -vowel, -x"""
	# Order matters - check longer endings first
	if word.endswith('ede'):
		return '-ede'
	if word.endswith('ete'):
		return '-ete'
	if word.endswith('de'):
		return '-de'
	if word.endswith('te'):
		return '-te'
	if word.endswith('en'):
		return '-en'
	if word.endswith('e'):
		return '-e'
	if word.endswith('d'):
		return '-d'
	if word.endswith('t'):
		return '-t'
	if word.endswith('ey'):
		return '-ey'
	if word.endswith('y'):
		return '-y'
	if word.endswith('eyn'):
		return '-eyn'
	if word.endswith('yn'):
		return '-yn'
	# Check if ends in vowel (but not already caught by -e)
	if len(word) > 0 and word[-1] in 'aiou':
		return '-vowel'
	# Everything else
	return '-x'

problems = 0

def parse_tagged_text(text, tagging):
	"""Extract words from text and tags from tagging"""
	global problems
	
	if pd.isna(text) or pd.isna(tagging) or text == '' or tagging == '':
		return [], [], []

	words = re.sub(r'[.,!?°¶]', '', text.lower()).strip().split()
	tag_tokens = tagging.strip().split()
	tags = []
	headwords = []

	for token in tag_tokens:
		if '@' in token and token != "--@dash":
			parts = token.split('@')
			headword = parts[0].lower()
			tag = parts[1] if len(parts) > 1 else ''
			headwords.append(headword)
			tags.append(tag)

	if len(words) != len(tags):
		problems += 1

	min_len = min(len(words), len(tags), len(headwords))
	return words[:min_len], headwords[:min_len], tags[:min_len]

def analyze_verb_endings(df, results):
	"""Analyze verb endings by headword, verb_class, and tag"""
	for idx, row in df.iterrows():
		text = row['TEXT']
		tagging = row['TAGGING']

		words, headwords, tags = parse_tagged_text(text, tagging)

		for j in range(len(tags)):
			if j >= len(words) or j >= len(headwords):
				continue

			word = words[j].lower()
			headword = headwords[j]
			tag = clean_tag(tags[j])
			
			# Exclude end of line
			if j == len(tags) - 1:
				continue
			
			# Exclude elision
			if is_elided(word, words[j + 1], tags[j + 1]):
				continue

			# Only process verb tags
			if not tag.startswith('v'):
				continue

			# Determine verb class
			verb_class = 'weak'
			if headword in verbs_dict:
				if is_strong(headword):
					verb_class = 'strong'
				elif is_weak(headword):
					verb_class = 'weak'
				elif is_pretpres(headword):
					verb_class = 'pretpres'
				else:
					verb_class = 'irregular'

			# Debug check for weak preterite
			if verb_class == 'weak' and tag in ["v%pt_1", "v%pt_3"] and not word.endswith(('de', 'te', 't', 'd')):
				print('WEAK PRETERITE ISSUE:', text)
				print(row)

			# Classify ending
			ending = classify_ending_detailed(word)

			# Count by (headword, verb_class, tag, ending)
			key = (headword, verb_class, tag, ending)
			results['breakdown'][key] += 1
			
			# Track unique words for this headword/tag combo
			word_key = (headword, verb_class, tag)
			results['word_examples'][word_key].add(word)

	return results

def process_csv_directory(csv_dir):
	"""Process all CSV files in directory"""
	results = {
		'breakdown': Counter(),
		'word_examples': defaultdict(set)
	}

	file_count = 0
	for root, dirs, files in os.walk(csv_dir):
		for file in files:
			if not file.endswith('.csv'):
				continue
			file_count += 1
			try:
				df = pd.read_csv(os.path.join(root, file), encoding='utf-8')
				required_columns = ['TAGGING', 'TEXT']
				if not all(col in df.columns for col in required_columns):
					print(f"Skipping {file}: missing columns")
					continue
				results = analyze_verb_endings(df, results)
			except Exception as e:
				print(f"Error processing {file}: {e}")
				continue
	
	print(f"Processed {file_count} files")
	return results

def write_breakdown(results, output_file):
	"""Write breakdown for Gower texts"""
	
	# Get all unique (headword, verb_class, tag) combinations
	headword_tag_combos = set()
	for headword, verb_class, tag, ending in results['breakdown'].keys():
		headword_tag_combos.add((headword, verb_class, tag))
	
	# Sort by headword, then tag
	sorted_combos = sorted(headword_tag_combos, key=lambda x: (x[0], x[2]))
	
	# Define ending types in order
	ending_types = ['-ede', '-ete', '-de', '-te', '-e', '-en', '-d', '-t', '-ey', '-y', '-eyn', '-yn', '-vowel', '-x']
	
	with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
		# Create fieldnames: headword, verb_class, tag, total, then each ending type
		fieldnames = ['headword', 'verb_class', 'tag', 'total'] + [e.lstrip('-') for e in ending_types] + ['example_words']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()
		
		for headword, verb_class, tag in sorted_combos:
			row = {
				'headword': headword,
				'verb_class': verb_class,
				'tag': tag,
				'total': 0
			}
			
			# Count each ending type
			for ending_type in ending_types:
				count = results['breakdown'][(headword, verb_class, tag, ending_type)]
				row[ending_type.lstrip('-')] = count
				row['total'] += count
			
			# Add example words (up to 5)
			word_key = (headword, verb_class, tag)
			example_words = sorted(list(results['word_examples'][word_key]))
			row['example_words'] = '; '.join(example_words)
			
			writer.writerow(row)

# Main execution
if __name__ == "__main__":
	print("Processing Gower texts...")
	gower_results = process_csv_directory(base_csv_dir)
	
	output_dir = 'gower_verb_ending_breakdown_output'
	os.makedirs(output_dir, exist_ok=True)
	
	print("\nWriting breakdown file...")
	output_file = os.path.join(output_dir, 'gower_verb_endings.csv')
	write_breakdown(gower_results, output_file)
	print(f"  - Gower breakdown: {output_file}")
	
	print(f"\nMismatched word/tag counts: {problems}")
	print("\nAnalysis complete!")
