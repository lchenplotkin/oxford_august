import csv
import os
import re
import pandas as pd
from collections import defaultdict, Counter

# Configuration
base_csv_dir = '../dataset'
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
	if next_word[0] in 'aeiou' or (next_word[0]=='h' and next_tag.startswith('pron')) or next_word in ELISION_FOLLOWERS:
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
def parse_tagged_text(text, tagging, text_type):
	global problems
	"""Extract words from text and tags from tagging"""
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

	if len(words)!=len(tags):
		problems+=1
		#print(problems)
		#print(text, words, tags)

	min_len = min(len(words), len(tags), len(headwords))
	return words[:min_len], headwords[:min_len], tags[:min_len]

def analyze_verb_endings(df, results, text_type):
	"""Analyze verb endings by headword, verb_class, and tag"""
	text_col = f'{text_type}_TEXT'
	tagging_col = f'{text_type}_TAGGING'

	for idx, row in df.iterrows():
		text = row[text_col]
		tagging = row[tagging_col]

		words, headwords, tags = parse_tagged_text(text, tagging, text_type)

		for j in range(len(tags)):
			if j >= len(words) or j >= len(headwords):
				continue

			word = words[j].lower()
			headword = headwords[j]
			tag = clean_tag(tags[j])
			if j == len(tags)-1: 
				continue #exclude end of line
			if is_elided(word,words[j+1],tags[j+1]):
				continue #exclude elision

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

			if verb_class=='weak' and tag in ["v%pt_1", "v%pt_3"] and not(word.endswith(('de','te','t','d'))):
				print('HERE',text)
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

def process_csv_directory(csv_dir, text_type):
	"""Process all CSV files in directory"""
	results = {
		'breakdown': Counter(),
		'word_examples': defaultdict(set)
	}

	file_count = 0
	for root, dirs, files in os.walk(csv_dir):
		for file in files:
			if not file.endswith('_gui_complete.csv'):
				continue
			file_count += 1
			try:
				df = pd.read_csv(os.path.join(root, file), encoding='utf-8')
				required_columns = [f'{text_type}_TAGGING', f'{text_type}_TEXT']
				if not all(col in df.columns for col in required_columns):
					print(f"Skipping {file} for {text_type}: missing columns")
					continue
				results = analyze_verb_endings(df, results, text_type)
			except Exception as e:
				print(f"Error processing {file} for {text_type}: {e}")
				continue
	
	print(f"Processed {file_count} files for {text_type}")
	return results

def write_individual_breakdown(results, output_file, text_type):
	"""Write breakdown for a single text type (Oxford or Riverside)"""
	
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

def write_combined_breakdown(oxford_results, riverside_results, output_file):
	"""Write combined breakdown comparing Oxford and Riverside"""
	
	# Get all unique (headword, verb_class, tag) combinations from both
	oxford_combos = set()
	for headword, verb_class, tag, ending in oxford_results['breakdown'].keys():
		oxford_combos.add((headword, verb_class, tag))
	
	riverside_combos = set()
	for headword, verb_class, tag, ending in riverside_results['breakdown'].keys():
		riverside_combos.add((headword, verb_class, tag))
	
	all_combos = oxford_combos | riverside_combos
	sorted_combos = sorted(all_combos, key=lambda x: (x[0], x[2]))
	
	# Define ending types
	ending_types = ['-ede', '-ete', '-de', '-te', '-e', '-en', '-d', '-t', '-ey', '-y', '-eyn', '-yn', '-vowel', '-x']
	
	with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
		# Fieldnames: headword, verb_class, tag, then Oxford and Riverside counts for each ending
		fieldnames = ['headword', 'verb_class', 'tag', 'oxford_total', 'riverside_total', 'combined_total']
		
		# Add Oxford columns for each ending
		for ending in ending_types:
			fieldnames.append(f'ox_{ending.lstrip("-")}')
		
		# Add Riverside columns for each ending
		for ending in ending_types:
			fieldnames.append(f'rv_{ending.lstrip("-")}')
		
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()
		
		for headword, verb_class, tag in sorted_combos:
			row = {
				'headword': headword,
				'verb_class': verb_class,
				'tag': tag,
				'oxford_total': 0,
				'riverside_total': 0,
				'combined_total': 0
			}
			
			# Count Oxford endings
			for ending_type in ending_types:
				ox_count = oxford_results['breakdown'][(headword, verb_class, tag, ending_type)]
				row[f'ox_{ending_type.lstrip("-")}'] = ox_count
				row['oxford_total'] += ox_count
			
			# Count Riverside endings
			for ending_type in ending_types:
				rv_count = riverside_results['breakdown'][(headword, verb_class, tag, ending_type)]
				row[f'rv_{ending_type.lstrip("-")}'] = rv_count
				row['riverside_total'] += rv_count
			
			row['combined_total'] = row['oxford_total'] + row['riverside_total']
			
			writer.writerow(row)

# Main execution
if __name__ == "__main__":
	print("Processing Oxford texts...")
	oxford_results = process_csv_directory(base_csv_dir, 'OXFORD')
	
#	print("Processing Riverside texts...")
#	riverside_results = process_csv_directory(base_csv_dir, 'RIVERSIDE')
	
	output_dir = 'verb_ending_breakdown_output'
	os.makedirs(output_dir, exist_ok=True)
	
	print("\nWriting breakdown files...")
	
	# Write individual files
	oxford_file = os.path.join(output_dir, 'oxford_verb_endings.csv')
	write_individual_breakdown(oxford_results, oxford_file, 'Oxford')
	print(f"  - Oxford breakdown: {oxford_file}")
	
#	riverside_file = os.path.join(output_dir, 'riverside_verb_endings.csv')
#	write_individual_breakdown(riverside_results, riverside_file, 'Riverside')
#	print(f"  - Riverside breakdown: {riverside_file}")
	
	# Write combined file
#	combined_file = os.path.join(output_dir, 'combined_verb_endings.csv')
#	write_combined_breakdown(oxford_results, riverside_results, combined_file)
#	print(f"  - Combined breakdown: {combined_file}")
	
	print("\nAnalysis complete!")
