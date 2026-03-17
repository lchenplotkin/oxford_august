# Strong vs Weak Verb Classification Analysis - Oxford and Riverside
# Using vowel changes for preterite classification

import os
import re
import pandas as pd
from collections import defaultdict
import csv

# Configuration
base_csv_dir = 'data/csvs'


ELISION_FOLLOWERS = ["have", "haven", "haveth", "havest", "had", "hadde",
					"hadden", "his", "her", "him", "hers", "hide", "hir",
					"hire", "hires", "hirs", "han"]

def is_elided(word, next_word):
	"""Check if word is elided before next word"""
	return False
	if next_word[0] in 'aeiou' or next_word in ELISION_FOLLOWERS:
		return True
	return False

# Define verb tags to analyze
verb_tags = ['v%ppl', 'v%pt_1', 'v%pt_3']  # Past participle, 1st/3rd person preterite
infinitive_tag = 'v%inf'  # Tag that marks infiInfinitive
# List of known strong verbs
known_strong_verbs = {
	'abyden', 'arisen', 'beswiken', 'biten', 'chynen', 'cleven', 'driven', 'flyten',
	'gliden', 'gniden', 'gripen', 'liþen', 'miȝen', 'overriden', 'riden', 'risen',
	'schinen', 'schiten', 'shryven', 'sliden', 'smiten', 'stien', 'striven', 'stryken',
	'thryven', 'writen', 'writhen', 'beden', 'bowen', 'brewen', 'chesen', 'cleven',
	'crepen', 'dresen', 'fleten', 'flien', 'forsethen', 'forten', 'fresen', 'louten',
	'rewen', 'schoven', 'sethen', 'souken', 'ten', 'theen', 'yeten', 'atbresten',
	'awerpen', 'awinnen', 'berken', 'berwen', 'bidelven', 'binden', 'biwinnen',
	'blinnen', 'breiden', 'bresten', 'climben', 'clingen', 'delven', 'dingen', 'drinken',
	'fighten', 'finden', 'flyngen', 'forbreiden', 'forkerven', 'forwerpen', 'grynden',
	'ȝelpen', 'helpen', 'iwerpen', 'iwinnen', 'kerven', 'linnen', 'melten', 'rennen',
	'schrynken', 'singen', 'slyngen', 'smerten', 'spinnen', 'spryngen', 'sterven',
	'styngen', 'stynken', 'swellen', 'swelten', 'swerven', 'swymmen', 'swyngen',
	'swynken', 'synken', 'threschen', 'trynen', 'underdelven', 'unhelpen', 'upbreiden',
	'werpen', 'winnen', 'wryngen', 'wynden', 'yellen', 'beren', 'breken', 'cleven',
	'comen', 'drepen', 'fornimen', 'forsweren', 'foryeten', 'foryiven', 'kneden',
	'nimen', 'quelen', 'roten', 'scheren', 'speken', 'stelen', 'sweren', 'teren',
	'tocomen', 'treden', 'weren', 'weven', 'yiven', 'bidden', 'biquethen', 'eten',
	'forfreten', 'foryeten', 'foryiven', 'freten', 'geten', 'kneden', 'mysspeken',
	'quethen', 'seen', 'speken', 'treden', 'weven', 'yiven', 'aken', 'baken', 'drawen',
	'faren', 'forsweren', 'gnawen', 'graven', 'laughen', 'quaken', 'schaken', 'schaven',
	'soupen', 'spruten', 'steppen', 'stonden', 'sweren', 'taken', 'toschaken',
	'understonden', 'waden', 'waken', 'wasshen', 'waxen', 'aknowen', 'beten', 'biflowen',
	'biholden', 'blowen', 'clawen', 'crowen', 'fallen', 'flowen', 'folden', 'forholden',
	'glowen', 'gnawen', 'growen', 'hangen', 'holden', 'knowen', 'leten', 'mowen',
	'ofholden', 'overflowen', 'rowen', 'salten', 'scheden', 'slepen', 'sowen', 'spewen',
	'steppen', 'swopen', 'throwen', 'toscheden', 'upholden', 'walken', 'wasshen', 'waxen', 'wepen'
}

# Vowel patterns for strong verb classification
vowel_groups = [
	{'i', 'a', 'u'},  # sing, sang, sung
	{'e', 'a', 'o'},   # speak, spoke, spoken
	{'i', 'o'},		# write, wrote
	{'a', 'o'},		# fall, fell
	{'e', 'o'},		# break, broke
	{'a', 'e'},		# fare, fōr (OE fōr)
	{'u', 'o'},		# choose, chose
	{'i', 'a'},		# sit, sat
	{'e', 'a'},		# eat, ate
	{'a', 'æ'},		# OE patterns
	{'e', 'æ'},
	{'i', 'y'},
	{'eo', 'e'},
	{'ea', 'e'}
]

verb_dict = {}
headword_to_infinitive = {}  # Maps headwords to their infinitives
infinitive_vowels = {}	   # Stores root vowels of infinitives

def get_root_vowel(word):
	"""Extract the first vowel from a word"""
	vowels = {'a', 'e', 'i', 'o', 'u', 'y', 'æ', 'œ', 'á', 'é', 'í', 'ó', 'ú', 'ý'} 
	for char in 'aeiou':
		word = word.lstrip(char)
	for char in word:
		if char in vowels:
			return char
	return None

def parse_tagged_text(text, tagging, text_type):
	"""Extract words from text and tags from tagging"""
	if pd.isna(text) or pd.isna(tagging) or text == '' or tagging == '':
		return [], [], []

	# Clean and split the text
	words = text.lower().translate(str.maketrans('', '', ',.!?°¶')).split()
	
	# Parse tags from tagging
	tag_tokens = tagging.strip().split()
	tags = []
	headwords = []

	for token in tag_tokens:
		if '@' in token and token not in ["--@dash", ".@ellipsis"]:
			parts = token.split('@')
			headword = parts[0].lower()
			tag_part = parts[1] if len(parts) > 1 else ''
			if tag_part.startswith('v') and len(tag_part)>1:
				if tag_part[1] != '%':
					tag_part = tag_part[0] + tag_part[2:]
			
			headwords.append(headword)
			tags.append(tag_part)
			
			# If this is an infinitive, store its root vowel
			if tag_part == infinitive_tag:
				headword_to_infinitive[headword] = headword
				infinitive_vowels[headword] = get_root_vowel(headword)

	min_len = min(len(words), len(tags), len(headwords))
	return words[:min_len], headwords[:min_len], tags[:min_len]

def classify_verb_form(headword, word, tag):
	"""Classify verb form based on vowel changes and endings"""
	global verb_dict
	
	if tag not in verb_tags:
		return
		
	infinitive = headword_to_infinitive.get(headword)
	if not infinitive:
		if headword.endswith('en'):
			infinitive = headword
		else:
			infinitive = headword+'en'
		#return
		
	if infinitive not in verb_dict:
		verb_dict[infinitive] = {
			'headword': headword,
			'known_strong': infinitive in known_strong_verbs,
			'ppl_strong_forms': {},
			'ppl_weak_forms': {},
			'ppl_unclear_forms': {},
			'pt_strong_forms': {},
			'pt_weak_forms': {},
			'pt_unclear_forms': {},
			'ppl_strong_count': 0,
			'ppl_weak_count': 0,
			'ppl_unclear_count': 0,
			'pt_strong_count': 0,
			'pt_weak_count': 0,
			'pt_unclear_count': 0,
			'total_occurrences': 0
		}

	verb_dict[infinitive]['total_occurrences'] += 1
	word_vowel = get_root_vowel(word)
	inf_vowel = infinitive_vowels.get(headword)

	if tag == 'v%ppl':  # Past participle
		if word.endswith('en') or word.endswith('e'):
			verb_dict[infinitive]['ppl_strong_count'] += 1
			verb_dict[infinitive]['ppl_strong_forms'][word] = verb_dict[infinitive]['ppl_strong_forms'].get(word, 0) + 1
		elif word.endswith('t') or word.endswith('d'):
			verb_dict[infinitive]['ppl_weak_count'] += 1
			verb_dict[infinitive]['ppl_weak_forms'][word] = verb_dict[infinitive]['ppl_weak_forms'].get(word, 0) + 1
		else:
			verb_dict[infinitive]['ppl_unclear_count'] += 1
			verb_dict[infinitive]['ppl_unclear_forms'][word] = verb_dict[infinitive]['ppl_unclear_forms'].get(word, 0) + 1

	elif tag in ['v%pt_1', 'v%pt_3']:  # Preterite
		# Check for vowel change first
		if word_vowel and inf_vowel and word_vowel != inf_vowel:
			# Check if this is a known vowel alternation pattern
			for group in vowel_groups:
				if inf_vowel in group and word_vowel in group:
					verb_dict[infinitive]['pt_strong_count'] += 1
					verb_dict[infinitive]['pt_strong_forms'][word] = verb_dict[infinitive]['pt_strong_forms'].get(word, 0) + 1
					return
			
			# If no known pattern but vowels differ, still count as strong
			verb_dict[infinitive]['pt_strong_count'] += 1
			verb_dict[infinitive]['pt_strong_forms'][word] = verb_dict[infinitive]['pt_strong_forms'].get(word, 0) + 1
		
		# No vowel change - check endings
		elif word.endswith('t') or word.endswith('d') or word.endswith('te') or word.endswith('de'):
			verb_dict[infinitive]['pt_weak_count'] += 1
			verb_dict[infinitive]['pt_weak_forms'][word] = verb_dict[infinitive]['pt_weak_forms'].get(word, 0) + 1
		else:
			verb_dict[infinitive]['pt_unclear_count'] += 1
			verb_dict[infinitive]['pt_unclear_forms'][word] = verb_dict[infinitive]['pt_unclear_forms'].get(word, 0) + 1

def process_csv_file(df, text_type):
	"""Process a single CSV file for verb classification"""
	text_col = f'{text_type}_TEXT'
	tagging_col = f'{text_type}_TAGGING'
	filename_col = f'{text_type}_FILENAME'
	
	# First pass: collect all infinitives (build headword-to-infinitive mapping)
	for idx, row in df.iterrows():
		if row["MATCH"] != "DIFF" or text_type == "RIVERSIDE":
			text = row[text_col]
			tagging = row[tagging_col]
			parse_tagged_text(text, tagging, text_type)
	
	# Second pass: classify verb forms
	for idx, row in df.iterrows():
		if row["MATCH"] != "DIFF" or text_type == "RIVERSIDE":
			text = row[text_col]
			tagging = row[tagging_col]
			words, headwords, tags = parse_tagged_text(text, tagging, text_type)
			
			for i in range(len(words)-1):
				word = words[i]
				next_word = words[i+1]
				if not is_elided(word,next_word):
					tag = tags[i]
					headword = headwords[i]
					classify_verb_form(headword, word, tag)
			#for word, headword, tag in zip(words, headwords, tags):
			 #   classify_verb_form(headword, word, tag)

def process_csv_directory(csv_dir):
	"""Process all _gui.csv files in the directory for both Oxford and Riverside"""
	global verb_dict, headword_to_infinitive
	verb_dict = {}
	headword_to_infinitive = {}
	
	file_count = 0
	for root, dirs, files in os.walk(csv_dir):
		for file in files:
			if not file.endswith('_gui.csv'):
				continue

			csv_path = os.path.join(root, file)
			file_count += 1

			try:
				# Read CSV file
				df = pd.read_csv(csv_path, encoding='utf-8')

				# Process Oxford text if columns exist
				oxford_columns = ['OXFORD_TAGGING', 'OXFORD_TEXT', 'LINE_NUMBER', 'OXFORD_FILENAME']
				if all(col in df.columns for col in oxford_columns):
					process_csv_file(df, 'OXFORD')
				
				# Process Riverside text if columns exist
				riverside_columns = ['RIVERSIDE_TAGGING', 'RIVERSIDE_TEXT', 'LINE_NUMBER', 'RIVERSIDE_FILENAME']
				if all(col in df.columns for col in riverside_columns):
					process_csv_file(df, 'RIVERSIDE')

			except Exception as e:
				print(f"Error processing {file}: {e}")
				continue

	print(f"Processed {file_count} files")
	print(f"Found {len(headword_to_infinitive)} headword-infinitive mappings")
	return verb_dict

def categorize_verbs(verb_dict):
	"""Categorize verbs based on vowel changes and endings"""
	strong_verbs = {}
	weak_verbs = {}
	unclear_verbs = {}
	conflict_verbs = {}
	
	for infinitive, data in verb_dict.items():
		if data['total_occurrences'] == 0:
			continue
			
		if data['known_strong']:
			strong_verbs[infinitive] = data
			continue
			
		# Calculate total evidence
		ppl_strong = data['ppl_strong_count']
		ppl_weak = data['ppl_weak_count']
		pt_strong = data['pt_strong_count']
		pt_weak = data['pt_weak_count']
		
		total_strong = ppl_strong + pt_strong
		total_weak = ppl_weak + pt_weak
		
		# Classification rules:
		# 1. If any strong evidence and no weak evidence -> strong
		# 2. If any weak evidence and no strong evidence -> weak
		# 3. If both or neither -> conflict
		if total_strong > 0 and total_weak == 0:
			strong_verbs[infinitive] = data
		elif total_weak > 0 and total_strong == 0:
			weak_verbs[infinitive] = data
		else:
			conflict_verbs[infinitive] = data
	
	return strong_verbs, weak_verbs, unclear_verbs, conflict_verbs


def write_verb_csv(verbs, filename, category):
	"""Write verb classification to CSV"""
	with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(['infinitive', 'headword', 'category', 'known_strong', 
						'ppl_strong_forms', 'ppl_weak_forms', 'ppl_unclear_forms',
						'pt_strong_forms', 'pt_weak_forms', 'pt_unclear_forms',
						'ppl_strong_count', 'ppl_weak_count', 'ppl_unclear_count',
						'pt_strong_count', 'pt_weak_count', 'pt_unclear_count',
						'total_occurrences'])
		
		for infinitive in sorted(verbs.keys()):
			data = verbs[infinitive]
			ppl_strong_forms = '; '.join([f"{form}({count})" for form, count in data['ppl_strong_forms'].items()])
			ppl_weak_forms = '; '.join([f"{form}({count})" for form, count in data['ppl_weak_forms'].items()])
			ppl_unclear_forms = '; '.join([f"{form}({count})" for form, count in data['ppl_unclear_forms'].items()])
			pt_strong_forms = '; '.join([f"{form}({count})" for form, count in data['pt_strong_forms'].items()])
			pt_weak_forms = '; '.join([f"{form}({count})" for form, count in data['pt_weak_forms'].items()])
			pt_unclear_forms = '; '.join([f"{form}({count})" for form, count in data['pt_unclear_forms'].items()])
			
			writer.writerow([
				infinitive, 
				verb_dict[infinitive]['headword'],
				category, 
				data['known_strong'],
				ppl_strong_forms,
				ppl_weak_forms,
				ppl_unclear_forms,
				pt_strong_forms,
				pt_weak_forms,
				pt_unclear_forms,
				data['ppl_strong_count'],
				data['ppl_weak_count'],
				data['ppl_unclear_count'],
				data['pt_strong_count'],
				data['pt_weak_count'],
				data['pt_unclear_count'],
				data['total_occurrences']
			])

def write_simple_verb_csv(all_verbs, filename):
	"""Write a simple CSV with infinitive, preterite, past participle, classification, and notes"""
	with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(['headword','infinitive', 'preterite', 'past_participle', 'classification', 'notes'])
		
		for infinitive in sorted(all_verbs.keys()):
			data = all_verbs[infinitive]
			headword = data['headword']
			
			# Determine classification
			if data['known_strong']:
				classification = 'strong'
				notes = 'known strong verb'
			else:
				ppl_strong = data['ppl_strong_count']
				ppl_weak = data['ppl_weak_count']
				pt_strong = data['pt_strong_count']
				pt_weak = data['pt_weak_count']
				
				total_strong = ppl_strong + pt_strong
				total_weak = ppl_weak + pt_weak
				
				if total_strong > 0 and total_weak == 0:
					classification = 'strong'
					notes = 'classified by morphological evidence'
				elif total_weak > 0 and total_strong == 0:
					classification = 'weak'
					notes = 'classified by morphological evidence'
				else:
					classification = 'unclear'
					notes = 'mixed or insufficient evidence'
			
			# Collect all preterite forms (pt_1 and pt_3) and sort by frequency
			all_pt_forms = {}
			all_pt_forms.update(data['pt_strong_forms'])
			all_pt_forms.update(data['pt_weak_forms'])
			all_pt_forms.update(data['pt_unclear_forms'])
			
			# Sort by frequency (descending)
			sorted_pt_forms = sorted(all_pt_forms.items(), key=lambda x: x[1], reverse=True)
			preterite_str = ', '.join([f"{form}({count})" for form, count in sorted_pt_forms])
			
			# Collect all past participle forms and sort by frequency
			all_ppl_forms = {}
			all_ppl_forms.update(data['ppl_strong_forms'])
			all_ppl_forms.update(data['ppl_weak_forms'])
			all_ppl_forms.update(data['ppl_unclear_forms'])
			
			# Sort by frequency (descending)
			sorted_ppl_forms = sorted(all_ppl_forms.items(), key=lambda x: x[1], reverse=True)
			past_participle_str = ', '.join([f"{form}({count})" for form, count in sorted_ppl_forms])
			
			writer.writerow([
				headword,
				infinitive,
				preterite_str,
				past_participle_str,
				classification,
				notes
			])

def write_summary_report(strong_verbs, weak_verbs, unclear_verbs, output_dir):
	"""Write a summary report of the verb classification"""
	with open(os.path.join(output_dir, 'verb_classification_summary.txt'), 'w', encoding='utf-8') as f:
		f.write("Strong vs Weak Verb Classification Summary\n")
		f.write("Using Vowel Changes for Preterite Classification\n")
		f.write("=" * 50 + "\n\n")
		
		f.write("Classification Rules:\n")
		f.write("1. Past Participle:\n")
		f.write("   - Strong: ends in -en or -e\n")
		f.write("   - Weak: ends in -t or -d\n")
		f.write("   - Unclear: other endings\n")
		f.write("2. Preterite:\n")
		f.write("   - Strong: root vowel differs from infinitive\n")
		f.write("   - Weak: ends in -t, -d, -te, or -de\n")
		f.write("   - Unclear: no vowel change and doesn't end in dental\n")
		f.write("3. Verbs in the known strong list are automatically classified as strong\n\n")
		
		f.write(f"Total verbs analyzed: {len(strong_verbs) + len(weak_verbs) + len(unclear_verbs)}\n")
		f.write(f"Strong verbs: {len(strong_verbs)}\n")
		f.write(f"  - Known strong verbs: {len([v for v in strong_verbs.values() if v['known_strong']])}\n")
		f.write(f"  - Newly identified strong verbs: {len([v for v in strong_verbs.values() if not v['known_strong']])}\n")
		f.write(f"Weak verbs: {len(weak_verbs)}\n") 
		f.write(f"Verbs needing review: {len(unclear_verbs)}\n\n")
		
		# Count total occurrences
		def count_total_occurrences(verbs):
			return sum(data['total_occurrences'] for data in verbs.values())
		
		strong_total = count_total_occurrences(strong_verbs)
		weak_total = count_total_occurrences(weak_verbs)
		unclear_total = count_total_occurrences(unclear_verbs)
		
		f.write("Total occurrences:\n")
		f.write(f"  Strong verb forms: {strong_total}\n")
		f.write(f"  Weak verb forms: {weak_total}\n")
		f.write(f"  Unclear verb forms: {unclear_total}\n")
		f.write(f"  Total: {strong_total + weak_total + unclear_total}\n\n")
		
		f.write("Files generated:\n")
		f.write("- strong_verbs.csv: Verbs classified as strong\n")
		f.write("- weak_verbs.csv: Verbs classified as weak\n") 
		f.write("- verbs_to_clarify.csv: Verbs that need further review\n")
		f.write("- verb_forms_simple.csv: Simple summary of all verb forms\n")
		f.write("- verb_classification_summary.txt: This summary file\n")

def get_user_input_for_conflicts(conflict_verbs):
	"""Get user input for verbs with conflicting patterns"""
	user_strong = {}
	user_weak = {}
	user_unclear = {}
	
	print(f"\nFound {len(conflict_verbs)} verbs with mixed patterns that need clarification:")
	print("For each verb, please choose: (s)trong, (w)eak, or (u)nclear")
	print("=" * 60)
	
	for verb, data in sorted(conflict_verbs.items()):
		print(f"\nVerb: '{verb}'")
		print(f"Headword: {data['headword']}")
		print(f"Infinitive: {headword_to_infinitive.get(data['headword'])}")
		print(f"Known strong: {'Yes' if data['known_strong'] else 'No'}")
		print(f"Total occurrences: {data['total_occurrences']}")
		print("\nEvidence:")
		print(f"  Strong patterns:")
		print(f"	Past participle (-en/-e): {data['ppl_strong_count']}")
		if data['ppl_strong_forms']:
			print(f"	  Forms: {list(data['ppl_strong_forms'].keys())}")
		print(f"	Preterite (vowel change): {data['pt_strong_count']}")
		if data['pt_strong_forms']:
			print(f"	  Forms: {list(data['pt_strong_forms'].keys())}")
		
		print(f"\n  Weak patterns:")
		print(f"	Past participle (-t/-d): {data['ppl_weak_count']}")
		if data['ppl_weak_forms']:
			print(f"	  Forms: {list(data['ppl_weak_forms'].keys())}")
		print(f"	Preterite (-t/-d endings): {data['pt_weak_count']}")
		if data['pt_weak_forms']:
			print(f"	  Forms: {list(data['pt_weak_forms'].keys())}")
		
		print(f"\n  Unclear patterns: {data['ppl_unclear_count'] + data['pt_unclear_count']}")
		
		while True:
			choice = input(f"Classify '{verb}' as (s)trong, (w)eak, or (u)nclear: ").lower().strip()
			if choice in ['s', 'strong']:
				user_strong[verb] = data
				break
			elif choice in ['w', 'weak']:
				user_weak[verb] = data
				break
			elif choice in ['u', 'unclear']:
				user_unclear[verb] = data
				break
			else:
				user_unclear[verb] = data
				break
				#print("Please enter 's', 'w', or 'u'")
	
	return user_strong, user_weak, user_unclear

# Main execution
if __name__ == "__main__":
	# Create output directory
	output_dir = 'verb_classification_output'
	os.makedirs(output_dir, exist_ok=True)
	
	print("Processing CSV files for verb classification...")
	print("Using vowel changes for preterite classification...")
	print(f"Comparing against {len(known_strong_verbs)} known strong verbs...")
	
	# Process all CSV files
	all_verbs = process_csv_directory(base_csv_dir)
	
	print(f"\nFound {len(all_verbs)} unique verbs with relevant forms")
	
	# Categorize verbs
	strong_verbs, weak_verbs, unclear_verbs, conflict_verbs = categorize_verbs(all_verbs)
	
	print(f"\nInitial classification:")
	print(f"  Known strong verbs: {len([v for v in strong_verbs.values() if v['known_strong']])}")
	print(f"  Other strong verbs: {len([v for v in strong_verbs.values() if not v['known_strong']])}")
	print(f"  Weak verbs: {len(weak_verbs)}")
	print(f"  Verbs needing review: {len(unclear_verbs) + len(conflict_verbs)}")
	
	# Get user input for conflicting verbs
	if conflict_verbs:
		user_strong, user_weak, user_unclear = get_user_input_for_conflicts(conflict_verbs)
		strong_verbs.update(user_strong)
		weak_verbs.update(user_weak)
		unclear_verbs.update(user_unclear)
	
	print(f"\nFinal classification:")
	print(f"  Strong verbs: {len(strong_verbs)}")
	print(f"  Weak verbs: {len(weak_verbs)}")
	print(f"  Verbs to clarify later: {len(unclear_verbs)}")
	
	# Write CSV files
	write_verb_csv(strong_verbs, os.path.join(output_dir, 'strong_verbs.csv'), 'strong')
	write_verb_csv(weak_verbs, os.path.join(output_dir, 'weak_verbs.csv'), 'weak')
	write_verb_csv(unclear_verbs, os.path.join(output_dir, 'verbs_to_clarify.csv'), 'unclear')
	
	# Write simple CSV with all verbs
	write_simple_verb_csv(all_verbs, os.path.join(output_dir, 'verb_forms_simple.csv'))
	
	# Write summary report
	write_summary_report(strong_verbs, weak_verbs, unclear_verbs, output_dir)
	
	print(f"\nVerb classification complete!")
	print(f"Results saved to: {output_dir}/")
	print("Files created:")
	print("- strong_verbs.csv")
	print("- weak_verbs.csv") 
	print("- verbs_to_clarify.csv")
	print("- verb_forms_simple.csv")
	print("- verb_classification_summary.txt")
