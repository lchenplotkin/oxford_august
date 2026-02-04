import os
import re
import string
import csv
import pandas as pd
from collections import defaultdict, Counter
import matplotlib.pyplot as plt



def clean_tag(tag):
	"""Remove digits before % in a tag (e.g., v2%pt_1 -> v%pt_1)."""
	if tag.startswith('ger') and '%' not in tag:
		return 'ger'
	return re.sub(r'\d+(?=%)', '', tag)

def parse_key_value_string(input_string):
    """
    Parses a string of format 'key1(value1), key2(value2), ...'
    into a dictionary with string keys and integer values.
    """
    pattern = r'([^,\(]+)\((\d+)\)'
    return {key.strip(): int(value) for key, value in re.findall(pattern, input_string)}

def parse_tagged_text(oxford_text, oxford_tagging):
	"""Extract words from oxford_text and tags from oxford_tagging"""
	if pd.isna(oxford_text) or pd.isna(oxford_tagging) or oxford_text == '' or oxford_tagging == '':
		return [], [], []

	# Clean and split the text
	words = oxford_text.lower().replace(',', '').replace('.', '').replace('!', '').replace('?', '').replace('°', '').replace('¶', '').strip().split()

	# Parse tags from oxford_tagging
	tag_tokens = oxford_tagging.strip().split()
	tags = []
	headwords = []

	for token in tag_tokens:
		if '@' in token:
			if token not in ["--@dash", ".@ellipsis"]:
				parts = token.split('@')
				headword = parts[0].lower()
				tag_part = parts[1] if len(parts) > 1 else ''
				#tag_part = re.sub('r\d','',tag_part)
				#tag_part = tag_part.replace('_','')

				tag = tag_part

				headwords.append(headword)
				tags.append(tag)

	# Ensure all lists are the same length (trim to shortest)
	if len(words) != len(tags):
		print(oxford_text, oxford_tagging, words, tags)
		print(len(words), len(tags))
	min_len = min(len(words), len(tags), len(headwords))
	return words[:min_len], headwords[:min_len], tags[:min_len]

#import docx

# Initialize Word document
#doc = docx.Document()
#doc.add_heading('Verb Rules Analysis in the Oxford Chaucer', 0)
#
# Configuration
base_csv_dir = 'for_gui/done'
verb_list_file = 'verb_forms_gold.csv'
verb_list_file = 'fixed_with_headwords.csv'
verb_df = pd.read_csv(verb_list_file, encoding='utf-8')
#known_headwords = list(verb_df['headword'])

print(verb_df)
verb_dict = {}
for root, dirs, files in os.walk(base_csv_dir):
	for file in files:
		if not file.endswith('complete.csv'):
			continue

		csv_path = os.path.join(root, file)
		rel_path = os.path.relpath(root, base_csv_dir)
		file_id = os.path.join(rel_path, file) if rel_path != '.' else file

		df = pd.read_csv(csv_path, encoding='utf-8')

		for idx, row in df.iterrows():
			oxford_text = row['OXFORD_TEXT']
			oxford_tagging = row['OXFORD_TAGGING']
			line_number = row['LINE_NUMBER']
			filename = row['OXFORD_FILENAME']

			words, headwords, tags = parse_tagged_text(oxford_text, oxford_tagging)

			for j, (word, headword, tag) in enumerate(zip(words, headwords, tags)):
				tag = clean_tag(tag)
				if headword not in verb_dict.keys() and tag in ["v%pt_1","v%pt_3","v%ppl"]:
					verb_dict[headword] = [{},{},{},'','']
					if headword in verb_df.keys():
						verb_dict[headword][4] = verb_df[headword]['classification']
						verb_dict[headword][5] = verb_df[headword]['notes']
		
				#if tag == "v%inf":
				#	if word not in verb_dict[headword][0].keys():
				#		verb_dict[headword][0][word] = 0
				#	verb_dict[headword][0][word] += 1
				
				if tag in ["v%pt_1","v%pt_3"]:
					if word not in verb_dict[headword][1].keys():
						verb_dict[headword][1][word] = 0
					verb_dict[headword][1][word]+=1
				if tag in ["v%ppl"]:
					if word not in verb_dict[headword][2].keys():
						verb_dict[headword][2][word] = 0
					verb_dict[headword][2][word] += 1
							

for headword in verb_dict:
	row = verb_dict[headword]
	if row[3]=='': #(we do not already know classification)
		weak_evidence = 0
		strong_evidence = 0
		prets = row[1]
		parts = row[2]

		for pret in prets.keys():
			if pret.endswith("de") or pret.endswith("te") or pret.endswith("d") or pret.endswith("t"):
				weak_evidence += 3*prets[pret]
			else:
				strong_evidence+=prets[pret]
		for part in parts.keys():
			if part.endswith('e') or part.endswith('en'):
				strong_evidence += 3*parts[part]
			elif part.endswith('d') or part.endswith('t'):
				weak_evidence += 2*parts[part] 
			else:
				weak_evidence += parts[part]
		
		#print(row)
		#print(strong_evidence, weak_evidence)
		
		if strong_evidence == 0:
			verb_dict[headword][3] = 'weak'
		elif weak_evidence == 0:
			verb_dict[headword][3] = 'strong'
		else:
			verb_dict[headword][3] = 'TODO'
		verb_dict[headword][4] = 'classified by computer (CHECK)'
print(verb_dict)	
